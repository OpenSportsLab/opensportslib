# Adapted from arturxe2/T-DEED, commit d953b4de8a2ad941c565e1683c5b07255cbe644e.
# SPDX-License-Identifier: GPL-3.0-only
# See LICENSE and LICENSE-E2E-Spot in this directory for upstream terms.
import torch
from torch import nn
import timm
import torchvision.transforms as T
from .modules import EDSGPMIXERLayers,FCLayers,FC2Layers
from .shift import make_temporal_shift

class TDEEDNetwork(nn.Module):

    def __init__(self, args = None):
        super().__init__()
        self._modality = args.modality
        assert self._modality == 'rgb', 'Only RGB supported for now'
        in_channels = {'rgb': 3}[self._modality]
        self._temp_arch = args.temporal_arch
        self._radi_displacement = args.radi_displacement
        self._feature_arch = args.feature_arch
        assert 'rny' in self._feature_arch, 'Only rny supported for now'
        self._double_head = False

        if self._feature_arch.startswith(('rny002', 'rny008')):
            pretrained = getattr(args, "pretrained", True)
            checkpoint = None
            if pretrained and getattr(args, 'pretrained_revision', None):
                from huggingface_hub import hf_hub_download
                checkpoint = hf_hub_download(args.pretrained_model_id, 'model.safetensors',
                                             revision=args.pretrained_revision)
            features = timm.create_model({
                'rny002': 'regnety_002',
                'rny008': 'regnety_008',
            }[self._feature_arch.rsplit('_', 1)[0]], pretrained=pretrained and checkpoint is None,
                **({'checkpoint_path': checkpoint} if checkpoint else {}))
            feat_dim = features.head.fc.in_features

            # Remove final classification layer
            features.head.fc = nn.Identity()
            self._d = feat_dim

        else:
            raise NotImplementedError(args._feature_arch)

        # Add Temporal Shift Modules
        self._require_clip_len = -1
        if self._feature_arch.endswith('_gsm'):
            make_temporal_shift(features, args.clip_len, mode='gsm', stages=getattr(args, 'shift_scope', 'half'))
            self._require_clip_len = args.clip_len
        elif self._feature_arch.endswith('_gsf'):
            make_temporal_shift(features, args.clip_len, mode='gsf', stages=getattr(args, 'shift_scope', 'half'))
            self._require_clip_len = args.clip_len

        self._features = features
        self._feat_dim = self._d
        feat_dim = self._d

        #Positional encoding
        self.temp_enc = nn.Parameter(torch.normal(mean = 0, std = 1 / args.clip_len, size = (args.clip_len, self._d)))

        if self._temp_arch == 'ed_sgp_mixer':
            self._temp_fine = EDSGPMIXERLayers(feat_dim, args.clip_len, num_layers=args.n_layers, ks = args.sgp_ks, k = args.sgp_r, concat = True)
            self._pred_fine = FCLayers(self._feat_dim, args.num_classes+1)
        else:
            from .ablations import build_temporal
            self._temp_fine = build_temporal(self._temp_arch, feat_dim, args.clip_len,
                args.n_layers, args.sgp_ks, args.sgp_r)
            self._pred_fine = FCLayers(self._feat_dim, args.num_classes+1)

        if self._radi_displacement > 0:
            self._pred_displ = FCLayers(self._feat_dim, 1)

        #Augmentations and crop
        self.augmentation = T.Compose([
            T.RandomApply([T.ColorJitter(hue = 0.2)], p = 0.25),
            T.RandomApply([T.ColorJitter(saturation = (0.7, 1.2))], p = 0.25),
            T.RandomApply([T.ColorJitter(brightness = (0.7, 1.2))], p = 0.25),
            T.RandomApply([T.ColorJitter(contrast = (0.7, 1.2))], p = 0.25),
            T.RandomApply([T.GaussianBlur(5)], p = 0.25),
            T.RandomHorizontalFlip(),
        ])

        #Standarization
        self.standarization = T.Compose([
            T.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)) #Imagenet mean and std
        ])

        #Augmentation at test time
        self.augmentationI = T.Compose([
            T.RandomHorizontalFlip(p = 1.0)
        ])

        #Croping in case of using it
        self.croping = args.crop_dim
        if self.croping != None:
            self.cropT = T.RandomCrop((self.croping, self.croping))
            self.cropI = T.CenterCrop((self.croping, self.croping))
        else:
            self.cropT = torch.nn.Identity()
            self.cropI = torch.nn.Identity()

    def forward(self, x, y = None, inference=False, augment_inference=False):
        if x.ndim != 5 or x.shape[1] != self.temp_enc.shape[0] or x.shape[2] != 3:
            raise ValueError('T-DEED expects fixed-length (batch, clip_len, 3, height, width) RGB clips')
        x = self.normalize(x) #Normalize to 0-1
        batch_size, true_clip_len, channels, height, width = x.shape

        if not inference:
            x.view(-1, channels, height, width)
            if self.croping != None:
                height = self.croping
                width = self.croping
            x = self.cropT(x) #same crop for all frames
            x = x.view(batch_size, true_clip_len, channels, height, width)
            x = self.augment(x) #augmentation per-batch
            x = self.standarize(x) #standarization imagenet stats

        else:
            x = x.view(-1, channels, height, width)
            if self.croping != None:
                height = self.croping
                width = self.croping
            x = self.cropI(x) #same center crop for all frames
            x = x.view(batch_size, true_clip_len, channels, height, width)
            if augment_inference:
                x = self.augmentI(x)
            x = self.standarize(x)

        clip_len = true_clip_len

        im_feat = self._features(
            x.view(-1, channels, height, width)
        ).reshape(batch_size, clip_len, self._d)

        im_feat = im_feat + self.temp_enc.expand(batch_size, -1, -1)

        if hasattr(self, '_temp_fine'):
            im_feat = self._temp_fine(im_feat)
            if self._radi_displacement > 0:
                displ_feat = self._pred_displ(im_feat).squeeze(-1)
                im_feat = self._pred_fine(im_feat)
                return {'im_feat': im_feat, 'displ_feat': displ_feat}, y
            im_feat = self._pred_fine(im_feat)
            return im_feat, y

        else:
            raise NotImplementedError(self._temp_arch)

    def normalize(self, x):
        return x / 255.

    def augment(self, x):
        for i in range(x.shape[0]):
            x[i] = self.augmentation(x[i])
        return x

    def augmentI(self, x):
        for i in range(x.shape[0]):
            x[i] = self.augmentationI(x[i])
        return x

    def standarize(self, x):
        for i in range(x.shape[0]):
            x[i] = self.standarization(x[i])
        return x

    def update_pred_head(self, num_classes = [1, 1]):
        self._pred_fine = FC2Layers(self._feat_dim, num_classes)
        self._pred_fine = self._pred_fine.cuda()
        self._double_head = True

    def print_stats(self):
        print('Model params:',
            sum(p.numel() for p in self.parameters()))
        print('  CNN features:',
            sum(p.numel() for p in self._features.parameters()))
        print('  Temporal:',
            sum(p.numel() for p in self._temp_fine.parameters()))
        print('  Head:',
            sum(p.numel() for p in self._pred_fine.parameters()))
