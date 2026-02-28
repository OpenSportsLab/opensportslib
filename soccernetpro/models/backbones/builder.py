"""
Copyright 2022 James Hong, Haotian Zhang, Matthew Fisher, Michael Gharbi,
Kayvon Fatahalian

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import torch
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


from soccernetpro.models.utils.shift import make_temporal_shift


def build_backbone(cfg, default_args=None):
    """Build a backbone from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        default_args (dict | None, optional): Default initialization arguments.
            Default: None.

    Returns:
        backbone: The constructed backbone.
    """
    
    if cfg.type == "graph_conv":
        backbone = GraphEncoder(
            input_dim=default_args["input_dim"],
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            conv_type=cfg.encoder,
            dropout=cfg.dropout,
        )
    elif cfg.type == "PreExtactedFeatures":
        backbone = PreExtactedFeatures(
            feature_dim=cfg.feature_dim, output_dim=cfg.output_dim
        )
    elif cfg.type in ["rn18", "rn18_tsm", "rn18_gsm", "rn50", "rn50_tsm", "rn50_gsm"]:
        backbone = ResnetExtractFeatures(
            cfg.type, cfg.clip_len, cfg.is_rgb, cfg.in_channels
        )
    elif cfg.type in [
        "rny002",
        "rny002_tsm",
        "rny002_gsm",
        "rny008",
        "rny008_tsm",
        "rny008_gsm",
    ]:
        backbone = RegnetyExtractFeatures(
            cfg.type, cfg.clip_len, cfg.is_rgb, cfg.in_channels
        )
    elif cfg.type in ["convnextt", "convnextt_tsm", "convnextt_gsm"]:
        backbone = ConvNextTinyExtractFeatures(
            cfg.type, cfg.clip_len, cfg.is_rgb, cfg.in_channels
        )
    elif cfg.type in ["r3d_18", "mc3_18", "r2plus1d_18", "s3d", "mvit_v2_s"]:
        backbone = TorchvisionVideoExtractFeatures(cfg.type)
    elif cfg.type in ["dinov3", "clip", "videomae", "videomae2"]:
        backbone = VideoBackbone(cfg)
    else:
        backbone = None

    return backbone


def Add_Temporal_Shift_Modules(
    feature_arch,
    features,
    clip_len,
):
    """Add temporal shift modules to a model.

    Args:
        feature_arch (string): name of the feature.
        features: The model.
        clip_len (int): The length of the clip.

    Returns:
        require_clip_len (int): The required length of clip.
    """
    require_clip_len = -1
    if feature_arch.endswith("_tsm"):
        make_temporal_shift(features, clip_len, is_gsm=False)
        require_clip_len = clip_len
    elif feature_arch.endswith("_gsm"):
        make_temporal_shift(features, clip_len, is_gsm=True)
        require_clip_len = clip_len
    return require_clip_len


class BaseExtractFeatures(nn.Module):
    """Base parent class for feature extractor model used by the E2E method.
    They all share the same forward method.
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        batch_size, true_clip_len, channels, height, width = inputs.shape

        clip_len = true_clip_len
        if self._require_clip_len > 0:
            assert (
                true_clip_len <= self._require_clip_len
            ), "Expected {}, got {}".format(self._require_clip_len, true_clip_len)
            if true_clip_len < self._require_clip_len:
                inputs = F.pad(
                    inputs, (0,) * 7 + (self._require_clip_len - true_clip_len,)
                )
                clip_len = self._require_clip_len

        im_feat = self._features(inputs.view(-1, channels, height, width)).reshape(
            batch_size, clip_len, self._feat_dim
        )

        if true_clip_len != clip_len:
            im_feat = im_feat[:, :true_clip_len, :]

        return im_feat


class ConvNextTinyExtractFeatures(BaseExtractFeatures):
    """Feature extractor which is based on the "convnext_tiny" of the timm models.
    The model is adapted for this task by adding temporal shift modules.

    Args:
        feature_arch (string): Feature extractor architecture.
        clip_len (int): Length of the clips.
        is_rgb (bool): Whether images are rgb or not.
        in_channels (int): Number of channels of images.
    """
    def __init__(self, feature_arch, clip_len, is_rgb, in_channels):
        super().__init__()
        import timm
        features = timm.create_model("convnext_tiny", pretrained=is_rgb)
        feat_dim = features.head.fc.in_features
        features.head.fc = nn.Identity()

        if not is_rgb:
            features.stem[0] = nn.Conv2d(in_channels, 96, kernel_size=4, stride=4)

        # Add Temporal Shift Modules
        self._require_clip_len = Add_Temporal_Shift_Modules(
            feature_arch, features, clip_len
        )

        self._features = features
        self._feat_dim = feat_dim


class RegnetyExtractFeatures(BaseExtractFeatures):
    """Feature extractor which is based on the "regnet" models of the timm models.
    The model is adapted for this task by adding temporal shift modules.

    Args:
        feature_arch (string): Feature extractor architecture.
        clip_len (int): Length of the clips.
        is_rgb (bool): Whether images are rgb or not.
        in_channels (int): Number of channels of images.
    """

    def __init__(self, feature_arch, clip_len, is_rgb, in_channels):
        super().__init__()
        import timm
        features = timm.create_model(
            {
                "rny002": "regnety_002",
                "rny008": "regnety_008",
            }[feature_arch.rsplit("_", 1)[0]],
            pretrained=is_rgb,
        )
        feat_dim = features.head.fc.in_features
        features.head.fc = nn.Identity()
        if not is_rgb:
            features.stem.conv = nn.Conv2d(
                in_channels,
                32,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                bias=False,
            )

        # Add Temporal Shift Modules
        self._require_clip_len = Add_Temporal_Shift_Modules(
            feature_arch, features, clip_len
        )

        self._features = features
        self._feat_dim = feat_dim


class ResnetExtractFeatures(nn.Module):
    """Feature extractor which is based on the "resnet" models of the torchvision models.
    The model is adapted for this task by adding temporal shift modules.

    Args:
        feature_arch (string): Feature extractor architecture.
        clip_len (int): Length of the clips.
        is_rgb (bool): Whether images are rgb or not.
        in_channels (int): Number of channels of images.
    """

    def __init__(self, feature_arch, clip_len, is_rgb, in_channels):
        super().__init__()

        resnet_name = feature_arch.split("_")[0].replace("rn", "resnet")
        features = getattr(torchvision.models, resnet_name)(pretrained=is_rgb)
        feat_dim = features.fc.in_features
        features.fc = nn.Identity()
        # import torchsummary
        # print(torchsummary.summary(features.to('cuda'), (3, 224, 224)))

        # Flow has only two input channels
        if not is_rgb:
            # FIXME: args maybe wrong for larger resnet
            features.conv1 = nn.Conv2d(
                in_channels,
                64,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
            )
        # Add Temporal Shift Modules
        self._require_clip_len = Add_Temporal_Shift_Modules(
            feature_arch, features, clip_len
        )

        self._features = features
        self._feat_dim = feat_dim


class PreExtactedFeatures(torch.nn.Module):
    """Class used when processing features directly. This class is used to adapt dimensions of the inputs if they do not correspond between layers."""

    def __init__(self, feature_dim, output_dim):
        super(PreExtactedFeatures, self).__init__()
        self.feature_dim = feature_dim
        self.output_dim = output_dim

        self.reduceDim = not self.feature_dim == self.output_dim
        if self.reduceDim:
            self.feature_extractor = torch.nn.Linear(self.feature_dim, self.output_dim)

    def forward(self, inputs):
        if len(inputs.shape) == 4:
            BS, D2, FR, IC = inputs.shape
            if self.reduceDim:
                inputs = inputs.reshape(BS * FR * D2, IC)
                inputs = self.feature_extractor(inputs)
                inputs = inputs.reshape(BS, D2, FR, -1)
        else:
            BS, FR, IC = inputs.shape
            if self.reduceDim:
                inputs = inputs.reshape(BS * FR, IC)
                inputs = self.feature_extractor(inputs)
                inputs = inputs.reshape(BS, FR, -1)
        return inputs


class TorchvisionVideoExtractFeatures(nn.Module):

    def __init__(self, name):
        super().__init__()
        from torchvision.models.video import r3d_18, R3D_18_Weights, MC3_18_Weights, mc3_18
        from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights, s3d, S3D_Weights
        from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights, mvit_v1_b, MViT_V1_B_Weights

        self.name = name
        print("Building Torchvision Video Backbone:", name)
        if name == "r3d_18":
            weights = R3D_18_Weights.DEFAULT
            model = r3d_18(weights=weights)
            self.feat_dim = 512
        elif name == "mc3_18":
            weights = MC3_18_Weights.DEFAULT
            model = mc3_18(weights=weights)
            self.feat_dim = 512
        elif name == "r2plus1d_18":
            weights = R2Plus1D_18_Weights.DEFAULT
            model = r2plus1d_18(weights=weights)
            self.feat_dim = 512
        elif name == "s3d":
            weights = S3D_Weights.DEFAULT
            model = s3d(weights=weights)
            self.feat_dim = 400
        elif name == "mvit_v2_s":
            weights = MViT_V2_S_Weights.DEFAULT
            model = mvit_v2_s(weights=weights)
            self.feat_dim = 400
        else:
            raise ValueError(f"Unknown backbone {name}")

        # Remove classification head → feature extractor
        model.fc = nn.Sequential()
        self.model = model

    def forward(self, x):
        # x: (B, C, T, H, W) or (C, T, H, W)
        if x.dim() == 4:
            x = x.unsqueeze(0)
        return self.model(x)


class GraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, conv_type='gin', dropout=0.1):
        super().__init__()

        from torch_geometric.nn import DeepGCNLayer
        self.conv_type = conv_type
        self.feat_dim = hidden_dim

        # initial projection for node features
        self.node_encoder = nn.Linear(input_dim, hidden_dim)

        # build GCN layers based on conv_type
        self.gcn_layers = nn.ModuleList()
        for _ in range(num_layers):
            conv = self._build_conv_layer(conv_type, hidden_dim, dropout)
            layer = DeepGCNLayer(
                conv=conv,
                norm=nn.LayerNorm(hidden_dim),
                act=nn.ReLU(inplace=True),
                block='res+',
                dropout=dropout
            )
            self.gcn_layers.append(layer)
        
    def _build_conv_layer(self, conv_type, hidden_dim, dropout):
        from torch_geometric.nn import (
            GATv2Conv, EdgeConv, SAGEConv, 
            GINConv, GENConv, GraphConv, MultiAggregation
        )

        if conv_type == 'graphconv':
            return GraphConv(
                hidden_dim, hidden_dim,
                aggr='add', bias=True,
            )

        elif conv_type == 'gat':
            return GATv2Conv(
                hidden_dim, hidden_dim // 4, heads=4,
                concat=True, dropout=dropout,
                add_self_loops=True, edge_dim=None,
                fill_value='mean', bias=True, share_weights=False
            )

        elif conv_type == 'edgeconv':
            return EdgeConv(
                nn=nn.Sequential(
                    nn.Linear(2 * hidden_dim, hidden_dim * 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU()
                ),
                aggr='max'
            )
            
        elif conv_type == 'sageconv':
            return SAGEConv(
                hidden_dim, hidden_dim,
                aggr=MultiAggregation(['mean', 'max', 'std']),
                normalize=False, project=True, bias=True
            )

        elif conv_type == 'gin':
            return GINConv(
                nn=nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                ),
                train_eps=True, eps=0.0
            )

        elif conv_type == 'gen':
            return GENConv(
                hidden_dim, hidden_dim,
                aggr='softmax', t=1.0, learn_t=True,
                p=1.0, learn_p=True, msg_norm=True,
                learn_msg_scale=True, norm='layer',
                num_layers=2, eps=1e-7
            )

        else:
            raise ValueError(f"Unknown conv type: {conv_type}")
            
    def forward(self, x, edge_index, batch):
        from torch_geometric.nn import global_mean_pool
        
        x = self.node_encoder(x)

        for layer in self.gcn_layers:
            if self.conv_type == 'edgeconv':
                edge_index = self._compute_edge_dynamic(x, batch)
            x = layer(x, edge_index)

        graph_embedding = global_mean_pool(x, batch)
        return graph_embedding

    def _compute_edge_dynamic(self, x, batch, k=8):
        """Compute dynamic KNN edges for EdgeConv layers."""
        from torch_geometric.nn import knn_graph
        edge_index = knn_graph(x, k=k, batch=batch, loop=False)
        return edge_index

        
# -----------------------------------------------------------------------
# new custom path: pure feature extractor backbone
# -----------------------------------------------------------------------

class VideoBackbone(nn.Module):
    """pure feature extractor for the custom frames_npy path.

    supports four backbone types:
        - dinov3:   DINOv2 ViT, processes frames independently, CLS token output.
        - clip:     CLIP ViT vision encoder, processes frames independently.
        - videomae: VideoMAE (MCG-NJU/videomae-base), processes full clip.  
        - videomae2: VideoMAEv2 (OpenGVLab/VideoMAEv2-Base), processes full clip.

    Args:
        cfg: MODEL.backbone config node. required fields:
            type (str): one of VIDEO_BACKBONE_TYPES.
            pretrained_model (str): HuggingFace model ID.
            hidden_dim (int): expected output feature dimension.
            freeze (bool): if True, freeze all backbone weights first.
            unfreeze_last_n_layers (int): number of encoder layers to thaw
                after freezing. 0 means fully frozen backbone.
    """

    def __init__(self, cfg):
        super().__init__()

        self.backbone_type = cfg.type
        self.feat_dim = cfg.hidden_dim

        pretrained = cfg.pretrained_model
        print(f"Building VideoBackbone: {cfg.type} from {pretrained}")

        self.model = self._build_model(pretrained)
        self._apply_freeze(cfg)
        self._fully_frozen =  (
            getattr(cfg, "freeze", True)
            and getattr(cfg, "unfreeze_last_n_layers", 0) == 0
        )
        self._log_trainable()

    def _build_model(self, pretrained):
        if self.backbone_type == "dinov3":
            from transformers import AutoModel
            return AutoModel.from_pretrained(pretrained)

        elif self.backbone_type == "clip":
            from transformers import CLIPVisionModel
            return CLIPVisionModel.from_pretrained(pretrained)

        elif self.backbone_type == "videomae":
            from transformers import VideoMAEModel
            return VideoMAEModel.from_pretrained(
                pretrained,
                ignore_mismatched_sizes=True,
            )

        elif self.backbone_type == "videomae2":
            from transformers import AutoModel, AutoConfig
            config = AutoConfig.from_pretrained(pretrained, trust_remote_code=True)
            return AutoModel.from_pretrained(
                pretrained,
                config=config,
                trust_remote_code=True,
                use_safetensors=True,
            )
        else:
            raise ValueError(f"Unknown VideoBackbone type: {self.backbone_type}")

    def _get_encoder_layers(self):
        """return the list of encoder layer modules for selective unfreezing."""
        if self.backbone_type == "dinov3":
            return list(self.model.encoder.layer)
        elif self.backbone_type == "clip":
            return list(self.model.vision_model.encoder.layers)
        elif self.backbone_type == "videomae":
            return list(self.model.encoder.layer)

        elif self.backbone_type == "videomae2":
            # OpenGVLab custom model; inspect structure at runtime
            if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "layer"):
                return list(self.model.encoder.layer)
            elif hasattr(self.model, "blocks"):
                return list(self.model.blocks)
            return []

    def _apply_freeze(self, cfg):
        freeze = getattr(cfg, "freeze", True)
        if not freeze:
            # full finetuning: leave all params trainable
            return

        for param in self.model.parameters():
            param.requires_grad = False

        n = getattr(cfg, "unfreeze_last_n_layers", 0)
        if n == 0:
            return

        layers = self._get_encoder_layers()
        if not layers:
            return

        for layer in layers[-n:]:
            for param in layer.parameters():
                param.requires_grad = True

    def _log_trainable(self):
        trainable = [n for n, p in self.model.named_parameters() if p.requires_grad]
        print(f"VideoBackbone trainable params: {len(trainable)}")
        for n in trainable:
            print(" ", n)

    def forward(self, x):
        """
        Args:
            x: (B, T, H, W, C) float32, imagenet-normalized.

        Returns:
            (B, T, hidden_dim) for image backbones
            (B, 1, hidden_dim) for video backbones
        """
        B, T, H, W, C = x.shape

        with torch.set_grad_enabled(self.training and not self._fully_frozen):
                
            if self.backbone_type == "dinov3":
                # flatten time into batch, permute to (B*T, C, H, W)
                x_flat = x.view(B * T, H, W, C).permute(0, 3, 1, 2)
                out = self.model(pixel_values=x_flat)
                # CLS token is index 0 of last_hidden_state
                feat = out.last_hidden_state[:, 0, :]        # (B*T, hidden_dim)
                return feat.view(B, T, -1)                    # (B, T, hidden_dim)

            elif self.backbone_type == "clip":
                x_flat = x.view(B * T, H, W, C).permute(0, 3, 1, 2)
                out = self.model(pixel_values=x_flat)
                feat = out.pooler_output                      # (B*T, hidden_dim)
                return feat.view(B, T, -1)                    # (B, T, hidden_dim)

            elif self.backbone_type == "videomae":
                x_vid = x.permute(0, 1, 4, 2, 3)             # (B, T, C, H, W)
                out = self.model(pixel_values=x_vid)
                feat = out.last_hidden_state.mean(dim=1)      # (B, hidden_dim)
                return feat.unsqueeze(1)                      # (B, 1, hidden_dim)

            elif self.backbone_type == "videomae2":
                x_vid = x.permute(0, 4, 1, 2, 3)             # (B, C, T, H, W)
                feat = self.model.extract_features(pixel_values=x_vid)  # (B, hidden_dim)
                return feat.unsqueeze(1)                      # (B, 1, hidden_dim)

