# opensportslib/models/base/video.py

"""video backbone and model for frames_npy modality.

this file contains two independent things:

1. the existing VideoMAE HuggingFace full-model builder functions
   (build_video_mae_backbone, load_video_mae_checkpoint).
   these are left exactly as they were and route through MODEL.type == "huggingface".

2. the new VideoBackbone + VideoModel classes for the custom frames_npy path,
   supporting dinov3, clip, videomae, videomae2 as pure feature extractors
   wired to the library's existing TemporalAggregation neck and
   TrackingClassifierHead head.
"""

import torch
import torch.nn as nn

from opensportslib.models.backbones.builder import build_backbone
from opensportslib.models.neck.builder import build_neck
from opensportslib.models.heads.builder import build_head


# -----------------------------------------------------------------------
# video mae backbone for MVFoul
# -----------------------------------------------------------------------

def build_video_mae_backbone(config, device, ckpt_path=None, infer=False):
    """
    Build HuggingFace VideoMAE model for video classification.
    This includes both backbone and classification head.
    """
    from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

    num_classes = config.MODEL.num_classes
    pretrained_model_name = ckpt_path if ckpt_path else config.MODEL.pretrained_model
    processor = VideoMAEImageProcessor.from_pretrained(config.MODEL.pretrained_model)
    model = VideoMAEForVideoClassification.from_pretrained(
        pretrained_model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
        trust_remote_code=True,
        device_map=device
    )

    for param in model.parameters():
        param.requires_grad = False

    if not infer:
        if config.MODEL.unfreeze_head:
            for p in model.classifier.parameters():
                p.requires_grad = True

        n_unfreeze = getattr(config.MODEL, "unfreeze_last_n_layers", 0)
        if n_unfreeze > 0:
            for layer in model.videomae.encoder.layer[-n_unfreeze:]:
                for p in layer.parameters():
                    p.requires_grad = True

    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    print("Number of trainable params:", len(trainable))
    for n in trainable:
        print(n)

    return model, processor


def load_video_mae_checkpoint(config, device, ckpt_path, infer=True):
    """
    Load fine-tuned VideoMAE checkpoint from a HuggingFace-style directory.
    """
    return build_video_mae_backbone(config, device, ckpt_path, infer=infer)


# -----------------------------------------------------------------------
# new custom path: full model
# -----------------------------------------------------------------------

class VideoModel(nn.Module):
    """Video classification model for the frames_npy modality.

    follows the same backbone -> neck -> head pattern as TrackingModel.

    the backbone is a VideoBackbone (pure feature extractor).
    the neck is TemporalAggregation.
    the head is TrackingClassifierHead.

    Args:
        config: full YAML config
        device: torch device string
    """

    def __init__(self, config, device):
        super().__init__()
        print("Building VideoModel")

        self.device = device
        self.num_frames = config.DATA.num_frames

        # backbone: pure feature extractor
        self.backbone = build_backbone(config.MODEL.backbone)

        # neck: temporal aggregation over the frame sequence
        self.neck = build_neck(
            config.MODEL.neck,
            default_args={"window_size": self.num_frames}
        )

        # head: linear classifier
        self.head = build_head(
            config.MODEL.head,
            default_args={"input_dim": self.neck.feat_dim}
        )

    def forward(self, batch):
        """
        Args:
            batch: dict with key "pixel_values" of shape (B, T, H, W, C).

        Returns:
            logits: (B, num_classes)
        """
        x = batch["pixel_values"]           # (B, T, H, W, C)

        x = self.backbone(x)                # (B, T, hidden_dim) or (B, 1, hidden_dim)
        x = self.neck(x)                    # (B, hidden_dim)
        logits = self.head(x)               # (B, num_classes)

        return logits
