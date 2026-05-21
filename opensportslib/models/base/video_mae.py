# models/backbones/video_mae.py

from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
import os
from opensportslib.core.config.accessors import (
    get_component_by_kind,
    get_component_params_by_kind,
    get_data_num_classes,
)

def build_video_mae_backbone(config, device, ckpt_path=None, infer=False):
    """
    Build HuggingFace VideoMAE model for video classification.
    This includes both backbone and classification head.
    """
    encoder_cfg = get_component_by_kind(config, "encoder") or {}
    encoder_source = encoder_cfg.get("source", {})
    encoder_params = get_component_params_by_kind(config, "encoder")
    head_params = get_component_params_by_kind(config, "head")

    num_classes = head_params.get("num_classes", get_data_num_classes(config))
    pretrained_ref = (
        encoder_params.get("pretrained_model")
        or encoder_source.get("repo_id")
        or encoder_source.get("name")
    )
    pretrained_model_name = ckpt_path if ckpt_path else pretrained_ref
    processor = VideoMAEImageProcessor.from_pretrained(pretrained_ref)
    model = VideoMAEForVideoClassification.from_pretrained(
        pretrained_model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
        trust_remote_code=True,
        device_map=device
    )

    # freeze everything 
    for param in model.parameters():
        param.requires_grad = False
    
    if not infer:
        # Unfreeze classification head 
        if encoder_params.get("unfreeze_head", False):
            for p in model.classifier.parameters():
                p.requires_grad = True

        # -------- Unfreeze last N VideoMAE encoder layers --------
        n_unfreeze = encoder_params.get("unfreeze_last_n_layers", 0)
        # unfreeze last encoder layer
        if n_unfreeze > 0:
            for layer in model.videomae.encoder.layer[-n_unfreeze:]:
                for p in layer.parameters():
                    p.requires_grad = True

    trainable = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            trainable.append(name)

    print("Number of trainable params:", len(trainable))
    for n in trainable:
        print(n)
    return model, processor


def load_video_mae_checkpoint(config, device, ckpt_path, infer=True):
    """
    Load fine-tuned VideoMAE checkpoint from a HuggingFace-style directory.

    Supports:
      - model.safetensors
      - pytorch_model.bin
      - config.json
    """
    return build_video_mae_backbone(config, device, ckpt_path, infer=infer)
