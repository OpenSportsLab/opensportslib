# models/backbones/video_mae.py

from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
import os

def build_video_mae_backbone(config, device, ckpt_path=None, infer=False):
    """
    Build HuggingFace VideoMAE model for video classification.
    This includes both backbone and classification head.
    """
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

    # freeze everything 
    for param in model.parameters():
        param.requires_grad = False
    
    if not infer:
        # Unfreeze classification head 
        if config.MODEL.unfreeze_head:
            for p in model.classifier.parameters():
                p.requires_grad = True

        # -------- Unfreeze last N VideoMAE encoder layers --------
        n_unfreeze = getattr(config.MODEL, "unfreeze_last_n_layers", 0)
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
