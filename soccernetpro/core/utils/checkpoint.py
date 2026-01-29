# soccernetpro/core/utils/checkpoint.py

import torch
import os

def localization_remap(key):
    if key.startswith("_features"):
        return "backbone." + key
    elif key.startswith("_pred_fine"):
        return "head." + key
    return key


def save_checkpoint(model, path, processor=None, tokenizer=None, optimizer=None, epoch=None):
    """
    Save model checkpoint to `path`. Uses HF save_pretrained if available, 
    otherwise falls back to saving a PyTorch checkpoint.

    Args:
        model (torch.nn.Module or HF PreTrainedModel): model to save
        path (str): path to save torch checkpoint (file path, e.g., /.../checkpoint.pt)
        optimizer (torch.optim.Optimizer, optional): optimizer to save
        epoch (int, optional): current epoch number
        processor (optional): HF processor / feature extractor to save with model
        tokenizer (optional): HF tokenizer to save with model
    """
    os.makedirs(path, exist_ok=True)

    hf_saved = False

    # 1) Try to save HuggingFace model if available
    if hasattr(model, "save_pretrained"):
        try:
            model.save_pretrained(path)
            hf_saved = True
            print(f"[Checkpoint] HuggingFace model saved to {path}")
        except Exception as e:
            print(f"[Checkpoint] Warning: could not save HF model: {e}")
            hf_saved = False

    # 2) Save processor / tokenizer if provided (only if HF save succeeded)
    if hf_saved:
        if processor is not None:
            try:
                processor.save_pretrained(path)
                print(f"[Checkpoint] Processor saved to {path}")
            except Exception as e:
                print(f"[Checkpoint] Warning: could not save processor: {e}")

        if tokenizer is not None:
            try:
                tokenizer.save_pretrained(path)
                print(f"[Checkpoint] Tokenizer saved to {path}")
            except Exception as e:
                print(f"[Checkpoint] Warning: could not save tokenizer: {e}")

    # 3) Fallback: Save a PyTorch checkpoint if HF save is unavailable or failed
    if not hf_saved:
        checkpoint = {}
        if hasattr(model, "state_dict"):
            checkpoint["model_state_dict"] = model.state_dict()
        else:
            print("[Checkpoint] Warning: model has no state_dict(), skipping model_state_dict.")

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        if epoch is not None:
            checkpoint["epoch"] = epoch

        torch.save(checkpoint, path)
        print(f"[Checkpoint] Torch checkpoint saved at: {path}")


def load_checkpoint(model, path, optimizer=None, scheduler=None, device=None, key_remap_fn=None):
    """
    Load model checkpoint (.pt / .pth / .tar) safely.

    Supports:
      - checkpoint["model_state_dict"]
      - checkpoint["state_dict"]
      - raw state_dict
      - optimizer under:
            "optimizer", "optimizer_state_dict"
      - scheduler under:
            "scheduler", "scheduler_state_dict"
      - DDP "module." prefixes

    Returns:
        model, optimizer, scheduler, epoch
    """
    import torch

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # ---------------- MODEL STATE ----------------
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            # fallback: maybe it's already a state_dict
            state_dict = {
                k: v for k, v in checkpoint.items()
                if isinstance(v, torch.Tensor)
            }
    else:
        raise ValueError("Checkpoint format not recognized")

    # Remove DDP prefix
    clean_state_dict = {}
    for k, v in state_dict.items():
        # Remove DDP prefix
        k = k.replace("module.", "")

        # Optional remapping
        if key_remap_fn is not None:
            k = key_remap_fn(k)

        clean_state_dict[k] = v

    missing, unexpected = model.load_state_dict(clean_state_dict, strict=False)
    model.to(device)

    # ---------------- EPOCH ----------------
    epoch = None
    if isinstance(checkpoint, dict):
        epoch = checkpoint.get("epoch", None)

    # ---------------- OPTIMIZER ----------------
    if optimizer is not None and isinstance(checkpoint, dict):
        opt_state = checkpoint.get("optimizer") or checkpoint.get("optimizer_state_dict")
        if opt_state is not None:
            optimizer.load_state_dict(opt_state)

    # ---------------- SCHEDULER ----------------
    if scheduler is not None and isinstance(checkpoint, dict):
        sch_state = checkpoint.get("scheduler") or checkpoint.get("scheduler_state_dict")
        if sch_state is not None:
            scheduler.load_state_dict(sch_state)

    print(f"[Checkpoint] Loaded from {path} | epoch: {epoch}")
    print(f"Epoch: {epoch}")
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")

    return model, optimizer, scheduler, epoch


def load_huggingface_checkpoint(config, path, device):
    from soccernetpro.models.base.video_mae import load_video_mae_checkpoint
    return load_video_mae_checkpoint(config, device=device, ckpt_path=path)

