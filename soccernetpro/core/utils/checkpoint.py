# soccernetpro/core/utils/checkpoint.py

import torch
import os



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


def load_checkpoint(model, path, optimizer=None, device=None):
    """
    Load model checkpoint (.pt or .pt.tar) safely.
    Supports:
      - checkpoint["model_state_dict"]
      - checkpoint["state_dict"]
      - raw state_dict
      - DDP "module." prefixes
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(path, map_location=device)

    # Detect model state dict format
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        # Fallback: assume checkpoint IS a state_dict
        state_dict = checkpoint

    # Remove "module." prefix if the model was trained with DDP
    clean_state_dict = {
        k.replace("module.", ""): v for k, v in state_dict.items()
    }

    # Load weights
    model.load_state_dict(clean_state_dict, strict=False)
    model.to(device)

    # Extract epoch
    epoch = checkpoint.get("epoch", None)

    # Load optimizer if requested
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"[Checkpoint] Loaded from {path} | epoch: {epoch}")
    return model, optimizer, epoch


def load_huggingface_checkpoint(config, path, device):
    from soccernetpro.models.base.video_mae import load_video_mae_checkpoint
    return load_video_mae_checkpoint(config, device=device, ckpt_path=path)

