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
    os.makedirs(os.path.dirname(path), exist_ok=True)

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


def load_checkpoint(
    model,
    path,
    optimizer=None,
    scheduler=None,
    device=None,
    key_remap_fn=None,
    hf_filename="model.pth.tar",   # required if loading from HF repo
    hf_token=None,      # optional (for private repos / non-interactive envs)
):
    """
    Load checkpoint from:
      - local .pt/.pth/.tar
      - HuggingFace repo (repo_id)

    Auth behavior:
      - If logged in -> no token needed
      - If not logged in -> interactive prompt
      - If non-interactive -> hf_token required
    """

    import os
    import sys
    import torch
    from soccernetpro.core.utils.config import expand 

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = None
    hf_error = None

    # --------------------------------------------------
    # Try Hugging Face FIRST
    # --------------------------------------------------
    try:
        from huggingface_hub import hf_hub_download, whoami, login

        # Ensure auth if needed
        if hf_token is None:
            try:
                whoami()
            except Exception:
                if sys.stdin.isatty():
                    login()

        print(f"[HF] Trying HuggingFace repo: {path}")

        ckpt_path = hf_hub_download(
            repo_id=path,
            filename=hf_filename,
            token=hf_token,
        )

        print(f"[HF] Loaded from cache: {ckpt_path}")

    except Exception as e:
        hf_error = e

    # --------------------------------------------------
    # 2️⃣ Fallback to local
    # --------------------------------------------------
    path = expand(path)
    if ckpt_path is None:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Checkpoint not found on HuggingFace OR locally: {path}"
            ) from hf_error

        ckpt_path = path
        print(f"[Local] Using local checkpoint: {ckpt_path}")
    # --------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    # ---------------- MODEL STATE ----------------
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = {
                k: v for k, v in checkpoint.items()
                if isinstance(v, torch.Tensor)
            }
    else:
        raise ValueError("Checkpoint format not recognized")

    # Clean + remap keys
    model_keys = list(model.state_dict().keys())
    ckpt_keys  = list(state_dict.keys())

    ckpt_has_module  = ckpt_keys[0].startswith("module.")
    model_has_module = model_keys[0].startswith("module.")

    # Case 1: checkpoint has module., model doesn't
    if ckpt_has_module and not model_has_module:
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    # Case 2: checkpoint doesn't have module., model does
    elif not ckpt_has_module and model_has_module:
        state_dict = {f"module.{k}": v for k, v in state_dict.items()}

    # Optional custom remap
    if key_remap_fn:
        state_dict = {key_remap_fn(k): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("\n--- MISSING KEYS ---")
    for k in missing[:20]:
        print(k)

    print("\n--- UNEXPECTED KEYS ---")
    for k in unexpected[:20]:
        print(k)
    model.to(device)

    # ---------------- EPOCH ----------------
    epoch = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None

    # ---------------- OPTIMIZER ----------------
    if optimizer and isinstance(checkpoint, dict):
        opt_state = checkpoint.get("optimizer") or checkpoint.get("optimizer_state_dict")
        if opt_state:
            optimizer.load_state_dict(opt_state)

    # ---------------- SCHEDULER ----------------
    if scheduler and isinstance(checkpoint, dict):
        sch_state = checkpoint.get("scheduler") or checkpoint.get("scheduler_state_dict")
        if sch_state:
            scheduler.load_state_dict(sch_state)

    print(f"[Checkpoint] Loaded from {ckpt_path} | epoch: {epoch}")
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")

    return model, optimizer, scheduler, epoch



def load_huggingface_checkpoint(config, path, device):
    from soccernetpro.models.base.video_mae import load_video_mae_checkpoint
    return load_video_mae_checkpoint(config, device=device, ckpt_path=path)

