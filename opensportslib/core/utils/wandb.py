import wandb
import matplotlib.pyplot as plt
import numpy as np
import logging
import os
from opensportslib.core.utils.config import namespace_to_dict


def _flatten_config(data, parent_key="", sep="."):
    """Flatten nested dict/list config for W&B table-friendly columns."""
    items = {}

    if isinstance(data, dict):
        for k, v in data.items():
            key = f"{parent_key}{sep}{k}" if parent_key else str(k)
            items.update(_flatten_config(v, key, sep=sep))
        return items

    if isinstance(data, list):
        for i, v in enumerate(data):
            key = f"{parent_key}{sep}{i}" if parent_key else str(i)
            items.update(_flatten_config(v, key, sep=sep))
        return items

    if parent_key:
        items[parent_key] = data

    return items


def _pick_keys(data, keys):
    """Return a shallow dict containing only keys present in data."""
    if not isinstance(data, dict):
        return {}
    return {key: data[key] for key in keys if key in data}

def _wandb_ready():
    return getattr(wandb, "run", None) is not None

def init_wandb(cfg, run_id, use_wandb=False):
    """
    Initialize Weights & Biases if enabled.
    
    Args:
        cfg: config object with attributes:
             - use_wandb (bool)
             - project_name (str)
             - run_name (str)
    """

    if not use_wandb:
        logging.info("W&B disabled.")
        return None

    try:
        import wandb
    except ImportError:
        logging.warning("wandb not installed. Install with `pip install wandb`.")
        return None

    # Prevent multiple processes from initializing wandb
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    if rank != 0:
        return None

    # Prevent re-initialization
    if wandb.run is not None:
        return wandb

    if getattr(cfg.DATA, "data_modality", None):
        run_name = f"{cfg.MODEL.backbone.type}_{cfg.DATA.data_modality}"
    else:
        run_name = f"{cfg.MODEL.backbone.type}"

    config_plain = namespace_to_dict(cfg)
    config_flat = _flatten_config(config_plain)

    wandb.init(
        project=cfg.TASK,
        name=run_name,
        id=run_id,
        resume="allow",
        config=config_flat,
    )

    logging.info(f"Wandb initialised")
    return wandb

def log_table_wandb(name, rows, headers):
    """
    Log a table to Weights & Biases.

    Args:
        name (str): Name of the table in wandb.
        rows (list[list]): Table rows.
        headers (list[str]): Column headers.
    """
    if not _wandb_ready():
        return

    table = wandb.Table(columns=headers)

    for row in rows:
        table.add_data(*row)

    wandb.log({name: table})

def log_attention_wandb(attention, split_name):
    if not _wandb_ready():
        return

    attn = attention.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.imshow(attn, aspect="auto", cmap="viridis")
    ax.set_title(f"{split_name} Attention Map")
    ax.set_xlabel("Views / Time")
    ax.set_ylabel("Batch")

    wandb.log({
        f"{split_name}/attention_map": wandb.Image(fig)
    })

    plt.close(fig)


def log_sample_videos_wandb(mvclips, preds, labels, split_name, max_samples=2, fps=5):
    if not _wandb_ready():
        return


    # mvclips: (B, V, C, T, H, W)
    mvclips = mvclips.detach().cpu().numpy()

    for i in range(min(len(mvclips), max_samples)):
        views = mvclips[i]  # (V, C, T, H, W)

        # Log each view separately
        for v in range(views.shape[0]):
            video = views[v].transpose(1, 2, 3, 0)  # (T, H, W, C)
            video = (video * 255).astype(np.uint8) if video.max() <= 1.0 else video

            wandb.log({
                f"{split_name}/sample_{i}_view_{v}": wandb.Video(
                    video,
                    fps=fps,
                    caption=f"Pred: {preds[i]}, GT: {labels[i]}"
                )
            })


def log_confusion_matrix_wandb(y_true, y_pred, class_names, split_name):
    if not _wandb_ready():
        return
    wandb.log({
        f"{split_name}/confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=y_true,
            preds=y_pred,
            class_names=class_names
        )
    })
