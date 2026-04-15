import wandb
import matplotlib.pyplot as plt
import numpy as np
import logging
import os
from opensportslib.core.utils.config import namespace_to_dict

def build_wandb_config(cfg):
    """
    Extract minimal + useful config for W&B dashboard.
    Returns a FLAT dict (ready for wandb.init(config=...))
    """

    from opensportslib.core.utils.config import namespace_to_dict

    cfg_dict = namespace_to_dict(cfg)

    def get(d, path, default=None):
        """Safe nested get using dot notation"""
        keys = path.split(".")
        for k in keys:
            if not isinstance(d, dict) or k not in d:
                return default
            d = d[k]
        return d

    def pick(paths):
        """Pick only selected keys"""
        out = {}
        for p in paths:
            v = get(cfg_dict, p)
            if v is not None:
                out[p] = v
        return out

    # -------------------------
    # REQUIRED (core columns)
    # -------------------------
    REQUIRED_KEYS = [
        "TASK",

        "DATA.dataset_name",
        "DATA.data_modality",

        "MODEL.type",
        "MODEL.backbone.type",
        "MODEL.neck.type",
        "MODEL.head.type",

        "TRAIN.optimizer.type",
        "TRAIN.optimizer.lr",
        "TRAIN.scheduler.type",

        "TRAIN.monitor",
        "TRAIN.mode",

        "SYSTEM.device",
    ]

    # -------------------------
    # OPTIONAL (useful knobs)
    # -------------------------
    OPTIONAL_KEYS = [
        # DATA
        "DATA.num_frames",
        "DATA.clip_len",
        "DATA.input_fps",
        "DATA.extract_fps",
        "DATA.frame_size",
        "DATA.view_type",
        "DATA.num_classes",

        # MODEL
        "MODEL.backbone.encoder",
        "MODEL.backbone.hidden_dim",
        "MODEL.backbone.freeze",
        "MODEL.unfreeze_last_n_layers",
        "MODEL.neck.agr_type",
        "MODEL.edge",

        # TRAIN
        "TRAIN.epochs",
        "TRAIN.num_epochs",
        "TRAIN.max_epochs",
        "TRAIN.use_amp",
        "TRAIN.use_weighted_loss",
        "TRAIN.use_weighted_sampler",
        "TRAIN.mixup",
        "TRAIN.mixup_alpha",

        # SYSTEM
        "SYSTEM.GPU",
        "SYSTEM.seed",
    ]

    config = {}

    # pick required
    config.update(pick(REQUIRED_KEYS))

    # pick optional
    config.update(pick(OPTIONAL_KEYS))

    # -------------------------
    # SPECIAL HANDLING
    # -------------------------

    # Normalize batch_size (from nested dataloader)
    batch_size = get(cfg_dict, "DATA.train.dataloader.batch_size")
    if batch_size is not None:
        config["TRAIN.batch_size"] = batch_size

    # Normalize epochs (different configs use different names)
    epochs = (
        get(cfg_dict, "TRAIN.epochs")
        or get(cfg_dict, "TRAIN.num_epochs")
        or get(cfg_dict, "TRAIN.max_epochs")
    )
    if epochs is not None:
        config["TRAIN.total_epochs"] = epochs

    return config

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


def _wandb_ready():
    return getattr(wandb, "run", None) is not None

def init_wandb(cfg_path, cfg, run_id, use_wandb=False):
    """
    Initialize Weights & Biases if enabled.
    
    Args:
        cfg_path: Path to the configuration file.
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

    config_flat = build_wandb_config(cfg)

    wandb.init(
        project=cfg.TASK,
        name=run_name,
        id=run_id,
        resume="allow",
        config=config_flat,
    )

    artifact = wandb.Artifact(
        name=f"{cfg.TASK}-config",
        type="config",
        description="configuration (YAML)"
    )

    artifact.add_file(cfg_path)
    wandb.log_artifact(artifact)

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
