import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import wandb

from opensportslib.core.config.accessors import (
    get_component_name_by_kind,
    get_data_modality,
    get_train_epochs,
)
from opensportslib.core.utils.config import namespace_to_dict


def build_wandb_config(cfg):
    cfg_dict = namespace_to_dict(cfg)

    def get(d, path, default=None):
        keys = path.split(".")
        for k in keys:
            if not isinstance(d, dict) or k not in d:
                return default
            d = d[k]
        return d

    fields = [
        "TASK",
        "SYSTEM.device",
        "SYSTEM.gpu.count",
        "SYSTEM.reproducibility.seed",
        "DATA.common.dataset_name",
        "DATA.common.runtime.loader_backend",
        "TRAIN.trainer.type",
        "TRAIN.optimizer.type",
        "TRAIN.optimizer.lr",
        "TRAIN.scheduler.type",
        "TRAIN.epochs",
        "TRAIN.selection.monitor",
        "TRAIN.selection.mode",
    ]

    out = {k: get(cfg_dict, k) for k in fields if get(cfg_dict, k) is not None}
    out["MODEL.encoder"] = get_component_name_by_kind(cfg, "encoder")
    out["MODEL.head"] = get_component_name_by_kind(cfg, "head")
    out["DATA.modality"] = get_data_modality(cfg)
    out["TRAIN.total_epochs"] = get_train_epochs(cfg)

    train_bs = get(cfg_dict, "TRAIN.sampling.batch_size")
    if train_bs is not None:
        out["TRAIN.batch_size"] = train_bs

    return out


def _flatten_config(data, parent_key="", sep="."):
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
    if not use_wandb:
        logging.info("W&B disabled.")
        return None

    try:
        import wandb as wandb_pkg
    except ImportError:
        logging.warning("wandb not installed. Install with `pip install wandb`.")
        return None

    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    if rank != 0:
        return None

    if wandb_pkg.run is not None:
        return wandb_pkg

    encoder_name = get_component_name_by_kind(cfg, "encoder") or "model"
    modality = get_data_modality(cfg)
    arch_name = f"{encoder_name}_{modality}" if modality else encoder_name
    # Name the run after RUN_ID. Naming by architecture alone gave every run
    # of the same model an identical display name, so a project of many
    # experiments showed as a wall of indistinguishable entries; the
    # architecture is still recorded in the logged config.
    run_name = str(run_id) if run_id else arch_name

    config_flat = build_wandb_config(cfg)

    wandb_pkg.init(
        project=cfg.TASK,
        name=run_name,
        id=run_id,
        resume="allow",
        config=config_flat,
    )

    artifact = wandb_pkg.Artifact(
        name=f"{cfg.TASK}-config",
        type="config",
        description="configuration (YAML)",
    )

    artifact.add_file(cfg_path)
    wandb_pkg.log_artifact(artifact)

    logging.info("Wandb initialised")
    return wandb_pkg


def log_table_wandb(name, rows, headers):
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

    wandb.log({f"{split_name}/attention_map": wandb.Image(fig)})
    plt.close(fig)


def log_confusion_matrix_wandb(
    cm=None,
    class_names=None,
    split_name="valid",
    y_true=None,
    y_pred=None,
):
    if not _wandb_ready():
        return

    if cm is None:
        if y_true is None or y_pred is None:
            raise TypeError(
                "log_confusion_matrix_wandb() requires either `cm` or both "
                "`y_true` and `y_pred`."
            )

        if class_names is None:
            labels = sorted(set(y_true) | set(y_pred))
            class_names = [str(label) for label in labels]
        else:
            labels = list(range(len(class_names)))

        cm = np.zeros((len(labels), len(labels)), dtype=int)
        label_to_idx = {label: idx for idx, label in enumerate(labels)}
        for true_label, pred_label in zip(y_true, y_pred):
            true_idx = label_to_idx.get(true_label)
            pred_idx = label_to_idx.get(pred_label)
            if true_idx is None or pred_idx is None:
                continue
            cm[true_idx, pred_idx] += 1
    else:
        cm = np.asarray(cm)
        if class_names is None:
            class_names = [str(i) for i in range(cm.shape[0])]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=f"Confusion Matrix ({split_name})",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    wandb.log({f"{split_name}/confusion_matrix": wandb.Image(fig)})
    plt.close(fig)
