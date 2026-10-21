"""Local T-DEED training, inference and resumable experiment artifacts.

This localization runtime owns its displacement-specific data/loss/evaluation
contract. It does not change E2E-Spot or write predictions during infer().
"""

from functools import partial
import hashlib
import importlib.metadata
import json
import math
import os
import platform
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from opensportslib.core.config.runtime_adapter import namespace_to_plain_dict
from opensportslib.core.config.accessors import get_system_seed
from opensportslib.datasets.tdeed import TDEEDDataset, JointTDEEDDataset, load_videos
from opensportslib.metrics.tdeed import (
    frame_map,
    suppress,
    soccernet_map,
    finegym_start_classes,
    released_snas_clip_map,
)

REFERENCE_COMMIT = "d953b4de8a2ad941c565e1683c5b07255cbe644e"


def atomic_json(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def seed_worker(worker_id, epoch):
    # Reference Python sampling seed; PyTorch seeds the worker's NumPy/torch RNGs.
    random.seed(worker_id + epoch * 100)


def payload(api):
    return namespace_to_plain_dict(api.config)


def settings(api):
    cfg = payload(api)
    video = cfg["DATA"]["inputs"]["video"]
    components = cfg["MODEL"]["components"]
    head = next(c for c in components.values() if c.get("kind") == "head")["params"]
    sampling = video["sampling"]
    return cfg, video, head, sampling


def build_dataset(api, split, *, inference=False, path=None, auxiliary=False):
    cfg, video, head, sampling = settings(api)
    common = cfg["DATA"]["common"]
    split_cfg = common["splits"][split]
    if auxiliary:
        aux = cfg["TRAIN"]["sampling"]["auxiliary"]
        manifest = aux["splits"][split]["annotation_path"]
        root = aux["splits"][split].get("source_path")
        classes = aux["classes"]
        stride = aux.get("stride", 2)
    else:
        manifest = path or api._resolve_split_path(split)
        root = split_cfg.get("source_path")
        classes = common["classes"]
        stride = sampling.get("stride", 1)
    videos = load_videos(manifest, root)
    resize = video["transform"]["resize"]
    length = sampling["clip_len"]
    training = split == "train" and not inference
    options = dict(
        clip_len=length,
        stride=stride,
        radius=head.get("radi_displacement", 0),
        dilation=video.get("params", {}).get("dilate_len", 0),
        resize=(resize["height"], resize["width"]),
        training=training,
        mixup=video.get("params", {}).get("mixup", True),
        epoch_num_frames=max(
            length, sampling["epoch_num_frames"] // (1 if split == "train" else 4)
        ),
        overlap=0.5 if auxiliary or common["dataset_name"] == "soccernet" else 0.9,
        foreground_only=auxiliary or common["dataset_name"] == "soccernet",
        inference_overlap=(
            split_cfg.get("overlap_len", length // 4 * 3) if inference else None
        ),
    )
    if inference:
        options["foreground_only"] = False
    return TDEEDDataset(videos, classes, **options)


def loader(api, dataset, split, epoch=0, *, inference=False):
    cfg = payload(api)
    opts = cfg["DATA"]["common"]["splits"][split].get("dataloader", {})
    workers = int(opts.get("num_workers", 0))
    batch = int(opts.get("batch_size", 8))
    accumulation = int(cfg["TRAIN"]["execution"].get("acc_grad_iter", 1))
    if not inference:
        if batch % accumulation:
            raise ValueError(
                "Reference batch size must be divisible by accumulation steps"
            )
        batch //= accumulation
    generator = torch.Generator().manual_seed(get_system_seed(api.config) + epoch)
    return DataLoader(
        dataset,
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        pin_memory=api.model.device.type == "cuda",
        worker_init_fn=partial(seed_worker, epoch=epoch),
        generator=generator,
        **({"prefetch_factor": 2} if workers else {}),
    )


def ensure_model(api, weights=None):
    from opensportslib.models.builder import build_model
    from opensportslib.core.utils.config import select_device

    effective = weights or api.last_loaded_weights
    if api.model is None:
        api.model = build_model(api.config, select_device(api.config.SYSTEM))
        api._tdeed_loaded_checkpoint = None
    if effective and getattr(api, "_tdeed_loaded_checkpoint", None) != str(effective):
        checkpoint = torch.load(
            effective, map_location=api.model.device, weights_only=False
        )
        state = checkpoint.get("model_state_dict", checkpoint)
        api.model.load(state)
        api._tdeed_loaded_checkpoint = str(effective)
        api.last_loaded_weights = str(effective)
    return api.model


def infer_raw(api, split="test", path=None, weights=None, *, validation=False):
    model = ensure_model(api, weights)
    cfg, video, head, sampling = settings(api)
    data = build_dataset(api, split, inference=True, path=path)
    if validation:
        # Validation has no overlap and no flip, as in the reference release.
        data = TDEEDDataset(
            data.videos,
            cfg["DATA"]["common"]["classes"],
            clip_len=sampling["clip_len"],
            stride=data.stride,
            resize=(data.reader.height, data.reader.width),
            training=False,
            radius=0,
            inference_overlap=0,
        )
    classes = list(data.classes)
    flip = cfg["TRAIN"].get("selection", {}).get("test_flip", True) and not validation
    arrays = {
        i: (
            np.zeros((math.ceil(v.frames / data.stride), len(classes) + 1), np.float32),
            np.zeros(math.ceil(v.frames / data.stride), np.int32),
        )
        for i, v in enumerate(data.videos)
    }
    for batch in loader(api, data, split, inference=True):
        views = [model.predict(batch["frame"])[1]]
        if flip:
            views.append(model.predict(batch["frame"], augment_inference=True)[1])
        for j, (vi, start) in enumerate(
            zip(batch["video_index"].tolist(), batch["start"].tolist())
        ):
            scores, support = arrays[vi]
            begin, end = max(0, start), min(len(scores), start + sampling["clip_len"])
            for view in views:
                part = view[j, begin - start : end - start]
                scores[begin:end] += part
                support[begin:end] += (part.sum(-1) != 0).astype(np.int32)
    output = []
    for vi, v in enumerate(data.videos):
        scores, support = arrays[vi]
        # Relocation can leave empty positions; the reference treats these as zero scores.
        scores /= np.maximum(support, 1)[:, None]
        times, labels = np.where(scores[:, 1:] >= 0.01)
        events = [
            {"frame": int(t), "label": classes[c], "score": float(scores[t, c + 1])}
            for t, c in zip(times, labels)
        ]
        output.append(
            {"video": v.identity, "events": events, "fps": v.fps / data.stride}
        )
    return output, data


def postprocess(api, raw, *, validation=False, method=None):
    cfg, _, _, _ = settings(api)
    selection = cfg["TRAIN"].get("selection", {})
    mode = method or (
        "nms" if validation else selection.get("postprocessing", "soft_nms")
    )
    if mode not in {"none", "nms", "soft_nms"}:
        raise ValueError(f"Unsupported T-DEED postprocessing: {mode}")
    window = (
        selection.get("nms_window", 1)
        if mode == "nms"
        else selection.get("soft_nms_window", 3)
    )
    threshold = 0.10 if validation else 0.01
    return [
        {
            **row,
            "events": (
                suppress(
                    row["events"], window, soft=mode == "soft_nms", threshold=threshold
                )
                if mode != "none"
                else row["events"]
            ),
        }
        for row in raw
    ]


def score(api, data, predictions):
    cfg = payload(api)
    dataset = cfg["DATA"]["common"]["dataset_name"]
    if dataset in {"soccernet", "soccernetball"}:
        names = ["tight", "loose"] if dataset == "soccernet" else ["ball"]
        return {
            name: soccernet_map(
                data.videos, predictions, list(data.classes), data.stride, name
            )
            for name in names
        }
    truth = [
        {
            "video": v.identity,
            "events": [{**e, "frame": e["frame"] // data.stride} for e in v.events],
        }
        for v in data.videos
    ]
    result = frame_map(truth, predictions)
    if dataset == "finegym":
        result["FG-Start"] = frame_map(
            truth, predictions, classes=finegym_start_classes(data.classes)
        )
    return result


def infer(api, test_set=None, weights=None):
    raw, data = infer_raw(api, path=test_set, weights=weights)
    predictions = postprocess(api, raw)
    return {
        "version": "2.0",
        "task": "localization",
        "metadata": {
            "type": "predictions",
            "created_by": "T-DEED",
            "reference_commit": REFERENCE_COMMIT,
            "frame_axis": "sampled, performance-relative",
        },
        "data": [
            {
                "id": v.identity,
                "inputs": [{"type": "video", "path": str(v.path), "fps": row["fps"]}],
                "metadata": {
                    "video": v.identity,
                    "source_start_frame": v.source_start,
                    "source_stride": data.stride,
                },
                "events": [
                    {
                        **e,
                        "head": "action",
                        "position_ms": round(e["frame"] * 1000 / row["fps"]),
                        "confidence": e["score"],
                    }
                    for e in row["events"]
                ],
            }
            for v, row in zip(data.videos, predictions)
        ],
    }


def evaluate(api, test_set=None, weights=None, predictions=None):
    if predictions is not None:
        data = build_dataset(api, "test", inference=True, path=test_set)
        raw = [
            {
                "video": r.get("metadata", {}).get("video", r.get("id")),
                "events": [
                    {**e, "score": e.get("score", e.get("confidence"))}
                    for e in r["events"]
                ],
            }
            for r in predictions["data"]
        ]
        return score(api, data, raw)
    raw, data = infer_raw(api, path=test_set, weights=weights)
    # Report both fixed postprocessors, never choose hyperparameters using test labels.
    return {
        method: score(api, data, postprocess(api, raw, method=method))
        for method in ["none", "nms", "soft_nms"]
    }


def scheduler(optimizer, epochs, warmup, steps):
    if not 0 < warmup < epochs or steps < 1:
        raise ValueError(
            "T-DEED requires 0 < warmup epochs < epochs and positive optimizer steps"
        )
    # Deliberately reproduce the released ChainedScheduler, not SequentialLR.
    return torch.optim.lr_scheduler.ChainedScheduler(
        [
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup * steps
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, (epochs - warmup) * steps
            ),
        ]
    )


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def save_checkpoint(path, checkpoint):
    path = Path(path)
    temporary = path.with_suffix(".tmp.pt")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def train(api, train_set=None, valid_set=None, weights=None):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    cfg, _, _, _ = settings(api)
    seed = get_system_seed(api.config)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=False)
    model = ensure_model(api, weights)
    train_data = build_dataset(api, "train", path=train_set)
    valid_data = build_dataset(api, "valid", path=valid_set)
    if cfg["TRAIN"].get("sampling", {}).get("auxiliary"):
        train_data = JointTDEEDDataset(
            train_data, build_dataset(api, "train", auxiliary=True)
        )
        valid_data = JointTDEEDDataset(
            valid_data, build_dataset(api, "valid", auxiliary=True)
        )
    execution = cfg["TRAIN"]["execution"]
    epochs, accumulation = int(cfg["TRAIN"]["epochs"]), int(
        execution.get("acc_grad_iter", 1)
    )
    optimizer = torch.optim.AdamW(
        model._get_params(),
        lr=cfg["TRAIN"]["optimizer"]["lr"],
        weight_decay=cfg["TRAIN"]["optimizer"].get("weight_decay", 0.01),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=model.device.type == "cuda")
    schedule = scheduler(
        optimizer,
        epochs,
        cfg["TRAIN"]["scheduler"]["warm_up_epochs"],
        math.ceil(len(loader(api, train_data, "train")) / accumulation),
    )
    out = Path(api.save_dir)
    out.mkdir(parents=True, exist_ok=True)
    implementation = Path(__file__).resolve().parents[2]
    code_paths = [
        Path(__file__),
        implementation / "apis/localization.py",
        implementation / "models/builder.py",
        implementation / "datasets/tdeed.py",
        implementation / "metrics/tdeed.py",
        implementation / "models/base/tdeed.py",
        *sorted((implementation / "models/utils/impl/tdeed").glob("*.py")),
    ]
    code_hashes = {
        str(p.relative_to(implementation)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in code_paths
    }
    history, first, best, best_epoch = [], 0, -1.0, None
    recipe = {
        "MODEL": cfg["MODEL"],
        "DATA": cfg["DATA"],
        "TRAIN": cfg["TRAIN"],
        "seed": seed,
    }
    recipe["implementation_sha256"] = code_hashes
    digest = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    effective = weights or api.last_loaded_weights
    if effective:
        state = torch.load(effective, map_location=model.device, weights_only=False)
        if "optimizer_state_dict" in state:
            if state.get("recipe_sha256") != digest:
                raise ValueError(
                    "Resume recipe differs; use a new run for changed experiment settings"
                )
            optimizer.load_state_dict(state["optimizer_state_dict"])
            schedule.load_state_dict(state["scheduler_state_dict"])
            scaler.load_state_dict(state["scaler_state_dict"])
            restore_rng(state["rng"])
            first, history, best, best_epoch = (
                state["epoch"] + 1,
                state["history"],
                state["best_score"],
                state["best_epoch"],
            )
            if first and best_epoch is not None and not (out / "best.pt").exists():
                raise ValueError(
                    "Resume into the original output directory so its best checkpoint remains available"
                )
    environment = {
        package: importlib.metadata.version(package)
        for package in ["torch", "torchvision", "timm", "numpy", "SoccerNet"]
    }
    import cv2

    environment.update(
        python=platform.python_version(),
        architecture=platform.machine(),
        cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),
        opencv=cv2.__version__,
        gpu=(
            torch.cuda.get_device_name(model.device)
            if model.device.type == "cuda"
            else None
        ),
    )
    atomic_json(
        out / "run.json",
        {
            "reference_commit": REFERENCE_COMMIT,
            "recipe": recipe,
            "recipe_sha256": digest,
            "environment": environment,
            "device": str(model.device),
            "status": "running",
            "deterministic_algorithms": True,
            "schedule": "released ChainedScheduler",
            "temporal_resampling_backward": "deterministic gather/reduction; native forward",
            "result_claim": "not yet reproduced",
        },
    )
    for epoch in range(first, epochs):
        started = time.monotonic()

        def progress(phase, batch, batches, loss):
            record = {
                "epoch": epoch,
                "epochs": epochs,
                "phase": phase,
                "batch": batch,
                "batches": batches,
                "running_loss": loss,
                "elapsed_epoch_seconds": time.monotonic() - started,
            }
            atomic_json(out / "progress.json", record)
            print(json.dumps(record), flush=True)

        train_loss = model.epoch(
            loader(api, train_data, "train", epoch),
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=schedule,
            acc_grad_iter=accumulation,
            progress_callback=partial(progress, "train"),
        )
        clip_validation = (
            cfg["TRAIN"]["selection"].get("validation_protocol")
            == "released_snas_clips"
            and epoch >= execution["start_valid_epoch"]
        )
        valid_result = model.epoch(
            loader(api, valid_data, "valid", epoch),
            progress_callback=partial(progress, "valid_loss"),
            collect_predictions=clip_validation,
        )
        valid_loss = valid_result[0] if clip_validation else valid_result
        value, metrics = None, None
        if epoch >= execution["start_valid_epoch"]:
            if clip_validation:
                metrics = {
                    "tight": released_snas_clip_map(valid_result[1], valid_result[2])
                }
            else:
                raw, data = infer_raw(api, "valid", path=valid_set, validation=True)
                metrics = score(api, data, postprocess(api, raw, validation=True))
            value = metrics.get(
                "mean_mAP", metrics.get("ball", metrics.get("tight", {})).get("a_mAP")
            )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "valid_score": value,
                "metrics": metrics,
            }
        )
        improved = value is not None and value > best
        if improved:
            best, best_epoch = value, epoch
        state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": schedule.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "rng": rng_state(),
            "epoch": epoch,
            "history": history,
            "best_score": best,
            "best_epoch": best_epoch,
            "recipe_sha256": digest,
        }
        save_checkpoint(out / "last.pt", state)
        if improved:
            save_checkpoint(out / "best.pt", state)
        atomic_json(out / "history.json", history)
        print(
            json.dumps({k: v for k, v in history[-1].items() if k != "metrics"}),
            flush=True,
        )
    if best_epoch is None:
        raise RuntimeError(
            "No validation-selected checkpoint; check start_valid_epoch and complete the configured run"
        )
    api.best_checkpoint = api.last_loaded_weights = str(out / "best.pt")
    api._tdeed_loaded_checkpoint = None
    manifest = json.loads((out / "run.json").read_text())
    manifest.update(status="trained", best_epoch=best_epoch, best_score=best)
    atomic_json(out / "run.json", manifest)
    return api.best_checkpoint
