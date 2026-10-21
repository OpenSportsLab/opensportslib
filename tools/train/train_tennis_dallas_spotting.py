"""Train a Dallas video or skeleton action spotter until validation loss plateaus.

Use original full-broadcast localization annotations, select on fixed validation
windows, then evaluate the selected checkpoint on the held-out test match once.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.convert.dallas_to_classification import MATCHES
from tools.convert.tennis_to_classification import digest, write_json


def validate_data(root):
    """Check the original swing-only manifests and the agreed whole-match split."""
    root = Path(root).resolve()
    documents = {}
    for split, matches in MATCHES.items():
        document = json.loads((root / f"{split}.json").read_text())
        if (document.get("task") != "action_spotting"
                or document.get("labels", {}).get("swing", {}).get("labels") != ["swing"]):
            raise ValueError(f"{split}: expected Dallas swing-only localization annotations")
        records = document["data"]
        if len(records) != len(matches) or {sample["id"] for sample in records} != set(matches):
            raise ValueError(f"{split}: expected the agreed disjoint Dallas three/one/one game split")
        for sample in records:
            if len(sample["inputs"]) != 1 or sample["inputs"][0]["type"] != "video":
                raise ValueError(f"{sample['id']}: expected one full video")
            path = root / sample["inputs"][0]["path"]
            if not path.is_file():
                raise FileNotFoundError(path)
            if not sample.get("events") or any(
                    event["label"] != "swing" or not math.isfinite(float(event["position_ms"]))
                    or float(event["position_ms"]) < 0 for event in sample["events"]):
                raise ValueError(f"{sample['id']}: expected finite original swing events")
        documents[split] = document
    return documents


def training_config(root, model, output, *, epochs=100, patience=10, min_delta=1e-4,
                    batch_size=None, num_workers=0, device="cuda", epoch_num_frames=64000,
                    clip_len=64, cache_mode="off", cache_dir=None, video_backbone="rny008_gsm"):
    from opensportslib.apis import Config

    if epochs < 1 or patience < 0 or not math.isfinite(min_delta) or min_delta < 0:
        raise ValueError("Require positive epochs, nonnegative patience, and finite nonnegative min_delta")
    if clip_len < 2 or epoch_num_frames < 4 * clip_len or epoch_num_frames % clip_len:
        raise ValueError("epoch_num_frames must be a multiple of clip_len and include at least four clips")
    if num_workers < 0 or (batch_size is not None and batch_size < 1):
        raise ValueError("Require nonnegative workers and positive batch size")
    if device not in {"cpu", "cuda"} or cache_mode not in {"off", "read", "read_write"}:
        raise ValueError("Unsupported device or cache policy")
    if video_backbone not in {"rny008_gsm", "rny002_gsm", "convnextt"}:
        raise ValueError("Choose a supported per-frame spotting video backbone")
    root, output = Path(root).resolve(), Path(output).resolve()
    validate_data(root)
    template = Path(__file__).resolve().parents[2] / "examples/tennis/localization_skeleton.yaml"
    payload = Config.from_file(template).get_config()
    data = payload["DATA"]
    data["common"].update(dataset_name="tennis-dallas-spotting", data_root=str(root), classes=["swing"])
    data["common"]["runtime"]["loader_backend"] = "opencv"
    video = data["inputs"]["video"]
    video["sampling"].update(clip_len=clip_len, extract_fps=25, input_fps=25, epoch_num_frames=epoch_num_frames)
    video["params"].update(crop_dim=-1, mixup=False, dilate_len=0)
    components = payload["MODEL"]["components"]
    if model == "skeleton":
        batch_size = batch_size or 4
        components["tracking_encoder"]["params"].update(num_layers=4, hidden_dim=64)
        components["pose_tracking"]["params"]["cache"] = {
            "mode": cache_mode, "path": str(Path(cache_dir).resolve() if cache_dir else root / "pose-cache-clips")}
        # Pose inference sees uncropped source RGB; keep full resolution for the far player.
        video.pop("transform", None)
        lr = .001
    elif model == "video":
        batch_size = batch_size or 2
        components.pop("pose_tracking")
        components.pop("tracking_encoder")
        components["video_encoder"] = {
            "kind": "encoder", "source": {"provider": "opensportslib", "registry": "backbone", "name": video_backbone},
            "params": {"type": video_backbone}, "overrides": {}}
        payload["MODEL"]["topology"] = [{"from": "video_encoder", "to": "task_head"}]
        payload["IO"]["inputs"]["video"] = "video_encoder"
        video["transform"] = {"resize": {"height": 224, "width": 398},
                              "normalization": {"mean": [.485, .456, .406], "std": [.229, .224, .225]}}
        lr = .0001
    else:
        raise ValueError("Choose skeleton or video")
    for split, specification in data["common"]["splits"].items():
        source_split = "valid" if split == "valid_data_frames" else split
        specification.update(type="VideoGameWithOpencvVideo" if split in {"test", "valid_data_frames"}
                             else "VideoGameWithOpencv", source_path=str(root),
                             annotation_path=str(root / f"{source_split}.json"), overlap_len=clip_len//2,
                             metric="tight", nms_window=2, results=None)
        specification["dataloader"].update(batch_size=batch_size if split == "train" else 1,
                                          num_workers=num_workers, shuffle=split == "train", pin_memory=False)
    payload["MODEL"]["task"] = "localization"
    payload["SYSTEM"]["device"] = device
    payload["SYSTEM"]["gpu"]["count"] = 1
    payload["SYSTEM"]["paths"] = {"save_dir": str(output / "checkpoints"), "work_dir": str(output / "checkpoints"),
                                    "log_dir": str(output / "logs")}
    train = payload["TRAIN"]
    train["epochs"] = epochs
    train["optimizer"] = {"type": "AdamWithScaler", "lr": lr}
    train["scheduler"].update(num_epochs=epochs, warm_up_epochs=min(5, epochs-1), acc_grad_iter=1)
    train["execution"].update(criterion_valid="loss", base_num_valid_epochs=1, multi_gpu=False,
                              acc_grad_iter=1, validation_seed=42)
    train["sampling"].update(use_weighted_sampler=False, use_weighted_loss=False)
    train["selection"].update(monitor="valid_loss", mode="min", patience=patience, min_delta=min_delta)
    return Config(payload)


def prepare_predictions(predictions, root, documents, *, apply_nms):
    """Normalize source paths and apply the same two-frame NMS radius for RGB.

    RGB's offline route emits dense high-recall events. Use a dense local maximum
    filter instead of quadratic pairwise event comparisons on full broadcasts.
    The skeleton streaming route already performs this suppression within shots.
    """
    import numpy as np

    root = Path(root).resolve()
    result = copy.deepcopy(predictions)
    expected = {str(root / sample["inputs"][0]["path"]) for sample in documents["test"]["data"]}
    paths = []
    for sample in result["data"]:
        path = Path(sample["inputs"][0]["path"])
        normalized = str(path if path.is_absolute() else root / path)
        sample["inputs"][0]["path"] = normalized
        paths.append(normalized)
        # Preserve the dataset's single head for both inference routes.
        for event in sample["events"]:
            event["head"] = "swing"
        if apply_nms and sample["events"]:
            events = sample["events"]
            indices = np.asarray([int(e["frame"]) for e in events])
            if (indices < 0).any() or len(np.unique(indices)) != len(indices):
                raise ValueError("Expected one swing score per nonnegative frame")
            scores = np.full(int(indices.max()) + 5, -np.inf)
            scores[indices + 2] = [e["confidence"] for e in events]
            maxima = np.lib.stride_tricks.sliding_window_view(scores, 5).max(axis=-1)
            sample["events"] = [e for e, frame in zip(events, indices) if e["confidence"] >= maxima[frame]]
    if set(paths) != expected or len(paths) != len(expected):
        raise ValueError("Predictions must cover every held-out test video exactly once")
    return result


def run(root, model, output, *, config_only=False, **settings):
    output = Path(output).resolve()
    if (output / "results.json").exists():
        raise ValueError("Training results already exist; choose another output directory")
    config = training_config(root, model, output, **settings)
    output.mkdir(parents=True, exist_ok=True)
    import yaml
    config_path = output / "config.yaml"
    config_path.write_text(yaml.safe_dump(config.get_config(), sort_keys=False))
    if config_only:
        print(f"Configuration: {config_path}", flush=True)
        return config_path

    import torch
    from opensportslib.apis import LocalizationModel
    from opensportslib.apis._streaming import evaluate_spotting

    torch.set_num_threads(4)
    start = time.perf_counter()
    api = LocalizationModel(config=str(config_path))
    checkpoint = api.train(use_wandb=False)
    train_seconds = time.perf_counter() - start
    if not checkpoint:
        raise RuntimeError("Training returned no validation-selected checkpoint")
    summary = json.loads((Path(checkpoint).parent / "training_summary.json").read_text())
    write_json(output / "training_summary.json", summary)
    # Explicitly load the selected weights, even if the live model still holds
    # the final training epoch or a previously evaluated checkpoint.
    api.load_weights(checkpoint)
    test_started = time.perf_counter()
    predictions = prepare_predictions(api.infer(weights=checkpoint, use_wandb=False), root,
                                      validate_data(root), apply_nms=model == "video")
    api.save_predictions(str(output / "predictions.json"), predictions)
    metrics = evaluate_spotting(api.config, str(Path(root).resolve() / "test.json"), predictions)
    write_json(output / "metrics.json", metrics)
    result = {"task": "localization", "model": model, "classes": ["swing"],
              "epochs_requested": settings.get("epochs", 100), "epochs_completed": summary["epochs_completed"],
              "selected_epoch": summary["selected_epoch"], "validation_loss": summary["best_metric"],
              "stop_reason": summary["stop_reason"], "checkpoint": str(checkpoint), "metrics": metrics,
              "train_seconds_including_final_validation": train_seconds, "test_seconds": time.perf_counter()-test_started,
              "seed": 42, "device": str(api.model.device), "torch": torch.__version__,
              "selection": "minimum loss on fixed validation windows; test match used only after selection",
              "sampling": config.get_config()["DATA"]["inputs"]["video"]["sampling"],
              "manifest_sha256": {split: digest(Path(root) / f"{split}.json") for split in MATCHES}}
    write_json(output / "results.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("skeleton", "video"))
    parser.add_argument("--data-root", type=Path, default=Path("data/tennis-dallas-localization"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--epoch-num-frames", type=int, default=64000)
    parser.add_argument("--clip-len", type=int, default=64)
    parser.add_argument("--cache-mode", choices=("off", "read", "read_write"), default="off")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--video-backbone", choices=("rny008_gsm", "rny002_gsm", "convnextt"), default="rny008_gsm")
    parser.add_argument("--config-only", action="store_true", help="Validate local data and write YAML without constructing a model")
    args = vars(parser.parse_args())
    root, model, output = args.pop("data_root"), args.pop("model"), args.pop("output")
    run(root, model, output or Path("outputs/dallas-spotting") / model, **args)


if __name__ == "__main__":
    main()
