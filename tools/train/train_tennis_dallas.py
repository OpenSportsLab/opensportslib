"""Train Dallas video or skeleton classifiers with existing OSL models.

Select a checkpoint on validation loss, then evaluate the held-out test game once.
The RGB route uses a local, deterministic frame cache, never trainable features.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import math
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.convert.tennis_to_classification import digest, fingerprint, write_json
from tools.convert.dallas_to_classification import CLASSES

MODEL_REPO = "OpenGVLab/VideoMAEv2-Base"
MODEL_REVISION = "78c337a418cc4adaf7f1ff6c7ba343418393966b"


def validate_data(root):
    root = Path(root).resolve()
    report = json.loads((root / "reports/build.json").read_text())
    if report["status"] != "complete":
        raise ValueError("Complete Dallas clip conversion first")
    matches, documents = {}, {}
    for split in ("train", "valid", "test"):
        document = json.loads((root / f"{split}.json").read_text())
        if (document["task"] != "action_classification" or document["labels"]["action"]["labels"] != CLASSES
                or document["conversion_identity"] != report["identity"]
                or len(document["data"]) != report["split_counts"][split]):
            raise ValueError(f"{split}: inconsistent Dallas binary manifest")
        matches[split] = {sample["metadata"]["base_video"] for sample in document["data"]}
        for sample in document["data"]:
            if sample["labels"]["action"]["label"] not in CLASSES:
                raise ValueError("Unexpected classification label")
            if any(not (root / item["path"]).is_file() for item in sample["inputs"]):
                raise FileNotFoundError(f"Missing clip: {sample['id']}")
        counts = dict(Counter(sample["labels"]["action"]["label"] for sample in document["data"]))
        if counts != report["class_counts"][split] or matches[split] != set(report["matches"][split]):
            raise ValueError(f"{split}: class or match counts differ from the completed conversion")
        if len({sample["id"] for sample in document["data"]}) != len(document["data"]):
            raise ValueError(f"{split}: duplicate sample IDs")
        documents[split] = document
    if [len(matches[s]) for s in matches] != [3, 1, 1] or any(
            matches[a] & matches[b] for a, b in (("train", "valid"), ("train", "test"), ("valid", "test"))):
        raise ValueError("Expected disjoint three/one/one game splits")
    return report, documents


def prepare_video_frames(root, *, workers=2):
    """Cache the same 16 uniform source indices as the raw skeleton loader.

    Apply the pinned checkpoint's aspect-preserving 224-short-edge resize and
    center crop. uint8 arrays remain unnormalized for the existing VideoDataset.
    """
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from opensportslib.datasets.video_raw import read_raw_clip, probe_video, uniform_indices

    root = Path(root).resolve()
    report, documents = validate_data(root)
    destination = root / "video-frames"
    identity = fingerprint({"conversion": report["identity"], "num_frames": 16,
                            "resize_short_edge": 224, "crop": "center224", "resample": "PIL-bilinear",
                            "sampling": "uniform_indices_v1", "schema": 1})
    receipt_path = destination / "manifest.json"
    if receipt_path.is_file() and json.loads(receipt_path.read_text())["identity"] != identity:
        raise ValueError("Frame cache belongs to another preprocessing profile")
    destination.mkdir(parents=True, exist_ok=True)
    write_json(receipt_path, {"status": "running", "identity": identity})
    cv2.setNumThreads(1)
    torch.set_num_threads(1)

    def extract(sample):
        path = root / sample["inputs"][0]["path"]
        relative = Path("frames") / f"{sample['id']}.npy"
        target = destination / relative
        if target.is_file():
            existing = np.load(target, mmap_mode="r")
            if existing.shape == (16, 224, 224, 3) and existing.dtype == np.uint8:
                converted = copy.deepcopy(sample)
                converted["inputs"] = [{"type": "frames_npy", "path": relative.as_posix()}]
                return converted
            raise ValueError(f"Corrupt frame cache: {target}")
        info = probe_video(path)
        frames, metadata = read_raw_clip(path, uniform_indices(0, info["count"], 16))
        images = []
        for frame in frames:
            image = Image.fromarray(frame.permute(1, 2, 0).numpy())
            width, height = image.size
            if width < height:
                resized = (224, int(224 * height / width))
            else:
                resized = (int(224 * width / height), 224)
            image = image.resize(resized, resample=Image.Resampling.BILINEAR)
            left, top = (image.width-224)//2, (image.height-224)//2
            images.append(np.asarray(image.crop((left, top, left+224, top+224))))
        array = np.stack(images)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".part.npy")
        np.save(temporary, array, allow_pickle=False)
        os.replace(temporary, target)
        converted = copy.deepcopy(sample)
        converted["inputs"] = [{"type": "frames_npy", "path": relative.as_posix()}]
        converted["metadata"]["frame_cache"] = {"identity": identity, "selected_frames": metadata["frames"]}
        return converted

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for split, document in documents.items():
            samples = []
            for index, sample in enumerate(executor.map(extract, document["data"])):
                samples.append(sample)
                if (index+1) % 200 == 0:
                    print(f"RGB frames {split}: {index+1}/{len(document['data'])}", flush=True)
            converted = {**document, "modalities": ["frames_npy"], "data": samples}
            write_json(destination / f"{split}.json", converted)
    write_json(receipt_path, {"status": "complete", "identity": identity,
                             "source_conversion_identity": report["identity"]})
    return destination


def training_config(root, model, output, *, epochs=5, batch_size=None, pretrained_path=None,
                    num_workers=2, device="cuda", patience=10, min_delta=1e-4):
    from opensportslib.apis import Config

    if epochs < 1:
        raise ValueError("epochs must be positive")
    if patience < 0:
        raise ValueError("patience must be nonnegative (zero disables validation early stopping)")
    if not math.isfinite(min_delta) or min_delta < 0:
        raise ValueError("min_delta must be finite and nonnegative")
    if device not in {"cpu", "cuda"}:
        raise ValueError("Choose cpu or cuda")
    root, output = Path(root).resolve(), Path(output).resolve()
    validate_data(root)
    template = Path(__file__).resolve().parents[2] / "examples/tennis/classification_skeleton.yaml"
    payload = Config.from_file(template).get_config()
    data = payload["DATA"]
    data["common"].update(dataset_name="tennis-dallas-binary", classes=CLASSES, data_root=str(root))
    components = payload["MODEL"]["components"]
    if model == "skeleton":
        components["pose_tracking"]["params"]["cache"] = {
            "mode": "read" if device == "cpu" else "read_write", "path": str(root / "pose-cache")}
        components["tracking_encoder"]["params"]["num_layers"] = 4
        components["tracking_adapter"]["params"]["use_position_encoding"] = False
        batch_size = batch_size or 8
        input_root = root
        lr = .001
    elif model == "video":
        input_root = root / "video-frames"
        frame_cache = json.loads((input_root / "manifest.json").read_text())
        if frame_cache["status"] != "complete":
            raise ValueError("Prepare the RGB frame cache before video training")
        if pretrained_path is None or not (Path(pretrained_path) / "model.safetensors").is_file():
            raise ValueError("Provide the pinned local VideoMAEv2-Base snapshot")
        components.clear()
        components.update({
            "video_encoder": {"kind": "encoder", "source": {"provider": "opensportslib", "registry": "backbone", "name": "videomae2"},
                              "params": {"type": "videomae2", "pretrained_model": str(Path(pretrained_path).resolve()),
                                         "hidden_dim": 768, "freeze": True, "unfreeze_last_n_layers": 4}},
            "video_adapter": {"kind": "adapter", "source": {"provider": "opensportslib", "registry": "neck", "name": "TemporalAggregation"},
                              "params": {"type": "TemporalAggregation", "agr_type": "avgpool", "hidden_dim": 768,
                                         "dropout": .1, "use_position_encoding": False}},
            "task_head": {"kind": "head", "source": {"provider": "opensportslib", "registry": "head", "name": "TrackingClassifier"},
                          "params": {"type": "TrackingClassifier", "hidden_dim": 768, "dropout": .1, "num_classes": 2}}})
        payload["MODEL"]["topology"] = [{"from": "video_encoder", "to": "video_adapter"}, {"from": "video_adapter", "to": "task_head"}]
        payload["IO"]["inputs"]["video"] = "video_encoder"
        data["inputs"]["video"].update(modality="frames_npy", representation="frames", source={"format": "npy"})
        batch_size = batch_size or 4
        lr = .00005
    else:
        raise ValueError("Choose skeleton or video")
    components["task_head"]["params"]["num_classes"] = 2
    data["inputs"]["video"]["sampling"].update(num_frames=16, input_fps=25)
    for split, specification in data["common"]["splits"].items():
        specification.update(source_path=str(input_root), annotation_path=str(input_root / f"{split}.json"))
        specification["dataloader"].update(batch_size=batch_size, num_workers=num_workers,
            shuffle=split == "train", pin_memory=model == "video", mp_context="spawn" if num_workers else None,
            persistent_workers=num_workers > 0, prefetch_factor=2 if num_workers else None)
    payload["SYSTEM"]["paths"] = {"save_dir": str(output / "checkpoints"), "log_dir": str(output / "logs"), "work_dir": str(output)}
    payload["SYSTEM"]["device"] = device
    payload["TRAIN"].update(epochs=epochs)
    payload["TRAIN"]["optimizer"] = {"type": "Adam", "lr": lr, "weight_decay": .0001}
    payload["TRAIN"]["execution"].update(use_amp=model == "video" and device == "cuda", mixup_alpha=0., detailed_results=True)
    payload["TRAIN"]["selection"].update(monitor="loss", mode="min", patience=patience, min_delta=min_delta)
    payload["TRAIN"]["scheduler"].update(patience=2, factor=.5)
    payload["TRAIN"]["checkpoint"]["save_every"] = 5
    return Config(payload)


def run(root, model, output, **settings):
    import torch
    from opensportslib.apis import ClassificationModel
    from sklearn.metrics import classification_report, confusion_matrix

    output = Path(output).resolve()
    if (output / "results.json").exists():
        raise ValueError("Training results already exist; choose another output directory")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    config = training_config(root, model, output, **settings)
    config_path = output / "config.yaml"
    import yaml
    config_path.write_text(yaml.safe_dump(config.get_config(), sort_keys=False))
    start = time.perf_counter()
    api = ClassificationModel(config=str(config_path))
    checkpoint = api.train(use_wandb=False)
    train_seconds = time.perf_counter() - start
    if not checkpoint:
        raise RuntimeError("Training returned no validation-selected checkpoint")
    selected = torch.load(checkpoint, map_location="cpu", weights_only=False)
    selected_epoch, selected_loss = selected["epoch"], selected["best_metric"]
    del selected
    summary = json.loads((Path(checkpoint).parent.parent / "training_summary.json").read_text())
    write_json(output / "training_summary.json", summary)
    test_started = time.perf_counter()
    predictions = api.infer(weights=checkpoint, use_wandb=False)
    api.save_predictions(str(output / "predictions.json"), predictions)
    # Use OSL's existing evaluator for the public result contract.
    metrics = api.evaluate(predictions=predictions, use_wandb=False)
    truth = {sample["id"]: sample["labels"]["action"]["label"]
             for sample in json.loads((Path(root) / "test.json").read_text())["data"]}
    actual = {sample["id"]: sample["labels"]["action"]["label"] for sample in predictions["data"]}
    if set(actual) != set(truth) or len(actual) != len(predictions["data"]):
        raise ValueError("Predictions do not cover the entire held-out test split exactly once")
    ids = sorted(truth)
    target = [CLASSES.index(truth[key]) for key in ids]
    predicted = [CLASSES.index(actual[key]) for key in ids]
    metrics["per_class"] = classification_report(target, predicted, labels=[0, 1], target_names=CLASSES,
                                                 output_dict=True, zero_division=0)
    metrics["confusion_matrix"] = confusion_matrix(target, predicted, labels=[0, 1]).tolist()
    metrics["test_clips"] = len(ids)
    write_json(output / "metrics.json", metrics)
    result = {"model": model, "classes": CLASSES, "epochs_requested": settings.get("epochs", 5),
              "epochs_completed": summary["epochs_completed"], "stop_reason": summary["stop_reason"],
              "early_stopping": {key: summary[key] for key in ("monitor", "mode", "patience", "min_delta")},
              "selected_epoch": selected_epoch, "validation_loss": selected_loss, "checkpoint": str(checkpoint),
              "train_seconds": train_seconds, "test_seconds": time.perf_counter()-test_started,
              "metrics": metrics, "device": torch.cuda.get_device_name(0) if settings.get("device", "cuda") == "cuda" else "cpu", "torch": torch.__version__,
              "seed": 42, "selection": "minimum validation loss; test game used only after selection",
              "background_labels": "inferred, not verified negatives"}
    result["manifest_sha256"] = {split: digest(Path(root) / f"{split}.json") for split in ("train", "valid", "test")}
    if model == "video":
        result.update(pretrained_repo=MODEL_REPO, pretrained_revision=MODEL_REVISION, pretrained_license="cc-by-nc-4.0",
                      encoder_finetuning="last four transformer blocks; other encoder weights frozen",
                      pretrained_sha256=digest(Path(settings["pretrained_path"]) / "model.safetensors"))
    write_json(output / "results.json", result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("prepare-video", "skeleton", "video"))
    parser.add_argument("--data-root", type=Path, default=Path("data/tennis-dallas-classification"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--epochs", type=int, default=5, help="Maximum epochs; may stop earlier on validation loss")
    parser.add_argument("--patience", type=int, default=10, help="Epochs without sufficient validation improvement (0 disables)")
    parser.add_argument("--min-delta", type=float, default=1e-4, help="Absolute validation-loss improvement needed to reset patience")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda",
                        help="CPU skeleton training requires exact-clip caches already warmed on GPU")
    parser.add_argument("--pretrained-path", type=Path, default=Path("data/dallas-models/videomaev2-base"))
    args = parser.parse_args()
    if args.model == "prepare-video":
        prepare_video_frames(args.data_root, workers=max(1, args.num_workers))
    else:
        run(args.data_root, args.model, args.output or Path("outputs/dallas-pilot") / args.model,
            epochs=args.epochs, batch_size=args.batch_size, pretrained_path=args.pretrained_path,
            num_workers=args.num_workers, device=args.device, patience=args.patience, min_delta=args.min_delta)


if __name__ == "__main__":
    main()
