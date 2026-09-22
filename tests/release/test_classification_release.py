"""Release verification: ClassificationModel training across backbone families.

Prefers OpenSportsLab/OSL-XFoul, pinned in the OpenSportsLab/osl-ready-
datasets Hugging Face collection (parquet + webdataset shards). It's the
dataset opensportslib/configs/classification/video.yaml's own data_root
comment points at (dataset_name: mvfouls) and doubles as the VQA dataset
(see test_vqa_release.py) -- same dual-purpose RefPal/MVFouls-style clips
with both classification labels and referee Q&A. It's an unpublished
placeholder as of the last check, so this falls back to
OpenSportsLab/OSL-cls-UEFA-fouls (real, populated, gated: request access at
https://huggingface.co/datasets/OpenSportsLab/OSL-cls-UEFA-fouls and export
HF_TOKEN, or `hf auth login`, before running) until OSL-XFoul lands. Point
OSL_RELEASE_CLS_REPO / OSL_RELEASE_CLS_FALLBACK_REPO at different repos to
override either side.

The fallback has been verified against real repo content (with granted
access): it ships a proper OSL v2 `refpal_labels_processed.json` (top-level
`data`/`labels` keys, `inputs: [{"type": "video", "path":
"processed_videos/<id>.mp4"}]`) with two label heads -- `action` (18
foul-type classes, used here) and `offence` (7-level severity, 0-6, not used
here) -- so `_load_and_convert_labels` below takes its "already OSL v2" fast
path unchanged. It's a small dataset (13 clips as of the last check), so
treat this as a correctness spot-check across backbones, not a
scale/accuracy benchmark. The list/dict-of-records conversion path in the
same function is kept in case a differently-shaped dataset is swapped in via
OSL_RELEASE_CLS_FALLBACK_REPO; if the upstream schema changes shape
entirely, that function is the first place to look -- it fails with a clear
assertion rather than guessing silently.

Covers two backbone families dispatched by
opensportslib/models/builder.py::build_model_canonical for TASK=classification:

* MVNetwork family (video_encoder -> video_adapter -> task_head, provider
  "torchvision"): r3d_18, mc3_18, r2plus1d_18, s3d, mvit_v2_s. Verified
  end-to-end for r3d_18 against real downloaded video in a prior smoke test;
  the others share the exact same code path and canonical config
  (classification/video.yaml) and are expected to behave the same.
* HuggingFace VideoMAE full-model path (encoder_type == "video_mae").

Additional release coverage:
* The VideoModel "frames_npy" family (dinov3, clip, videomae, videomae2 as
  pure feature extractors) uses a bounded raw-video -> frame `.npy`
  materialization step before running the canonical frames pipeline.
* Tracking-based classification (graph_conv / classification/sngar_tracking.yaml)
  uses OpenSportsLab/SoccerNet-GAR; missing required release data fails explicitly.

Run through the repository's single command: bash scripts/run_tests.sh.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path

import pytest
import numpy as np

from opensportslib.apis.classification import ClassificationModel
from opensportslib.core.utils.video_processing import read_video

from ._release_common import (
    CACHE_ROOT,
    DATA_DIR,
    classes_from_osl_json,
    download_shard_split,
    training_overrides,
    materialize_config,
    max_items_for,
    prefer_osl_ready_dataset,
    report_step,
    record_release_metadata,
    require_release_enabled,
    require_repo_access,
    require_repo_populated,
    snapshot_dataset,
    system_block,
)

# OSL-XFoul is pinned in the OpenSportsLab/osl-ready-datasets collection
# (parquet + webdataset shards) and is the dataset
# opensportslib/configs/classification/video.yaml's own data_root comment
# points at (dataset_name: mvfouls) -- the same dual-purpose RefPal/MVFouls-
# style dataset used for VQA (see test_vqa_release.py). Prefer it. Its
# default ("main") branch is an empty placeholder; the real shards live on
# the "224p"/"720p" branches (train/valid/test each) -- default to "224p"
# (smaller/faster), override with OSL_RELEASE_CLS_REVISION=720p. Falls back
# to OSL-cls-UEFA-fouls (real, populated, gated, verified against real
# content with granted access -- see below) only if neither branch is up.
CLS_PRIMARY_REPO = os.environ.get("OSL_RELEASE_CLS_REPO", "OpenSportsLab/OSL-XFoul")
CLS_PRIMARY_REVISION = os.environ.get("OSL_RELEASE_CLS_REVISION", "224p")
CLS_FALLBACK_REPO = os.environ.get("OSL_RELEASE_CLS_FALLBACK_REPO", "OpenSportsLab/OSL-cls-UEFA-fouls")
TRACKING_CLS_REPO = os.environ.get("OSL_RELEASE_TRACKING_CLS_REPO", "OpenSportsLab/SoccerNet-GAR")
TRACKING_CLS_REVISION = os.environ.get("OSL_RELEASE_TRACKING_CLS_REVISION", "tracking")
PUBLISHED_CLASSIFICATION_MODEL = "OpenSportsLab/OSL-cls-action-mvitv2"

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("OSL_PRETRAINED_WEIGHTS", "0")


# --------------------------------------------------------------------------
# Dataset prep
# --------------------------------------------------------------------------

# opensportslib.datasets.classification_dataset.ClassificationDataset
# hardcodes exclude_labels = ["Unknown", "Dont know"] and drops them from
# the label space it actually builds (label_map, class weights, etc). If
# DATA.common.classes / num_classes don't apply the same exclusion, the
# model is built with the wrong output size and weighted-loss class weights
# come back the wrong shape ("weight tensor should be defined for all N
# classes but got shape [N-k]"). Keep this in sync with that hardcoded list.
_DATASET_EXCLUDED_LABELS = {"Unknown", "Dont know"}


def _exclude_dataset_labels(classes: list[str]) -> list[str]:
    return [c for c in classes if c not in _DATASET_EXCLUDED_LABELS]


def _resolve_label(record: dict) -> str | None:
    for key in ("label", "action_class", "foul_type", "class", "action"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, dict) and isinstance(value.get("label"), str):
            return value["label"]
    return None


def _resolve_id(record: dict, fallback_index: int) -> str:
    for key in ("id", "action_id", "clip_id", "name"):
        value = record.get(key)
        if value is not None:
            return str(value)
    return f"clip_{fallback_index:05d}"


def _resolve_video_relpath(record: dict, clip_id: str, videos_dir: Path) -> str | None:
    for key in ("video", "clip", "file", "filename", "path"):
        value = record.get(key)
        if isinstance(value, str) and value:
            rel = value if "/" in value else f"processed_videos/{value}"
            if (videos_dir.parent / rel).is_file():
                return rel
    guess = f"processed_videos/{clip_id}.mp4"
    if (videos_dir.parent / guess).is_file():
        return guess
    return None


def _load_and_convert_labels(root: Path, repo_id: str) -> dict:
    """Best-effort conversion of the dataset's label file into OSL v2 JSON.

    See the module docstring: this is the part most likely to need a fix if
    the upstream schema has moved on since this was written.
    """
    label_files = list(root.glob("*labels*.json"))
    assert label_files, (
        f"No '*labels*.json' file found under {root}. Inspect the downloaded "
        f"repo contents and update _load_and_convert_labels()."
    )
    raw = json.loads(label_files[0].read_text(encoding="utf-8"))

    if isinstance(raw, dict) and "data" in raw:
        # Already OSL v2 -- use as-is.
        return raw

    records = raw if isinstance(raw, list) else list(raw.values())
    assert isinstance(records, list) and records, (
        f"Unrecognized label file schema in {label_files[0].name}: expected a "
        f"list of records, or a dict keyed by clip id, or an OSL v2 payload "
        f"with a top-level 'data' key. Update _load_and_convert_labels()."
    )

    videos_dir = root / "processed_videos"
    items = []
    for idx, record in enumerate(records):
        assert isinstance(record, dict), f"Expected dict records, got {type(record)}"
        clip_id = _resolve_id(record, idx)
        label = _resolve_label(record)
        video_rel = _resolve_video_relpath(record, clip_id, videos_dir)
        if label is None or video_rel is None:
            continue
        items.append(
            {
                "id": clip_id,
                "inputs": [{"type": "video", "path": video_rel}],
                "labels": {"action": {"label": label}},
            }
        )
    assert items, (
        f"Converted 0/{len(records)} label records to OSL v2 -- the field "
        f"names in _resolve_label/_resolve_id/_resolve_video_relpath don't "
        f"match this dataset's schema. Inspect {label_files[0]} and fix them."
    )

    classes = sorted({item["labels"]["action"]["label"] for item in items})
    return {
        "version": "2.0",
        "task": "action_classification",
        "dataset_name": repo_id,
        "labels": {"action": {"type": "single_label", "labels": classes}},
        "data": items,
    }


def _split_dataset(payload: dict, seed: int = 0) -> dict[str, dict]:
    import random

    items = list(payload["data"])
    random.Random(seed).shuffle(items)
    n = len(items)
    n_train = max(1, int(n * 0.7))
    n_valid = max(1, int(n * 0.15))
    splits = {
        "train": items[:n_train],
        "valid": items[n_train : n_train + n_valid],
        "test": items[n_train + n_valid :] or items[-1:],
    }
    return {
        split: {**payload, "data": split_items} for split, split_items in splits.items()
    }


@pytest.fixture(scope="module")
def classification_dataset():
    require_release_enabled()
    require_repo_access(CLS_PRIMARY_REPO)

    repo_id, revision, is_sharded = prefer_osl_ready_dataset(
        CLS_PRIMARY_REPO, fallback=CLS_FALLBACK_REPO, primary_revision=CLS_PRIMARY_REVISION
    )

    if is_sharded:
        root = DATA_DIR / "classification" / f"{repo_id.split('/')[-1]}-{revision}"
        split_paths = {
            split: download_shard_split(repo_id, split, root, revision=revision)
            for split in ("train", "valid", "test")
        }
        classes = _exclude_dataset_labels(
            classes_from_osl_json(json.loads(split_paths["test"].read_text(encoding="utf-8")), head="action")
        )
        # Each split's media is extracted under its own split directory (see
        # download_dataset_split_from_hf's split_output_dir), and the json's
        # "inputs[].path" values (e.g. "train/action_0/clip_0.mp4") are
        # relative to that directory -- NOT to a single shared data_root, so
        # source_path must be resolved per split.
        source_paths = {split: path.parent for split, path in split_paths.items()}
        return {
            "data_root": root,
            "classes": classes,
            "split_paths": split_paths,
            "source_paths": source_paths,
        }

    # Fallback: known-good, real (gated) dataset -- not sharded, but small
    # (13 clips), so downloaded and split locally rather than capped.
    require_repo_access(repo_id)
    root = DATA_DIR / "classification" / repo_id.split("/")[-1]
    report_step(f"Downloading {repo_id} to {root}")
    snapshot_dataset(repo_id, root)

    payload = _load_and_convert_labels(root, repo_id)
    splits = _split_dataset(payload)

    split_dir = root / "_osl_v2_splits"
    split_dir.mkdir(exist_ok=True)
    split_paths = {}
    for split, split_payload in splits.items():
        path = split_dir / f"annotations-classification-{split}.json"
        path.write_text(json.dumps(split_payload, indent=2), encoding="utf-8")
        split_paths[split] = path

    classes = _exclude_dataset_labels(payload["labels"]["action"]["labels"])
    counts = Counter(item["labels"]["action"]["label"] for item in payload["data"])
    print(f"Classes ({len(classes)}): {classes}")
    print(f"Label distribution: {dict(counts)}")

    # All splits share one media root here (processed_videos/<id>.mp4
    # relative to `root`), unlike the sharded branch above.
    source_paths = {split: root for split in split_paths}

    return {
        "data_root": root,
        "classes": classes,
        "split_paths": split_paths,
        "source_paths": source_paths,
    }


# --------------------------------------------------------------------------
# Backbone matrix
# --------------------------------------------------------------------------

# (backbone name, torchvision provider) for the MVNetwork family. All share
# the same video_adapter (MV_Aggregate) / task_head (MV_LinearLayer) wiring.
MVNETWORK_BACKBONES = ["r3d_18", "mc3_18", "r2plus1d_18", "s3d", "mvit_v2_s"]
FRAME_BACKBONES = {
    "dinov3": ("facebook/dinov3-vitb16-pretrain-lvd1689m", 768),
    "clip": ("openai/clip-vit-large-patch14", 1024),
    "videomae": ("MCG-NJU/videomae-base", 768),
    "videomae2": ("OpenGVLab/VideoMAEv2-Base", 768),
}


def _run_classification_pipeline(config_path: str, dataset: dict, run_name: str) -> None:
    started = time.perf_counter()
    split_paths = dataset["split_paths"]

    report_step(f"[{run_name}] instantiate ClassificationModel")
    model = ClassificationModel(config=config_path, weights=None)

    report_step(f"[{run_name}] train()")
    checkpoint = model.train(
        train_set=str(split_paths["train"]),
        valid_set=str(split_paths["valid"]),
        use_wandb=False,
    )
    assert checkpoint and Path(checkpoint).exists(), f"[{run_name}] checkpoint was not written"
    record_release_metadata(
        "pipeline", task="classification", model_family=run_name,
        config_path=config_path, checkpoint_path=str(checkpoint),
    )

    report_step(f"[{run_name}] fresh checkpoint load + infer()")
    restored_model = ClassificationModel(config=config_path, weights=checkpoint)
    predictions = restored_model.infer(test_set=str(split_paths["test"]), use_wandb=False)
    assert isinstance(predictions, dict) and predictions.get("data"), f"[{run_name}] empty predictions"

    report_step(f"[{run_name}] save_predictions()")
    pred_path = CACHE_ROOT / "outputs" / f"classification_{run_name}_predictions.json"
    model.save_predictions(output_path=str(pred_path), predictions=predictions)
    assert pred_path.exists()
    record_release_metadata("prediction", task="classification", model_family=run_name, prediction_path=str(pred_path))

    report_step(f"[{run_name}] evaluate()")
    metrics = restored_model.evaluate(test_set=str(split_paths["test"]), predictions=predictions, use_wandb=False)
    assert isinstance(metrics, dict), f"[{run_name}] evaluate() did not return metrics"
    record_release_metadata(
        "result", task="classification", model_family=run_name,
        runtime_seconds=round(time.perf_counter() - started, 3),
    )
    print(f"[{run_name}] metrics: {metrics}")


@pytest.mark.release
def test_published_classification_model_inference(classification_dataset):
    """Published model artifact must load with its canonical inference path."""
    require_release_enabled()
    from ._release_common import require_model_repo

    require_model_repo(PUBLISHED_CLASSIFICATION_MODEL)
    dataset = classification_dataset
    overrides = system_block("published_classification_mvitv2")
    overrides["DATA"] = {
        "common": {
            "data_root": str(dataset["data_root"]), "classes": dataset["classes"],
            "splits": {split: {"annotation_path": str(path), "source_path": str(dataset["source_paths"][split])}
                       for split, path in dataset["split_paths"].items()},
        }
    }
    config_path = materialize_config("classification", "video", overrides, out_name="published_cls_mvitv2.yaml")
    model = ClassificationModel(config=config_path, weights=PUBLISHED_CLASSIFICATION_MODEL)
    predictions = model.infer(test_set=str(dataset["split_paths"]["test"]), use_wandb=False)
    assert isinstance(predictions, dict) and predictions.get("data")
    metrics = model.evaluate(test_set=str(dataset["split_paths"]["test"]), predictions=predictions, use_wandb=False)
    record_release_metadata("published_model_result", task="classification", model_id=PUBLISHED_CLASSIFICATION_MODEL, metrics=metrics)


@pytest.mark.release
@pytest.mark.parametrize("backbone", MVNETWORK_BACKBONES)
def test_classification_mvnetwork_backbone(classification_dataset, backbone):
    require_release_enabled()
    dataset = classification_dataset
    max_frames = max_items_for("OSL_RELEASE_CLS_NUM_FRAMES", 16) or 16

    overrides = system_block(f"classification_{backbone}")
    overrides["DATA"] = {
        "common": {
            "dataset_name": dataset["data_root"].name,
            "data_root": str(dataset["data_root"]),
            "classes": dataset["classes"],
            "splits": {
                split: {
                    "annotation_path": str(path),
                    "source_path": str(dataset["source_paths"][split]),
                }
                for split, path in dataset["split_paths"].items()
            },
        },
        "inputs": {
            "video": {
                "sampling": {"num_frames": max_frames},
                "params": {"view_type": "single", "num_classes": len(dataset["classes"])},
            }
        },
    }
    overrides["MODEL"] = {
        "components": {
            "video_encoder": {"source": {"name": backbone}, "params": {"pretrained_model": backbone}},
            "task_head": {"params": {"num_classes": len(dataset["classes"])}},
        }
    }
    overrides["TRAIN"] = training_overrides(2)

    config_path = materialize_config("classification", "video", overrides, out_name=f"cls_{backbone}.yaml")
    _run_classification_pipeline(config_path, dataset, backbone)


@pytest.mark.release
def test_classification_video_mae_huggingface_backend(classification_dataset):
    """Coverage of the HF VideoMAE full-model path
    (encoder_type == 'video_mae' in build_model_canonical). If it fails,
    check build_video_mae_backbone() in
    opensportslib/models/base/video.py for what it actually expects from the
    dataloader batch.
    """
    require_release_enabled()
    dataset = classification_dataset

    overrides = system_block("classification_video_mae")
    overrides["DATA"] = {
        "common": {
            "dataset_name": dataset["data_root"].name,
            "data_root": str(dataset["data_root"]),
            "classes": dataset["classes"],
            "splits": {
                split: {
                    "annotation_path": str(path),
                    "source_path": str(dataset["source_paths"][split]),
                }
                for split, path in dataset["split_paths"].items()
            },
        },
        "inputs": {
            "video": {
                "params": {"view_type": "single", "num_classes": len(dataset["classes"])},
            }
        },
    }
    overrides["MODEL"] = {
        "components": {
            "video_encoder": {
                "source": {"provider": "opensportslib", "registry": "backbone", "name": "video_mae"},
                "params": {"pretrained_model": "MCG-NJU/videomae-base"},
            },
            "task_head": {"params": {"num_classes": len(dataset["classes"])}},
        }
    }
    overrides["TRAIN"] = training_overrides(2)

    config_path = materialize_config("classification", "video", overrides, out_name="cls_video_mae.yaml")
    _run_classification_pipeline(config_path, dataset, "video_mae")


@pytest.fixture(scope="module")
def frames_npy_dataset(classification_dataset):
    """Decode the bounded real-video split into the production frames_npy format."""

    dataset = classification_dataset
    root = DATA_DIR / "classification" / "frames_npy"
    root.mkdir(parents=True, exist_ok=True)
    frame_count = max_items_for("OSL_RELEASE_CLS_NUM_FRAMES", 16) or 16
    split_paths = {}

    for split, annotation_path in dataset["split_paths"].items():
        payload = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
        output_items = []
        for index, item in enumerate(payload.get("data", [])):
            source_input = next(
                (value for value in item.get("inputs", []) if value.get("type") == "video"),
                None,
            )
            assert source_input, f"{split} item {item.get('id', index)!r} has no video input"
            source = Path(source_input["path"])
            if not source.is_absolute():
                source = Path(dataset["source_paths"][split]) / source
            assert source.is_file(), f"Release video does not exist: {source}"
            frames = read_video(str(source))
            assert frames, f"Release video decoded no frames: {source}"
            indices = np.linspace(0, len(frames) - 1, frame_count).astype(int)
            array = np.stack([frames[position] for position in indices]).astype(np.uint8)
            relative = Path(split) / f"{item.get('id', index)}.npy"
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.save(destination, array)
            converted = dict(item)
            converted["inputs"] = [{"type": "frames_npy", "path": str(relative)}]
            output_items.append(converted)
        converted_payload = {**payload, "data": output_items}
        output_path = root / f"{split}.json"
        output_path.write_text(json.dumps(converted_payload, indent=2), encoding="utf-8")
        split_paths[split] = output_path

    return {
        "data_root": root,
        "classes": dataset["classes"],
        "split_paths": split_paths,
        "source_paths": {split: root for split in split_paths},
    }


@pytest.fixture(scope="module")
def tracking_classification_dataset():
    """Materialize populated SoccerNet-GAR OSL tracking splits."""

    require_release_enabled()
    require_repo_access(TRACKING_CLS_REPO)
    require_repo_populated(TRACKING_CLS_REPO, revision=TRACKING_CLS_REVISION)
    root = DATA_DIR / "classification" / f"{TRACKING_CLS_REPO.split('/')[-1]}-{TRACKING_CLS_REVISION}"
    snapshot_dataset(TRACKING_CLS_REPO, root, revision=TRACKING_CLS_REVISION)
    split_paths = {}
    for split in ("train", "valid", "test"):
        candidates = sorted(root.rglob(f"*{split}*.json"))
        assert candidates, f"{TRACKING_CLS_REPO} has no {split} OSL annotation JSON"
        split_paths[split] = candidates[0]
    payload = json.loads(split_paths["train"].read_text(encoding="utf-8"))
    classes = _exclude_dataset_labels(classes_from_osl_json(payload, head="action"))
    assert classes, f"{TRACKING_CLS_REPO} training annotations declare no action classes"
    return {
        "data_root": root,
        "classes": classes,
        "split_paths": split_paths,
        "source_paths": {split: root for split in split_paths},
    }


@pytest.mark.release
@pytest.mark.release_gar
def test_classification_tracking_graph_conv(tracking_classification_dataset):
    """Tracking-modality classification (graph_conv backbone,
    classification/sngar_tracking.yaml canonical config). Uses
    OpenSportsLab/SoccerNet-GAR. Missing required data fails release verification.
    """
    require_release_enabled()
    dataset = tracking_classification_dataset
    overrides = system_block("classification_graph_conv")
    overrides["DATA"] = {
        "common": {
            "dataset_name": dataset["data_root"].name,
            "data_root": str(dataset["data_root"]),
            "classes": dataset["classes"],
            "splits": {
                split: {"annotation_path": str(path), "source_path": str(dataset["source_paths"][split])}
                for split, path in dataset["split_paths"].items()
            },
        },
        "inputs": {"video": {"params": {"max_samples": max_items_for("OSL_RELEASE_MAX_CLIPS", 40)}}},
    }
    overrides["MODEL"] = {"components": {"task_head": {"params": {"num_classes": len(dataset["classes"])}}}}
    overrides["TRAIN"] = training_overrides(1)
    config_path = materialize_config(
        "classification", "sngar_tracking", overrides, out_name="cls_graph_conv.yaml"
    )
    _run_classification_pipeline(config_path, dataset, "graph_conv")


@pytest.mark.release
@pytest.mark.parametrize("backbone", FRAME_BACKBONES)
def test_classification_frames_npy_backbones(frames_npy_dataset, backbone):
    require_release_enabled()
    dataset = frames_npy_dataset
    pretrained_model, hidden_dim = FRAME_BACKBONES[backbone]
    overrides = system_block(f"classification_frames_{backbone}")
    overrides["DATA"] = {
        "common": {
            "dataset_name": dataset["data_root"].name,
            "data_root": str(dataset["data_root"]),
            "classes": dataset["classes"],
            "splits": {
                split: {"annotation_path": str(path), "source_path": str(dataset["source_paths"][split])}
                for split, path in dataset["split_paths"].items()
            },
        },
        "inputs": {"video": {"sampling": {"num_frames": max_items_for("OSL_RELEASE_CLS_NUM_FRAMES", 16) or 16}}},
    }
    overrides["MODEL"] = {"components": {
        "video_encoder": {
            "source": {"name": backbone},
            "params": {"pretrained_model": pretrained_model, "hidden_dim": hidden_dim},
        },
        "video_adapter": {"params": {"hidden_dim": hidden_dim}},
        "task_head": {"params": {"hidden_dim": hidden_dim, "num_classes": len(dataset["classes"])}},
    }}
    overrides["TRAIN"] = training_overrides(1)
    config_path = materialize_config(
        "classification", "sngar_frames", overrides, out_name=f"cls_frames_{backbone}.yaml"
    )
    _run_classification_pipeline(config_path, dataset, f"frames_{backbone}")
