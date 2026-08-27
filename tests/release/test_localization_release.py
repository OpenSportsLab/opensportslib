"""Release verification: LocalizationModel training across model families.

Each dataset fixture prefers a dataset pinned in the
OpenSportsLab/osl-ready-datasets Hugging Face collection -- the org's
sharded (parquet + webdataset) datasets, a handful of large files per split
rather than one file per clip/game -- and falls back to a known-good, real,
but loose-file dataset only while the sharded one isn't populated yet (see
_release_common.prefer_osl_ready_dataset()). Both OSL-ready repos below are
unpublished placeholders as of the last check, so both fixtures currently
take the fallback path; re-run after either lands to switch automatically.

* Primary: OpenSportsLab/OSL-SNBAS. Fallback:
  OpenSportsLab/soccernetpro-localization-tennis (real, public, 3450 loose
  files, capped by default -- see OSL_RELEASE_MAX_CLIPS). Drives the E2E
  family (opensportslib/configs/localization/video_ocv.yaml, and
  video_dali.yaml if the `nvidia.dali` optional dependency is installed)
  across a curated backbone x head matrix. OSL-SNBAS is exactly the dataset
  video_ocv.yaml was written for (same 12-class Ball Action Spotting label
  set, same model-zoo checkpoints OSL-loc-snbas-2023-e2e /
  OSL-loc-snbas-2025-e2e); the tennis fallback + rn18/gru combo was verified
  end-to-end in a prior smoke-test session (see git history around the
  ResnetExtractFeatures fix in opensportslib/models/backbones/builder.py) --
  the other backbones/heads share that exact code path.

* Primary: OpenSportsLab/OSL-SoccerNet. Fallback:
  OpenSportsLab/SoccerNet-ActionSpotting-Features (real, public, 2210 loose
  ResNET_TF2_PCA512 feature files in a `ResNET_PCA512/<split>/...` +
  per-split `annotations.json` layout, capped by default -- see
  OSL_RELEASE_MAX_GAMES). Drives the classic ContextAware (CALF) and
  LearnablePooling (NetVLAD++) families, matching
  opensportslib/configs/localization/calf_resnetpca512.yaml and
  netvladpp_resnetpca512.yaml's own 17-class SoccerNet Action Spotting label
  set. Not verified end-to-end this session (the E2E/tennis path was); if a
  schema (either OSL-SoccerNet's shards, or the fallback's annotations.json)
  has moved on, that's the first place to check.

Run:
    RUN_OSL_RELEASE_TESTS=1 pytest tests/release/test_localization_release.py -v -s

    # bigger (but still capped) fallback subsets:
    RUN_OSL_RELEASE_TESTS=1 OSL_RELEASE_MAX_CLIPS=200 OSL_RELEASE_MAX_GAMES=50 \\
        pytest tests/release/test_localization_release.py -v -s

    # full-scale fallback feature dataset (~111GB, 2210 individual files):
    RUN_OSL_RELEASE_TESTS=1 OSL_RELEASE_MAX_GAMES=all \\
        pytest tests/release/test_localization_release.py -k feature -v -s
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import pytest

from opensportslib.apis.localization import LocalizationModel

from ._release_common import (
    CACHE_ROOT,
    DATA_DIR,
    classes_from_osl_json,
    download_files,
    download_shard_split,
    epochs_for,
    hf_token,
    materialize_config,
    max_items_for,
    optional_module_available,
    prefer_osl_ready_dataset,
    report_step,
    require_release_enabled,
    require_repo_access,
    snapshot_dataset,
    system_block,
)

# OSL-SNBAS is pinned in the OpenSportsLab/osl-ready-datasets collection
# (parquet + webdataset shards) and is exactly the dataset
# opensportslib/configs/localization/video_ocv.yaml was written for -- same
# 12-class Ball Action Spotting label set, same model-zoo checkpoints
# (OSL-loc-snbas-2023-e2e / OSL-loc-snbas-2025-e2e). Prefer it. Its default
# ("main") branch is an empty placeholder; the real shards live on
# "{224p,720p}-{2023,2024}" branches (train/valid/test/challenge each) --
# default to "224p-2024" (smallest, most recent), override with
# OSL_RELEASE_LOC_E2E_REVISION. Falls back to the tennis dataset (real,
# populated, loose files) only if that branch isn't up either.
#
# OSL-SoccerNet (below) *also* has "224p"/"720p" raw-video branches -- the
# classic 17-class SoccerNet Action Spotting task rather than SNBAS's
# 12-class Ball Action Spotting -- which work just as well for this E2E
# family. To use those instead, point OSL_RELEASE_LOC_E2E_REPO at
# OpenSportsLab/OSL-SoccerNet and OSL_RELEASE_LOC_E2E_REVISION at "224p" or
# "720p"; _e2e_overrides() derives its class list from the downloaded data
# either way, so no other change is needed.
E2E_PRIMARY_REPO = os.environ.get("OSL_RELEASE_LOC_E2E_REPO", "OpenSportsLab/OSL-SNBAS")
E2E_PRIMARY_REVISION = os.environ.get("OSL_RELEASE_LOC_E2E_REVISION", "224p-2024")
E2E_FALLBACK_REPO = os.environ.get("OSL_RELEASE_LOC_E2E_FALLBACK_REPO", "OpenSportsLab/soccernetpro-localization-tennis")

# OSL-SoccerNet is the other localization dataset pinned in the OSL-ready
# collection. Its "ResNET_PCA512" branch ships classic 17-class SoccerNet
# Action Spotting as pre-extracted features, matching calf_resnetpca512.yaml
# / netvladpp_resnetpca512.yaml's own class list -- prefer it. Falls back to
# the (non-sharded, 2210-file) SoccerNet-ActionSpotting-Features dataset only
# if that branch isn't up.
FEATURES_PRIMARY_REPO = os.environ.get("OSL_RELEASE_LOC_FEATURES_REPO", "OpenSportsLab/OSL-SoccerNet")
FEATURES_PRIMARY_REVISION = os.environ.get("OSL_RELEASE_LOC_FEATURES_REVISION", "ResNET_PCA512")
FEATURES_FALLBACK_REPO = os.environ.get(
    "OSL_RELEASE_LOC_FEATURES_FALLBACK_REPO", "OpenSportsLab/SoccerNet-ActionSpotting-Features"
)

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("OSL_PRETRAINED_WEIGHTS", "0")


# --------------------------------------------------------------------------
# E2E raw-video dataset prep
# --------------------------------------------------------------------------

TENNIS_CLASSES = [
    "far_court_bounce", "far_court_serve", "far_court_swing",
    "near_court_bounce", "near_court_serve", "near_court_swing",
]


@pytest.fixture(scope="module")
def e2e_localization_dataset():
    require_release_enabled()
    require_repo_access(E2E_PRIMARY_REPO)

    repo_id, revision, is_sharded = prefer_osl_ready_dataset(
        E2E_PRIMARY_REPO, fallback=E2E_FALLBACK_REPO, primary_revision=E2E_PRIMARY_REVISION
    )

    if is_sharded:
        root = DATA_DIR / "localization" / f"{repo_id.split('/')[-1]}-{revision}"
        split_paths = {
            split: download_shard_split(repo_id, split, root, revision=revision)
            for split in ("train", "valid", "test")
        }
        classes = classes_from_osl_json(json.loads(split_paths["test"].read_text(encoding="utf-8")))
        return {"data_root": root, "split_paths": split_paths, "classes": classes}

    # Fallback: known-good, real, but loose-file dataset. Capped by default
    # (the repo is 3450 individual files) -- set OSL_RELEASE_MAX_CLIPS=all
    # for the full ~1.4GB dataset.
    require_repo_access(repo_id)
    root = DATA_DIR / "localization" / "tennis"
    max_clips = max_items_for("OSL_RELEASE_MAX_CLIPS", 40)

    if max_clips is None:
        report_step(f"Downloading full {repo_id} to {root}")
        snapshot_dataset(repo_id, root)
        split_paths = {
            split: root / f"annotations-localization-{split}.json"
            for split in ("train", "valid", "test")
        }
    else:
        report_step(f"Downloading {repo_id} annotations + {max_clips} clips/split to {root}")
        root.mkdir(parents=True, exist_ok=True)
        split_paths = {}
        for split in ("train", "valid", "test"):
            full = download_files(repo_id, [f"annotations-localization-{split}.json"], root)[0]
            payload = json.loads(full.read_text(encoding="utf-8"))
            entries = [e for e in payload["data"] if e.get("events")][:max_clips]
            video_files = [
                inp["path"] for e in entries for inp in e["inputs"] if inp["type"] == "video"
            ]
            download_files(repo_id, video_files, root)
            subset_path = root / f"annotations-localization-{split}.json"
            subset_path.write_text(json.dumps({**payload, "data": entries}, indent=2), encoding="utf-8")
            split_paths[split] = subset_path

    return {"data_root": root, "split_paths": split_paths, "classes": TENNIS_CLASSES}


def _e2e_overrides(run_name: str, dataset: dict, backbone: str, head: str, *, loader_backend: str = "opencv") -> dict:
    root = dataset["data_root"]
    split_paths = dataset["split_paths"]
    overrides = system_block(f"localization_{run_name}")
    overrides["DATA"] = {
        "common": {
            "dataset_name": root.name,
            "data_root": str(root),
            "classes": dataset["classes"],
            "runtime": {"loader_backend": loader_backend},
            "splits": {
                "train": {"annotation_path": str(split_paths["train"]), "source_path": str(root)},
                "valid": {"annotation_path": str(split_paths["valid"]), "source_path": str(root)},
                "valid_data_frames": {"annotation_path": str(split_paths["valid"]), "source_path": str(root)},
                "test": {
                    "annotation_path": str(split_paths["test"]),
                    "source_path": str(root),
                    "results": str(CACHE_ROOT / "outputs" / f"localization_{run_name}_results"),
                    # overlap_len must stay < clip_len or ActionSpotVideoDataset
                    # builds an empty clip window (see the negative-step range
                    # bug worked around in the earlier smoke test).
                    "overlap_len": 0,
                },
            },
        },
    }
    overrides["MODEL"] = {
        "components": {
            "video_encoder": {"source": {"name": backbone}},
            "task_head": {"source": {"name": head}},
        }
    }
    overrides["TRAIN"] = {
        "epochs": epochs_for(2),
        "scheduler": {"num_epochs": epochs_for(2)},
    }
    return overrides


def _run_localization_pipeline(config_path: str, dataset: dict, run_name: str) -> None:
    split_paths = dataset["split_paths"]

    report_step(f"[{run_name}] instantiate LocalizationModel")
    model = LocalizationModel(config=config_path, weights=None)

    report_step(f"[{run_name}] train()")
    checkpoint = model.train(
        train_set=str(split_paths["train"]),
        valid_set=str(split_paths["valid"]),
        use_wandb=False,
    )
    assert checkpoint and Path(checkpoint).exists(), f"[{run_name}] checkpoint was not written"

    report_step(f"[{run_name}] infer()")
    predictions = model.infer(test_set=str(split_paths["test"]), weights=checkpoint, use_wandb=False)
    assert isinstance(predictions, dict) and predictions.get("data"), f"[{run_name}] empty predictions"

    report_step(f"[{run_name}] save_predictions()")
    pred_path = CACHE_ROOT / "outputs" / f"localization_{run_name}_predictions.json"
    model.save_predictions(output_path=str(pred_path), predictions=predictions)
    assert pred_path.exists()

    report_step(f"[{run_name}] evaluate()")
    metrics = model.evaluate(test_set=str(split_paths["test"]), use_wandb=False)
    print(f"[{run_name}] metrics: {metrics}")


# Curated backbone x head coverage. Full cross product of every
# backbone_type/head_type check_config() allows (see opensportslib/core/utils/
# load_annotations.py::check_config) would be 9 backbones x 4 heads = 36 runs;
# this picks a representative subset that touches every backbone family and
# every head at least once. Expand BACKBONES_TO_TEST / HEADS_TO_TEST for full
# coverage on a dedicated release-test machine.
E2E_BACKBONES = ["rn18", "rn50", "rny002", "rny008_gsm", "convnextt"]
E2E_HEADS = ["gru", "deeper_gru", "mstcn", "asformer"]
E2E_MATRIX = [(b, "gru") for b in E2E_BACKBONES] + [("rn18", h) for h in E2E_HEADS if h != "gru"]


@pytest.mark.release
@pytest.mark.parametrize("backbone,head", E2E_MATRIX, ids=[f"{b}-{h}" for b, h in E2E_MATRIX])
def test_localization_e2e_opencv(e2e_localization_dataset, backbone, head):
    require_release_enabled()
    run_name = f"{backbone}_{head}_opencv"
    overrides = _e2e_overrides(run_name, e2e_localization_dataset, backbone, head, loader_backend="opencv")
    config_path = materialize_config("localization", "video_ocv", overrides, out_name=f"loc_{run_name}.yaml")
    _run_localization_pipeline(config_path, e2e_localization_dataset, run_name)


@pytest.mark.release
@pytest.mark.skipif(
    not optional_module_available("nvidia.dali"),
    reason="DALI backend not installed; run `opensportslib setup --dali` first.",
)
def test_localization_e2e_dali(e2e_localization_dataset):
    require_release_enabled()
    run_name = "rny008_gsm_gru_dali"
    overrides = _e2e_overrides(run_name, e2e_localization_dataset, "rny008_gsm", "gru", loader_backend="dali")
    config_path = materialize_config("localization", "video_dali", overrides, out_name=f"loc_{run_name}.yaml")
    _run_localization_pipeline(config_path, e2e_localization_dataset, run_name)


# --------------------------------------------------------------------------
# Feature-based classic models (CALF / NetVLAD++) on real SoccerNet features
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def features_dataset():
    require_release_enabled()
    require_repo_access(FEATURES_PRIMARY_REPO)

    repo_id, revision, is_sharded = prefer_osl_ready_dataset(
        FEATURES_PRIMARY_REPO, fallback=FEATURES_FALLBACK_REPO, primary_revision=FEATURES_PRIMARY_REVISION
    )

    if is_sharded:
        root = DATA_DIR / "localization" / f"{repo_id.split('/')[-1]}-{revision}"
        split_paths = {
            split: download_shard_split(repo_id, split, root, revision=revision)
            for split in ("train", "valid", "test")
        }
        classes = classes_from_osl_json(json.loads(split_paths["test"].read_text(encoding="utf-8")))
        return {"data_root": root, "split_paths": split_paths, "classes": classes}

    # Fallback: known-good, real, but loose-file dataset (2210 individual
    # files) -- capped to OSL_RELEASE_MAX_GAMES games/split by default.
    require_repo_access(repo_id)
    root = DATA_DIR / "localization" / "soccernet_features"
    max_games = max_items_for("OSL_RELEASE_MAX_GAMES", 10)

    root.mkdir(parents=True, exist_ok=True)
    split_paths = {}
    for split in ("train", "valid", "test"):
        split_dir = root / "ResNET_PCA512" / split
        ann_path = download_files(
            repo_id, [f"ResNET_PCA512/{split}/annotations.json"], root
        )[0]
        payload = json.loads(ann_path.read_text(encoding="utf-8"))

        videos_by_game: dict[str, list[dict]] = defaultdict(list)
        for video in payload["videos"]:
            game = str(Path(video["path"]).parent)
            videos_by_game[game].append(video)

        games = list(videos_by_game)[: max_games] if max_games is not None else list(videos_by_game)
        kept_videos = [v for game in games for v in videos_by_game[game]]

        npy_paths = [f"ResNET_PCA512/{split}/{v['path']}" for v in kept_videos]
        report_step(f"Downloading {len(npy_paths)} feature files for split={split} ({len(games)} games)")
        download_files(repo_id, npy_paths, root)

        subset_path = split_dir / "annotations.json"
        subset_path.write_text(json.dumps({**payload, "videos": kept_videos}, indent=2), encoding="utf-8")
        split_paths[split] = subset_path

    classes = json.loads(split_paths["test"].read_text(encoding="utf-8"))["labels"]
    return {"data_root": root, "split_paths": split_paths, "classes": classes}


def _feature_family_overrides(run_name: str, dataset: dict) -> dict:
    split_paths = dataset["split_paths"]
    overrides = system_block(f"localization_{run_name}")
    overrides["DATA"] = {
        "common": {
            "dataset_name": dataset["data_root"].name,
            # Informational only -- each split below sets its own
            # source_path explicitly, since the sharded and loose-file
            # fallback layouts don't share a single common root.
            "data_root": str(split_paths["train"].parent),
            "classes": dataset["classes"],
            "splits": {
                "train": {
                    "annotation_path": str(split_paths["train"]),
                    "source_path": str(split_paths["train"].parent),
                },
                "valid": {
                    "annotation_path": str(split_paths["valid"]),
                    "source_path": str(split_paths["valid"].parent),
                },
                "test": {
                    "annotation_path": str(split_paths["test"]),
                    "source_path": str(split_paths["test"].parent),
                    "results": str(CACHE_ROOT / "outputs" / f"localization_{run_name}_results"),
                },
            },
        },
    }
    overrides["TRAIN"] = {"epochs": epochs_for(3)}
    return overrides


@pytest.mark.release
def test_localization_calf_contextaware(features_dataset):
    require_release_enabled()
    overrides = _feature_family_overrides("calf", features_dataset)
    config_path = materialize_config(
        "localization", "calf_resnetpca512", overrides, out_name="loc_calf.yaml"
    )
    _run_localization_pipeline(config_path, features_dataset, "calf")


@pytest.mark.release
def test_localization_netvladpp_learnablepooling(features_dataset):
    require_release_enabled()
    overrides = _feature_family_overrides("netvladpp", features_dataset)
    config_path = materialize_config(
        "localization", "netvladpp_resnetpca512", overrides, out_name="loc_netvladpp.yaml"
    )
    _run_localization_pipeline(config_path, features_dataset, "netvladpp")
