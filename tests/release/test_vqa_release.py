"""Release verification: VQAModel LoRA training across backends.

Dataset: OpenSportsLab/OSL-XFoul -- the dataset behind every VQA checkpoint
in the model zoo (OSL-VQA-XFOUL-XVARS-lora, OSL-VQA-XFOUL-qwen2.5-7B-VL-lora,
OSL-VQA-XFOUL-qwen3-8B-VL-lora). It's one of the three datasets pinned in the
OpenSportsLab/osl-ready-datasets Hugging Face collection, which the org
publishes as parquet + webdataset shards (a handful of large files per split,
not one file per sample) specifically so consumers don't need to pull
thousands of individual clips one at a time.

Like the other two OSL-ready repos, OSL-XFoul's default ("main") branch is an
empty placeholder -- the real data lives on named branches: "224p" and
"720p" (train/valid/test shards each). This module defaults to "224p"
(smaller/faster); override with OSL_RELEASE_VQA_REVISION=720p for the
higher-resolution branch. There's no non-sharded fallback dataset for VQA in
the org, so unlike the classification/localization fixtures this one has
nothing to fall back to -- it's OSL-XFoul@<revision> or skip.

download_shard_split() (in _release_common.py) downloads and converts each
split via opensportslib.tools.hf_transfer.download_dataset_split_from_hf(...,
download_format="parquet"), which expects the `<split>/metadata.parquet` +
`<split>/shard_manifest.parquet` + `<split>/shards/shard-*.tar` layout (see
opensportslib/tools/parquet_to_osl_json.py) and converts it back to a local
OSL v2 JSON with media extracted -- confirmed to match OSL-XFoul's real
branch layout.

Backends covered, each via its real canonical config
(opensportslib/configs/vqa/*.yaml) with TRAIN.execution.training_backend
already set correctly by that file's own defaults:

* xvars       -- X-VARS / VideoChatGPT + LoRA (configs/vqa/xvars.yaml).
                 Requires `opensportslib setup --vqa_xvars`.
* qwen_lora   -- CLIP features + Qwen LoRA (configs/vqa/qwen_lora.yaml).
                 Requires `opensportslib setup --vqa_qwen`.
* qwen3_vl_native -- full end-to-end QwenVL LoRA (configs/vqa/qwen3_vl_native.yaml).
                 Heaviest (downloads an 8B-parameter VLM). Requires
                 `opensportslib setup --vqa_qwen`.

Each test imports its backend's runtime module lazily and skips with the
actual ImportError message if the optional dependency profile hasn't been
installed, rather than guessing package names up front.

Run:
    RUN_OSL_RELEASE_TESTS=1 pytest tests/release/test_vqa_release.py -v -s

    # higher-resolution branch:
    RUN_OSL_RELEASE_TESTS=1 OSL_RELEASE_VQA_REVISION=720p \\
        pytest tests/release/test_vqa_release.py -v -s

    # skip the heaviest backend:
    RUN_OSL_RELEASE_TESTS=1 pytest tests/release/test_vqa_release.py -v -s \\
        -k "not qwen3_vl_native"
"""

from __future__ import annotations

import json
import os

import pytest

from opensportslib.apis.vqa import VQAModel

from ._release_common import (
    CACHE_ROOT,
    DATA_DIR,
    download_shard_split,
    epochs_for,
    materialize_config,
    prefer_osl_ready_dataset,
    report_step,
    require_release_enabled,
    require_repo_access,
    system_block,
)

XFOUL_REPO = os.environ.get("OSL_RELEASE_VQA_REPO", "OpenSportsLab/OSL-XFoul")
XFOUL_REVISION = os.environ.get("OSL_RELEASE_VQA_REVISION", "224p")

os.environ.setdefault("WANDB_MODE", "disabled")


@pytest.fixture(scope="module")
def xfoul_dataset():
    require_release_enabled()
    require_repo_access(XFOUL_REPO)

    repo_id, revision, is_sharded = prefer_osl_ready_dataset(
        XFOUL_REPO, fallback=None, primary_revision=XFOUL_REVISION
    )
    assert is_sharded  # fallback=None means we only ever get here via the primary

    root = DATA_DIR / "vqa" / f"OSL-XFoul-{revision}"
    split_paths = {}
    for split in ("train", "valid", "test"):
        split_paths[split] = download_shard_split(repo_id, split, root, revision=revision)
        payload = json.loads(split_paths[split].read_text(encoding="utf-8"))
        print(f"{split}: {len(payload.get('data', []))} samples -> {split_paths[split]}")

    return {"data_root": root, "split_paths": split_paths}


def _run_vqa_pipeline(config_path: str, dataset: dict, run_name: str) -> None:
    split_paths = dataset["split_paths"]

    report_step(f"[{run_name}] instantiate VQAModel")
    model = VQAModel(config=config_path, weights=None)

    report_step(f"[{run_name}] train() (LoRA)")
    checkpoint = model.train(
        train_set=str(split_paths["train"]),
        valid_set=str(split_paths["valid"]),
        use_wandb=False,
    )
    assert checkpoint, f"[{run_name}] train() did not return a checkpoint"

    report_step(f"[{run_name}] infer()")
    predictions = model.infer(test_set=str(split_paths["test"]), weights=checkpoint, use_wandb=False)
    assert isinstance(predictions, dict) and predictions.get("data"), f"[{run_name}] empty predictions"

    report_step(f"[{run_name}] save_predictions()")
    pred_path = CACHE_ROOT / "outputs" / f"vqa_{run_name}_predictions.json"
    model.save_predictions(output_path=str(pred_path), predictions=predictions)
    assert pred_path.exists()

    report_step(f"[{run_name}] evaluate()")
    metrics = model.evaluate(test_set=str(split_paths["test"]), predictions=predictions, use_wandb=False)
    print(f"[{run_name}] metrics: {metrics}")


def _dataset_overrides(run_name: str, dataset: dict) -> dict:
    overrides = system_block(f"vqa_{run_name}", gpu_count=1)
    split_paths = dataset["split_paths"]
    overrides["DATA"] = {
        "common": {
            "dataset_name": "OSL-XFoul",
            "data_root": str(dataset["data_root"]),
            "splits": {
                split: {"annotation_path": str(path), "source_path": str(dataset["data_root"])}
                for split, path in split_paths.items()
            },
        }
    }
    overrides["TRAIN"] = {"epochs": epochs_for(1)}
    return overrides


@pytest.mark.release
def test_vqa_xvars_videochatgpt_lora(xfoul_dataset):
    require_release_enabled()
    try:
        import peft  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"X-VARS LoRA deps not installed (run `opensportslib setup --vqa_xvars` first): {exc}")

    overrides = _dataset_overrides("xvars", xfoul_dataset)
    config_path = materialize_config("vqa", "xvars", overrides, out_name="vqa_xvars.yaml")
    _run_vqa_pipeline(config_path, xfoul_dataset, "xvars")


@pytest.mark.release
def test_vqa_clip_qwen_lora(xfoul_dataset):
    require_release_enabled()
    try:
        import peft  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"Qwen LoRA deps not installed (run `opensportslib setup --vqa_qwen` first): {exc}")

    overrides = _dataset_overrides("qwen_lora", xfoul_dataset)
    config_path = materialize_config("vqa", "qwen_lora", overrides, out_name="vqa_qwen_lora.yaml")
    _run_vqa_pipeline(config_path, xfoul_dataset, "qwen_lora")


@pytest.mark.release
@pytest.mark.slow
def test_vqa_qwen3_vl_native_lora(xfoul_dataset):
    """Heaviest backend: downloads an 8B-parameter end-to-end VLM."""
    require_release_enabled()
    try:
        import peft  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"Qwen VL native LoRA deps not installed (run `opensportslib setup --vqa_qwen` first): {exc}")

    overrides = _dataset_overrides("qwen3_vl_native", xfoul_dataset)
    config_path = materialize_config(
        "vqa", "qwen3_vl_native", overrides, out_name="vqa_qwen3_vl_native.yaml"
    )
    _run_vqa_pipeline(config_path, xfoul_dataset, "qwen3_vl_native")
