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
nothing to fall back to -- an enabled release run fails with a remediation
message until OSL-XFoul@<revision> is available.

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

Each test imports its backend lazily and fails with the actual ImportError
message when the required release dependency profile is incomplete.

Run through the repository's single command: bash scripts/run_tests.sh.
"""

from __future__ import annotations

import json
import os
import time

import pytest

from opensportslib.apis.vqa import VQAModel

from ._release_common import (
    CACHE_ROOT,
    DATA_DIR,
    download_shard_split,
    training_overrides,
    materialize_config,
    prefer_osl_ready_dataset,
    report_step,
    record_release_metadata,
    download_model_snapshot,
    require_model_repo,
    run_xvars_preprocessing,
    require_release_enabled,
    require_repo_access,
    system_block,
)

XFOUL_REPO = os.environ.get("OSL_RELEASE_VQA_REPO", "OpenSportsLab/OSL-XFoul")
XFOUL_REVISION = os.environ.get("OSL_RELEASE_VQA_REVISION", "224p")
XVARS_BASE_MODEL = "OpenSportsLab/base_model_videoChatGPT"
XVARS_VISUAL_MODEL = "OpenSportsLab/trained-clip-vit-large-patch14"
PUBLISHED_VQA_MODELS = {
    "xvars": ("OpenSportsLab/OSL-VQA-XFOUL-XVARS-lora", "xvars"),
    "qwen25": ("OpenSportsLab/OSL-VQA-XFOUL-qwen2.5-7B-VL-lora", "qwen3_vl_native"),
    "qwen3": ("OpenSportsLab/OSL-VQA-XFOUL-qwen3-8B-VL-lora", "qwen3_vl_native"),
}

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
    started = time.perf_counter()
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
    record_release_metadata(
        "pipeline", task="vqa", model_family=run_name,
        config_path=config_path, checkpoint_path=str(checkpoint),
    )

    report_step(f"[{run_name}] fresh checkpoint load + infer()")
    restored_model = VQAModel(config=config_path, weights=checkpoint)
    predictions = restored_model.infer(test_set=str(split_paths["test"]), use_wandb=False)
    assert isinstance(predictions, dict) and predictions.get("data"), f"[{run_name}] empty predictions"

    report_step(f"[{run_name}] save_predictions()")
    pred_path = CACHE_ROOT / "outputs" / f"vqa_{run_name}_predictions.json"
    model.save_predictions(output_path=str(pred_path), predictions=predictions)
    assert pred_path.exists()
    record_release_metadata("prediction", task="vqa", model_family=run_name, prediction_path=str(pred_path))

    report_step(f"[{run_name}] evaluate()")
    metrics = restored_model.evaluate(test_set=str(split_paths["test"]), predictions=predictions, use_wandb=False)
    record_release_metadata(
        "result", task="vqa", model_family=run_name,
        runtime_seconds=round(time.perf_counter() - started, 3),
    )
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
    overrides["TRAIN"] = training_overrides(1)
    return overrides


@pytest.mark.release
@pytest.mark.vqa_xvars
def test_xvars_dependency_models_and_preprocessing(xfoul_dataset):
    """Download/cache the real X-VARS prerequisites and materialize features."""
    require_release_enabled()
    dependency_root = CACHE_ROOT / "models" / "xvars"
    for repo_id in (XVARS_BASE_MODEL, XVARS_VISUAL_MODEL):
        require_model_repo(repo_id)
        path = download_model_snapshot(repo_id, dependency_root / repo_id.rsplit("/", 1)[-1])
        assert any(path.rglob("*")), f"{repo_id} downloaded no files"
    visual_root = dependency_root / XVARS_VISUAL_MODEL.rsplit("/", 1)[-1]
    checkpoints = list(visual_root.rglob("14_model.pth.tar"))
    assert checkpoints, f"{XVARS_VISUAL_MODEL} is missing required 14_model.pth.tar"
    feature_root = DATA_DIR / "vqa" / "xvars_features"
    run_xvars_preprocessing(xfoul_dataset["data_root"], feature_root, weights_path=checkpoints[0])


@pytest.mark.release
@pytest.mark.parametrize("name,model_id,preset", [
    pytest.param(name, model_id, preset, marks=pytest.mark.vqa_xvars if name == "xvars" else ())
    for name, (model_id, preset) in PUBLISHED_VQA_MODELS.items()
])
def test_published_vqa_model_inference(xfoul_dataset, name, model_id, preset):
    require_release_enabled()
    require_model_repo(model_id)
    overrides = _dataset_overrides(f"published_{name}", xfoul_dataset)
    if name == "qwen25":
        overrides.setdefault("MODEL", {}).setdefault("components", {}).setdefault("llm_decoder", {}).setdefault("params", {})["repo_id"] = "Qwen/Qwen2.5-VL-7B-Instruct"
    config_path = materialize_config("vqa", preset, overrides, out_name=f"published_{name}.yaml")
    model = VQAModel(config=config_path, weights=model_id)
    predictions = model.infer(test_set=str(xfoul_dataset["split_paths"]["test"]), use_wandb=False)
    assert isinstance(predictions, dict) and predictions.get("data"), f"{name} produced no predictions"
    metrics = model.evaluate(test_set=str(xfoul_dataset["split_paths"]["test"]), predictions=predictions, use_wandb=False)
    record_release_metadata("published_model_result", task="vqa", model_id=model_id, metrics=metrics)


@pytest.mark.release
@pytest.mark.vqa_xvars
def test_vqa_xvars_videochatgpt_lora(xfoul_dataset):
    require_release_enabled()
    try:
        import peft  # noqa: F401
    except ImportError as exc:
        pytest.fail(f"X-VARS LoRA deps not installed (run `opensportslib setup --vqa_xvars` first): {exc}")

    overrides = _dataset_overrides("xvars", xfoul_dataset)
    config_path = materialize_config("vqa", "xvars", overrides, out_name="vqa_xvars.yaml")
    _run_vqa_pipeline(config_path, xfoul_dataset, "xvars")


@pytest.mark.release
@pytest.mark.vqa_qwen
def test_vqa_clip_qwen_lora(xfoul_dataset):
    require_release_enabled()
    try:
        import peft  # noqa: F401
    except ImportError as exc:
        pytest.fail(f"Qwen LoRA deps not installed (run `opensportslib setup --vqa_qwen` first): {exc}")

    overrides = _dataset_overrides("qwen_lora", xfoul_dataset)
    config_path = materialize_config("vqa", "qwen_lora", overrides, out_name="vqa_qwen_lora.yaml")
    _run_vqa_pipeline(config_path, xfoul_dataset, "qwen_lora")


@pytest.mark.release
@pytest.mark.slow
@pytest.mark.vqa_qwen
def test_vqa_qwen3_vl_native_lora(xfoul_dataset):
    """Heaviest backend: downloads an 8B-parameter end-to-end VLM."""
    require_release_enabled()
    try:
        import peft  # noqa: F401
    except ImportError as exc:
        pytest.fail(f"Qwen VL native LoRA deps not installed (run `opensportslib setup --vqa_qwen` first): {exc}")

    overrides = _dataset_overrides("qwen3_vl_native", xfoul_dataset)
    config_path = materialize_config(
        "vqa", "qwen3_vl_native", overrides, out_name="vqa_qwen3_vl_native.yaml"
    )
    _run_vqa_pipeline(config_path, xfoul_dataset, "qwen3_vl_native")
