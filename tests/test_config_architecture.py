from pathlib import Path

import pytest

from opensportslib.core.config import load_config, migrate_config, resolve_config, validate_config
from opensportslib.core.config.accessors import (
    get_vqa_xvars_feature_mode,
    get_xvars_feature_token_len_for_mode,
    get_xvars_infer_video_token_len,
    get_xvars_train_video_token_len,
)
from opensportslib.core.config.conflicts import assert_no_legacy_aliases
from opensportslib.models.builder import build_model_from_config


def test_public_config_api_is_canonical_first():
    import opensportslib.core.config as config_module

    assert hasattr(config_module, "load_config")
    assert hasattr(config_module, "validate_config")
    assert hasattr(config_module, "migrate_config")


def test_legacy_inputs_route_through_migration(tmp_path):
    config_path = tmp_path / "legacy.yaml"
    config_path.write_text(
        "\n".join(
            [
                "DATA:",
                f"  data_dir: {tmp_path / 'data'}",
                "  annotations:",
                f"    train: {tmp_path / 'train.json'}",
                "MODEL:",
                "  backbone:",
                "    type: smoke_backbone",
                "SYSTEM:",
                f"  save_dir: {tmp_path / 'ckpt'}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    loaded = load_config(str(config_path), as_namespace=False)

    assert loaded["VERSION"] == 2
    assert loaded["MODEL"]["schema_version"] == 3
    assert "components" in loaded["MODEL"]
    assert "dali" not in loaded


def test_task_defaults_compose_shared_root_defaults():
    config_path = Path("opensportslib/configs/classification/default.yaml")
    loaded = load_config(str(config_path), as_namespace=False)

    assert loaded["VERSION"] == 2
    assert loaded["TASK"] == "classification"
    assert loaded["SYSTEM"]["paths"]["log_dir"] == "./logs"
    assert loaded["SYSTEM"]["paths"]["save_dir"] == "./checkpoints"
    assert loaded["MODEL"]["runtime"]["dtype"] == "fp32"
    assert loaded["DATA"]["common"]["runtime"]["loader_backend"] == "opencv"


def test_classification_experiment_composes_all_layers():
    cfg = load_config("opensportslib/configs/classification/video.yaml", as_namespace=False)

    assert cfg["VERSION"] == 2
    assert cfg["TASK"] == "classification"
    assert cfg["SYSTEM"]["paths"]["log_dir"] == "./logs"
    assert cfg["SYSTEM"]["paths"]["save_dir"] == "./checkpoints_video"
    assert cfg["SYSTEM"]["gpu"]["count"] == 4
    assert cfg["DATA"]["common"]["dataset_name"] == "mvfouls"
    assert cfg["MODEL"]["components"]["video_encoder"]["source"]["name"] == "mvit_v2_s"


def test_localization_experiment_composes_all_layers():
    cfg = load_config("opensportslib/configs/localization/video_dali.yaml", as_namespace=False)

    assert cfg["VERSION"] == 2
    assert cfg["TASK"] == "localization"
    assert cfg["SYSTEM"]["paths"]["log_dir"] == "./logs"
    assert cfg["SYSTEM"]["paths"]["save_dir"] == "./checkpoints"
    assert cfg["SYSTEM"]["paths"]["work_dir"] == "./checkpoints"
    assert cfg["DATA"]["common"]["runtime"]["loader_backend"] == "dali"
    assert cfg["MODEL"]["components"]["video_encoder"]["source"]["name"] == "rny008_gsm"


def test_resolve_config_cpu_normalizes_canonical_dali_localization_config():
    cfg = load_config("opensportslib/configs/localization/video_dali.yaml", as_namespace=False)
    cfg["SYSTEM"]["device"] = "cpu"

    resolved = migrate_config(cfg, as_namespace=False)
    resolved = validate_config(resolved)
    resolved = resolve_config(resolved, as_namespace=False)

    assert resolved["DATA"]["common"]["runtime"]["loader_backend"] == "opencv"
    assert resolved["DATA"]["common"]["splits"]["train"]["type"] == "VideoGameWithOpencv"
    assert resolved["DATA"]["common"]["splits"]["valid"]["type"] == "VideoGameWithOpencv"
    assert resolved["DATA"]["common"]["splits"]["test"]["type"] == "VideoGameWithOpencvVideo"


def test_vqa_xvars_experiment_composes_all_layers():
    cfg = load_config("opensportslib/configs/vqa/xvars.yaml", as_namespace=False)

    assert cfg["VERSION"] == 2
    assert cfg["TASK"] == "vqa"
    assert cfg["SYSTEM"]["paths"]["log_dir"] == "./logs"
    assert cfg["SYSTEM"]["paths"]["save_dir"] == "./checkpoints_vqa_lora"
    assert cfg["SYSTEM"]["paths"]["work_dir"] == "./checkpoints_vqa_lora"
    assert cfg["DATA"]["common"]["runtime"]["loader_backend"] == "opencv"
    assert cfg["MODEL"]["metadata"]["backend"] == "xvars_videochatgpt"
    assert cfg["TRAIN"]["execution"]["hf"]["tokenizer_id"] == "/home/vorajv/X-VARS/weights/base_model_videoChatGPT"


def test_vqa_qwen_experiment_composes_all_layers():
    cfg = load_config("opensportslib/configs/vqa/qwen.yaml", as_namespace=False)

    assert cfg["VERSION"] == 2
    assert cfg["TASK"] == "vqa"
    assert cfg["SYSTEM"]["paths"]["log_dir"] == "./logs"
    assert cfg["SYSTEM"]["paths"]["save_dir"] == "./checkpoints_vqa_qwen"
    assert cfg["SYSTEM"]["paths"]["work_dir"] == "./checkpoints_vqa_qwen"
    assert cfg["DATA"]["common"]["runtime"]["loader_backend"] == "opencv"
    assert cfg["MODEL"]["metadata"]["backend"] == "qwen_xvars_infer"
    assert cfg["MODEL"]["components"]["llm_decoder"]["source"]["name"] == "Qwen/Qwen3.5-9B-Base"
    assert cfg["TRAIN"]["execution"]["hf"]["offload_folder"] == "./hf_offload_qwen"


def test_vqa_qwen_vl_native_experiment_composes_all_layers():
    cfg = load_config("opensportslib/configs/vqa/qwen3_vl_native.yaml", as_namespace=False)

    assert cfg["VERSION"] == 2
    assert cfg["TASK"] == "vqa"
    assert cfg["SYSTEM"]["paths"]["log_dir"] == "./logs"
    assert cfg["SYSTEM"]["paths"]["save_dir"] == "./checkpoints_vqa_qwen3_vl_native"
    assert cfg["SYSTEM"]["paths"]["work_dir"] == "./checkpoints_vqa_qwen3_vl_native"
    assert cfg["DATA"]["common"]["runtime"]["loader_backend"] == "opencv"
    assert cfg["MODEL"]["metadata"]["backend"] == "qwen_vl_native_infer"
    assert cfg["MODEL"]["components"]["llm_decoder"]["source"]["name"] == "Qwen/Qwen3-VL-8B-Instruct"
    assert cfg["TRAIN"]["execution"]["training_backend"] == "qwen_vl_native_lora"


def test_validation_accepts_canonical_schema():
    canonical = load_config(
        "opensportslib/configs/localization/default.yaml",
        as_namespace=False,
    )

    validated = validate_config(canonical)

    assert validated["VERSION"] == 2


def test_builder_exposes_version_neutral_dispatcher():
    assert callable(build_model_from_config)


def test_layered_merge_preserves_sibling_keys():
    cfg = load_config("opensportslib/configs/classification/video.yaml", as_namespace=False)

    assert cfg["SYSTEM"]["paths"]["log_dir"] == "./logs"
    assert cfg["SYSTEM"]["paths"]["save_dir"] == "./checkpoints_video"
    assert cfg["MODEL"]["runtime"]["device"] == "auto"


def test_migrate_config_rejects_legacy_aliases_in_canonical_payload():
    canonical_with_alias = {
        "TASK": "classification",
        "VERSION": 3,
        "SYSTEM": {"paths": {}, "gpu": {}, "reproducibility": {}, "device": "cpu"},
        "DATA": {"common": {"runtime": {}, "splits": {}, "classes": [], "dataset_name": "x"}, "inputs": {}},
        "MODEL": {"schema_version": 3, "task": "classification", "components": {"video_encoder": {"kind": "encoder", "source": {"provider": "opensportslib"}}}, "topology": [], "backbone": {}},
        "TRAIN": {"trainer": {"type": "classification"}, "epochs": 1},
        "IO": {"inputs": {}, "outputs": {}},
    }
    with pytest.raises(ValueError, match="Legacy alias"):
        assert_no_legacy_aliases(canonical_with_alias)


def test_migrate_config_accepts_legacy_and_returns_canonical_without_aliases():
    legacy = {
        "TASK": "localization",
        "DATA": {
            "dali": True,
            "data_dir": "/tmp/data",
            "classes": ["PASS", "SHOT"],
            "train": {"path": "/tmp/train.json", "video_path": "/tmp"},
        },
        "MODEL": {"backbone": {"type": "rny008_gsm"}, "head": {"type": "gru"}},
        "TRAIN": {"type": "trainer_e2e", "num_epochs": 2},
        "SYSTEM": {"GPU": 1, "device": "cuda"},
    }
    canonical = migrate_config(legacy, as_namespace=False)
    assert "dali" not in canonical
    assert "annotations" not in canonical["DATA"]
    assert canonical["TRAIN"]["epochs"] == 2
