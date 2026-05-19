from pathlib import Path
from copy import deepcopy

import yaml

from opensportslib.core.config import (
    adapt_config_to_runtime,
    load_config,
    migrate_config,
    validate_config,
)
from opensportslib.models.builder import build_model_from_config
from opensportslib.core.utils.load_annotations import check_config


def test_public_config_api_is_version_neutral():
    import opensportslib.core.config as config_module

    assert hasattr(config_module, "load_config")
    assert hasattr(config_module, "validate_config")
    assert hasattr(config_module, "migrate_config")
    assert not hasattr(config_module, "load_config_v3")
    assert not hasattr(config_module, "validator_v3")


def test_version_1_inputs_route_through_migration(tmp_path):
    config_path = tmp_path / "legacy.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "DATA": {
                    "data_dir": str(tmp_path / "data"),
                    "annotations": {"train": str(tmp_path / "train.json")},
                },
                "MODEL": {"backbone": {"type": "smoke_backbone"}},
                "SYSTEM": {"save_dir": str(tmp_path / "ckpt")},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    loaded = load_config(str(config_path), compatibility=False, as_namespace=False)

    assert loaded["VERSION"] == 3
    assert loaded["MODEL"]["schema_version"] == 3
    assert "components" in loaded["MODEL"]


def test_version_3_inputs_load_directly_from_same_api():
    config_path = Path("opensportslib/configs/classification/default.yaml")
    loaded = load_config(str(config_path), compatibility=False, as_namespace=False)

    assert loaded["VERSION"] == 3
    assert loaded["TASK"] == "classification"
    assert loaded["MODEL"]["schema_version"] == 3


def test_runtime_adapter_preserves_legacy_surface():
    canonical = load_config(
        "opensportslib/configs/classification/default.yaml",
        compatibility=False,
        as_namespace=False,
    )
    runtime = adapt_config_to_runtime(canonical, as_namespace=False)

    assert runtime["SYSTEM"]["save_dir"] == "./checkpoints"
    assert runtime["SYSTEM"]["GPU"] == 1
    assert runtime["DATA"]["train"]["path"] == "/path/to/train.json"
    assert runtime["MODEL"]["backbone"]["type"] is None


def test_validation_accepts_canonical_v3():
    canonical = load_config(
        "opensportslib/configs/localization/default.yaml",
        compatibility=False,
        as_namespace=False,
    )

    validated = validate_config(canonical)

    assert validated["VERSION"] == 3


def test_builder_exposes_version_neutral_dispatcher():
    assert callable(build_model_from_config)


def test_v1_top_level_dali_migrates_to_v3_loader_backend():
    runtime = load_config(
        "opensportslib/config/localization.yaml",
        compatibility=True,
        as_namespace=False,
    )

    assert runtime["dali"] is True
    assert runtime["DATA"]["common"]["runtime"]["loader_backend"] == "dali"


def test_v1_data_level_dali_migrates_to_v3_loader_backend():
    legacy = {
        "TASK": "localization",
        "DATA": {
            "dali": True,
            "data_dir": "/tmp/data",
            "classes": ["PASS", "SHOT"],
            "train": {"path": "/tmp/train.json", "video_path": "/tmp"},
        },
        "MODEL": {"backbone": {"type": "rny008_gsm"}, "head": {"type": "gru"}},
        "TRAIN": {"type": "trainer_e2e"},
        "SYSTEM": {"GPU": 1, "device": "cuda"},
    }
    runtime = adapt_config_to_runtime(migrate_config(legacy, as_namespace=False), as_namespace=False)

    assert runtime["dali"] is True
    assert runtime["DATA"]["common"]["runtime"]["loader_backend"] == "dali"


def test_runtime_adapter_sets_dali_true_for_v3_dali_backend():
    canonical = load_config(
        "opensportslib/configs/localization/video/localization-dali.yaml",
        compatibility=False,
        as_namespace=False,
    )
    runtime = adapt_config_to_runtime(canonical, as_namespace=False)

    assert runtime["dali"] is True


def test_runtime_adapter_sets_dali_false_for_non_dali_or_missing_backend():
    canonical = load_config(
        "opensportslib/configs/localization/video/localization-dali.yaml",
        compatibility=False,
        as_namespace=False,
    )

    non_dali = deepcopy(canonical)
    non_dali["DATA"]["common"]["runtime"]["loader_backend"] = "opencv"
    for split_cfg in non_dali["DATA"]["common"]["splits"].values():
        split_cfg["type"] = str(split_cfg.get("type", "")).replace("Dali", "Opencv")
    runtime_non_dali = adapt_config_to_runtime(non_dali, as_namespace=False)
    assert runtime_non_dali["dali"] is False

    missing_runtime = deepcopy(canonical)
    missing_runtime["DATA"]["common"].pop("runtime", None)
    for split_cfg in missing_runtime["DATA"]["common"]["splits"].values():
        split_cfg["type"] = str(split_cfg.get("type", "")).replace("Dali", "Opencv")
    runtime_missing = adapt_config_to_runtime(missing_runtime, as_namespace=False)
    assert runtime_missing["dali"] is False


def test_runtime_adapter_non_v3_passthrough_unchanged():
    legacy = {"dali": True, "DATA": {"classes": ["PASS", "SHOT"]}, "MODEL": {"runner": {"type": "runner_e2e"}}}
    runtime = adapt_config_to_runtime(legacy, as_namespace=False)
    assert runtime == legacy


def test_localization_check_config_accepts_v3_runtime_without_dali_attribute_error():
    runtime_cfg = load_config(
        "opensportslib/configs/localization/video/localization-dali.yaml",
        compatibility=True,
        as_namespace=True,
    )
    check_config(runtime_cfg, split="train")
    assert hasattr(runtime_cfg, "dali")
