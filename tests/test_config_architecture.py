from pathlib import Path

import yaml

from opensportslib.core.config import (
    adapt_config_to_runtime,
    load_config,
    migrate_config,
    validate_config,
)
from opensportslib.models.builder import build_model_from_config


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
