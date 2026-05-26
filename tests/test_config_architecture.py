from pathlib import Path

import pytest
import yaml

from opensportslib.core.config import load_config, migrate_config, validate_config
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

    loaded = load_config(str(config_path), as_namespace=False)

    assert loaded["VERSION"] == 2
    assert "schema_version" not in loaded["MODEL"] or loaded["MODEL"]["schema_version"] == 3
    assert "components" in loaded["MODEL"]
    assert "dali" not in loaded


def test_canonical_inputs_load_directly_from_same_api():
    config_path = Path("opensportslib/configs/classification/default.yaml")
    loaded = load_config(str(config_path), as_namespace=False)

    assert loaded["VERSION"] == 2
    assert loaded["TASK"] == "classification"
    assert "components" in loaded["MODEL"]


def test_validation_accepts_canonical_schema():
    canonical = load_config(
        "opensportslib/configs/localization/default.yaml",
        as_namespace=False,
    )

    validated = validate_config(canonical)

    assert validated["VERSION"] == 2


def test_builder_exposes_version_neutral_dispatcher():
    assert callable(build_model_from_config)


def test_legacy_dali_migrates_to_canonical_loader_backend():
    cfg = load_config("opensportslib/configs/localization/default.yaml", as_namespace=False)

    assert cfg["DATA"]["common"]["runtime"]["loader_backend"] == "opencv"


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
