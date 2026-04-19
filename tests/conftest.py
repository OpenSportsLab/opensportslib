from pathlib import Path
import json

import pytest


def _write_config(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    return str(path)


def make_classification_config(tmp_path: Path) -> str:
    data_dir = tmp_path / "classification_data"
    save_dir = tmp_path / "classification_save"
    log_dir = tmp_path / "classification_logs"
    data_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "DATA": {
            "data_dir": str(data_dir),
            "annotations": {
                "train": str(tmp_path / "train.json"),
                "valid": str(tmp_path / "valid.json"),
            },
        },
        "MODEL": {"backbone": {"type": "smoke_backbone"}},
        "SYSTEM": {"save_dir": str(save_dir), "log_dir": str(log_dir)},
    }
    return _write_config(tmp_path / "classification.json", payload)


def make_localization_config(tmp_path: Path) -> str:
    data_dir = tmp_path / "localization_data"
    save_dir = tmp_path / "localization_save"
    log_dir = tmp_path / "localization_logs"
    data_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "DATA": {"data_dir": str(data_dir), "classes": ["PASS", "SHOT"]},
        "MODEL": {"backbone": {"type": "smoke_backbone"}},
        "SYSTEM": {"save_dir": str(save_dir), "log_dir": str(log_dir)},
    }
    return _write_config(tmp_path / "localization.json", payload)


@pytest.fixture
def classification_config_path(tmp_path: Path) -> str:
    return make_classification_config(tmp_path)


@pytest.fixture
def localization_config_path(tmp_path: Path) -> str:
    return make_localization_config(tmp_path)
