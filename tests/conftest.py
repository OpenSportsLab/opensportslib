from pathlib import Path
import json

import pytest
import yaml


def _write_config(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False)
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
    return _write_config(tmp_path / "classification.yaml", payload)


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
    return _write_config(tmp_path / "localization.yaml", payload)


@pytest.fixture
def classification_config_path(tmp_path: Path) -> str:
    return make_classification_config(tmp_path)


@pytest.fixture
def localization_config_path(tmp_path: Path) -> str:
    return make_localization_config(tmp_path)


def _write_annotation(path: Path, num_samples: int = 2) -> str:
    classes = ["PASS", "SHOT"]
    items = []
    for idx in range(num_samples):
        label = classes[idx % len(classes)]
        position_ms = (idx + 1) * 1000
        game_time = f"1 - 00:{idx + 1:02d}"

        items.append(
            {
                "id": f"sample_{idx:05d}",
                "metadata": {
                    "game_id": f"game_{idx // 2:03d}",
                    "clip_id": idx,
                },
                "inputs": [
                    {
                        "type": "video",
                        "path": f"clips/video_{idx:05d}.mp4",
                        "fps": 25,
                    }
                ],
                # Classification-style annotation
                "labels": {
                    "action": {"label": label},
                    # Keep foul_type for backward compatibility with legacy utilities.
                    "foul_type": {"label": label},
                },
                # Localization-style annotation
                "events": [
                    {
                        "label": label,
                        "position_ms": position_ms,
                        "position": position_ms,
                        "gameTime": game_time,
                        "half": "1",
                    }
                ],
            }
        )

    payload = {
        "labels": {
            "action": {"labels": classes},
            "foul_type": {"labels": classes},
        },
        "data": items,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    return str(path)


@pytest.fixture
def classification_integration_assets(tmp_path: Path) -> dict:
    data_dir = tmp_path / "classification_data"
    save_dir = tmp_path / "classification_ckpt"
    log_dir = tmp_path / "classification_logs"
    data_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    train_path = _write_annotation(tmp_path / "classification-train.json", num_samples=4)
    valid_path = _write_annotation(tmp_path / "classification-valid.json", num_samples=2)
    test_path = _write_annotation(tmp_path / "classification-test.json", num_samples=2)

    payload = {
        "DATA": {
            "data_dir": str(data_dir),
            "annotations": {
                "train": train_path,
                "valid": valid_path,
                "test": test_path,
            },
        },
        "MODEL": {
            "type": "custom",
            "backbone": {"type": "smoke_backbone"},
        },
        "SYSTEM": {
            "save_dir": str(save_dir),
            "log_dir": str(log_dir),
            "GPU": 0,
            "device": "cpu",
            "use_seed": False,
            "seed": 0,
        },
        "TRAIN": {"epochs": 1},
    }
    config_path = _write_config(tmp_path / "classification-integration.yaml", payload)

    return {
        "config": config_path,
        "train": train_path,
        "valid": valid_path,
        "test": test_path,
    }


@pytest.fixture
def localization_integration_assets(tmp_path: Path) -> dict:
    data_dir = tmp_path / "localization_data"
    save_dir = tmp_path / "localization_ckpt"
    log_dir = tmp_path / "localization_logs"
    data_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    train_path = _write_annotation(tmp_path / "localization-train.json", num_samples=2)
    valid_path = _write_annotation(tmp_path / "localization-valid.json", num_samples=1)
    test_path = _write_annotation(tmp_path / "localization-test.json", num_samples=1)
    result_path = tmp_path / "localization-results.json"

    payload = {
        "dali": False,
        "DATA": {
            "data_dir": str(data_dir),
            "classes": ["PASS", "SHOT"],
            "mixup": False,
            "train": {
                "path": train_path,
                "video_path": str(data_dir),
                "dataloader": {
                    "batch_size": 1,
                    "num_workers": 0,
                    "pin_memory": False,
                },
            },
            "valid": {
                "path": valid_path,
                "video_path": str(data_dir),
                "dataloader": {
                    "batch_size": 1,
                    "num_workers": 0,
                    "pin_memory": False,
                },
            },
            "test": {
                "path": test_path,
                "video_path": str(data_dir),
                "results": str(result_path),
                "dataloader": {
                    "batch_size": 1,
                    "num_workers": 0,
                    "pin_memory": False,
                },
            },
        },
        "MODEL": {
            "backbone": {"type": "smoke_backbone"},
            "multi_gpu": False,
        },
        "TRAIN": {
            "max_epochs": 1,
            "evaluation_frequency": 1,
            "batch_size": 1,
            "type": "localization",
        },
        "SYSTEM": {
            "save_dir": str(save_dir),
            "log_dir": str(log_dir),
            "work_dir": str(save_dir),
            "GPU": 0,
            "device": "cpu",
            "seed": 0,
        },
    }
    config_path = _write_config(tmp_path / "localization-integration.yaml", payload)

    return {
        "config": config_path,
        "train": train_path,
        "valid": valid_path,
        "test": test_path,
        "results": str(result_path),
    }


@pytest.fixture
def classification_public_dataset_assets(tmp_path: Path) -> dict:
    data_dir = tmp_path / "mvfouls"
    save_dir = tmp_path / "classification_public_ckpt"
    log_dir = tmp_path / "classification_public_logs"

    train_path = _write_annotation(
        data_dir / "train" / "annotations-train.json",
        num_samples=4,
    )
    valid_path = _write_annotation(
        data_dir / "valid" / "annotations-valid.json",
        num_samples=2,
    )
    test_path = _write_annotation(
        data_dir / "test" / "annotations-test.json",
        num_samples=2,
    )

    payload = {
        "DATA": {
            "dataset_name": "mvfouls",
            "data_dir": str(data_dir),
            "train": {"path": train_path},
            "valid": {"path": valid_path},
            "test": {"path": test_path},
        },
        "MODEL": {
            "type": "custom",
            "backbone": {"type": "smoke_backbone"},
        },
        "SYSTEM": {
            "save_dir": str(save_dir),
            "log_dir": str(log_dir),
            "GPU": 0,
            "device": "cpu",
            "use_seed": False,
            "seed": 0,
        },
        "TRAIN": {"epochs": 1},
    }
    config_path = _write_config(tmp_path / "classification-public.yaml", payload)

    return {
        "config": config_path,
        "train": train_path,
        "valid": valid_path,
        "test": test_path,
    }


@pytest.fixture
def localization_public_dataset_assets(tmp_path: Path) -> dict:
    data_dir = tmp_path / "soccernet"
    save_dir = tmp_path / "localization_public_ckpt"
    log_dir = tmp_path / "localization_public_logs"
    data_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    train_path = _write_annotation(
        data_dir / "train" / "annotations-2024-224p-train.json",
        num_samples=2,
    )
    valid_path = _write_annotation(
        data_dir / "valid" / "annotations-2024-224p-valid.json",
        num_samples=1,
    )
    test_path = _write_annotation(
        data_dir / "test" / "annotations-2024-224p-test.json",
        num_samples=1,
    )
    result_path = tmp_path / "results_spotting_test"

    payload = {
        "dali": False,
        "DATA": {
            "dataset_name": "SoccerNet",
            "data_dir": str(data_dir),
            "classes": ["PASS", "SHOT"],
            "mixup": False,
            "train": {
                "path": train_path,
                "video_path": str(data_dir / "train"),
                "dataloader": {"batch_size": 1, "num_workers": 0, "pin_memory": False},
            },
            "valid": {
                "path": valid_path,
                "video_path": str(data_dir / "valid"),
                "dataloader": {"batch_size": 1, "num_workers": 0, "pin_memory": False},
            },
            "test": {
                "path": test_path,
                "video_path": str(data_dir / "test"),
                "results": str(result_path),
                "dataloader": {"batch_size": 1, "num_workers": 0, "pin_memory": False},
            },
        },
        "MODEL": {"backbone": {"type": "smoke_backbone"}, "multi_gpu": False},
        "TRAIN": {"max_epochs": 1, "evaluation_frequency": 1, "batch_size": 1, "type": "localization"},
        "SYSTEM": {
            "save_dir": str(save_dir),
            "log_dir": str(log_dir),
            "work_dir": str(save_dir),
            "GPU": 0,
            "device": "cpu",
            "seed": 0,
        },
    }
    config_path = _write_config(tmp_path / "localization-public.yaml", payload)

    return {
        "config": config_path,
        "train": train_path,
        "valid": valid_path,
        "test": test_path,
        "results": str(result_path),
    }
