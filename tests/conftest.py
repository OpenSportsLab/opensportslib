from pathlib import Path
import json

import pytest
import yaml


def _write_config(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False)
    return str(path)


def _system_block(save_dir: Path, log_dir: Path, gpu_count: int = 0) -> dict:
    return {
        "paths": {
            "save_dir": str(save_dir),
            "work_dir": str(save_dir),
            "log_dir": str(log_dir),
        },
        "device": "cpu",
        "gpu": {"count": gpu_count, "id": 0},
        "reproducibility": {"use_seed": False, "seed": 0},
    }


def _classification_payload(
    data_root: Path,
    save_dir: Path,
    log_dir: Path,
    train_path: str | None = None,
    valid_path: str | None = None,
    test_path: str | None = None,
) -> dict:
    return {
        "TASK": "classification",
        "VERSION": 3,
        "SYSTEM": _system_block(save_dir, log_dir),
        "DATA": {
            "common": {
                "dataset_name": "mvfouls",
                "data_root": str(data_root),
                "classes": ["PASS", "SHOT"],
                "runtime": {"loader_backend": "opencv"},
                "splits": {
                    "train": {
                        "type": "annotations_train.json",
                        "annotation_path": train_path,
                        "source_path": str(data_root),
                        "dataloader": {
                            "batch_size": 1,
                            "shuffle": True,
                            "num_workers": 0,
                            "pin_memory": False,
                            "mp_context": "spawn",
                            "persistent_workers": False,
                        },
                    },
                    "valid": {
                        "type": "annotations_valid.json",
                        "annotation_path": valid_path,
                        "source_path": str(data_root),
                        "dataloader": {
                            "batch_size": 1,
                            "shuffle": False,
                            "num_workers": 0,
                            "pin_memory": False,
                            "mp_context": "spawn",
                            "persistent_workers": False,
                        },
                    },
                    "test": {
                        "type": "annotations_test.json",
                        "annotation_path": test_path,
                        "source_path": str(data_root),
                        "dataloader": {
                            "batch_size": 1,
                            "shuffle": False,
                            "num_workers": 0,
                            "pin_memory": False,
                            "mp_context": "spawn",
                            "persistent_workers": False,
                        },
                    },
                },
            },
            "inputs": {
                "video": {
                    "modality": "video",
                    "representation": "raw",
                    "source": {"format": "mp4"},
                    "sampling": {
                        "num_frames": 16,
                        "input_fps": 25,
                        "target_fps": 17,
                        "start_frame": 63,
                        "end_frame": 87,
                    },
                    "transform": {"resize": {"height": 224, "width": 224}},
                    "augmentations": {},
                    "params": {"view_type": "single", "num_classes": 2},
                }
            },
        },
        "MODEL": {
            "schema_version": 3,
            "task": "classification",
            "runtime": {
                "dtype": "fp32",
                "device": "auto",
                "compile": False,
                "freeze": False,
            },
            "load": {
                "checkpoint_path": None,
                "pretrained": False,
                "strict": True,
                "map_location": None,
                "format": "auto",
            },
            "components": {
                "video_encoder": {
                    "kind": "encoder",
                    "source": {
                        "provider": "opensportslib",
                        "registry": "backbone",
                        "name": "smoke_backbone",
                    },
                    "params": {},
                    "overrides": {},
                },
                "task_head": {
                    "kind": "head",
                    "source": {
                        "provider": "opensportslib",
                        "registry": "head",
                        "name": "smoke_head",
                    },
                    "params": {"num_classes": 2},
                    "overrides": {},
                },
            },
            "topology": [{"from": "video_encoder", "to": "task_head"}],
        },
        "IO": {
            "inputs": {"video": "video_encoder", "label": "task_head"},
            "outputs": {"logits": "task_head"},
        },
        "TRAIN": {
            "trainer": {"type": "classification"},
            "epochs": 1,
            "criterion": {"type": "CrossEntropyLoss"},
            "optimizer": {"type": "SGD", "lr": 0.1},
            "scheduler": {"type": "StepLR", "step_size": 1, "gamma": 0.1},
            "execution": {"enabled": True, "log_interval": 1},
            "sampling": {
                "use_weighted_sampler": False,
                "use_weighted_loss": False,
            },
            "selection": {"monitor": "loss", "mode": "min"},
            "checkpoint": {"save_every": 1, "save_best": True},
        },
    }


def _localization_payload(
    data_root: Path,
    save_dir: Path,
    log_dir: Path,
    train_path: str | None = None,
    valid_path: str | None = None,
    test_path: str | None = None,
    result_name: str | None = None,
) -> dict:
    return {
        "TASK": "localization",
        "VERSION": 3,
        "SYSTEM": _system_block(save_dir, log_dir),
        "DATA": {
            "common": {
                "dataset_name": "SoccerNet",
                "data_root": str(data_root),
                "classes": ["PASS", "SHOT"],
                "runtime": {"loader_backend": "opencv"},
                "splits": {
                    "train": {
                        "type": "VideoGameWithOpencv",
                        "annotation_path": train_path,
                        "source_path": str(data_root),
                        "output_map": ["data", "label"],
                        "dataloader": {
                            "batch_size": 1,
                            "shuffle": True,
                            "num_workers": 0,
                            "pin_memory": False,
                            "mp_context": "spawn",
                            "persistent_workers": False,
                        },
                    },
                    "valid": {
                        "type": "VideoGameWithOpencv",
                        "annotation_path": valid_path,
                        "source_path": str(data_root),
                        "output_map": ["data", "label"],
                        "dataloader": {
                            "batch_size": 1,
                            "shuffle": False,
                            "num_workers": 0,
                            "pin_memory": False,
                            "mp_context": "spawn",
                            "persistent_workers": False,
                        },
                    },
                    "valid_data_frames": {
                        "type": "VideoGameWithOpencvVideo",
                        "annotation_path": valid_path,
                        "source_path": str(data_root),
                        "output_map": ["data", "label"],
                        "overlap_len": 0,
                        "dataloader": {
                            "batch_size": 1,
                            "shuffle": False,
                            "num_workers": 0,
                            "pin_memory": False,
                            "mp_context": "spawn",
                            "persistent_workers": False,
                        },
                    },
                    "test": {
                        "type": "VideoGameWithOpencvVideo",
                        "annotation_path": test_path,
                        "source_path": str(data_root),
                        "output_map": ["data", "label"],
                        "results": result_name,
                        "metric": "tight",
                        "nms_window": 2,
                        "overlap_len": 50,
                        "dataloader": {
                            "batch_size": 1,
                            "shuffle": False,
                            "num_workers": 0,
                            "pin_memory": False,
                            "mp_context": "spawn",
                            "persistent_workers": False,
                        },
                    },
                },
            },
            "inputs": {
                "video": {
                    "modality": "video",
                    "representation": "raw",
                    "source": {"format": "mp4"},
                    "sampling": {
                        "epoch_num_frames": 64,
                        "clip_len": 16,
                        "input_fps": 25,
                        "extract_fps": 2,
                    },
                    "transform": {
                        "resize": {"height": 224, "width": 224},
                        "normalization": {
                            "mean": [0.485, 0.456, 0.406],
                            "std": [0.229, 0.224, 0.225],
                        },
                    },
                    "augmentations": {},
                    "params": {"crop_dim": -1, "mixup": False, "dilate_len": 0},
                }
            },
        },
        "MODEL": {
            "schema_version": 3,
            "task": "localization",
            "runtime": {
                "dtype": "fp32",
                "device": "auto",
                "compile": False,
                "freeze": False,
            },
            "load": {
                "checkpoint_path": None,
                "pretrained": False,
                "strict": True,
                "map_location": None,
                "format": "auto",
            },
            "components": {
                "video_encoder": {
                    "kind": "encoder",
                    "source": {
                        "provider": "opensportslib",
                        "registry": "backbone",
                        "name": "smoke_backbone",
                    },
                    "params": {},
                    "overrides": {},
                },
                "task_head": {
                    "kind": "head",
                    "source": {
                        "provider": "opensportslib",
                        "registry": "head",
                        "name": "gru",
                    },
                    "params": {},
                    "overrides": {},
                },
            },
            "topology": [{"from": "video_encoder", "to": "task_head"}],
        },
        "IO": {
            "inputs": {"video": "video_encoder", "label": "task_head"},
            "outputs": {"logits": "task_head"},
        },
        "TRAIN": {
            "trainer": {"type": "trainer_e2e"},
            "epochs": 1,
            "criterion": {"type": "CrossEntropyLoss"},
            "optimizer": {"type": "AdamWithScaler", "lr": 0.01},
            "scheduler": {
                "type": "ChainedSchedulerE2E",
                "acc_grad_iter": 1,
                "num_epochs": 1,
                "warm_up_epochs": 0,
            },
            "execution": {
                "enabled": True,
                "multi_gpu": False,
                "acc_grad_iter": 1,
                "base_num_valid_epochs": 1,
                "start_valid_epoch": 0,
                "valid_map_every": 1,
                "criterion_valid": "map",
            },
            "sampling": {},
            "selection": {"monitor": "valid_loss", "mode": "min"},
            "checkpoint": {"save_best": True},
        },
    }


def make_classification_config(tmp_path: Path) -> str:
    data_dir = tmp_path / "classification_data"
    save_dir = tmp_path / "classification_save"
    log_dir = tmp_path / "classification_logs"
    data_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    payload = _classification_payload(
        data_root=data_dir,
        save_dir=save_dir,
        log_dir=log_dir,
        train_path=str(tmp_path / "train.json"),
        valid_path=str(tmp_path / "valid.json"),
    )
    return _write_config(tmp_path / "classification.yaml", payload)


def make_localization_config(tmp_path: Path) -> str:
    data_dir = tmp_path / "localization_data"
    save_dir = tmp_path / "localization_save"
    log_dir = tmp_path / "localization_logs"
    data_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    payload = _localization_payload(
        data_root=data_dir,
        save_dir=save_dir,
        log_dir=log_dir,
    )
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
                "labels": {
                    "action": {"label": label},
                    "foul_type": {"label": label},
                },
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

    payload = _classification_payload(
        data_root=data_dir,
        save_dir=save_dir,
        log_dir=log_dir,
        train_path=train_path,
        valid_path=valid_path,
        test_path=test_path,
    )
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

    payload = _localization_payload(
        data_root=data_dir,
        save_dir=save_dir,
        log_dir=log_dir,
        train_path=train_path,
        valid_path=valid_path,
        test_path=test_path,
        result_name=str(result_path),
    )
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

    payload = _classification_payload(
        data_root=data_dir,
        save_dir=save_dir,
        log_dir=log_dir,
        train_path=train_path,
        valid_path=valid_path,
        test_path=test_path,
    )
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

    payload = _localization_payload(
        data_root=data_dir,
        save_dir=save_dir,
        log_dir=log_dir,
        train_path=train_path,
        valid_path=valid_path,
        test_path=test_path,
        result_name=str(result_path),
    )
    config_path = _write_config(tmp_path / "localization-public.yaml", payload)

    return {
        "config": config_path,
        "train": train_path,
        "valid": valid_path,
        "test": test_path,
        "results": str(result_path),
    }
