import json
import os
from pathlib import Path

import pytest
import yaml

from opensportslib.apis.classification import ClassificationAPI
from opensportslib.apis.localization import LocalizationAPI


pytestmark = pytest.mark.integration


def _enabled():
    return os.environ.get("RUN_OSL_SUBSET_INTEGRATION", "0").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _subset_json(src_path: Path, dst_path: Path, count: int):
    data = json.loads(src_path.read_text(encoding="utf-8"))
    data["data"] = data.get("data", [])[:count]
    dst_path.write_text(json.dumps(data), encoding="utf-8")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _require_enabled():
    if not _enabled():
        pytest.skip("Set RUN_OSL_SUBSET_INTEGRATION=1 to run train+infer integration tests.")


def test_classification_train_and_infer_subset(tmp_path, monkeypatch):
    _require_enabled()

    repo = _repo_root()
    data_dir = repo / "SoccerNet" / "mvfouls"
    cfg_src = repo / "opensportslib" / "config" / "classification.yaml"

    train_src = data_dir / "train" / "annotations_train.json"
    valid_src = data_dir / "valid" / "annotations_valid.json"
    test_src = data_dir / "test" / "annotations_test.json"

    for path in (cfg_src, train_src, valid_src, test_src):
        if not path.exists():
            pytest.skip(f"Required file not found: {path}")

    train_subset = tmp_path / "classification-train-subset.json"
    valid_subset = tmp_path / "classification-valid-subset.json"
    test_subset = tmp_path / "classification-test-subset.json"
    _subset_json(train_src, train_subset, count=8)
    _subset_json(valid_src, valid_subset, count=4)
    _subset_json(test_src, test_subset, count=4)

    cfg = yaml.safe_load(cfg_src.read_text(encoding="utf-8"))
    cfg["DATA"]["data_dir"] = str(data_dir)
    cfg["SYSTEM"]["device"] = "cpu"
    cfg["SYSTEM"]["GPU"] = 0
    cfg["SYSTEM"]["save_dir"] = str(tmp_path / "classification-checkpoints")
    cfg["SYSTEM"]["log_dir"] = str(tmp_path / "classification-logs")
    cfg["TRAIN"]["epochs"] = 1
    cfg["TRAIN"]["save_every"] = 1
    cfg["TRAIN"]["use_weighted_loss"] = False
    cfg["MODEL"]["backbone"]["type"] = "r3d_18"
    cfg["MODEL"]["pretrained_model"] = "r3d_18"
    cfg["DATA"]["train"]["dataloader"]["batch_size"] = 2
    cfg["DATA"]["valid"]["dataloader"]["batch_size"] = 2
    cfg["DATA"]["test"]["dataloader"]["batch_size"] = 2
    cfg["DATA"]["train"]["dataloader"]["num_workers"] = 0
    cfg["DATA"]["valid"]["dataloader"]["num_workers"] = 0
    cfg["DATA"]["test"]["dataloader"]["num_workers"] = 0

    cfg_path = tmp_path / "classification-subset.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    monkeypatch.setenv("OSL_PRETRAINED_WEIGHTS", "0")
    monkeypatch.setenv("WANDB_MODE", "disabled")

    api = ClassificationAPI(config=str(cfg_path), save_dir=str(tmp_path / "classification-runs"))
    checkpoint = api.train(
        train_set=str(train_subset),
        valid_set=str(valid_subset),
        use_wandb=False,
    )

    assert checkpoint is not None
    assert Path(checkpoint).exists()

    metrics = api.infer(
        test_set=str(test_subset),
        pretrained=checkpoint,
        use_wandb=False,
    )
    assert isinstance(metrics, dict)


def test_localization_train_and_infer_subset(tmp_path, monkeypatch):
    _require_enabled()

    cfg_src = Path(
        "/home/vorajv/opensportslib/opensportslib/config/localization-json_netvlad++_resnetpca512.yaml"
    )
    data_root = Path("/home/vorajv/soccernetpro/SoccerNet/SNAS-ResNET_PCA512")

    train_src = data_root / "annotations-train.json"
    valid_src = data_root / "annotations-valid.json"
    test_src = data_root / "annotations-test.json"

    for path in (cfg_src, train_src, valid_src, test_src):
        if not path.exists():
            pytest.skip(f"Required file not found: {path}")

    train_subset = tmp_path / "localization-train-subset.json"
    valid_subset = tmp_path / "localization-valid-subset.json"
    test_subset = tmp_path / "localization-test-subset.json"
    _subset_json(train_src, train_subset, count=1)
    _subset_json(valid_src, valid_subset, count=1)
    _subset_json(test_src, test_subset, count=1)

    cfg = yaml.safe_load(cfg_src.read_text(encoding="utf-8"))
    cfg["dali"] = False
    cfg["DATA"]["data_dir"] = str(data_root)
    cfg["DATA"]["mixup"] = False

    for split in ("train", "valid", "test"):
        cfg["DATA"][split]["video_path"] = str(data_root)
        cfg["DATA"][split]["dataloader"]["batch_size"] = 8 if split != "test" else 1
        cfg["DATA"][split]["dataloader"]["num_workers"] = 0
        cfg["DATA"][split]["dataloader"]["pin_memory"] = False

    cfg["DATA"]["train"]["path"] = str(train_subset)
    cfg["DATA"]["valid"]["path"] = str(valid_subset)
    cfg["DATA"]["test"]["path"] = str(test_subset)

    cfg["TRAIN"]["max_epochs"] = 1
    cfg["TRAIN"]["evaluation_frequency"] = 1
    cfg["TRAIN"]["batch_size"] = 8

    cfg["SYSTEM"]["device"] = "cpu"
    cfg["SYSTEM"]["GPU"] = 0
    cfg["SYSTEM"]["save_dir"] = str(tmp_path / "localization-checkpoints")
    cfg["SYSTEM"]["log_dir"] = str(tmp_path / "localization-logs")
    cfg["SYSTEM"]["work_dir"] = cfg["SYSTEM"]["save_dir"]

    cfg_path = tmp_path / "localization-subset.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    monkeypatch.setenv("OSL_PRETRAINED_WEIGHTS", "0")
    monkeypatch.setenv("WANDB_MODE", "disabled")

    api = LocalizationAPI(config=str(cfg_path), save_dir=str(tmp_path / "localization-runs"))
    checkpoint = api.train(
        train_set=str(train_subset),
        valid_set=str(valid_subset),
        use_wandb=False,
    )
    assert checkpoint is not None
    assert Path(checkpoint).exists()

    metrics = api.infer(
        test_set=str(test_subset),
        pretrained=checkpoint,
        use_wandb=False,
    )
    # Localization evaluator may return a formatted table string or a dict,
    # depending on evaluator backend/version.
    assert isinstance(metrics, (dict, str))
