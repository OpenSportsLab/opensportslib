from pathlib import Path

import pytest
pytest.importorskip("omegaconf")
from omegaconf import OmegaConf

from opensportslib.apis.classification import ClassificationAPI
from opensportslib.apis.localization import LocalizationAPI
from opensportslib.core.utils import config as config_utils
from opensportslib.core.utils.config import (
    fetch_and_merge_pretrained_config,
    namespace_to_dict,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _as_plain(obj):
    return namespace_to_dict(obj)


def test_fetch_merge_compatibility_keeps_runtime_and_adopts_hf_model(tmp_path):
    target_cfg = {
        "TASK": "classification",
        "VERSION": 3,
        "MODEL": {
            "schema_version": 3,
            "task": "classification",
            "components": {
                "video_encoder": {
                    "kind": "encoder",
                    "source": {"provider": "opensportslib", "name": "local_backbone"},
                    "params": {},
                    "overrides": {},
                }
            },
            "topology": [],
            "local_only_key": 7,
        },
        "DATA": {
            "common": {
                "data_root": "/local/data",
                "classes": ["A", "B"],
                "splits": {"test": {"annotation_path": "/local/test.json"}},
            },
            "inputs": {"video": {"params": {"num_classes": 8}}},
        },
        "SYSTEM": {
            "device": "cpu",
            "gpu": {"count": 1, "id": 0},
            "paths": {"save_dir": "/local/save"},
        },
        "TRAIN": {"epochs": 12},
    }
    hf_cfg = {
        "TASK": "classification",
        "VERSION": 3,
        "MODEL": {
            "schema_version": 3,
            "task": "classification",
            "components": {
                "video_encoder": {
                    "kind": "encoder",
                    "source": {"provider": "huggingface", "name": "hf_backbone"},
                    "params": {},
                    "overrides": {},
                }
            },
            "topology": [],
            "hf_key": 42,
        },
        "DATA": {
            "common": {"data_root": "/hf/data", "classes": ["X", "Y"]},
            "inputs": {"video": {"params": {"num_classes": 10}}},
        },
        "SYSTEM": {"device": "cuda", "gpu": {"count": 8, "id": 0}},
    }

    ckpt_dir = tmp_path / "hf_ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "config.yaml").write_text(OmegaConf.to_yaml(OmegaConf.create(hf_cfg)))

    merged = fetch_and_merge_pretrained_config(
        target_cfg,
        str(ckpt_dir / "model.pt"),
        merge_policy="compatibility",
    )
    merged = _as_plain(merged)

    assert merged["MODEL"]["components"]["video_encoder"]["source"]["provider"] == "huggingface"
    assert merged["MODEL"]["components"]["video_encoder"]["source"]["name"] == "hf_backbone"
    assert merged["MODEL"]["hf_key"] == 42
    assert merged["MODEL"]["local_only_key"] == 7

    assert merged["DATA"]["common"]["data_root"] == "/local/data"
    assert merged["DATA"]["inputs"]["video"]["params"]["num_classes"] == 8
    assert merged["DATA"]["common"]["classes"] == ["A", "B"]
    assert merged["SYSTEM"]["device"] == "cpu"
    assert merged["SYSTEM"]["gpu"]["count"] == 1
    assert merged["TRAIN"]["epochs"] == 12


def test_namespace_to_dict_handles_omegaconf_without_recursion():
    cfg = OmegaConf.create(
        {
            "MODEL": {"backbone": {"type": "mvit_v2_s"}},
            "SYSTEM": {"device": "cpu"},
            "VALUE": "${SYSTEM.device}",
        }
    )

    out = namespace_to_dict(cfg)

    assert out["MODEL"]["backbone"]["type"] == "mvit_v2_s"
    assert out["VALUE"] == "cpu"


@pytest.mark.parametrize(
    "api_cls,cfg_path",
    [
        (
            ClassificationAPI,
            REPO_ROOT / "opensportslib" / "legacy_config" / "classification.yaml",
        ),
        (
            LocalizationAPI,
            REPO_ROOT / "opensportslib" / "legacy_config" / "localization.yaml",
        ),
    ],
)
def test_infer_uses_compatibility_merge_policy(monkeypatch, tmp_path, api_cls, cfg_path):
    calls = []

    def fake_fetch(cfg, pretrained, hf_token=None, merge_policy="full"):
        calls.append(
            {
                "pretrained": pretrained,
                "merge_policy": merge_policy,
            }
        )
        return cfg

    class _StopFlow(RuntimeError):
        pass

    def stop_after_merge(cfg):
        raise _StopFlow

    monkeypatch.setattr(config_utils, "fetch_and_merge_pretrained_config", fake_fetch)
    monkeypatch.setattr(config_utils, "resolve_config_omega", stop_after_merge)

    api = api_cls(config=str(cfg_path))

    with pytest.raises(_StopFlow):
        api.infer(
            test_set=str(tmp_path / "test_annotations.json"),
            pretrained="OpenSportsLab/dummy-repo",
            predictions=str(tmp_path / "predictions.json"),
            use_wandb=False,
        )

    assert calls
    assert calls[0]["pretrained"] == "OpenSportsLab/dummy-repo"
    assert calls[0]["merge_policy"] == "compatibility"


@pytest.mark.parametrize(
    "api_cls,cfg_path,train_kwargs",
    [
        (
            ClassificationAPI,
            REPO_ROOT / "opensportslib" / "legacy_config" / "classification.yaml",
            {
                "train_set": "train.json",
                "valid_set": "valid.json",
            },
        ),
        (
            LocalizationAPI,
            REPO_ROOT / "opensportslib" / "legacy_config" / "localization.yaml",
            {
                "train_set": "train.json",
                "valid_set": "valid.json",
            },
        ),
    ],
)
def test_train_uses_compatibility_merge_policy(
    monkeypatch, api_cls, cfg_path, train_kwargs
):
    calls = []

    def fake_fetch(cfg, pretrained, hf_token=None, merge_policy="full"):
        calls.append(
            {
                "pretrained": pretrained,
                "merge_policy": merge_policy,
            }
        )
        return cfg

    class _StopFlow(RuntimeError):
        pass

    def stop_after_merge(cfg):
        raise _StopFlow

    monkeypatch.setattr(config_utils, "fetch_and_merge_pretrained_config", fake_fetch)
    monkeypatch.setattr(config_utils, "resolve_config_omega", stop_after_merge)

    api = api_cls(config=str(cfg_path))

    with pytest.raises(_StopFlow):
        api.train(
            pretrained="OpenSportsLab/dummy-repo",
            use_wandb=False,
            **train_kwargs,
        )

    assert calls
    assert calls[0]["pretrained"] == "OpenSportsLab/dummy-repo"
    assert calls[0]["merge_policy"] == "compatibility"
