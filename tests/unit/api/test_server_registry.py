import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


SERVER_ROOT = Path(__file__).parents[3] / "server"
sys.path.insert(0, str(SERVER_ROOT))

from config.model_registry import (  # noqa: E402
    ModelRegistry,
    generated_local_model_id,
    public_record,
    resolve_local_source,
    resolve_huggingface_source,
    infer_source_mode,
    validate_local_model_id,
)


class FakeRedis:
    def __init__(self):
        self.hashes = {}

    def hget(self, key, field):
        return self.hashes.get(key, {}).get(field)

    def hvals(self, key):
        return list(self.hashes.get(key, {}).values())

    def hset(self, key, field, value):
        self.hashes.setdefault(key, {})[field] = value

    def hsetnx(self, key, field, value):
        values = self.hashes.setdefault(key, {})
        if field in values:
            return 0
        values[field] = value
        return 1

    def hdel(self, key, field):
        self.hashes.get(key, {}).pop(field, None)

    def hgetall(self, key):
        return dict(self.hashes.get(key, {}))

    def setex(self, key, ttl, value):
        self.hashes.setdefault("strings", {})[key] = value

    def getdel(self, key):
        return self.hashes.setdefault("strings", {}).pop(key, None)

    def delete(self, key):
        self.hashes.setdefault("strings", {}).pop(key, None)


def record(model_id="model", source="org/model"):
    return {
        "model_id": model_id,
        "task_type": "classification",
        "source_type": "huggingface",
        "source": source,
        "config_path": None,
        "status": "ready",
        "generation": 1,
        "created_at": "now",
        "updated_at": "now",
        "error": None,
    }


def test_registry_reserves_idempotently_and_rejects_conflicts():
    registry = ModelRegistry(SimpleNamespace(redis_url="redis://unused"), FakeRedis())
    _, created = registry.reserve(record())
    assert created
    existing, created = registry.reserve(record())
    assert not created
    assert existing["source"] == "org/model"
    with pytest.raises(FileExistsError):
        registry.reserve(record(source="other/model"))


def test_local_source_stays_inside_model_root(tmp_path):
    root = tmp_path / "models"
    folder = root / "custom"
    folder.mkdir(parents=True)
    (folder / "config.yaml").write_text("MODEL: {}")
    settings = SimpleNamespace(model_root=root)

    weights, config = resolve_local_source(settings, str(folder), None)
    assert weights == folder
    assert config == folder / "config.yaml"
    assert generated_local_model_id(folder).startswith("local:custom-")

    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(ValueError, match="OSL_MODEL_ROOT"):
        resolve_local_source(settings, str(outside), None)


def test_custom_id_and_public_local_record_validation():
    assert validate_local_model_id("football-model:v2") == "football-model:v2"
    with pytest.raises(ValueError):
        validate_local_model_id("folder/model")
    local = record()
    local.update({"source_type": "local", "source": "/secret/model", "config_path": "/secret/config", "error": "at /secret/model"})
    visible = public_record(local)
    assert "source" not in visible
    assert "config_path" not in visible
    assert "/secret" not in visible["error"]


@pytest.mark.parametrize("runner_section", ["MODEL", "TRAIN"])
def test_config_driven_runner_is_weightless(tmp_path, runner_section):
    config = tmp_path / "config.yaml"
    config.write_text(f"{runner_section}:\n  runner:\n    type: runner_h5_header_rule\n")
    assert infer_source_mode(str(config), "org/config-only") == "config_only"


def test_ordinary_config_requires_checkpoint(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("TRAIN:\n  runner:\n    type: default\n")
    assert infer_source_mode(str(config), "org/model") == "checkpoint"


def test_operation_secret_is_consumed_and_not_part_of_records():
    settings = SimpleNamespace(redis_url="redis://unused", model_operation_timeout_seconds=60)
    registry = ModelRegistry(settings, FakeRedis())
    registry.store_operation_secret("operation", "hf_secret")
    assert registry.consume_operation_secret("operation") == "hf_secret"
    assert registry.consume_operation_secret("operation") is None
    assert registry.list() == []


def _mock_hub(monkeypatch, config_path, files):
    import huggingface_hub

    download = Mock(return_value=str(config_path))
    listing = Mock(return_value=files)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(huggingface_hub, "list_repo_files", listing)
    return download, listing


@pytest.mark.parametrize("variant", [
    "h5_header_skeleton", "h5_header_skeleton_recall", "h5_header_skeleton_max_recall",
])
def test_hf_rule_source_loads_localization_service_without_weights(tmp_path, monkeypatch, variant):
    from opensportslib.core.config import load_config
    from opensportslib.core.config.loader import save_config
    from services.localization_service import LocalizationService

    canonical = load_config(
        SERVER_ROOT.parent / "opensportslib/configs/localization/h5_header_skeleton.yaml",
        as_namespace=False,
    )
    canonical["MODEL"]["components"]["rule"]["source"]["name"] = variant
    canonical["SYSTEM"]["paths"]["save_dir"] = str(tmp_path / "outputs")
    canonical["DATA"]["common"]["data_root"] = str(tmp_path)
    canonical["DATA"]["common"]["splits"]["test"]["source_path"] = str(tmp_path)
    config_path = tmp_path / "config.yaml"
    save_config(canonical, config_path)
    download, listing = _mock_hub(monkeypatch, config_path, ["config.yaml"])
    model_id = "example/skeleton"

    resolved_path, source_mode, weights = resolve_huggingface_source(model_id, token="test-token")
    assert (resolved_path, source_mode, weights) == (str(config_path), "config_only", None)
    download.assert_called_once_with(repo_id=model_id, filename="config.yaml", token="test-token")
    listing.assert_called_once_with(model_id, token="test-token")

    registry = ModelRegistry(SimpleNamespace(redis_url="redis://unused"), FakeRedis())
    registered = record(model_id=model_id, source=model_id)
    registered.update(task_type="localization", config_path=resolved_path, source_mode=source_mode)
    settings = registry.model_settings(registered)
    assert settings.weights is None
    service = LocalizationService(settings)
    service.preload()
    assert service.model.last_loaded_weights is None
    assert service.model.config.MODEL.components.rule.source.name == variant
    assert service.label_schema["action"]["labels"] == ["header"]
    # Older persisted records without source_mode must infer the same behavior.
    registered.pop("source_mode")
    assert registry.model_settings(registered).weights is None
    assert download.call_count == 1


@pytest.mark.parametrize("checkpoint", ["model.pt", "model.pth.tar", "adapter_model.safetensors"])
def test_hf_checkpoint_source_keeps_weights(tmp_path, monkeypatch, checkpoint):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("MODEL:\n  runner:\n    type: runner_e2e\n")
    _mock_hub(monkeypatch, config_path, ["config.yaml", checkpoint])
    model_id = "example/checkpoint"

    path, mode, weights = resolve_huggingface_source(model_id)
    assert (path, mode, weights) == (str(config_path), "checkpoint", model_id)
    registry = ModelRegistry(SimpleNamespace(redis_url="redis://unused"), FakeRedis())
    registered = record(model_id=model_id, source=model_id)
    registered.update(config_path=path, source_mode=mode)
    assert registry.model_settings(registered).weights == model_id


@pytest.mark.parametrize("content", [
    "MODEL:\n  runner:\n    type: runner_e2e\n",
    "{}",
    "[]",
    "MODEL: [",
])
def test_hf_weightless_source_without_supported_runner_is_rejected(tmp_path, monkeypatch, content):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(content)
    _mock_hub(monkeypatch, config_path, ["config.yaml"])

    with pytest.raises(ValueError, match="no supported checkpoint"):
        resolve_huggingface_source("example/unsupported")


def test_hf_repository_access_failure_is_reported(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("MODEL: {}")
    _, listing = _mock_hub(monkeypatch, config_path, [])
    listing.side_effect = RuntimeError("repository access denied")
    with pytest.raises(ValueError, match="Could not access Hugging Face repository"):
        resolve_huggingface_source("example/gated", token="test-token")
