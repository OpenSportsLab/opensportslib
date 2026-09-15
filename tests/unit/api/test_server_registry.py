import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


SERVER_ROOT = Path(__file__).parents[3] / "server"
sys.path.insert(0, str(SERVER_ROOT))

from config.model_registry import (  # noqa: E402
    ModelRegistry,
    generated_local_model_id,
    public_record,
    resolve_local_source,
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
