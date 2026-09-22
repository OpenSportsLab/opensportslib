from pathlib import Path

from types import SimpleNamespace

from opensportslib.core.config.accessors import get_hf_cuda_device_index, get_hf_prefer_cuda
from opensportslib.core.utils.config import (
    dict_to_namespace,
    expand,
    load_classes,
    load_gz_json,
    load_json,
    store_gz_json,
    store_json,
)


def test_json_store_and_load_roundtrip(tmp_path):
    payload = {"a": 1, "b": ["x", "y"]}

    json_path = tmp_path / "payload.json"
    gz_path = tmp_path / "payload.json.gz"

    store_json(str(json_path), payload, pretty=True)
    store_gz_json(str(gz_path), payload)

    assert load_json(str(json_path)) == payload
    assert load_gz_json(str(gz_path)) == payload


def test_expand_returns_absolute_path(tmp_path):
    rel_path = Path("relative") / "path"
    expanded = expand(str(rel_path))

    assert Path(expanded).is_absolute()


def test_load_classes_supports_list():
    classes = ["PASS", "SHOT", "GOAL"]
    mapping = load_classes(classes)

    assert mapping == {"PASS": 1, "SHOT": 2, "GOAL": 3}


def test_dict_to_namespace_preserves_classes_shape():
    data = {"DATA": {"classes": ["A", "B"], "num_classes": 2}}
    ns = dict_to_namespace(data)

    assert ns.DATA.classes == ["A", "B"]
    assert ns.DATA.num_classes == 2


def test_get_hf_cuda_device_index_prefers_explicit_value(monkeypatch):
    cfg = SimpleNamespace(
        SYSTEM=SimpleNamespace(gpu=SimpleNamespace(id=3)),
        TRAIN=SimpleNamespace(execution={"hf": {"cuda_device_index": 1}}),
    )

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    assert get_hf_cuda_device_index(cfg, {"cuda_device_index": "2"}) == 2


def test_get_hf_cuda_device_index_falls_back_to_system_gpu(monkeypatch):
    cfg = SimpleNamespace(SYSTEM=SimpleNamespace(gpu=SimpleNamespace(id=4)), TRAIN=SimpleNamespace(execution={"hf": {}}))

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    assert get_hf_cuda_device_index(cfg, {}) == 4


def test_get_hf_cuda_device_index_ignores_cuda_when_visible_devices_is_set(monkeypatch):
    cfg = SimpleNamespace(SYSTEM=SimpleNamespace(gpu=SimpleNamespace(id=4)), TRAIN=SimpleNamespace(execution={"hf": {"cuda_device_index": 1}}))

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")

    assert get_hf_cuda_device_index(cfg, {"cuda_device_index": 1}) is None


def test_get_hf_cuda_device_index_returns_none_when_unset(monkeypatch):
    cfg = SimpleNamespace(SYSTEM=SimpleNamespace(), TRAIN=SimpleNamespace(execution={"hf": {}}))

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    assert get_hf_cuda_device_index(cfg, {}) is None


def test_get_hf_prefer_cuda_respects_explicit_override():
    cfg = SimpleNamespace(
        SYSTEM=SimpleNamespace(device="cpu"),
        TRAIN=SimpleNamespace(execution={"hf": {"prefer_cuda": True}}),
    )

    assert get_hf_prefer_cuda(cfg, {"prefer_cuda": True}) is True


def test_get_hf_prefer_cuda_maps_system_cpu_to_false():
    cfg = SimpleNamespace(
        SYSTEM=SimpleNamespace(device="cpu"),
        TRAIN=SimpleNamespace(execution={"hf": {}}),
    )

    assert get_hf_prefer_cuda(cfg, {}) is False


def test_get_hf_prefer_cuda_maps_system_cuda_to_true():
    cfg = SimpleNamespace(
        SYSTEM=SimpleNamespace(device="cuda"),
        TRAIN=SimpleNamespace(execution={"hf": {}}),
    )

    assert get_hf_prefer_cuda(cfg, {}) is True


def test_get_hf_prefer_cuda_maps_system_auto_to_true():
    cfg = SimpleNamespace(
        SYSTEM=SimpleNamespace(device="auto"),
        TRAIN=SimpleNamespace(execution={"hf": {}}),
    )

    assert get_hf_prefer_cuda(cfg, {}) is True
