import pytest

from opensportslib.apis import base_task_model as api


class DummyModel(api.BaseTaskModel):
    def load_weights(self, weights=None, **kwargs):
        self.loaded = weights

    def train(self, **kwargs):
        pass

    def infer(self, **kwargs):
        pass

    def evaluate(self, **kwargs):
        pass


@pytest.fixture
def init_calls(monkeypatch, tmp_path):
    calls = []
    config = {
        "SYSTEM": {"paths": {"save_dir": str(tmp_path)}},
        "MODEL": {"components": {}},
    }
    class FakeEditor:
        weights = None
        def __init__(self, source):
            self.source = source
        def get_config(self):
            return config
        def apply_to(self, value):
            return config
        def remote_overrides(self):
            return {}
    def resolve(value):
        calls.append(value)
        return str(tmp_path / "config.yaml")
    monkeypatch.setattr(api, "resolve_config_path", resolve)
    monkeypatch.setattr(api.Config, "from_file", classmethod(lambda cls, path: FakeEditor(path)))
    monkeypatch.setattr(api, "fetch_and_merge_config_from_HF", lambda cfg, *a, **kw: cfg)
    monkeypatch.setattr(api, "resolve_inference_class_metadata", lambda cfg: cfg)
    monkeypatch.setattr(api, "get_component_name_by_kind", lambda *a: None)
    return calls


def test_hf_weights_supply_config(init_calls):
    model = DummyModel(weights="OpenSportsLab/OSL-cls-action-mvitv2")
    assert init_calls[0] == "OpenSportsLab/OSL-cls-action-mvitv2"
    assert model.loaded == model.last_loaded_weights


def test_explicit_config_still_wins(init_calls):
    DummyModel(config="custom.yaml", weights="org/model")
    assert init_calls == ["custom.yaml"]


@pytest.mark.parametrize("weights", [None, "", "/missing/model", "./model", "model.pt", "a/b/c"])
def test_missing_config_requires_hf_id(weights, init_calls):
    with pytest.raises(ValueError, match="config path is required"):
        DummyModel(weights=weights)
    assert init_calls == []


def test_local_directory_requires_config(tmp_path, init_calls):
    with pytest.raises(ValueError, match="config path is required"):
        DummyModel(weights=str(tmp_path))
    assert init_calls == []


def test_missing_hf_config_explains_fix(monkeypatch, init_calls):
    def missing(value):
        raise OSError("not found")
    monkeypatch.setattr(api, "resolve_config_path", missing)
    with pytest.raises(ValueError, match="provide config explicitly") as error:
        DummyModel(weights="org/model")
    assert isinstance(error.value.__cause__, OSError)


def test_remote_hf_config_does_not_load_local_weights(init_calls):
    model = DummyModel(weights="org/model", remote="http://localhost:8000")
    assert init_calls[0] == "org/model"
    assert not hasattr(model, "loaded")
