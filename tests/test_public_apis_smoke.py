from pathlib import Path
import sys
from types import SimpleNamespace

from opensportslib import model
from opensportslib.apis import (
    ClassificationAPI,
    LocalizationAPI,
    classification,
    localization,
)

def test_classification_factory_initializes(classification_config_path):
    api = classification(config=classification_config_path)

    assert isinstance(api, ClassificationAPI)
    assert Path(api.config.DATA.data_dir).is_absolute()
    assert Path(api.config.SYSTEM.save_dir).exists()


def test_localization_factory_initializes(localization_config_path):
    api = localization(config=localization_config_path)

    assert isinstance(api, LocalizationAPI)
    assert Path(api.config.DATA.data_dir).is_absolute()
    assert Path(api.config.SYSTEM.save_dir).exists()


def test_model_namespace_exposes_factories():
    assert callable(model.classification)
    assert callable(model.localization)


# def test_classification_factory_accepts_hf_dataset_at_init(
#     monkeypatch, tmp_path, classification_config_path
# ):
#     calls = {}
#     cached_path = tmp_path / "hf-classification"
#     cached_path.mkdir()

#     def fake_snapshot_download(**kwargs):
#         calls.update(kwargs)
#         return str(cached_path)

#     monkeypatch.setitem(
#         sys.modules,
#         "huggingface_hub",
#         SimpleNamespace(snapshot_download=fake_snapshot_download),
#     )

#     api = classification(
#         config=classification_config_path,
#         dataset_repo_id="OpenSportsLab/example-classification",
#         dataset_revision="mvfouls",
#         hf_token="secret",
#     )

#     assert api.config.DATA.data_dir == str(cached_path)
#     assert calls["repo_id"] == "OpenSportsLab/example-classification"
#     assert calls["repo_type"] == "dataset"
#     assert calls["revision"] == "mvfouls"
#     assert calls["token"] == "secret"


# def test_localization_factory_accepts_hf_dataset_at_init(
#     monkeypatch, tmp_path, localization_config_path
# ):
#     calls = {}
#     cached_path = tmp_path / "hf-localization"
#     cached_path.mkdir()

#     def fake_snapshot_download(**kwargs):
#         calls.update(kwargs)
#         return str(cached_path)

#     monkeypatch.setitem(
#         sys.modules,
#         "huggingface_hub",
#         SimpleNamespace(snapshot_download=fake_snapshot_download),
#     )

#     api = localization(
#         config=localization_config_path,
#         dataset_repo_id="OpenSportsLab/example-localization",
#         dataset_revision="main",
#     )

#     assert api.config.DATA.data_dir == str(cached_path)
#     assert calls["repo_id"] == "OpenSportsLab/example-localization"
#     assert calls["repo_type"] == "dataset"
#     assert calls["revision"] == "main"
