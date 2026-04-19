from pathlib import Path

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
