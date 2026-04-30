from pathlib import Path

from opensportslib import model
from opensportslib.apis import (
    BaseTaskModel,
    ClassificationModel,
    LocalizationModel,
)

def test_classification_model_initializes(classification_config_path):
    api = ClassificationModel(config=classification_config_path)

    assert isinstance(api, BaseTaskModel)
    assert isinstance(api, ClassificationModel)
    assert Path(api.config.DATA.data_dir).is_absolute()
    assert Path(api.config.SYSTEM.save_dir).exists()
    assert callable(api.load_weights)
    assert callable(api.train)
    assert callable(api.infer)
    assert callable(api.evaluate)


def test_localization_model_initializes(localization_config_path):
    api = LocalizationModel(config=localization_config_path)

    assert isinstance(api, BaseTaskModel)
    assert isinstance(api, LocalizationModel)
    assert Path(api.config.DATA.data_dir).is_absolute()
    assert Path(api.config.SYSTEM.save_dir).exists()
    assert callable(api.load_weights)
    assert callable(api.train)
    assert callable(api.infer)
    assert callable(api.evaluate)


def test_model_namespace_exposes_model_classes():
    assert callable(model.ClassificationModel)
    assert callable(model.LocalizationModel)
