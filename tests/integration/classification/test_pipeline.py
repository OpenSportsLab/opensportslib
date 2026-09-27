from pathlib import Path

from opensportslib.apis.classification import ClassificationModel
from tests.helpers.pipeline_stubs import install_classification_pipeline_stub


def _exercise(assets, monkeypatch, tmp_path):
    install_classification_pipeline_stub(monkeypatch, tmp_path)
    monkeypatch.setenv("OSL_PRETRAINED_WEIGHTS", "0")
    monkeypatch.setenv("WANDB_MODE", "disabled")
    api = ClassificationModel(config=assets["config"])
    checkpoint = api.train(use_wandb=False)
    assert checkpoint and Path(checkpoint).exists()
    predictions = api.infer(weights=checkpoint, use_wandb=False)
    assert predictions["task"] == "action_classification"
    assert isinstance(api.evaluate(use_wandb=False), dict)


def test_classification_synthetic_subset_pipeline(classification_integration_assets, monkeypatch, tmp_path):
    assets = classification_integration_assets
    install_classification_pipeline_stub(monkeypatch, tmp_path)
    api = ClassificationModel(config=assets["config"])
    checkpoint = api.train(train_set=assets["train"], valid_set=assets["valid"], use_wandb=False)
    assert checkpoint and Path(checkpoint).exists()
    assert api.infer(test_set=assets["test"], weights=checkpoint, use_wandb=False)["task"] == "action_classification"
    assert isinstance(api.evaluate(test_set=assets["test"], use_wandb=False), dict)


def test_classification_public_format_pipeline(classification_public_dataset_assets, monkeypatch, tmp_path):
    _exercise(classification_public_dataset_assets, monkeypatch, tmp_path)

