from pathlib import Path

from opensportslib.apis.localization import LocalizationModel
from tests.helpers.pipeline_stubs import install_localization_pipeline_stub


def _exercise(assets, monkeypatch, tmp_path, explicit_paths):
    install_localization_pipeline_stub(monkeypatch, tmp_path)
    monkeypatch.setenv("OSL_PRETRAINED_WEIGHTS", "0")
    monkeypatch.setenv("WANDB_MODE", "disabled")
    api = LocalizationModel(config=assets["config"])
    kwargs = {"train_set": assets["train"], "valid_set": assets["valid"]} if explicit_paths else {}
    checkpoint = api.train(use_wandb=False, **kwargs)
    assert checkpoint and Path(checkpoint).exists()
    infer_kwargs = {"test_set": assets["test"]} if explicit_paths else {}
    assert api.infer(weights=checkpoint, use_wandb=False, **infer_kwargs)["task"] == "localization"
    assert isinstance(api.evaluate(use_wandb=False, **infer_kwargs), dict)


def test_localization_synthetic_subset_pipeline(localization_integration_assets, monkeypatch, tmp_path):
    _exercise(localization_integration_assets, monkeypatch, tmp_path, True)


def test_localization_public_format_pipeline(localization_public_dataset_assets, monkeypatch, tmp_path):
    _exercise(localization_public_dataset_assets, monkeypatch, tmp_path, False)

