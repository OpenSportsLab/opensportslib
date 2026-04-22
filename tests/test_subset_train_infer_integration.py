from pathlib import Path

import pytest

from opensportslib.apis.classification import ClassificationModel
from opensportslib.apis.localization import LocalizationModel


pytestmark = pytest.mark.integration


def _install_classification_stubs(monkeypatch, tmp_path: Path):
    checkpoint_path = tmp_path / "classification-best.pt"

    def fake_train(
        self,
        train_set=None,
        valid_set=None,
        test_set=None,
        *,
        weights=None,
        use_ddp=False,
        use_wandb=True,
        **kwargs,
    ):
        del test_set, weights, use_ddp, use_wandb, kwargs
        train_path = Path(self._resolve_split_path("train", train_set))
        valid_path = Path(self._resolve_split_path("valid", valid_set))
        assert train_path.exists()
        assert valid_path.exists()

        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text("dummy checkpoint", encoding="utf-8")
        self.best_checkpoint = str(checkpoint_path)
        self.last_loaded_weights = str(checkpoint_path)
        return str(checkpoint_path)

    def fake_infer(
        self,
        test_set=None,
        *,
        weights=None,
        use_ddp=False,
        use_wandb=True,
        **kwargs,
    ):
        del use_ddp, use_wandb, kwargs

        test_path = Path(self._resolve_split_path("test", test_set))
        assert test_path.exists()

        if weights is not None:
            self.last_loaded_weights = weights

        predictions = {
            "version": "2.0",
            "task": "action_classification",
            "metadata": {"type": "predictions"},
            "data": [],
        }
        return predictions

    def fake_evaluate(
        self,
        test_set=None,
        *,
        weights=None,
        use_ddp=False,
        use_wandb=True,
        **kwargs,
    ):
        del kwargs
        test_path = Path(self._resolve_split_path("test", test_set))
        assert test_path.exists()

        predictions = self.infer(
            test_set=str(test_path),
            weights=weights,
            use_ddp=use_ddp,
            use_wandb=use_wandb,
        )

        return {"f1": 1.0, "predictions": predictions}

    monkeypatch.setattr(ClassificationModel, "train", fake_train)
    monkeypatch.setattr(ClassificationModel, "infer", fake_infer)
    monkeypatch.setattr(ClassificationModel, "evaluate", fake_evaluate)


def _install_localization_stubs(monkeypatch, tmp_path: Path):
    checkpoint_path = tmp_path / "localization-best.ckpt"
    def fake_load_weights(
        self,
        weights=None,
        **kwargs,
    ):
        del kwargs
        if weights is None:
            raise ValueError("`weights` must be provided to load_weights().")
        self.model = object()
        self.last_loaded_weights = weights
        self.best_checkpoint = weights

    def fake_train(
        self,
        train_set=None,
        valid_set=None,
        *,
        weights=None,
        use_wandb=True,
        **kwargs,
    ):
        del weights, use_wandb, kwargs
        train_path = Path(self._resolve_split_path("train", train_set))
        valid_path = Path(self._resolve_split_path("valid", valid_set))
        assert train_path.exists()
        assert valid_path.exists()

        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text("dummy localization checkpoint", encoding="utf-8")
        self.best_checkpoint = str(checkpoint_path)
        self.last_loaded_weights = str(checkpoint_path)
        return str(checkpoint_path)

    def fake_infer(
        self,
        test_set=None,
        *,
        weights=None,
        use_wandb=True,
        **kwargs,
    ):
        del use_wandb, kwargs

        test_path = Path(self._resolve_split_path("test", test_set))
        assert test_path.exists()

        if weights is not None:
            self.load_weights(weights=weights)

        predictions = {
            "version": "2.0",
            "task": "localization",
            "metadata": {"type": "predictions"},
            "data": [],
        }
        return predictions

    def fake_evaluate(
        self,
        test_set=None,
        *,
        weights=None,
        use_wandb=True,
        **kwargs,
    ):
        del use_wandb, weights, kwargs
        test_path = Path(self._resolve_split_path("test", test_set))
        assert test_path.exists()
        predictions = self.infer(test_set=str(test_path))

        return {"a_mAP": 0.0, "predictions": predictions}

    monkeypatch.setattr(LocalizationModel, "load_weights", fake_load_weights)
    monkeypatch.setattr(LocalizationModel, "train", fake_train)
    monkeypatch.setattr(LocalizationModel, "infer", fake_infer)
    monkeypatch.setattr(LocalizationModel, "evaluate", fake_evaluate)


def test_classification_train_and_infer_subset(
    tmp_path,
    monkeypatch,
    classification_integration_assets,
):
    _install_classification_stubs(monkeypatch, tmp_path)
    assets = classification_integration_assets

    monkeypatch.setenv("OSL_PRETRAINED_WEIGHTS", "0")
    monkeypatch.setenv("WANDB_MODE", "disabled")

    api = ClassificationModel(
        config=assets["config"],
    )
    checkpoint = api.train(
        train_set=assets["train"],
        valid_set=assets["valid"],
        use_wandb=False,
    )

    assert checkpoint is not None
    assert Path(checkpoint).exists()

    predictions = api.infer(
        test_set=assets["test"],
        weights=checkpoint,
        use_wandb=False,
    )
    assert isinstance(predictions, dict)
    assert predictions.get("task") == "action_classification"

    metrics = api.evaluate(
        test_set=assets["test"],
        use_wandb=False,
    )
    assert isinstance(metrics, dict)


def test_localization_train_and_infer_subset(
    tmp_path,
    monkeypatch,
    localization_integration_assets,
):
    _install_localization_stubs(monkeypatch, tmp_path)
    assets = localization_integration_assets

    monkeypatch.setenv("OSL_PRETRAINED_WEIGHTS", "0")
    monkeypatch.setenv("WANDB_MODE", "disabled")

    api = LocalizationModel(
        config=assets["config"],
    )
    checkpoint = api.train(
        train_set=assets["train"],
        valid_set=assets["valid"],
        use_wandb=False,
    )
    assert checkpoint is not None
    assert Path(checkpoint).exists()

    predictions = api.infer(
        test_set=assets["test"],
        weights=checkpoint,
        use_wandb=False,
    )
    assert isinstance(predictions, dict)
    assert predictions.get("task") == "localization"

    metrics = api.evaluate(
        test_set=assets["test"],
        use_wandb=False,
    )
    assert isinstance(metrics, dict)
