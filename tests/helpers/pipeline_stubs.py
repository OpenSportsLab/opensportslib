"""Reusable external-boundary stubs for fast task-wrapper integration tests."""

from __future__ import annotations

from pathlib import Path

from opensportslib.apis.classification import ClassificationModel
from opensportslib.apis.localization import LocalizationModel


def install_classification_pipeline_stub(monkeypatch, tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "classification-best.pt"

    def train(self, train_set=None, valid_set=None, test_set=None, **kwargs):
        del test_set, kwargs
        assert Path(self._resolve_split_path("train", train_set)).exists()
        assert Path(self._resolve_split_path("valid", valid_set)).exists()
        checkpoint_path.write_text("synthetic classification checkpoint", encoding="utf-8")
        self.best_checkpoint = self.last_loaded_weights = str(checkpoint_path)
        return str(checkpoint_path)

    def infer(self, test_set=None, weights=None, **kwargs):
        del kwargs
        assert Path(self._resolve_split_path("test", test_set)).exists()
        if weights is not None:
            self.last_loaded_weights = weights
        return {"version": "2.0", "task": "action_classification", "metadata": {"type": "predictions"}, "data": []}

    def evaluate(self, test_set=None, weights=None, **kwargs):
        predictions = self.infer(test_set=test_set, weights=weights, **kwargs)
        return {"f1": 1.0, "predictions": predictions}

    monkeypatch.setattr(ClassificationModel, "train", train)
    monkeypatch.setattr(ClassificationModel, "infer", infer)
    monkeypatch.setattr(ClassificationModel, "evaluate", evaluate)


def install_localization_pipeline_stub(monkeypatch, tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "localization-best.ckpt"

    def load_weights(self, weights=None, **kwargs):
        del kwargs
        if weights is None:
            raise ValueError("`weights` must be provided to load_weights().")
        self.model = object()
        self.last_loaded_weights = self.best_checkpoint = weights

    def train(self, train_set=None, valid_set=None, **kwargs):
        del kwargs
        assert Path(self._resolve_split_path("train", train_set)).exists()
        assert Path(self._resolve_split_path("valid", valid_set)).exists()
        checkpoint_path.write_text("synthetic localization checkpoint", encoding="utf-8")
        self.best_checkpoint = self.last_loaded_weights = str(checkpoint_path)
        return str(checkpoint_path)

    def infer(self, test_set=None, weights=None, **kwargs):
        del kwargs
        assert Path(self._resolve_split_path("test", test_set)).exists()
        if weights is not None:
            self.load_weights(weights=weights)
        return {"version": "2.0", "task": "localization", "metadata": {"type": "predictions"}, "data": []}

    def evaluate(self, test_set=None, weights=None, **kwargs):
        del weights
        predictions = self.infer(test_set=test_set, **kwargs)
        return {"a_mAP": 0.0, "predictions": predictions}

    monkeypatch.setattr(LocalizationModel, "load_weights", load_weights)
    monkeypatch.setattr(LocalizationModel, "train", train)
    monkeypatch.setattr(LocalizationModel, "infer", infer)
    monkeypatch.setattr(LocalizationModel, "evaluate", evaluate)
