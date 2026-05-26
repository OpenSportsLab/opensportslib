"""Public API for VQA tasks."""

from __future__ import annotations

from opensportslib.apis.base_task_model import BaseTaskModel
from opensportslib.core.config.accessors import get_split_annotation_path
from opensportslib.core.utils.config import expand, resolve_config_omega


class VQAModel(BaseTaskModel):
    """Top-level task wrapper for VQA."""

    def _resolve_split_path(self, split: str, override: str | None = None) -> str:
        if override is not None:
            return expand(override)
        path = get_split_annotation_path(self.config, split)
        if not path:
            raise ValueError(
                f"Could not resolve path for split '{split}'. "
                f"Expected DATA.common.splits.{split}.annotation_path."
            )
        return expand(path)

    def load_weights(self, weights: str | None = None, **kwargs) -> None:
        del kwargs
        if weights is None:
            raise ValueError("`weights` must be provided to load_weights().")
        from opensportslib.core.trainer.vqa_trainer import Trainer_VQA

        self.trainer = Trainer_VQA(self.config)
        self.trainer.load(weights)
        self.last_loaded_weights = weights
        self.best_checkpoint = weights

    def train(
        self,
        train_set: str | None = None,
        valid_set: str | None = None,
        weights: str | None = None,
        use_wandb: bool = True,
        **kwargs,
    ) -> str | None:
        del use_wandb, kwargs
        from opensportslib.core.trainer.vqa_trainer import Trainer_VQA
        from opensportslib.datasets.builder import build_dataset
        from opensportslib.models.builder import build_model
        from opensportslib.core.utils.config import select_device

        self.config = resolve_config_omega(self.config, weights=weights)
        train_set = self._resolve_split_path("train", train_set)
        valid_set = self._resolve_split_path("valid", valid_set)
        device = select_device(self.config.SYSTEM)

        model, _ = build_model(self.config, device)
        train_data = build_dataset(self.config, train_set, None, split="train")
        valid_data = build_dataset(self.config, valid_set, None, split="valid")
        self.trainer = Trainer_VQA(self.config)
        ckpt = self.trainer.train(model, train_data, valid_data)
        self.best_checkpoint = ckpt
        self.last_loaded_weights = ckpt
        return ckpt

    def infer(
        self,
        test_set: str | None = None,
        weights: str | None = None,
        use_wandb: bool = True,
        **kwargs,
    ) -> dict:
        del use_wandb, kwargs
        from opensportslib.core.trainer.vqa_trainer import Trainer_VQA
        from opensportslib.datasets.builder import build_dataset
        from opensportslib.models.builder import build_model
        from opensportslib.core.utils.config import select_device

        self.config = resolve_config_omega(self.config, weights=weights)
        effective_weights = weights if weights is not None else self.last_loaded_weights
        test_set = self._resolve_split_path("test", test_set)
        device = select_device(self.config.SYSTEM)
        model, _ = build_model(self.config, device)
        test_data = build_dataset(self.config, test_set, None, split="test")
        self.trainer = Trainer_VQA(self.config)
        if effective_weights is not None:
            self.trainer.load(effective_weights)
        return self.trainer.infer(model, test_data)

    def evaluate(
        self,
        test_set: str | None = None,
        weights: str | None = None,
        predictions: str | dict | None = None,
        use_wandb: bool = True,
        **kwargs,
    ) -> dict | str | None:
        del use_wandb, kwargs
        from opensportslib.core.trainer.vqa_trainer import Trainer_VQA
        from opensportslib.datasets.builder import build_dataset

        self.config = resolve_config_omega(self.config, weights=weights)
        test_set = self._resolve_split_path("test", test_set)
        test_data = build_dataset(self.config, test_set, None, split="test")
        if predictions is None:
            predictions = self.infer(test_set=test_set, weights=weights, use_wandb=False)
        elif isinstance(predictions, str):
            import json

            with open(expand(predictions), encoding="utf-8") as f:
                predictions = json.load(f)
        self.trainer = self.trainer or Trainer_VQA(self.config)
        return self.trainer.evaluate(predictions, test_data)
