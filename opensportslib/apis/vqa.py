"""Public API for VQA tasks."""

from __future__ import annotations

import logging
import os

from opensportslib.apis.base_task_model import BaseTaskModel
from opensportslib.core.config.accessors import get_split_annotation_path, get_system_gpu_count, get_train_execution
from opensportslib.core.utils.config import expand, resolve_config_omega


def _set_model_checkpoint_path(config, weights: str | None) -> None:
    if weights is None:
        return
    model = getattr(config, "MODEL", None)
    if model is None:
        return
    load = getattr(model, "load", None)
    if load is None:
        from types import SimpleNamespace

        load = SimpleNamespace()
        setattr(model, "load", load)
    if isinstance(load, dict):
        load["checkpoint_path"] = weights
    else:
        setattr(load, "checkpoint_path", weights)


class VQAModel(BaseTaskModel):
    """Top-level task wrapper for VQA."""
    @staticmethod
    def _worker_ddp(
        rank,
        world_size,
        config_path,
        config,
        return_queue=None,
        train_set=None,
        valid_set=None,
        use_wandb=False,
    ):
        del config_path, use_wandb
        import torch
        from opensportslib.core.trainer.vqa_trainer import Trainer_VQA
        from opensportslib.core.utils.ddp import ddp_cleanup, ddp_setup
        from opensportslib.datasets.builder import build_dataset

        logging.basicConfig(
            level=logging.INFO,
            format=f"[RANK {rank}] %(asctime)s | %(levelname)s | %(message)s",
            force=True,
        )
        if rank != 0:
            logging.getLogger().setLevel(logging.ERROR)

        is_ddp = world_size > 1
        if is_ddp:
            os.environ["RANK"] = str(rank)
            os.environ["WORLD_SIZE"] = str(world_size)
            os.environ["LOCAL_RANK"] = str(rank)
            torch.cuda.set_device(rank)
            ddp_setup(rank, world_size)

        try:
            train_data = build_dataset(config, train_set, None, split="train")
            valid_data = build_dataset(config, valid_set, None, split="valid")
            trainer = Trainer_VQA(config)
            ckpt = trainer.train(None, train_data, valid_data, rank=rank, world_size=world_size)
            if rank == 0 and return_queue is not None:
                return_queue.put(ckpt)
        finally:
            if is_ddp:
                ddp_cleanup()

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

        _set_model_checkpoint_path(self.config, weights)
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
        del kwargs
        import torch
        import torch.multiprocessing as mp

        self.config = resolve_config_omega(self.config, weights=weights)
        train_set = self._resolve_split_path("train", train_set)
        valid_set = self._resolve_split_path("valid", valid_set)
        execution = get_train_execution(self.config)
        backend = str(execution.get("training_backend", "placeholder")).lower()
        if backend == "xvars_lora":
            world_size = torch.cuda.device_count() or get_system_gpu_count(self.config)
            requested_gpus = get_system_gpu_count(self.config)
            use_ddp = world_size > 1 and int(requested_gpus) > 1
            logging.info(
                "VQA train launch | mode=%s | world_size=%s",
                "ddp" if use_ddp else "single",
                world_size if use_ddp else 1,
            )

            ctx = mp.get_context("spawn")
            queue = ctx.SimpleQueue()
            if use_ddp:
                mp.spawn(
                    VQAModel._worker_ddp,
                    args=(world_size, self.config_path, self.config, queue, train_set, valid_set, use_wandb),
                    nprocs=world_size,
                )
            else:
                VQAModel._worker_ddp(
                    rank=0,
                    world_size=1,
                    config_path=self.config_path,
                    config=self.config,
                    return_queue=queue,
                    train_set=train_set,
                    valid_set=valid_set,
                    use_wandb=use_wandb,
                )
            ckpt = queue.get()
            self.best_checkpoint = ckpt
            self.last_loaded_weights = ckpt
            return ckpt

        from opensportslib.core.trainer.vqa_trainer import Trainer_VQA
        from opensportslib.datasets.builder import build_dataset
        from opensportslib.models.builder import build_model
        from opensportslib.core.utils.config import select_device

        model = None
        device = select_device(self.config.SYSTEM)
        model, _ = build_model(self.config, device)
        train_data = build_dataset(self.config, train_set, None, split="train")
        valid_data = build_dataset(self.config, valid_set, None, split="valid")
        self.trainer = Trainer_VQA(self.config)
        ckpt = self.trainer.train(model, train_data, valid_data, rank=0, world_size=1)
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
        _set_model_checkpoint_path(self.config, effective_weights)
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
