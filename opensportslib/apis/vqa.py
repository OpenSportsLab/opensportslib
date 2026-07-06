"""Public API for VQA tasks."""

from __future__ import annotations

import logging
import os
from typing import Any

from opensportslib.apis.base_task_model import BaseTaskModel
from opensportslib.core.config.accessors import get_split_annotation_path, get_system_gpu_count, get_train_execution, get_vqa_backend
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

    def _init_wandb(self, use_wandb: bool) -> None:
        from opensportslib.core.utils.wandb import init_wandb

        init_wandb(
            self.config_path,
            self.config,
            run_id=os.environ["RUN_ID"],
            use_wandb=use_wandb,
        )

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
        import torch
        from opensportslib.core.trainer.vqa_trainer import Trainer_VQA
        from opensportslib.core.utils.ddp import ddp_cleanup, ddp_setup
        from opensportslib.core.utils.wandb import init_wandb
        from opensportslib.datasets.builder import build_dataset

        logging.basicConfig(
            level=logging.INFO,
            format=f"[RANK {rank}] %(asctime)s | %(levelname)s | %(message)s",
            force=True,
        )
        if rank != 0:
            logging.getLogger().setLevel(logging.ERROR)
        # Keep INFO on all ranks while debugging multi-GPU startup/hangs.
        # This makes DDP failures visible instead of appearing as a silent stall.
        #logging.getLogger().setLevel(logging.INFO)

        is_ddp = world_size > 1
        if is_ddp:
            os.environ["RANK"] = str(rank)
            os.environ["WORLD_SIZE"] = str(world_size)
            os.environ["LOCAL_RANK"] = str(rank)
            if "TORCH_DISTRIBUTED_DEBUG" not in os.environ:
                os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
            torch.cuda.set_device(rank)
            ddp_setup(rank, world_size)
            logging.info(
                "Initialized VQA DDP worker | rank=%s | world_size=%s | torch_distributed_debug=%s",
                rank,
                world_size,
                os.environ.get("TORCH_DISTRIBUTED_DEBUG"),
            )

        try:
            if rank == 0:
                init_wandb(
                    config_path,
                    config,
                    run_id=os.environ["RUN_ID"],
                    use_wandb=use_wandb,
                )
            train_data = build_dataset(config, train_set, None, split="train")
            valid_data = build_dataset(config, valid_set, None, split="valid")
            trainer = Trainer_VQA(config)
            ckpt = trainer.train(
                None,
                train_data,
                valid_data,
                rank=rank,
                world_size=world_size,
                use_wandb=use_wandb,
            )
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
        if get_vqa_backend(self.config) == "qwen_xvars_infer":
            raise ValueError("The 'qwen_xvars_infer' backend is inference-only and does not support train().")
        train_set = self._resolve_split_path("train", train_set)
        valid_set = self._resolve_split_path("valid", valid_set)
        execution = get_train_execution(self.config)
        backend = str(execution.get("training_backend", "placeholder")).lower()
        if backend == "xvars_videochatgpt_lora":
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

        raise ValueError(
            f"Unsupported VQA training backend '{backend}'. "
            "Only 'xvars_videochatgpt_lora' is supported."
        )

    def infer(
        self,
        test_set: str | None = None,
        weights: str | None = None,
        use_wandb: bool = True,
        video_path: str | None = None,
        question: str | None = None,
        **kwargs,
    ) -> dict:
        del kwargs
        from opensportslib.core.trainer.vqa_trainer import Trainer_VQA
        from opensportslib.datasets.builder import build_dataset
        from opensportslib.models.builder import build_model
        from opensportslib.core.utils.config import select_device

        direct_requested = video_path is not None or question is not None
        if direct_requested and test_set is not None:
            raise ValueError("Provide either `test_set` or `video_path`/`question`, not both.")
        if direct_requested and (not video_path or not str(question or "").strip()):
            raise ValueError("Direct VQA inference requires both `video_path` and a non-empty `question`.")

        self.config = resolve_config_omega(self.config, weights=weights)
        backend = get_vqa_backend(self.config)
        effective_weights = weights if weights is not None else self.last_loaded_weights
        if backend == "qwen_xvars_infer" and effective_weights is not None:
            raise ValueError("The 'qwen_xvars_infer' backend does not support adapter weights for infer().")
        _set_model_checkpoint_path(self.config, effective_weights)
        self.trainer = Trainer_VQA(self.config)
        if effective_weights is not None:
            # Validate OpenSportsLib adapter metadata before allocating the base model.
            self.trainer.load(effective_weights)
        resolved_video_path = None
        if direct_requested:
            resolved_video_path = expand(str(video_path))
            if not os.path.isfile(resolved_video_path):
                raise FileNotFoundError(f"Video file not found: {resolved_video_path}")
        device = select_device(self.config.SYSTEM)
        model, _ = build_model(self.config, device)
        if direct_requested:
            test_data = [
                {
                    "id": os.path.splitext(os.path.basename(resolved_video_path))[0],
                    "question": str(question).strip(),
                    "references": [],
                    "video_path": resolved_video_path,
                    "video_spatio_temporal_features": None,
                    "prior_prediction_text": "",
                    "labels": {},
                    "metadata": {},
                    "_xvars_demo_parity_direct_infer": True,
                }
            ]
        else:
            test_set = self._resolve_split_path("test", test_set)
            test_data = build_dataset(self.config, test_set, None, split="test")
        self._init_wandb(use_wandb=use_wandb)
        return self.trainer.infer(model, test_data, use_wandb=use_wandb)

    def evaluate(
        self,
        test_set: str | None = None,
        weights: str | None = None,
        predictions: str | dict | None = None,
        use_wandb: bool = True,
        **kwargs,
    ) -> dict | str | None:
        del kwargs
        from opensportslib.core.trainer.vqa_trainer import Trainer_VQA
        from opensportslib.datasets.builder import build_dataset

        self.config = resolve_config_omega(self.config, weights=weights)
        test_set = self._resolve_split_path("test", test_set)
        test_data = build_dataset(self.config, test_set, None, split="test")
        self._init_wandb(use_wandb=use_wandb)
        if predictions is None:
            predictions = self.infer(test_set=test_set, weights=weights, use_wandb=use_wandb)
        elif isinstance(predictions, str):
            import json

            with open(expand(predictions), encoding="utf-8") as f:
                predictions = json.load(f)
        self.trainer = self.trainer or Trainer_VQA(self.config)
        return self.trainer.evaluate(predictions, test_data, use_wandb=use_wandb)

    def save_predictions(
        self,
        output_path: str,
        predictions: dict,
        output_format: str = "osl",
    ) -> str:
        """Persist VQA predictions, optionally as X-VARS-compatible rows."""

        if str(output_format).lower() != "xvars":
            return super().save_predictions(output_path, predictions)

        import json

        payload = self._to_xvars_prediction_rows(predictions)
        dst = expand(output_path)
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        with open(dst, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        return dst

    @staticmethod
    def _to_xvars_prediction_rows(predictions: dict[str, Any]) -> list[dict[str, Any]]:
        rows = []
        for item in predictions.get("data", []) if isinstance(predictions, dict) else []:
            video_path = str(item.get("video_path") or "")
            video_name = os.path.splitext(os.path.basename(video_path))[0] if video_path else str(item.get("id"))
            rows.append(
                {
                    "id": item.get("id"),
                    "video_name": video_name,
                    "Q": item.get("question"),
                    "pred": item.get("answer_text"),
                }
            )
        return rows
