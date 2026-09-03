# opensportslib/apis/classification.py

"""Public API for classification tasks."""

import logging
import os
import json

from opensportslib.apis.base_task_model import BaseTaskModel
from opensportslib.core.config.accessors import (
    get_component_provider_by_kind,
    get_data_modality,
    get_split_annotation_path,
    get_system_gpu_count,
    get_system_seed,
    get_system_use_seed,
)
from opensportslib.core.utils.config import expand, resolve_inference_class_metadata


def _is_tracking_graph_modality(modality):
    return str(modality).lower() in {
        "tracking",
        "tracking_parquet",
        "tracking_h5",
        "player_centroids_h5",
        "player_joints_h5",
    }


class ClassificationModel(BaseTaskModel):
    """Top-level task wrapper for classification."""

    def _resolve_split_path(self, split: str, override: str | None = None) -> str:
        if override is not None:
            return expand(override)

        path = get_split_annotation_path(self.config, split)
        if path:
            return expand(path)

        raise ValueError(
            f"Could not resolve path for split '{split}'. "
            f"Expected DATA.common.splits.{split}.annotation_path."
        )

    # -----------------------------------------------------------------
    # internal DDP worker
    # -----------------------------------------------------------------
    @staticmethod
    def _worker_ddp(
        rank,
        world_size,
        mode,
        config_path,
        config,
        return_queue=None,
        train_set=None,
        valid_set=None,
        test_set=None,
        weights=None,
        use_wandb=False,
    ):
        """Execute one training/inference job on a single process."""
        import torch
        import wandb
        from opensportslib.core.trainer.classification_trainer import Trainer_Classification
        from opensportslib.core.utils.ddp import ddp_cleanup, ddp_setup
        from opensportslib.core.utils.wandb import init_wandb
        from opensportslib.core.utils.seed import set_reproducibility
        from opensportslib.datasets.builder import build_dataset
        from opensportslib.models.builder import build_model

        logging.basicConfig(
            level=logging.INFO,
            format=f"[RANK {rank}] %(asctime)s | %(levelname)s | %(message)s",
            force=True,
        )
        if rank != 0:
            logging.getLogger().setLevel(logging.ERROR)

        if rank == 0:
            init_wandb(
                config_path,
                config,
                run_id=os.environ["RUN_ID"],
                use_wandb=use_wandb,
            )

        if get_system_use_seed(config):
            set_reproducibility(get_system_seed(config))

        is_ddp = world_size > 1
        if is_ddp:
            torch.cuda.set_device(rank)
            ddp_setup(rank, world_size)
            device = torch.device(f"cuda:{rank}")
        else:
            from opensportslib.core.utils.config import select_device

            device = select_device(config.SYSTEM)

        trainer = Trainer_Classification(config)
        trainer.device = device

        if weights:
            # resume_training=True only for the training path: it makes
            # load() also rebuild the optimizer/scheduler and restore their
            # state from the checkpoint (momentum, LR schedule, best-metric
            # tracking) so `trainer.train(..., resume_from=trainer._resume_state)`
            # below continues training rather than restarting it. Inference
            # has no use for that state, so it stays out of scope there.
            model, processor, _, _ = trainer.load(
                weights, resume_training=(mode == "train")
            )
        else:
            model, processor = build_model(config, device)

        trainer.model = model

        modality = get_data_modality(config)
        use_tracking_collate = _is_tracking_graph_modality(modality)
        logging.info(
            "Worker setup | mode=%s | modality=%s | tracking_collate=%s",
            mode,
            modality,
            use_tracking_collate,
        )

        try:
            if mode == "train":
                train_data = build_dataset(config, train_set, processor, split="train")
                valid_data = build_dataset(config, valid_set, processor, split="valid")
                best_ckpt = trainer.train(
                    model,
                    train_data,
                    valid_data,
                    rank=rank,
                    world_size=world_size,
                    resume_from=trainer._resume_state,
                )
                if rank == 0 and return_queue is not None:
                    best_ckpt = best_ckpt or getattr(trainer.trainer, "best_checkpoint_path", None)
                    return_queue.put(best_ckpt)

            elif mode == "infer":
                test_data = build_dataset(config, test_set, processor, split="test")
                _ = trainer.infer(
                    test_data,
                    rank=rank,
                    world_size=world_size,
                )
                if rank == 0 and return_queue is not None:
                    if world_size > 1:
                        return_queue.put(getattr(trainer, "predictions_path", None))
                    else:
                        return_queue.put(getattr(trainer, "predictions_payload", None))
        finally:
            if rank == 0 and use_wandb and getattr(wandb, "run", None) is not None:
                try:
                    wandb.finish(quiet=True)
                except Exception:
                    pass
            if is_ddp:
                ddp_cleanup()

    def load_weights(
        self,
        weights: str | None = None,
        **kwargs,
    ) -> None:
        from opensportslib.core.trainer.classification_trainer import Trainer_Classification

        del kwargs
        if weights is None:
            raise ValueError("`weights` must be provided to load_weights().")

        self.trainer = Trainer_Classification(self.config)
        loaded = self.trainer.load(weights)
        self.model = loaded[0]

        if get_component_provider_by_kind(self.config, "encoder") == "huggingface":
            self.processor = loaded[1]

        self.last_loaded_weights = weights
        self.best_checkpoint = weights

    # -----------------------------------------------------------------
    # public training interface
    # -----------------------------------------------------------------

    def train(
        self,
        train_set=None,
        valid_set=None,
        test_set=None,
        weights=None,
        use_ddp=False,
        use_wandb=True,
        **kwargs,
    ):
        """Run full training and return best checkpoint path."""
        import torch
        import torch.multiprocessing as mp
        from opensportslib.core.utils.config import resolve_config_omega

        del test_set  # retained for API compatibility

        train_set = self._resolve_split_path("train", train_set)
        valid_set = self._resolve_split_path("valid", valid_set)
        
        self.config = resolve_config_omega(self.config, weights=weights)
        logging.info("Configuration:")
        logging.info(self.config)

        del kwargs

        effective_weights = weights if weights is not None else self.last_loaded_weights

        world_size = torch.cuda.device_count() or get_system_gpu_count(self.config)
        use_ddp = use_ddp and world_size > 1

        ctx = mp.get_context("spawn")
        queue = ctx.SimpleQueue()

        if use_ddp:
            logging.info(f"Launching DDP on {world_size} GPUs")
            mp.spawn(
                ClassificationModel._worker_ddp,
                args=(
                    world_size,
                    "train",
                    self.config_path,
                    self.config,
                    queue,
                    train_set,
                    valid_set,
                    None,
                    effective_weights,
                    use_wandb,
                ),
                nprocs=world_size,
            )
        else:
            logging.info("Single GPU training")
            ClassificationModel._worker_ddp(
                rank=0,
                world_size=1,
                mode="train",
                config_path=self.config_path,
                config=self.config,
                return_queue=queue,
                train_set=train_set,
                valid_set=valid_set,
                weights=effective_weights,
                use_wandb=use_wandb,
            )

        self.best_checkpoint = queue.get()
        self.last_loaded_weights = self.best_checkpoint
        return self.best_checkpoint

    def infer(
        self,
        test_set=None,
        weights=None,
        use_ddp=False,
        use_wandb=True,
        **kwargs,
    ):
        """Run model inference and return predictions in OSL JSON format."""
        del kwargs

        import torch
        import torch.multiprocessing as mp
        from opensportslib.core.utils.config import resolve_config_omega

        test_set = self._resolve_split_path("test", test_set)

        self.config = resolve_config_omega(self.config, weights=weights)
        self.config = resolve_inference_class_metadata(self.config)
        logging.info("Configuration:")
        logging.info(self.config)

        effective_weights = weights if weights is not None else self.last_loaded_weights

        world_size = torch.cuda.device_count()
        use_ddp = use_ddp and world_size > 1

        ctx = mp.get_context("spawn")
        queue = ctx.Queue()

        if use_ddp:
            mp.spawn(
                ClassificationModel._worker_ddp,
                args=(
                    world_size,
                    "infer",
                    self.config_path,
                    self.config,
                    queue,
                    None,
                    None,
                    test_set,
                    effective_weights,
                    use_wandb,
                ),
                nprocs=world_size,
            )
        else:
            ClassificationModel._worker_ddp(
                rank=0,
                world_size=1,
                mode="infer",
                config_path=self.config_path,
                config=self.config,
                return_queue=queue,
                test_set=test_set,
                weights=effective_weights,
                use_wandb=use_wandb,
            )

        predictions = queue.get()
        if use_ddp and isinstance(predictions, str):
            with open(predictions, encoding="utf-8") as f:
                predictions = json.load(f)
        return predictions

    def evaluate(
        self,
        test_set=None,
        weights=None,
        predictions=None,
        use_ddp=False,
        use_wandb=True,
        **kwargs,
    ):
        """Run inference on test set and return evaluation metrics."""
        del kwargs

        from opensportslib.datasets.builder import build_dataset
        from opensportslib.core.trainer.classification_trainer import Trainer_Classification
        from opensportslib.core.utils.config import resolve_config_omega

        test_set = self._resolve_split_path("test", test_set)

        self.config = resolve_config_omega(self.config, weights=weights)
        self.config = resolve_inference_class_metadata(self.config)
        logging.info("Configuration:")
        logging.info(self.config)
        if predictions is None:
            predictions = self.infer(
                test_set=test_set,
                weights=weights,
                use_ddp=use_ddp,
                use_wandb=use_wandb,
            )

        self.trainer = self.trainer or Trainer_Classification(self.config)
        test_data = build_dataset(self.config, test_set, None, split="test")
        metrics = self.trainer.evaluate(
            pred_path=predictions,
            gt_path=test_set,
            class_names=test_data.label_map,
            exclude_labels=test_data.exclude_labels,
        )

        logging.info(f"TEST METRICS : {metrics}")
        return metrics
