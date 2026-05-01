import logging
import os
import time

from opensportslib.apis.base_task_model import BaseTaskModel
from opensportslib.core.utils.config import expand


class LocalizationModel(BaseTaskModel):
    """Top-level task wrapper for localization / spotting."""

    def __init__(self, config=None, weights=None):
        super().__init__(config=config, weights=None)
        if weights is not None:
            self.last_loaded_weights = weights
            self.best_checkpoint = weights

        self.train_flag = False  # Flag to indicate whether we're in training mode (affects checkpoint loading behavior)

    def _resolve_split_path(self, split: str, override: str | None = None) -> str:
        if override is not None:
            return expand(override)

        data_cfg = getattr(self.config, "DATA", None)
        split_cfg = getattr(data_cfg, split, None)
        path = getattr(split_cfg, "path", None) if split_cfg is not None else None
        if path:
            return expand(path)

        annotations_cfg = getattr(data_cfg, "annotations", None)
        path = (
            getattr(annotations_cfg, split, None)
            if annotations_cfg is not None
            else None
        )
        if path:
            return expand(path)

        raise ValueError(
            f"Could not resolve path for split '{split}'. "
            f"Expected DATA.{split}.path or DATA.annotations.{split}."
        )

    def _set_split_path(self, split: str, value: str) -> str:
        resolved = expand(value)
        data_cfg = getattr(self.config, "DATA", None)
        split_cfg = getattr(data_cfg, split, None)

        if split_cfg is not None and hasattr(split_cfg, "path"):
            split_cfg.path = resolved
            return resolved

        annotations_cfg = getattr(data_cfg, "annotations", None)
        if annotations_cfg is not None and hasattr(annotations_cfg, split):
            setattr(annotations_cfg, split, resolved)
            return resolved

        raise ValueError(
            f"Could not set path for split '{split}'. "
            f"Expected DATA.{split}.path or DATA.annotations.{split}."
        )

    def load_weights(
        self,
        weights: str | None = None,
        **kwargs,
    ) -> None:
        from opensportslib.models.builder import build_model
        from opensportslib.core.utils.config import is_local_path, select_device
        from opensportslib.core.utils.checkpoint import (
            load_checkpoint,
            localization_remap,
        )
        from opensportslib.core.optimizer.builder import build_optimizer
        from opensportslib.core.scheduler.builder import build_scheduler
        default_args = kwargs.get("default_args", None)
        del kwargs
        if weights is None:
            raise ValueError("`weights` must be provided to load_weights().")

        model_cfg = getattr(self.config, "MODEL", None)
        if not self.train_flag:
            original_multi_gpu = getattr(model_cfg, "multi_gpu", None)
            if model_cfg is not None and original_multi_gpu is not None:
                model_cfg.multi_gpu = False

        device = select_device(self.config.SYSTEM)
        if self.model is None:
            self.model = build_model(self.config, device=device)

        inner_model = getattr(self.model, "_model", None)
        if inner_model is None:
            inner_model = getattr(self.model, "model", self.model)

        if is_local_path(weights):
            self.config.SYSTEM.work_dir = os.path.dirname(os.path.abspath(weights))

        if default_args is not None:
            logging.info("Building optimizer + scaler for checkpoint restore...")
            optimizer, scaler = build_optimizer(
                inner_model.parameters(),  # or _get_params() if required
                self.config.TRAIN.optimizer
            )
            
            logging.info("Building scheduler for checkpoint restore...")
            scheduler = build_scheduler(
                optimizer,
                self.config.TRAIN.scheduler,
                default_args
            )
        else:
            optimizer = scheduler = scaler = None

        inner_model, optimizer, scheduler, scaler, epoch, checkpoint = load_checkpoint(
            model=inner_model,
            path=weights,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            key_remap_fn=localization_remap,
        )

        if hasattr(self.model, "_model"):
            self.model._model = inner_model
        elif hasattr(self.model, "model"):
            self.model.model = inner_model
        else:
            self.model = inner_model

        self.last_loaded_weights = weights
        self.best_checkpoint = weights

        best_epoch = checkpoint.get("best_epoch", 0)

        best_criterion_valid = checkpoint.get(
            "best_criterion_valid",
            0 if self.config.TRAIN.criterion_valid == "map" else float("inf")
        )
        self._resume_state = {
            "optimizer": optimizer,
            "scheduler": scheduler,
            "scaler": scaler,
            "epoch": epoch if epoch is not None else 0,
            "best_epoch": best_epoch,
            "best_criterion_valid": best_criterion_valid,
        }

        if not self.train_flag:
            if model_cfg is not None and original_multi_gpu is not None:
                model_cfg.multi_gpu = original_multi_gpu

    def train(
        self,
        train_set=None,
        valid_set=None,
        weights=None,
        use_wandb=True,
        **kwargs,
    ):
        from opensportslib.datasets.builder import build_dataset
        from opensportslib.models.builder import build_model
        from opensportslib.core.trainer.localization_trainer import build_trainer
        from opensportslib.core.utils.default_args import (
            get_default_args_train,
            get_default_args_trainer,
        )
        from opensportslib.core.utils.config import resolve_config_omega, select_device
        from opensportslib.core.utils.load_annotations import check_config
        from opensportslib.core.utils.wandb import init_wandb
        import random
        import numpy as np
        import torch

        del kwargs

        train_set = self._resolve_split_path("train", train_set)
        valid_set = self._resolve_split_path("valid", valid_set)
        self._set_split_path("train", train_set)
        self._set_split_path("valid", valid_set)

        self.config = resolve_config_omega(self.config)
        check_config(self.config, split="train")
        init_wandb(
            self.config_path,
            self.config,
            run_id=os.environ["RUN_ID"],
            use_wandb=use_wandb,
        )

        logging.info("Configuration:")
        logging.info(self.config)

        effective_weights = weights if weights is not None else self.last_loaded_weights

        def set_seed(seed):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)

        set_seed(self.config.SYSTEM.seed)

        start = time.time()

        data_obj_train = build_dataset(self.config, split="train")
        dataset_train = data_obj_train.building_dataset(
            cfg=data_obj_train.cfg,
            gpu=self.config.SYSTEM.GPU,
            default_args=data_obj_train.default_args,
        )
        train_loader = data_obj_train.building_dataloader(
            dataset_train,
            cfg=data_obj_train.cfg.dataloader,
            gpu=self.config.SYSTEM.GPU,
            dali=self.config.dali,
        )

        data_obj_valid = build_dataset(self.config, split="valid")
        dataset_valid = data_obj_valid.building_dataset(
            cfg=data_obj_valid.cfg,
            gpu=self.config.SYSTEM.GPU,
            default_args=data_obj_valid.default_args,
        )
        valid_loader = data_obj_valid.building_dataloader(
            dataset_valid,
            cfg=data_obj_valid.cfg.dataloader,
            gpu=self.config.SYSTEM.GPU,
            dali=self.config.dali,
        )

        default_args = get_default_args_trainer(self.config, len(train_loader))
        
        self.train_flag = True  # Set flag to indicate training mode for checkpoint loading
        if effective_weights is not None:
            if self.model is None or self.last_loaded_weights != effective_weights:
                self.load_weights(weights=effective_weights, default_args=default_args)
        elif self.model is None:
            device = select_device(self.config.SYSTEM)
            self.model = build_model(self.config, device=device)

        self.trainer = build_trainer(
            cfg=self.config,
            model=self.model,
            default_args=default_args,
            resume_from=self._resume_state if hasattr(self, "_resume_state") else None,
        )

        logging.info("Start training")

        self.trainer.train(
            **get_default_args_train(
                self.model,
                train_loader,
                valid_loader,
                self.config.DATA.classes,
                self.config.TRAIN.type,
            )
        )

        self.best_checkpoint = self.trainer.best_checkpoint_path
        self.last_loaded_weights = self.best_checkpoint

        logging.info(f"Total Execution Time is {time.time()-start} seconds")
        return self.best_checkpoint

    def infer(
        self,
        test_set=None,
        weights=None,
        use_wandb=True,
        **kwargs,
    ):
        """Run model inference and return predictions in OSL JSON format."""
        from opensportslib.datasets.builder import build_dataset
        from opensportslib.models.builder import build_model
        from opensportslib.core.trainer.localization_trainer import build_inferer
        from opensportslib.core.utils.config import resolve_config_omega, select_device
        from opensportslib.core.utils.load_annotations import (
            check_config,
            whether_infer_split,
        )
        from opensportslib.core.utils.wandb import init_wandb

        del kwargs

        test_set = self._resolve_split_path("test", test_set)
        self._set_split_path("test", test_set)

        self.config.MODEL.multi_gpu = False
        self.config = resolve_config_omega(self.config)
        check_config(self.config, split="test")
        self.config.infer_split = whether_infer_split(self.config.DATA.test)

        init_wandb(
            self.config_path,
            self.config,
            run_id=os.environ["RUN_ID"],
            use_wandb=use_wandb,
        )

        logging.info("Configuration:")
        logging.info(self.config)

        start = time.time()

        effective_weights = weights if weights is not None else self.last_loaded_weights

        if effective_weights is not None:
            if self.model is None or self.last_loaded_weights != effective_weights:
                self.load_weights(weights=effective_weights)
        elif self.model is None:
            device = select_device(self.config.SYSTEM)
            self.model = build_model(self.config, device=device)

        data_obj_test = build_dataset(self.config, split="test")
        dataset_test = data_obj_test.building_dataset(
            cfg=data_obj_test.cfg,
            gpu=self.config.SYSTEM.GPU,
            default_args=data_obj_test.default_args,
        )
        test_loader = data_obj_test.building_dataloader(
            dataset_test,
            cfg=data_obj_test.cfg.dataloader,
            gpu=self.config.SYSTEM.GPU,
            dali=self.config.dali,
        )

        inferer = build_inferer(cfg=self.config.MODEL, model=self.model)
        predictions = inferer.infer(
            cfg=self.config,
            data=dataset_test,
            dataloader=test_loader,
        )

        logging.info(f"Total Execution Time is {time.time()-start} seconds")
        return predictions

    def evaluate(
        self,
        test_set=None,
        weights=None,
        predictions=None,
        use_wandb=True,
        **kwargs,
    ):
        from opensportslib.core.trainer.localization_trainer import build_evaluator
        from opensportslib.core.utils.config import resolve_config_omega
        from opensportslib.core.utils.load_annotations import (
            check_config,
            has_localization_events,
            whether_infer_split,
        )
        from opensportslib.core.utils.wandb import init_wandb

        del kwargs

        test_set = self._resolve_split_path("test", test_set)
        self._set_split_path("test", test_set)

        self.config.MODEL.multi_gpu = False
        self.config = resolve_config_omega(self.config)
        check_config(self.config, split="test")
        self.config.infer_split = whether_infer_split(self.config.DATA.test)

        init_wandb(
            self.config_path,
            self.config,
            run_id=os.environ["RUN_ID"],
            use_wandb=use_wandb,
        )

        if predictions is None:
            predictions = self.infer(
                test_set=test_set,
                weights=weights,
                use_wandb=use_wandb,
            )

        metrics = None

        if has_localization_events(self.config.DATA.test.path):
            logging.info("Ground truth labels detected -> running evaluation")
            evaluator = build_evaluator(cfg=self.config)
            eval_input = (
                self.config.DATA.test.results
                if isinstance(predictions, dict)
                else predictions
            )
            metrics = evaluator.evaluate(
                cfg_testset=self.config.DATA.test,
                json_gz_file=eval_input,
            )
        else:
            logging.info("No labels found in annotation file -> skipping evaluation")

        return metrics
