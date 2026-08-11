import logging
import os
import time
from types import SimpleNamespace

from opensportslib.apis.base_task_model import BaseTaskModel
from opensportslib.core.config.accessors import (
    get_data_classes,
    get_loader_backend,
    get_system_gpu_count,
    get_system_seed,
    set_system_path,
    get_train_trainer_type,
    get_train_execution,
    get_split_annotation_path,
    get_split_cfg,
    set_split_annotation_path,
    set_loader_backend,
    get_model_family,
)
from opensportslib.core.utils.config import expand, resolve_inference_class_metadata
from opensportslib.core.config.loader import _dali_available


class LocalizationModel(BaseTaskModel):
    """Top-level task wrapper for localization / spotting."""

    _HF_BACKEND_SPLIT_TYPES = {
        "dali": {
            "VideoGameWithOpencv": "VideoGameWithDali",
            "VideoGameWithOpencvVideo": "VideoGameWithDaliVideo",
        },
        "opencv": {
            "VideoGameWithDali": "VideoGameWithOpencv",
            "VideoGameWithDaliVideo": "VideoGameWithOpencvVideo",
        },
    }

    # def __init__(self, config=None, weights=None):
    #     super().__init__(config=config, weights=None)
    #     if weights is not None:
    #         self.last_loaded_weights = weights
    #         self.best_checkpoint = weights

    #     self.train_flag = False  # Flag to indicate whether we're in training mode (affects checkpoint loading behavior)

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

    def _set_split_path(self, split: str, value: str) -> str:
        resolved = expand(value)
        set_split_annotation_path(self.config, split, resolved)
        return resolved

    def _gate_multi_gpu_by_device(self, device) -> None:
        """Disable TRAIN.execution.multi_gpu when effective device is CPU."""
        execution = getattr(getattr(self.config, "TRAIN", None), "execution", None)
        if execution is None:
            return

        multi_gpu = bool(getattr(execution, "multi_gpu", False))
        if device.type == "cpu" and multi_gpu:
            execution.multi_gpu = False
            logging.warning(
                "Detected SYSTEM.device=%s; forcing TRAIN.execution.multi_gpu=false for localization runtime.",
                device,
            )

    @staticmethod
    def _device_type(device) -> str:
        device_type = getattr(device, "type", device)
        return str(device_type).split(":", 1)[0].lower()

    @staticmethod
    def _is_hf_repo_weights(weights: str | None) -> bool:
        if not weights:
            return False
        from opensportslib.core.utils.config import is_local_path

        return not is_local_path(weights)

    @staticmethod
    def _iter_split_items(splits):
        if splits is None:
            return []
        if isinstance(splits, dict):
            return list(splits.items())
        return list(vars(splits).items())

    @staticmethod
    def _ensure_namespace(parent, attr: str):
        current = getattr(parent, attr, None)
        if current is None:
            current = SimpleNamespace()
            setattr(parent, attr, current)
        return current

    def _normalize_opencv_dataloader(self, split_name: str, split_cfg) -> None:
        dataloader = getattr(split_cfg, "dataloader", None)
        if dataloader is None:
            dataloader = SimpleNamespace()
            setattr(split_cfg, "dataloader", dataloader)

        if getattr(dataloader, "batch_size", None) is None:
            dataloader.batch_size = 1
        if getattr(dataloader, "shuffle", None) is None:
            dataloader.shuffle = split_name == "train"
        if getattr(dataloader, "num_workers", None) is None:
            dataloader.num_workers = 0
        if getattr(dataloader, "pin_memory", None) is None:
            dataloader.pin_memory = False

    def _adapt_hf_backend_for_device(self, weights: str | None) -> None:
        if not self._is_hf_repo_weights(weights):
            return

        from opensportslib.core.utils.config import select_device

        device = select_device(self.config.SYSTEM)
        backend = "dali" if self._device_type(device) == "cuda" and _dali_available() else "opencv"
        current_backend = get_loader_backend(self.config)
        set_loader_backend(self.config, backend)

        common = getattr(getattr(self.config, "DATA", None), "common", None)
        if common is None:
            return
        splits = self._ensure_namespace(common, "splits")
        remap = self._HF_BACKEND_SPLIT_TYPES[backend]

        for split_name, split_cfg in self._iter_split_items(splits):
            split_type = getattr(split_cfg, "type", None)
            if split_type in remap:
                setattr(split_cfg, "type", remap[split_type])
            if backend == "opencv":
                self._normalize_opencv_dataloader(split_name, split_cfg)

        if current_backend != backend:
            logging.info(
                "HF localization runtime backend override: %s -> %s for weights=%s",
                current_backend,
                backend,
                weights,
            )

    def _configure_test_time_adaptation(self) -> None:
        """Start a fresh configured adaptation stream for localization inference."""
        model_cfg = getattr(self.config, "MODEL", None)
        policies = getattr(model_cfg, "policies", None)
        if isinstance(policies, dict):
            adaptation_cfg = policies.get("test_time_adaptation")
        else:
            adaptation_cfg = getattr(policies, "test_time_adaptation", None)

        if adaptation_cfg is None:
            if hasattr(self.model, "configure_test_time_adaptation"):
                self.model.configure_test_time_adaptation(None)
            return

        enabled = (
            adaptation_cfg.get("enabled", False)
            if isinstance(adaptation_cfg, dict)
            else getattr(adaptation_cfg, "enabled", False)
        )
        if enabled and str(get_model_family(self.config)).strip().lower() != "e2e":
            raise ValueError("SpoTTA is currently integrated only for the E2ESpot family.")
        test_split_type = getattr(get_split_cfg(self.config, "test"), "type", None)
        if enabled and test_split_type != "VideoGameWithOpencvVideo":
            raise ValueError(
                "The reproducible SpoTTA E2ESpot recipe requires a "
                "VideoGameWithOpencvVideo test split."
            )
        if enabled:
            set_loader_backend(self.config, "opencv")
            import random

            import numpy as np
            import torch

            seed = get_system_seed(self.config)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        if not hasattr(self.model, "configure_test_time_adaptation"):
            if enabled:
                raise ValueError("This localization model does not support test-time adaptation.")
            return
        self.model.configure_test_time_adaptation(adaptation_cfg)

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

        self._adapt_hf_backend_for_device(weights)
        device = select_device(self.config.SYSTEM)
        self._gate_multi_gpu_by_device(device)
        if self.model is None:
            self.model = build_model(self.config, device=device)

        if hasattr(self.model, "configure_test_time_adaptation"):
            self.model.configure_test_time_adaptation(None)

        inner_model = getattr(self.model, "_model", None)
        if inner_model is None:
            inner_model = getattr(self.model, "model", self.model)

        if is_local_path(weights):
            set_system_path(
                self.config,
                "work_dir",
                os.path.dirname(os.path.abspath(weights)),
            )

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
            0 if get_train_execution(self.config).get("criterion_valid") == "map" else float("inf")
        )
        self._resume_state = {
            "optimizer": optimizer,
            "scheduler": scheduler,
            "scaler": scaler,
            "epoch": epoch if epoch is not None else 0,
            "best_epoch": best_epoch,
            "best_criterion_valid": best_criterion_valid,
        }

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

        if str(get_model_family(self.config)).lower() == "rulebased":
            raise NotImplementedError(
                "RuleBased localization models are inference-only; call infer() instead of train()."
            )

        train_set = self._resolve_split_path("train", train_set)
        valid_set = self._resolve_split_path("valid", valid_set)
        self._set_split_path("train", train_set)
        self._set_split_path("valid", valid_set)
        # E2E validation mAP uses the `valid_data_frames` split; keep it in sync
        # with explicit valid annotation overrides.
        self._set_split_path("valid_data_frames", valid_set)

        self.config = resolve_config_omega(self.config, weights=weights)
        self.config = resolve_inference_class_metadata(self.config)
        effective_weights = weights if weights is not None else self.last_loaded_weights
        self._adapt_hf_backend_for_device(effective_weights)
        check_config(self.config, split="train")
        init_wandb(
            self.config_path,
            self.config,
            run_id=os.environ["RUN_ID"],
            use_wandb=use_wandb,
        )

        logging.info("Configuration:")
        logging.info(self.config)

        def set_seed(seed):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)

        set_seed(get_system_seed(self.config))

        start = time.time()

        data_obj_train = build_dataset(self.config, split="train")
        dataset_train = data_obj_train.building_dataset(
            cfg=data_obj_train.cfg,
            gpu=get_system_gpu_count(self.config),
            default_args=data_obj_train.default_args,
        )
        train_loader = data_obj_train.building_dataloader(
            dataset_train,
            cfg=data_obj_train.cfg.dataloader,
            gpu=get_system_gpu_count(self.config),
            dali=(get_loader_backend(self.config) == "dali"),
        )

        data_obj_valid = build_dataset(self.config, split="valid")
        dataset_valid = data_obj_valid.building_dataset(
            cfg=data_obj_valid.cfg,
            gpu=get_system_gpu_count(self.config),
            default_args=data_obj_valid.default_args,
        )
        valid_loader = data_obj_valid.building_dataloader(
            dataset_valid,
            cfg=data_obj_valid.cfg.dataloader,
            gpu=get_system_gpu_count(self.config),
            dali=(get_loader_backend(self.config) == "dali"),
        )

        default_args = get_default_args_trainer(self.config, len(train_loader))
        
        self.train_flag = True  # Set flag to indicate training mode for checkpoint loading
        if effective_weights is not None:
            if self.model is None or self.last_loaded_weights != effective_weights:
                self.load_weights(weights=effective_weights, default_args=default_args)
        elif self.model is None:
            device = select_device(self.config.SYSTEM)
            self._gate_multi_gpu_by_device(device)
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
                get_data_classes(self.config),
                get_train_trainer_type(self.config),
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

        self.config = resolve_config_omega(self.config, weights=weights)
        self.config = resolve_inference_class_metadata(self.config)
        effective_weights = weights if weights is not None else self.last_loaded_weights
        self._adapt_hf_backend_for_device(effective_weights)
        check_config(self.config, split="test")
        self.config = resolve_inference_class_metadata(self.config)
        self.config.infer_split = whether_infer_split(get_split_cfg(self.config, "test"))

        init_wandb(
            self.config_path,
            self.config,
            run_id=os.environ["RUN_ID"],
            use_wandb=use_wandb,
        )

        logging.info("Configuration:")
        logging.info(self.config)

        start = time.time()

        if effective_weights is not None:
            if self.model is None or self.last_loaded_weights != effective_weights:
                self.load_weights(weights=effective_weights)
        elif self.model is None:
            device = select_device(self.config.SYSTEM)
            self._gate_multi_gpu_by_device(device)
            self.model = build_model(self.config, device=device)

        self._configure_test_time_adaptation()

        data_obj_test = build_dataset(self.config, split="test")
        dataset_test = data_obj_test.building_dataset(
            cfg=data_obj_test.cfg,
            gpu=get_system_gpu_count(self.config),
            default_args=data_obj_test.default_args,
        )
        test_loader = data_obj_test.building_dataloader(
            dataset_test,
            cfg=data_obj_test.cfg.dataloader,
            gpu=get_system_gpu_count(self.config),
            dali=(get_loader_backend(self.config) == "dali"),
        )

        inferer = build_inferer(cfg=self.config, model=self.model)
        predictions = inferer.infer(
            cfg=self.config,
            data=dataset_test,
            dataloader=test_loader,
        )

        adaptation_stats = getattr(self.model, "test_time_adaptation_stats", None)
        if adaptation_stats is not None:
            logging.info("SpoTTA inference stats: %s", adaptation_stats)

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
        self.config = resolve_config_omega(self.config, weights=weights)
        self.config = resolve_inference_class_metadata(self.config)
        effective_weights = weights if weights is not None else self.last_loaded_weights
        self._adapt_hf_backend_for_device(effective_weights)
        check_config(self.config, split="test")
        self.config = resolve_inference_class_metadata(self.config)
        self.config.infer_split = whether_infer_split(get_split_cfg(self.config, "test"))

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

        test_path = get_split_annotation_path(self.config, "test")
        if has_localization_events(test_path):
            logging.info("Ground truth labels detected -> running evaluation")
            evaluator = build_evaluator(cfg=self.config)
            eval_input = (
                getattr(get_split_cfg(self.config, "test"), "results")
                if isinstance(predictions, dict)
                else predictions
            )
            metrics = evaluator.evaluate(
                cfg_testset=get_split_cfg(self.config, "test"),
                json_gz_file=eval_input,
            )
        else:
            logging.info("No labels found in annotation file -> skipping evaluation")

        return metrics
