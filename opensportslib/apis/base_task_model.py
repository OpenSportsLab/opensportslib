"""Shared task-level wrapper base for OpenSportsLib APIs."""

from __future__ import annotations

import json
import logging
import os
import uuid
from abc import ABC, abstractmethod
from typing import Any

from opensportslib.core.utils.config import expand, load_config_omega


class BaseTaskModel(ABC):
    """Thin shared contract for task-level OpenSportsLib wrappers."""

    def __init__(self, config=None, weights=None):
        self._configure_logging()

        if config is None:
            raise ValueError("config path is required")

        self.config_path = expand(config)
        self.config = load_config_omega(self.config_path)

        data_cfg = getattr(self.config, "DATA", None)
        if data_cfg is not None and hasattr(data_cfg, "data_dir"):
            data_cfg.data_dir = expand(data_cfg.data_dir)
            logging.info(f"Data directory: {data_cfg.data_dir}")

        self.run_id = os.environ.get("RUN_ID") or str(uuid.uuid4())[:8]
        os.environ["RUN_ID"] = self.run_id

        system_cfg = getattr(self.config, "SYSTEM", None)
        if system_cfg is not None:
            base_save_dir = expand(getattr(system_cfg, "save_dir", None) or "./checkpoints")
            model_cfg = getattr(self.config, "MODEL", None)
            backbone_cfg = getattr(model_cfg, "backbone", None)
            model_name = getattr(backbone_cfg, "type", None) or "model"
            run_save_dir = os.path.join(base_save_dir, model_name, self.run_id)
            self.save_dir = run_save_dir
            system_cfg.save_dir = run_save_dir
            if hasattr(system_cfg, "work_dir"):
                system_cfg.work_dir = run_save_dir
            os.makedirs(run_save_dir, exist_ok=True)
        else:
            self.save_dir = expand("./checkpoints")
            os.makedirs(self.save_dir, exist_ok=True)

        logging.info(f"Save directory: {self.save_dir}")

        self.model = None
        self.processor = None
        self.trainer = None
        self.best_checkpoint = None
        self.last_loaded_weights = None

        if weights is not None:
            self.load_weights(weights=weights)

    @staticmethod
    def _configure_logging() -> None:
        root_logger = logging.getLogger()
        if not root_logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s | %(levelname)s | %(message)s",
            )
        elif root_logger.level > logging.INFO:
            root_logger.setLevel(logging.INFO)

    @abstractmethod
    def load_weights(
        self,
        weights: str | None = None,
        **kwargs,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def train(
        self,
        train_set: str | None = None,
        valid_set: str | None = None,
        weights: str | None = None,
        use_wandb: bool = True,
        **kwargs,
    ) -> str | None:
        raise NotImplementedError

    @abstractmethod
    def infer(
        self,
        test_set: str | None = None,
        weights: str | None = None,
        use_wandb: bool = True,
        **kwargs,
    ) -> dict:
        raise NotImplementedError

    @abstractmethod
    def evaluate(
        self,
        test_set: str | None = None,
        weights: str | None = None,
        predictions: str | dict[str, Any] | None = None,
        use_wandb: bool = True,
        **kwargs,
    ) -> dict | str | None:
        raise NotImplementedError

    def save_predictions(
        self,
        output_path: str,
        predictions: dict,
    ) -> str:
        """Persist in-memory prediction JSON payload to a target file path."""

        dst = expand(output_path)
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)

        if not isinstance(predictions, dict):
            raise TypeError(
                f"Unsupported predictions type: {type(predictions).__name__}. "
                "Expected dict."
            )

        with open(dst, "w", encoding="utf-8") as f:
            json.dump(predictions, f)
        return dst
