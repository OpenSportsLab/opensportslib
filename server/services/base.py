from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from config.settings import ModelSettings


class BaseTaskService(ABC):
    def __init__(self, model_settings: ModelSettings):
        self.model_settings = model_settings

    @abstractmethod
    def preload(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, request_payload: dict[str, Any], working_dir: Path) -> dict[str, Any]:
        raise NotImplementedError

    def unload(self) -> None:
        self._release_model_refs()

    def _release_model_refs(self) -> None:
        for attr in ("model", "trainer", "processor"):
            if hasattr(self, attr):
                setattr(self, attr, None)
