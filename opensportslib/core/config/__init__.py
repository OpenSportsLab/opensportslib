"""Version-neutral configuration APIs for OpenSportsLib."""

from .loader import load_config, load_config_omega, resolve_config, save_config
from .migrate import migrate_config
from .runtime_adapter import adapt_config_to_runtime
from .schema import ConfigDocument, DataConfig, IOConfig, ModelConfig, SystemConfig, TrainConfig
from .validate import validate_config

__all__ = [
    "ConfigDocument",
    "DataConfig",
    "IOConfig",
    "ModelConfig",
    "SystemConfig",
    "TrainConfig",
    "adapt_config_to_runtime",
    "load_config",
    "load_config_omega",
    "migrate_config",
    "resolve_config",
    "save_config",
    "validate_config",
]
