"""Version-neutral configuration APIs for OpenSportsLib."""

from .loader import load_config, load_config_omega, resolve_config, save_config
from .migrate import migrate_config, migrate_legacy_to_canonical
from .schema import ConfigDocument, DataConfig, IOConfig, ModelConfig, SystemConfig, TrainConfig
from .validate import validate_config

__all__ = [
    "ConfigDocument",
    "DataConfig",
    "IOConfig",
    "ModelConfig",
    "SystemConfig",
    "TrainConfig",
    "load_config",
    "load_config_omega",
    "migrate_legacy_to_canonical",
    "migrate_config",
    "resolve_config",
    "save_config",
    "validate_config",
]
