from __future__ import annotations

from config.settings import ModelSettings, Settings


def get_enabled_models(settings: Settings) -> dict[str, ModelSettings]:
    return {
        item.model_id: item
        for item in settings.model_settings()
        if item.enabled and item.config_path
    }


def get_configured_model_ids(settings: Settings) -> list[str]:
    return [item.model_id for item in settings.model_settings() if item.enabled and item.config_path]
