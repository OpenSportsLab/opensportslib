from __future__ import annotations

from pathlib import Path
from typing import Any

from config.settings import ModelSettings
from services.base import BaseTaskService
from services.configuration import configured_prediction
from storage.manifests import build_localization_manifest


class LocalizationService(BaseTaskService):
    def __init__(self, model_settings: ModelSettings):
        super().__init__(model_settings)
        self.model = None
        self.trainer = None
        self.processor = None
        self.label_schema: dict[str, Any] | None = None

    def preload(self) -> None:
        from opensportslib.apis import LocalizationModel
        from opensportslib.core.utils.config import namespace_to_dict

        self.model = LocalizationModel(
            config=self.model_settings.config_path,
            weights=self.model_settings.weights,
        )
        config_dict = namespace_to_dict(self.model.config)
        self.label_schema = _build_label_schema_from_config(config_dict)

    @configured_prediction
    def predict(self, request_payload: dict[str, Any], working_dir: Path) -> dict[str, Any]:
        if self.model is None:
            self.preload()
        uploaded_manifest = request_payload.get("resolved_test_set_path")
        video_path = request_payload.get("resolved_video_path") or request_payload.get("video_path")
        task_options = dict(request_payload.get("task_options") or {})
        if self.label_schema and "label_schema" not in task_options:
            task_options["label_schema"] = self.label_schema
        manifest_path = Path(uploaded_manifest) if uploaded_manifest else build_localization_manifest(
            working_dir=working_dir, video_path=video_path, task_options=task_options, config=self.model.config
        )
        predictions = self.model.infer(
            test_set=str(manifest_path),
            use_wandb=False,
        )
        return {
            "task_type": "localization",
            "model_id": self.model_settings.model_id,
            "manifest_path": str(manifest_path),
            "predictions": predictions,
            "summary": _extract_localization_summary(predictions),
        }


def _extract_localization_summary(predictions: dict[str, Any]) -> dict[str, Any]:
    first_item = {}
    for item in predictions.get("data", []) if isinstance(predictions, dict) else []:
        first_item = item
        break
    events = first_item.get("events") or []
    return {
        "sample_id": first_item.get("id"),
        "num_events": len(events),
        "events_preview": events[:5],
    }


def _build_label_schema_from_config(config_dict: dict[str, Any]) -> dict[str, Any] | None:
    data_common = ((config_dict.get("DATA") or {}).get("common") or {})
    classes = data_common.get("classes")
    if classes is None:
        return None

    if isinstance(classes, dict):
        ordered = [label for label, _ in sorted(classes.items(), key=lambda item: int(item[1]))]
    elif isinstance(classes, list):
        ordered = [str(label) for label in classes]
    else:
        return None

    if not ordered:
        return None

    return {
        "action": {
            "type": "single_label",
            "labels": ordered,
        }
    }
