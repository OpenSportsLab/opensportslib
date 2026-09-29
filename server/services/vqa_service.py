from __future__ import annotations

from pathlib import Path
from typing import Any

from config.settings import ModelSettings
from services.base import BaseTaskService
from services.configuration import configured_prediction


class VQAService(BaseTaskService):
    def __init__(self, model_settings: ModelSettings):
        super().__init__(model_settings)
        self.model = None
        self.trainer = None
        self.processor = None

    def preload(self) -> None:
        from opensportslib.apis import VQAModel

        self.model = VQAModel(
            config=self.model_settings.config_path,
            weights=self.model_settings.weights,
        )

    @configured_prediction
    def predict(self, request_payload: dict[str, Any], working_dir: Path) -> dict[str, Any]:
        if self.model is None:
            self.preload()
        uploaded_manifest = request_payload.get("resolved_test_set_path")
        video_path = request_payload.get("resolved_video_path") or request_payload.get("video_path")
        question = str(request_payload.get("question") or "").strip()
        conversation_history = list(request_payload.get("conversation_history") or [])
        history_mode = "single_turn"
        if conversation_history:
            history_mode = "question_only_fallback"
        if uploaded_manifest:
            predictions = self.model.infer(test_set=str(uploaded_manifest), use_wandb=False)
        else:
            predictions = self.model.infer(video_path=video_path, question=question, use_wandb=False)
        return {
            "task_type": "vqa",
            "model_id": self.model_settings.model_id,
            "predictions": predictions,
            "summary": _extract_vqa_summary(predictions),
            "history_mode": history_mode,
            "conversation_history_used": conversation_history,
        }


def _extract_vqa_summary(predictions: dict[str, Any]) -> dict[str, Any]:
    first_item = {}
    for item in predictions.get("data", []) if isinstance(predictions, dict) else []:
        first_item = item
        break
    return {
        "answer_text": first_item.get("answer_text"),
        "sample_id": first_item.get("id"),
        "question": first_item.get("question"),
    }
