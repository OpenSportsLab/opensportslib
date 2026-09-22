"""Prediction-format contracts for localization metrics."""

from __future__ import annotations

from opensportslib.metrics.localization_metric import build_snpro_prediction_json


def test_localization_prediction_builder_emits_osl_v2_shape():
    payload = build_snpro_prediction_json([
        {
            "video": "clips/match.mp4",
            "fps": 25,
            "events": [{
                "label": "GOAL",
                "gameTime": "1 - 00:10",
                "frame": 250,
                "position": 10_000,
                "confidence": 0.9,
            }],
        }
    ], split="test", created_by="test")
    assert payload["version"] == "2.0"
    assert payload["task"] == "localization"
    assert payload["metadata"] == {"type": "predictions", "created_by": "test", "split": "test"}
    assert payload["data"][0]["events"][0]["position_ms"] == 10_000
