"""Prediction-format contracts for localization metrics."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from opensportslib.metrics import localization_metric as metrics
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


@pytest.mark.parametrize("predictions,tolerance,expected", [
    ([], 0, 0.),
    ([("match", 10, .9), ("match", 30, .8)], 0, 1.),
    ([("match", 10, .9), ("match", 10, .8), ("match", 30, .7)], 0, 5 / 6),
    ([("other", 10, .9), ("match", 10, .8), ("match", 30, .7)], 0, 2 / 3),
    ([("match", 12, .9), ("match", 32, .8)], 2, 1.),
    ([("match", 12, .9), ("match", 32, .8)], 1, 0.),
])
def test_average_precision_matches_each_event_once(predictions, tolerance, expected):
    assert metrics.compute_average_precision(
        predictions, {"match": [10, 30]}, tolerance=tolerance,
    ) == pytest.approx(expected)


def test_average_precision_requires_confidence_order():
    with pytest.raises(AssertionError):
        metrics.compute_average_precision([("match", 10, .2), ("match", 30, .9)], {"match": [10, 30]})


def test_map_averages_classes_equally_and_respects_tracking_clock(monkeypatch):
    monkeypatch.setattr(metrics, "log_table_wandb", lambda **kwargs: None)
    truth = [{"path": "match", "events": [
        {"label": "PASS", "frame": 1, "eval_frame": 10},
        {"label": "PASS", "frame": 2, "eval_frame": 30},
        {"label": "SHOT", "frame": 50},
    ]}]
    predictions = [{"video": "match", "events": [
        {"label": "SHOT", "frame": 52, "confidence": .7},
        {"label": "PASS", "frame": 30, "confidence": .8},
        {"label": "PASS", "frame": 10, "confidence": .9},
    ]}]
    scores, tolerances = metrics.compute_mAPs_E2E(truth, predictions, tolerances=[0, 2])
    assert scores == pytest.approx([.5, 1.])
    assert tolerances == [0, 2]
    assert metrics.get_predictions(predictions, label="PASS") == [("match", 10, .9), ("match", 30, .8)]
    assert metrics.get_predictions([{"video": "empty", "events": []}]) == []


def test_map_rejects_mismatched_video_sets():
    with pytest.raises(AssertionError, match="Video set mismatch"):
        metrics.compute_mAPs_E2E([{"path": "missing", "events": []}], [])


def test_nms_respects_labels_and_window_boundary_without_mutation():
    predictions = [{"video": "match", "events": [
        {"label": "PASS", "frame": 10, "confidence": .9},
        {"label": "PASS", "frame": 12, "confidence": .4},
        {"label": "SHOT", "frame": 12, "confidence": .3},
        {"label": "PASS", "frame": 13, "confidence": .5},
    ]}]
    original = deepcopy(predictions)
    result = metrics.non_maximum_supression(predictions, window=2)
    assert [(e["label"], e["frame"]) for e in result[0]["events"]] == [("PASS", 10), ("SHOT", 12), ("PASS", 13)]
    assert result[0]["num_events"] == 3
    assert predictions == original
    assert metrics.non_maximum_supression([], window=2) == []


def test_dense_ground_truth_handles_frame_time_visibility_and_end_boundary():
    labels = [
        {"label": "PASS", "frame": 1},
        {"label": "SHOT", "position": "1000", "gameTime": "1 - 00:01", "visibility": "not shown"},
        {"label": "PASS", "gameTime": "1 - 00:02"},
        {"label": "SHOT", "frame": 999},
    ]
    actual = metrics.label2vector(labels, num_classes=2, framerate=2, EVENT_DICTIONARY={"PASS": 0, "SHOT": 1}, vector_size=6)
    expected = np.zeros((6, 2))
    expected[1, 0] = expected[4, 0] = expected[5, 1] = 1
    expected[2, 1] = -1
    np.testing.assert_array_equal(actual, expected)


def test_dense_predictions_preserve_confidence_and_missing_sentinel():
    predictions = [
        {"label": "PASS", "frame": 1, "confidence": .8},
        {"label": "SHOT", "frame": None, "position": "1000", "confidence": .6},
        {"label": "PASS", "frame": 999, "confidence": .4},
    ]
    actual = metrics.predictions2vector(predictions, num_classes=2, framerate=2, EVENT_DICTIONARY={"PASS": 0, "SHOT": 1}, vector_size=4)
    expected = np.full((4, 2), -1.)
    expected[1, 0], expected[2, 1], expected[3, 0] = .8, .6, .4
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(metrics.label2vector([], num_classes=2, vector_size=4), np.zeros((4, 2)))
    np.testing.assert_array_equal(metrics.predictions2vector([], num_classes=2, vector_size=4), np.full((4, 2), -1.))


def test_foreground_f1_separates_detection_from_classification():
    score = metrics.ForegroundF1()
    assert score.get(None) == 0
    for true, predicted in [(1, 1), (1, 2), (0, 1), (2, 0), (0, 0)]:
        score.update(true, predicted)
    assert score.tp_fp_fn(None) == (2, 1, 1)
    assert score.get(None) == pytest.approx(2 / 3)
    assert score.tp_fp_fn(1) == (1, 1, 1)
    assert score.get(1) == .5
    assert score.get(2) == 0
