from __future__ import annotations

import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from opensportslib.core.trainer.localization_trainer import Evaluator
from opensportslib.core.utils.load_annotations import (
    annotationstoe2eformat,
    expand_localization_intervals,
)
from opensportslib.datasets.localization_dataset import FrameReader


def _record(status="verified"):
    return {
        "id": "sample-01",
        "inputs": [{"type": "video", "path": "videos/sample.mp4"}],
        "metadata": {
            "annotation_status": status,
            "intervals": [
                {
                    "period": "segment_a",
                    "start_time_ms": 10_000,
                    "end_time_ms": 12_000,
                },
                {
                    "period": "segment_b",
                    "start_time_ms": 20_000,
                    "end_time_ms": 22_000,
                },
            ],
        },
        "events": [
            {"head": "Actions", "label": "Action", "position_ms": 10_500},
            {"head": "Actions", "label": "Action", "position_ms": 20_500},
        ],
    }


def _document(records):
    return {
        "version": "2.0",
        "task": "localization",
        "metadata": {
            "interval_semantics": "half-open [start_time_ms, end_time_ms)"
        },
        "labels": {"Actions": {"labels": ["Action"]}},
        "data": records,
    }


def test_expand_localization_intervals_rebases_half_open_events():
    segments = expand_localization_intervals(_record())

    assert [segment["period"] for segment in segments] == [
        "segment_a",
        "segment_b",
    ]
    assert [segment["events"][0]["position_ms"] for segment in segments] == [
        500,
        500,
    ]
    assert segments[0]["logical_path"].endswith(
        "sample.interval-01-segment_a.mp4"
    )
    assert segments[1]["logical_path"].endswith(
        "sample.interval-02-segment_b.mp4"
    )

    invalid = _record()
    invalid["events"].append(
        {"head": "Actions", "label": "Action", "position_ms": 12_000}
    )
    with pytest.raises(ValueError, match="outside its declared"):
        expand_localization_intervals(invalid)


def test_expand_localization_intervals_preserves_legacy_and_omits_excluded():
    legacy = _record()
    del legacy["metadata"]["intervals"]

    segments = expand_localization_intervals(legacy)

    assert len(segments) == 1
    assert segments[0]["logical_path"] == "videos/sample.mp4"
    assert segments[0]["start_time_ms"] == 0
    assert segments[0]["end_time_ms"] is None
    assert segments[0]["events"] == legacy["events"]

    excluded = {
        "id": "known-bad-record",
        "metadata": {"annotation_status": "excluded", "intervals": []},
    }
    assert expand_localization_intervals(excluded) == []


class _MetadataCapture:
    def __init__(self, *_):
        self.released = False

    def isOpened(self):
        return True

    def get(self, key):
        return {
            cv2.CAP_PROP_FRAME_WIDTH: 398,
            cv2.CAP_PROP_FRAME_HEIGHT: 224,
            cv2.CAP_PROP_FPS: 25.0,
            cv2.CAP_PROP_FRAME_COUNT: 300_000,
        }[key]

    def release(self):
        self.released = True


def test_annotation_loader_expands_intervals_and_source_bounds(
    tmp_path, monkeypatch
):
    annotation_path = tmp_path / "annotations.json"
    annotation_path.write_text(json.dumps(_document([_record()])))
    video_path = tmp_path / "videos" / "sample.mp4"
    video_path.parent.mkdir()
    video_path.touch()
    monkeypatch.setattr(cv2, "VideoCapture", _MetadataCapture)

    labels, task_name = annotationstoe2eformat(
        str(annotation_path), str(tmp_path), 25, 2, False
    )

    assert task_name == "Actions"
    assert len(labels) == 2
    assert [item["annotation_status"] for item in labels] == [
        "verified",
        "verified",
    ]
    assert labels[0]["source_start_frame"] == 250
    assert labels[0]["source_end_frame"] == 300
    assert labels[0]["num_frames_base"] == 50
    assert labels[0]["events"] == [{"frame": 1, "label": "Action"}]

    with pytest.raises(ValueError, match="requires the OpenCV backend"):
        annotationstoe2eformat(
            str(annotation_path), str(tmp_path), 25, 2, True
        )


class _FrameCapture:
    instances = []

    def __init__(self, *_):
        self.position = 0
        self.seek_positions = []
        self.read_positions = []
        self.__class__.instances.append(self)

    def get(self, key):
        return {
            cv2.CAP_PROP_FPS: 10.0,
            cv2.CAP_PROP_FRAME_COUNT: 100,
        }.get(key, 0)

    def set(self, key, value):
        assert key == cv2.CAP_PROP_POS_FRAMES
        self.position = int(value)
        self.seek_positions.append(self.position)
        return True

    def read(self):
        if self.position >= 100:
            return False, None
        self.read_positions.append(self.position)
        self.position += 1
        return True, np.zeros((2, 2, 3), dtype=np.uint8)

    def release(self):
        pass


def test_frame_reader_seeks_and_stops_inside_interval(monkeypatch):
    _FrameCapture.instances.clear()
    monkeypatch.setattr(cv2, "VideoCapture", _FrameCapture)
    reader = FrameReader(
        "rgb",
        crop_transform=None,
        img_transform=lambda image: image,
        same_transform=False,
        sample_fps=2,
        TARGET_HEIGHT=2,
        TARGET_WIDTH=2,
    )

    frames = reader.load_frames_ocv(
        "unused.mp4",
        -1,
        5,
        pad=True,
        source_start_frame=20,
        source_end_frame=50,
    )

    capture = _FrameCapture.instances[-1]
    assert frames.shape == (6, 3, 2, 2)
    assert capture.seek_positions == [20]
    assert min(capture.read_positions) == 20
    assert max(capture.read_positions) < 50


def test_v2_evaluator_scores_only_verified_logical_intervals(tmp_path):
    verified = _record()
    unlabeled = _record(status="unlabeled")
    unlabeled["id"] = "adaptation-only"
    unlabeled["inputs"][0]["path"] = "videos/adaptation-only.mp4"
    unlabeled["events"] = []
    document = _document([verified, unlabeled])
    annotation_path = tmp_path / "annotations.json"
    annotation_path.write_text(json.dumps(document))

    prediction_items = []
    for record in (verified, unlabeled):
        for segment in expand_localization_intervals(record):
            events = []
            if segment["annotation_status"] == "verified":
                events.append(
                    {
                        "head": "Actions",
                        "label": "Action",
                        "frame": 1,
                        "position_ms": 500,
                        "confidence": 0.9,
                    }
                )
            else:
                events.append(
                    {
                        "head": "Actions",
                        "label": "Action",
                        "frame": 1,
                        "position_ms": 500,
                        "confidence": 1.0,
                    }
                )
            prediction_items.append(
                {
                    "inputs": [
                        {
                            "type": "video",
                            "path": segment["logical_path"],
                            "fps": 2.0,
                        }
                    ],
                    "events": events,
                }
            )

    predictions_path = tmp_path / "predictions.json"
    predictions_path.write_text(
        json.dumps({"version": "2.0", "data": prediction_items})
    )
    evaluator = object.__new__(Evaluator)
    evaluator.extract_fps = 2
    cfg = SimpleNamespace(
        annotation_path=str(annotation_path), classes=["Action"]
    )

    result = evaluator.evaluate_common_JSON(
        cfg, str(predictions_path), metric="at1"
    )

    assert "Average mAP" in result
    assert "Average mAP    100" in result


def test_v2_evaluator_counts_missing_verified_interval_as_empty(tmp_path):
    annotation_path = tmp_path / "annotations.json"
    annotation_path.write_text(json.dumps(_document([_record()])))

    first_segment = expand_localization_intervals(_record())[0]
    predictions_path = tmp_path / "predictions.json"
    predictions_path.write_text(
        json.dumps(
            {
                "version": "2.0",
                "data": [
                    {
                        "inputs": [
                            {
                                "type": "video",
                                "path": first_segment["logical_path"],
                                "fps": 2.0,
                            }
                        ],
                        "events": [
                            {
                                "label": "Action",
                                "frame": 1,
                                "position_ms": 500,
                                "confidence": 0.9,
                            }
                        ],
                    }
                ],
            }
        )
    )
    evaluator = object.__new__(Evaluator)
    evaluator.extract_fps = 2
    cfg = SimpleNamespace(
        annotation_path=str(annotation_path), classes=["Action"]
    )

    result = evaluator.evaluate_common_JSON(
        cfg, str(predictions_path), metric="at1"
    )

    assert "Average mAP" in result
    assert "Average mAP  54.55" in result
