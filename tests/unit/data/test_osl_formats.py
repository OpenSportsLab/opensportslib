"""Minimal executable contracts for common OSL annotation documents."""

from __future__ import annotations

import json

import pytest

from opensportslib.core.utils.load_annotations import load_annotations


pytestmark = pytest.mark.unit


def _osl_document(task: str, sample: dict, classes: list[str]) -> dict:
    return {
        "metadata": {"schema_version": 2, "task": task},
        "labels": {"action": {"labels": classes}},
        "data": [sample],
    }


def _classification_sample(label="PASS"):
    return {
        "id": "sample-1",
        "inputs": [{"type": "video", "path": "clips/sample.mp4"}],
        "labels": {"action": {"label": label}},
    }


def test_classification_osl_document_loads(tmp_path):
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(_osl_document("classification", _classification_sample(), ["PASS", "SHOT"])))
    samples, labels = load_annotations(path)
    assert labels == {"PASS": 0, "SHOT": 1}
    assert samples == [{
        "video_paths": ["clips/sample.mp4"], "label": 0,
        "inputs": [{"type": "video", "path": "clips/sample.mp4"}],
        "id": "sample-1", "metadata": {},
    }]


def test_unknown_label_is_not_silently_remapped(tmp_path):
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(_osl_document("classification", _classification_sample("UNKNOWN"), ["PASS"])))
    samples, labels = load_annotations(path)
    assert labels == {"PASS": 0}
    assert samples[0]["label"] is None


@pytest.mark.parametrize("payload", [{}, {"labels": {}}, {"labels": {"action": {"labels": []}}}])
def test_malformed_osl_documents_fail_loudly(tmp_path, payload):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload))
    with pytest.raises((KeyError, TypeError)):
        load_annotations(path)
