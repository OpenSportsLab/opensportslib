"""Known-value classification metric tests."""

from __future__ import annotations

import numpy as np

from opensportslib.metrics.classification_metric import compute_classification_metrics


def test_perfect_classification_metrics_are_one():
    logits = np.array([[5.0, 0.0], [0.0, 5.0]])
    metrics = compute_classification_metrics((logits, np.array([0, 1])), top_k=1)
    for name in ("accuracy", "balanced_accuracy", "f1", "precision", "recall", "top_1_accuracy"):
        assert metrics[name] == 1.0


def test_one_hot_labels_are_normalized():
    logits = np.array([[5.0, 0.0], [0.0, 5.0]])
    labels = np.array([[1, 0], [0, 1]])
    assert compute_classification_metrics((logits, labels))["accuracy"] == 1.0

