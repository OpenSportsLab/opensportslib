"""T-DEED frame AP and greedy hard/soft NMS, independent of video FPS.

AP integrates the interpolated precision at each newly recalled event, matching
the reference's PES scorer (rather than SoccerNet's 11-point AP).
"""

from collections import defaultdict
import copy
import math

import numpy as np


def finegym_start_classes(classes):
    """E2E-Spot's 16-class FG-Start protocol includes the two interior vault timestamps."""
    return [c for c in classes if c.endswith("_start") or c in {"VT_1", "VT_2"}]


def released_snas_clip_map(labels, predictions):
    """Released checkpoint-selection protocol, including its explicit 6.25-fps clock.

    This is a validation compatibility mode, separate from full-game test mAP.
    """
    from SoccerNet.Evaluation.ActionSpotting import average_mAP

    targets = [x.numpy()[:, 1:] for x in labels]
    detections = [x.numpy()[:, 1:] for x in predictions]
    closest = []
    for target in targets:
        near = np.full_like(target, -1)
        for c in range(target.shape[1]):
            indices = np.flatnonzero(target[:, c]).tolist()
            if indices:
                padded = [-indices[0]] + indices + [2 * len(target)]
                for i, frame in enumerate(indices, 1):
                    near[
                        max(0, (padded[i - 1] + frame) // 2) : min(
                            len(target), (frame + padded[i + 1]) // 2
                        ),
                        c,
                    ] = target[frame, c]
        closest.append(near)
    values = average_mAP(
        targets, detections, closest, framerate=6.25, deltas=np.arange(1, 6)
    )
    return {
        "a_mAP": float(values[0]),
        "a_mAP_per_class": np.asarray(values[1]).tolist(),
        "protocol": "released SN-AS clip validation at 6.25fps; full-game test is separate",
    }


def suppress(events, window, *, soft=False, threshold=0.01):
    if window < 0 or (soft and window == 0):
        raise ValueError("Invalid suppression window")
    by_label = defaultdict(list)
    for event in copy.deepcopy(events):
        if not math.isfinite(event["score"]) or not 0 <= event["score"] <= 1:
            raise ValueError("Event scores must be finite probabilities")
        by_label[event["label"]].append(event)
    output = []
    for group in by_label.values():
        while group:
            index = max(range(len(group)), key=lambda i: group[i]["score"])
            best = group.pop(index)
            if best["score"] < threshold:
                break
            output.append(copy.deepcopy(best))
            if soft:
                for event in group:
                    distance = abs(event["frame"] - best["frame"])
                    if distance <= window:
                        event["score"] = event["score"] * distance**2 / window**2
            else:
                group = [e for e in group if abs(e["frame"] - best["frame"]) > window]
    return sorted(output, key=lambda e: e["frame"])


def average_precision(predictions, truth, tolerance):
    total = sum(len(v) for v in truth.values())
    if total == 0:
        return None
    recalled, precision = set(), []
    for rank, (video, frame, score) in enumerate(
        sorted(predictions, key=lambda p: p[2], reverse=True), 1
    ):
        remaining = [f for f in truth.get(video, []) if (video, f) not in recalled]
        closest = min(remaining, key=lambda f: abs(f - frame)) if remaining else None
        if closest is not None and abs(closest - frame) <= tolerance:
            recalled.add((video, closest))
            precision.append(len(recalled) / rank)
    return (
        float(np.maximum.accumulate(precision[::-1]).sum() / total)
        if precision
        else 0.0
    )


def frame_map(truth, predictions, tolerances=(1, 2, 4), classes=None):
    if len({v["video"] for v in predictions}) != len(predictions):
        raise ValueError("Duplicate prediction video IDs")
    if {v["video"] for v in truth} != {v["video"] for v in predictions}:
        raise ValueError("Truth and prediction video sets differ")
    labels = sorted(classes or {e["label"] for v in truth for e in v["events"]})
    result = {}
    for tolerance in tolerances:
        per_class = {}
        for label in labels:
            gt = {
                v["video"]: [e["frame"] for e in v["events"] if e["label"] == label]
                for v in truth
            }
            pred = [
                (v["video"], e["frame"], e["score"])
                for v in predictions
                for e in v["events"]
                if e["label"] == label
            ]
            per_class[label] = average_precision(pred, gt, tolerance)
        valid = [x for x in per_class.values() if x is not None]
        result[str(tolerance)] = {
            "mAP": float(np.mean(valid)) if valid else None,
            "AP": per_class,
        }
    valid = [r["mAP"] for r in result.values() if r["mAP"] is not None]
    return {
        "tolerances": result,
        "mean_mAP": float(np.mean(valid)) if valid else None,
        "units": "frames",
        "scale": "0..1",
        "metric": "PES interpolated event AP",
    }


def soccernet_map(videos, predictions, classes, stride, metric="tight"):
    """Use SoccerNet's official scorer; tolerance denotes a full match window."""
    from SoccerNet.Evaluation.ActionSpotting import average_mAP

    ground_truth, detections, visibility = [], [], []
    mapping = {c: i for i, c in enumerate(classes)}
    by_id = {v["video"]: v for v in predictions}
    if set(by_id) != {v.identity for v in videos}:
        raise ValueError("SoccerNet prediction video set differs")
    rates = {round(v.fps / stride, 8) for v in videos}
    if len(rates) != 1:
        raise ValueError("Official SoccerNet scoring requires a shared FPS")
    for video in videos:
        n = max(
            math.ceil(video.frames / stride),
            max((e["frame"] // stride + 1 for e in video.events), default=0),
        )
        target = np.zeros((n, len(classes)))
        detection = np.full_like(target, -1)
        for event in video.events:
            target[event["frame"] // stride, mapping[event["label"]]] = (
                -1 if event.get("visibility") == "not shown" else 1
            )
        for event in by_id[video.identity]["events"]:
            index = int(event["frame"])
            if 0 <= index < n:
                col = mapping[event["label"]]
                detection[index, col] = max(detection[index, col], event["score"])
        closest = np.full_like(target, -1)
        for col in range(len(classes)):
            indices = np.flatnonzero(target[:, col])
            if len(indices):
                boundaries = (
                    [0]
                    + [int((a + b) // 2) for a, b in zip(indices[:-1], indices[1:])]
                    + [n]
                )
                for i, frame in enumerate(indices):
                    closest[boundaries[i] : boundaries[i + 1], col] = target[frame, col]
        ground_truth.append(target)
        detections.append(detection)
        visibility.append(closest)
    deltas = {
        "tight": np.arange(1, 6),
        "loose": np.arange(5, 61, 5),
        "ball": np.array([1.0]),
    }
    if metric not in deltas:
        raise ValueError(f"Unknown SoccerNet tolerance protocol: {metric}")
    values = average_mAP(
        ground_truth,
        detections,
        visibility,
        framerate=next(iter(rates)),
        deltas=deltas[metric],
    )
    return {
        "a_mAP": float(values[0]),
        "a_mAP_per_class": np.asarray(values[1]).tolist(),
        "metric": metric,
        "scale": "0..1",
        "scorer": "SoccerNet.Evaluation.ActionSpotting.average_mAP",
    }
