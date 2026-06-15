"""Legacy-to-canonical duplicate key conflict detection."""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any


def detect_legacy_conflicts(legacy_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Detect conflicting duplicate semantics in legacy payload.

    Returns structured findings consumed by migration and validators.
    """
    findings: list[dict[str, Any]] = []
    data = legacy_cfg.get("DATA", {}) if isinstance(legacy_cfg, dict) else {}
    model = legacy_cfg.get("MODEL", {}) if isinstance(legacy_cfg, dict) else {}
    train = legacy_cfg.get("TRAIN", {}) if isinstance(legacy_cfg, dict) else {}

    _check_pair(findings, "TRAIN.epochs", train, "epochs", "num_epochs")
    _check_pair(findings, "TRAIN.epochs", train, "epochs", "max_epochs")
    _check_pair(findings, "TRAIN.epochs", train, "num_epochs", "max_epochs")

    for split in ("train", "valid", "test", "valid_data_frames", "challenge", "infer"):
        split_cfg = data.get(split, {}) if isinstance(data, dict) else {}
        annotations = data.get("annotations", {}) if isinstance(data.get("annotations", {}), dict) else {}

        if isinstance(split_cfg, dict) and "path" in split_cfg and split in annotations:
            if split_cfg.get("path") != annotations.get(split):
                findings.append(
                    {
                        "key": f"DATA.common.splits.{split}.annotation_path",
                        "canonical": split_cfg.get("path"),
                        "legacy": annotations.get(split),
                        "effective": split_cfg.get("path"),
                        "deprecated_key": f"DATA.annotations.{split}",
                    }
                )

    if "multi_gpu" in model and isinstance(train.get("execution"), dict) and "multi_gpu" in train["execution"]:
        if model.get("multi_gpu") != train["execution"].get("multi_gpu"):
            findings.append(
                {
                    "key": "TRAIN.execution.multi_gpu",
                    "canonical": train["execution"].get("multi_gpu"),
                    "legacy": model.get("multi_gpu"),
                    "effective": train["execution"].get("multi_gpu"),
                    "deprecated_key": "MODEL.multi_gpu",
                }
            )

    return findings


def emit_conflict_warnings(findings: list[dict[str, Any]]) -> None:
    for finding in findings:
        logging.warning(
            "Config conflict resolved: key=%s canonical=%r legacy=%r effective=%r deprecated=%s",
            finding["key"],
            finding["canonical"],
            finding["legacy"],
            finding["effective"],
            finding["deprecated_key"],
        )


def assert_no_legacy_aliases(payload: dict[str, Any]) -> None:
    """Strict check: canonical payload must not contain legacy aliases."""
    alias_paths = [
        ("dali",),
        ("DATA", "annotations"),
        ("MODEL", "backbone"),
        ("MODEL", "neck"),
        ("MODEL", "head"),
        ("MODEL", "post_proc"),
        ("MODEL", "multi_gpu"),
        ("TRAIN", "num_epochs"),
        ("TRAIN", "max_epochs"),
    ]
    for path in alias_paths:
        if _has_path(payload, path):
            joined = ".".join(path)
            raise ValueError(f"Legacy alias {joined} is not allowed in strict canonical runtime.")

    splits = (
        payload.get("DATA", {})
        .get("common", {})
        .get("splits", {})
        if isinstance(payload.get("DATA", {}), dict)
        else {}
    )
    if isinstance(splits, dict):
        for split_name, split_cfg in splits.items():
            if not isinstance(split_cfg, dict):
                continue
            for key in ("path", "video_path"):
                if key in split_cfg:
                    raise ValueError(
                        f"Legacy alias DATA.common.splits.{split_name}.{key} is not allowed in strict canonical runtime."
                    )


def _check_pair(
    findings: list[dict[str, Any]],
    canonical_key: str,
    section: dict[str, Any],
    preferred: str,
    alias: str,
) -> None:
    if not isinstance(section, dict):
        return
    if preferred in section and alias in section and section[preferred] != section[alias]:
        findings.append(
            {
                "key": canonical_key,
                "canonical": deepcopy(section[preferred]),
                "legacy": deepcopy(section[alias]),
                "effective": deepcopy(section[preferred]),
                "deprecated_key": alias,
            }
        )


def _has_path(payload: dict[str, Any], path: tuple[str, ...]) -> bool:
    current: Any = payload
    for part in path:
        if not isinstance(current, dict) or part not in current:
            return False
        current = current[part]
    return True
