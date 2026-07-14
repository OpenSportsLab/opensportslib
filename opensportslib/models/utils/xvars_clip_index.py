"""Shared VQA feature and prediction index helpers."""

from __future__ import annotations

import json
import os
from typing import Any

import torch


def _load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _as_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return [row for row in payload["data"] if isinstance(row, dict)]
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        rows = []
        for k, v in payload.items():
            if isinstance(v, dict):
                row = dict(v)
                row.setdefault("id", k)
                rows.append(row)
        return rows
    return []


def _normalize_candidates(row: dict[str, Any], base_dir: str) -> list[str]:
    candidates: list[str] = []
    vals = row.get("feature_paths")
    if isinstance(vals, list):
        for p in vals:
            if not p:
                continue
            p = str(p)
            candidates.append(p if os.path.isabs(p) else os.path.abspath(os.path.join(base_dir, p)))
    dir_hint = row.get("feature_dir") or row.get("path")
    if dir_hint:
        d = str(dir_hint)
        d = d if os.path.isabs(d) else os.path.abspath(os.path.join(base_dir, d))
        for i in (1, 2, 3):
            candidates.append(os.path.join(d, f"PRE_CLIP_feature_clip_{i}.pkl"))
    return candidates


def load_feature_index(index_path: str, *, split: str | None = None) -> dict[str, list[str]]:
    payload = _load_json(index_path)
    rows = _as_rows(payload)
    root = os.path.dirname(os.path.abspath(index_path))
    out: dict[str, list[str]] = {}
    for row in rows:
        row_split = str(row.get("split") or "").strip().lower()
        if split and row_split and row_split != str(split).lower():
            continue
        rid = str(row.get("id") or row.get("sample_id") or row.get("clip_id") or "").strip()
        if not rid:
            continue
        candidates = _normalize_candidates(row, root)
        if candidates:
            out[rid] = candidates
    return out


def load_prediction_index(index_path: str, *, split: str | None = None) -> dict[str, dict[str, Any]]:
    payload = _load_json(index_path)
    rows = _as_rows(payload)
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_split = str(row.get("split") or "").strip().lower()
        if split and row_split and row_split != str(split).lower():
            continue
        rid = str(row.get("id") or row.get("sample_id") or row.get("clip_id") or "").strip()
        if not rid:
            continue
        out[rid] = row
    return out


def validate_xvars_feature_tensor(
    features: torch.Tensor,
    *,
    expected_tokens: int | None = None,
    context: str = "X-VARS features",
) -> torch.Tensor:
    if not isinstance(features, torch.Tensor):
        features = torch.as_tensor(features, dtype=torch.float32)
    if features.ndim != 2:
        raise ValueError(f"{context} must be a 2D tensor [tokens, dim], got shape {tuple(features.shape)}")
    if expected_tokens is not None and int(features.shape[0]) != int(expected_tokens):
        raise ValueError(
            f"{context} token count mismatch: expected {int(expected_tokens)}, got {int(features.shape[0])}. "
            "Check that the configured X-VARS feature mode matches the extracted feature files."
        )
    return features
