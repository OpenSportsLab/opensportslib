"""X-VARS CLIP feature/prediction index helpers."""

from __future__ import annotations

import json
import os
from typing import Any


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


def load_feature_index(index_path: str) -> dict[str, list[str]]:
    payload = _load_json(index_path)
    rows = _as_rows(payload)
    root = os.path.dirname(os.path.abspath(index_path))
    out: dict[str, list[str]] = {}
    for row in rows:
        rid = str(row.get("id") or row.get("sample_id") or row.get("clip_id") or "").strip()
        if not rid:
            continue
        candidates = _normalize_candidates(row, root)
        if candidates:
            out[rid] = candidates
    return out


def load_prediction_index(index_path: str) -> dict[str, dict[str, Any]]:
    payload = _load_json(index_path)
    rows = _as_rows(payload)
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        rid = str(row.get("id") or row.get("sample_id") or row.get("clip_id") or "").strip()
        if not rid:
            continue
        out[rid] = row
    return out


def build_xvars_prior_from_prediction(pred: dict[str, Any] | None) -> str:
    pred = pred or {}
    action = str(pred.get("Action class") or pred.get("action") or "").strip()
    offence = str(pred.get("Offence") or pred.get("offence") or "").strip()
    severity = str(pred.get("Severity") or pred.get("severity") or "").strip()

    action_map = {
        "tackling": "a tackle",
        "standing tackling": "a foot duel",
        "elbowing": "using his elbows or arms",
        "holding": "holding",
        "high leg": "a high leg",
        "pushing": "pushing",
        "challenge": "a shoulder challenge",
        "dive": "a simulation",
    }
    action = action_map.get(action.lower(), action)

    if offence.lower() == "offence":
        offence = "foul"
    elif offence.lower() == "no offence":
        offence = "no foul"
    if severity == "3.0":
        severity = "yellow card"
    elif severity == "5.0":
        severity = "red card"
    elif severity == "1.0":
        severity = "no card"
    parts = [p for p in (action, offence, severity) if p]
    return ", ".join(parts)
