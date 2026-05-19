"""Canonical schema v3 detection helpers."""

from __future__ import annotations

from typing import Any


def is_schema_v3(payload: dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return False
    version = payload.get("VERSION")
    model = payload.get("MODEL", {})
    return version in (3, "3") or model.get("schema_version") == 3
