"""Legacy v1 schema detection helpers."""

from __future__ import annotations

from typing import Any


def is_schema_v1(payload: dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return False
    version = payload.get("VERSION")
    if version in (1, "1"):
        return True
    return "MODEL" in payload and "components" not in payload.get("MODEL", {})
