"""Legacy schema detection helpers."""

from __future__ import annotations

from typing import Any


def is_legacy_schema(payload: dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return False

    model = payload.get("MODEL", {})
    if not isinstance(model, dict):
        return False

    # Legacy configs do not expose canonical component graph.
    return "MODEL" in payload and "components" not in model
