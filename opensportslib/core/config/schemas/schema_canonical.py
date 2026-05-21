"""Canonical schema detection helpers."""

from __future__ import annotations

from typing import Any


def is_canonical_schema(payload: dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return False

    model = payload.get("MODEL", {})
    if not isinstance(model, dict):
        return False

    # VERSION / schema_version remain compatibility markers.
    return (
        "components" in model
        and "topology" in model
        #and payload.get("VERSION") in (2, "2")
        #and model.get("schema_version") == 3
    )
