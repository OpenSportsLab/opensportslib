"""Reusable VQA prompt and prior-text helpers."""

from __future__ import annotations

from typing import Any


def build_prior_text(
    labels: dict[str, Any] | None,
    metadata: dict[str, Any] | None = None,
    include_fields: list[str] | None = None,
) -> str:
    """Build compact prior text from structured labels/metadata."""
    labels = labels or {}
    metadata = metadata or {}
    include_fields = include_fields or ["action", "offence", "contact", "bodypart"]

    chunks: list[str] = []
    for field in include_fields:
        value = ((labels.get(field) or {}).get("label")) if isinstance(labels.get(field), dict) else None
        if value:
            chunks.append(f"{field}={value}")

    league = metadata.get("league")
    if league:
        chunks.append(f"league={league}")

    return "; ".join(chunks)
