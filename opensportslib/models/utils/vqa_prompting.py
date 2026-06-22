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


def build_xvars_prompt(
    *,
    system_prompt: str,
    question: str,
    prior_text: str = "",
    video_token_len: int = 300,
) -> str:
    """Build a shared X-VARS/Video-ChatGPT-style prompt contract."""
    video_token_len = max(int(video_token_len), 0)
    question_text = str(question).strip()
    prior_text = str(prior_text).strip()

    user_turn = f"USER: {question_text}"
    if prior_text:
        user_turn = f"{user_turn} The prediction for this video is {prior_text}."
    if video_token_len > 0:
        user_turn = f"{user_turn}\n<vid_start>{'<vid_patch>' * video_token_len}<vid_end>"

    return "\n".join(
        [
            str(system_prompt).strip(),
            user_turn,
            "ASSISTANT:",
        ]
    )
