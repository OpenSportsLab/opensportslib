from __future__ import annotations

from types import SimpleNamespace
from typing import Any


def to_namespace(cfg: Any) -> Any:
    """Convert dict-like config payloads to attribute-access objects."""
    if cfg is None:
        return None
    if isinstance(cfg, SimpleNamespace):
        return cfg
    if isinstance(cfg, dict):
        return SimpleNamespace(**cfg)
    return cfg


def normalize_builder_cfg(cfg: Any, *, kind: str | None = None) -> Any:
    """Normalize builder config and validate required `type` key."""
    normalized = to_namespace(cfg)
    cfg_type = getattr(normalized, "type", None)
    if cfg_type is None:
        scope = f" for component '{kind}'" if kind else ""
        raise ValueError(
            f"Missing required 'type' in builder config{scope}. "
            "Expected canonical component params with source.name or params.type."
        )
    return normalized

