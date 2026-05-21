"""Internal config migration helpers."""

from .legacy_to_canonical import migrate_legacy_to_canonical

__all__ = ["migrate_legacy_to_canonical"]
