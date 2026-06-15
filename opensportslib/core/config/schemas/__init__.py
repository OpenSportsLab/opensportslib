"""Internal schema helpers for config resolution."""

from .schema_canonical import is_canonical_schema
from .schema_legacy import is_legacy_schema

__all__ = ["is_canonical_schema", "is_legacy_schema"]
