"""Version-neutral migration entrypoints."""

from __future__ import annotations

from typing import Any

from .migrations.legacy_to_canonical import migrate_legacy_to_canonical
from .runtime_adapter import maybe_namespace, namespace_to_plain_dict
from .schemas.schema_canonical import is_canonical_schema
from .schemas.schema_legacy import is_legacy_schema


def migrate_config(config: Any, *, as_namespace: bool = False) -> Any:
    payload = namespace_to_plain_dict(config)
    if is_canonical_schema(payload):
        migrated = payload
    elif is_legacy_schema(payload):
        migrated = migrate_legacy_to_canonical(payload)
    else:
        raise ValueError("Unsupported config schema. Only legacy input and canonical schema are supported.")
    return maybe_namespace(migrated, as_namespace=as_namespace)

