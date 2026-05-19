"""Version-neutral migration entrypoints."""

from __future__ import annotations

from typing import Any

from .migrations.v1_to_v3 import migrate_v1_to_v3
from .runtime_adapter import maybe_namespace, namespace_to_plain_dict
from .schemas.schema_v1 import is_schema_v1
from .schemas.schema_v3 import is_schema_v3


def migrate_config(config: Any, *, as_namespace: bool = False) -> Any:
    payload = namespace_to_plain_dict(config)
    if is_schema_v3(payload):
        migrated = payload
    elif is_schema_v1(payload):
        migrated = migrate_v1_to_v3(payload)
    else:
        raise ValueError("Unsupported config version. Only v1 input and canonical VERSION: 3 are supported.")
    return maybe_namespace(migrated, as_namespace=as_namespace)
