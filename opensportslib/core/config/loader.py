"""Version-neutral config loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .migrate import migrate_config
from .runtime_adapter import adapt_config_to_runtime, maybe_namespace, namespace_to_plain_dict


def load_raw_config(path: str | Path) -> dict[str, Any]:
    path = str(path)
    suffix = Path(path).suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            from omegaconf import OmegaConf

            return OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        except Exception:
            with open(path, "r", encoding="utf-8") as handle:
                return yaml.safe_load(handle)
    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    raise ValueError("Unsupported config format. Use YAML or JSON.")


def load_config(
    config_path: str | Path,
    *,
    validate: bool = True,
    compatibility: bool = True,
    as_namespace: bool = True,
) -> Any:
    raw = load_raw_config(config_path)
    canonical = migrate_config(raw, as_namespace=False)
    if validate:
        from .validate import validate_config

        validate_config(canonical)
    if compatibility:
        runtime = adapt_config_to_runtime(canonical, as_namespace=False)
        runtime = _ensure_legacy_runtime_flags(runtime)
        return maybe_namespace(runtime, as_namespace=as_namespace)
    return maybe_namespace(canonical, as_namespace=as_namespace)


def load_config_omega(
    path: str | Path,
    *,
    validate: bool = True,
    compatibility: bool = True,
    as_namespace: bool = True,
) -> Any:
    return load_config(
        path,
        validate=validate,
        compatibility=compatibility,
        as_namespace=as_namespace,
    )


def resolve_config(
    config: Any,
    *,
    compatibility: bool = True,
    as_namespace: bool = True,
) -> Any:
    payload = namespace_to_plain_dict(config)
    canonical = migrate_config(payload, as_namespace=False)
    if compatibility:
        runtime = adapt_config_to_runtime(canonical, as_namespace=False)
        runtime = _ensure_legacy_runtime_flags(runtime)
        return maybe_namespace(runtime, as_namespace=as_namespace)
    return maybe_namespace(canonical, as_namespace=as_namespace)


def save_config(config_obj: Any, path: str | Path) -> None:
    payload = namespace_to_plain_dict(config_obj)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _ensure_legacy_runtime_flags(runtime: dict[str, Any]) -> dict[str, Any]:
    if "dali" not in runtime:
        runtime["dali"] = False
    return runtime
