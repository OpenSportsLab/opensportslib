"""Version-neutral config loading."""

from __future__ import annotations

import json
import os
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
    canonical = _sync_runtime_split_overrides(canonical)
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


def _sync_runtime_split_overrides(canonical: dict[str, Any]) -> dict[str, Any]:
    """Preserve split path overrides when resolving already-runtime-shaped configs.

    Runtime objects may carry overrides in DATA.<split>.path, while canonical v3
    stores source of truth in DATA.common.splits.<split>.annotation_path.
    If we do not sync these, adapt_config_to_runtime can regenerate stale paths.
    """
    data = canonical.get("DATA")
    if not isinstance(data, dict):
        return canonical

    common = data.get("common")
    if not isinstance(common, dict):
        return canonical

    splits = common.get("splits")
    if not isinstance(splits, dict):
        return canonical

    pre_override_valid_path = _normalized_path(
        splits.get("valid", {}).get("annotation_path")
        if isinstance(splits.get("valid"), dict)
        else None
    )

    for split_name, split_cfg in splits.items():
        runtime_split = data.get(split_name, {})
        if not isinstance(runtime_split, dict):
            continue

        runtime_path = runtime_split.get("path")
        if runtime_path:
            split_cfg["annotation_path"] = runtime_path

        runtime_video_path = runtime_split.get("video_path")
        if runtime_video_path:
            split_cfg["source_path"] = runtime_video_path

    _sync_dependent_valid_splits_from_runtime(
        data,
        splits,
        pre_override_valid_path=pre_override_valid_path,
    )
    return canonical


def _sync_dependent_valid_splits_from_runtime(
    data: dict[str, Any],
    splits: dict[str, Any],
    *,
    pre_override_valid_path: str | None,
) -> None:
    """Propagate valid split runtime override to dependent valid_data_frames split.

    This keeps localization map validation aligned when runtime code overrides
    DATA.valid.path but does not explicitly override DATA.valid_data_frames.path.
    """
    runtime_valid = data.get("valid", {})
    if not isinstance(runtime_valid, dict):
        return

    valid_override = runtime_valid.get("path")
    if not valid_override:
        return

    dependent_split_name = "valid_data_frames"
    dependent_canonical = splits.get(dependent_split_name)
    if not isinstance(dependent_canonical, dict):
        return

    runtime_dependent = data.get(dependent_split_name, {})
    runtime_dependent_path = (
        runtime_dependent.get("path") if isinstance(runtime_dependent, dict) else None
    )
    dependent_canonical_path = dependent_canonical.get("annotation_path")
    if runtime_dependent_path and _normalized_path(runtime_dependent_path) != pre_override_valid_path:
        # Explicit custom runtime override for valid_data_frames: respect it.
        return

    if (
        dependent_canonical_path
        and _normalized_path(dependent_canonical_path) != pre_override_valid_path
        and not runtime_dependent_path
    ):
        # Canonical dependent split already custom and runtime did not request override.
        return

    dependent_canonical["annotation_path"] = valid_override


def _normalized_path(path: Any) -> str | None:
    if not path or not isinstance(path, str):
        return None
    return os.path.normpath(path)
