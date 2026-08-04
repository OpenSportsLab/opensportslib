"""Canonical config loading."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any
import re

try:
    import yaml
except ImportError:  # pragma: no cover - optional fallback in lean environments
    yaml = None

from .conflicts import assert_no_legacy_aliases
from .migrate import migrate_config
from .runtime_adapter import maybe_namespace, namespace_to_plain_dict

_YAML_SUFFIXES = {".yaml", ".yml"}
_TASK_DIRS = {"classification", "localization", "vqa"}
_INTERPOLATION_RE = re.compile(r"\$\{([^}]+)\}")
_CPU_OPENCV_SPLIT_TYPES = {
    "VideoGameWithDali": "VideoGameWithOpencv",
    "VideoGameWithDaliVideo": "VideoGameWithOpencvVideo",
}


def _load_single_yaml(path: str | Path) -> Any:
    path = str(path)
    try:
        from omegaconf import OmegaConf

        return OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    except Exception:
        if yaml is None:
            raise
        with open(path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)


def _deep_merge(base: Any, override: Any) -> Any:
    if not isinstance(base, dict) or not isinstance(override, dict):
        return deepcopy(override)

    merged = deepcopy(base)
    for key, value in override.items():
        if key in merged:
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _compose_yaml_layers(path: Path) -> list[Path] | None:
    if path.suffix.lower() not in _YAML_SUFFIXES:
        return None

    task_dir = path.parent
    if task_dir.parent.name != "configs" or task_dir.name not in _TASK_DIRS:
        return None

    root_default = task_dir.parent / "default.yaml"
    task_default = task_dir / "default.yaml"
    if path.name == "default.yaml":
        return [root_default, task_default]
    return [root_default, task_default, path]


def _load_composed_yaml(path: Path) -> dict[str, Any]:
    layers = _compose_yaml_layers(path)
    if not layers:
        return _load_single_yaml(path)

    try:
        from omegaconf import OmegaConf

        merged = OmegaConf.merge(*(OmegaConf.load(str(layer)) for layer in layers))
        return OmegaConf.to_container(merged, resolve=True)
    except Exception:
        payload: dict[str, Any] = {}
        for layer in layers:
            payload = _deep_merge(payload, _load_single_yaml(layer))
        return _resolve_interpolations(payload)


def _resolve_interpolations(payload: Any) -> Any:
    def resolve(value: Any, trail: tuple[str, ...]) -> Any:
        if isinstance(value, dict):
            return {key: resolve(item, trail + (str(key),)) for key, item in value.items()}
        if isinstance(value, list):
            return [resolve(item, trail + (str(index),)) for index, item in enumerate(value)]
        if isinstance(value, str):
            match = _INTERPOLATION_RE.fullmatch(value)
            if match:
                resolved = _resolve_reference(payload, match.group(1), trail)
                return resolve(resolved, trail) if isinstance(resolved, (dict, list)) else resolved

            def replace(match_obj: re.Match[str]) -> str:
                resolved = _resolve_reference(payload, match_obj.group(1), trail)
                if resolved is None:
                    return match_obj.group(0)
                if isinstance(resolved, (dict, list)):
                    return str(resolved)
                return str(resolved)

            return _INTERPOLATION_RE.sub(replace, value)
        return value

    return resolve(payload, ())


def _resolve_reference(payload: Any, ref: str, trail: tuple[str, ...]) -> Any:
    current = payload
    for part in ref.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]

    if isinstance(current, str) and _INTERPOLATION_RE.search(current):
        if ref in trail:
            return current
        return current
    return current


def load_raw_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    suffix = config_path.suffix.lower()
    if suffix in _YAML_SUFFIXES:
        return _load_composed_yaml(config_path)
    if suffix == ".json":
        with open(config_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    raise ValueError("Unsupported config format. Use YAML or JSON.")


def _normalize_cpu_loader_backend(payload: dict[str, Any]) -> dict[str, Any]:
    system = payload.get("SYSTEM", {})
    if not isinstance(system, dict):
        return payload

    if str(system.get("device", "auto")).lower() != "cpu":
        return payload

    data = payload.get("DATA", {})
    if not isinstance(data, dict):
        return payload

    common = data.get("common", {})
    if not isinstance(common, dict):
        return payload

    runtime = common.get("runtime", {})
    if not isinstance(runtime, dict):
        runtime = {}
        common["runtime"] = runtime
    runtime["loader_backend"] = "opencv"

    splits = common.get("splits", {})
    if not isinstance(splits, dict):
        return payload

    for split_cfg in splits.values():
        if not isinstance(split_cfg, dict):
            continue
        split_type = split_cfg.get("type")
        if split_type in _CPU_OPENCV_SPLIT_TYPES:
            split_cfg["type"] = _CPU_OPENCV_SPLIT_TYPES[split_type]

    return payload


def load_config(
    config_path: str | Path,
    *,
    validate: bool = True,
    as_namespace: bool = True,
) -> Any:
    raw = load_raw_config(config_path)
    canonical = migrate_config(raw, as_namespace=False)
    canonical = _normalize_cpu_loader_backend(canonical)
    assert_no_legacy_aliases(canonical)

    if validate:
        from .validate import validate_config

        validate_config(canonical)
    return maybe_namespace(canonical, as_namespace=as_namespace)


def load_config_omega(
    path: str | Path,
    *,
    validate: bool = True,
    as_namespace: bool = True,
) -> Any:
    return load_config(
        path,
        validate=validate,
        as_namespace=as_namespace,
    )


def resolve_config(
    config: Any,
    *,
    as_namespace: bool = True,
) -> Any:
    payload = namespace_to_plain_dict(config)
    canonical = migrate_config(payload, as_namespace=False)
    canonical = _normalize_cpu_loader_backend(canonical)
    assert_no_legacy_aliases(canonical)
    return maybe_namespace(canonical, as_namespace=as_namespace)


def save_config(config_obj: Any, path: str | Path) -> None:
    payload = namespace_to_plain_dict(config_obj)
    with open(path, "w", encoding="utf-8") as handle:
        if yaml is not None:
            yaml.safe_dump(payload, handle, sort_keys=False)
            return

        from omegaconf import OmegaConf

        handle.write(OmegaConf.to_yaml(OmegaConf.create(payload), resolve=False))
