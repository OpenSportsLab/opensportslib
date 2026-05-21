"""Canonical config validation."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

from .loader import load_raw_config
from .migrate import migrate_config
from .conflicts import assert_no_legacy_aliases
from .schemas.schema_canonical import is_canonical_schema


def validate_config(config_or_path: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config_or_path, str):
        raw = load_raw_config(config_or_path)
    else:
        raw = config_or_path

    canonical = migrate_config(raw, as_namespace=False)
    _validate_canonical(canonical)
    return canonical


def _validate_canonical(cfg: dict[str, Any]) -> None:
    required_sections = ["TASK", "VERSION", "SYSTEM", "DATA", "MODEL", "TRAIN"]
    missing = [section for section in required_sections if section not in cfg]
    if missing:
        raise ValueError(f"Config missing required sections: {missing}")

    if not is_canonical_schema(cfg):
        raise ValueError("Config must resolve to canonical schema.")
    assert_no_legacy_aliases(cfg)

    model = cfg["MODEL"]
    if model.get("schema_version") != 3:
        raise ValueError("MODEL.schema_version must be 3")
    # if str(model.get("task", "")).lower() != str(cfg["TASK"]).lower():
    #     raise ValueError("MODEL.task must match TASK")

    components = model.get("components", {})
    if not isinstance(components, dict) or not components:
        raise ValueError("MODEL.components must be a non-empty mapping")

    for component_id, component_cfg in components.items():
        if not component_id.replace("_", "").isalnum() or component_id.startswith("_"):
            raise ValueError(f"Invalid component id: {component_id}")
        source = component_cfg.get("source", {})
        if not source.get("provider"):
            raise ValueError(f"Component {component_id} is missing source.provider")

    topology = model.get("topology", [])
    _validate_topology(components, topology)

    io_cfg = cfg.get("IO", {})
    for section in ("inputs", "outputs"):
        for public_name, component_id in io_cfg.get(section, {}).items():
            if component_id not in components:
                raise ValueError(
                    f"IO.{section}.{public_name} references unknown component {component_id}"
                )


def _validate_topology(components: dict[str, Any], topology: list[dict[str, Any]]) -> None:
    if not topology:
        return

    graph = defaultdict(list)
    indegree = {node: 0 for node in components}
    for edge in topology:
        src = edge.get("from")
        dst = edge.get("to")
        if src not in components or dst not in components:
            raise ValueError(f"Topology edge references unknown components: {edge}")
        graph[src].append(dst)
        indegree[dst] += 1

    queue = deque(node for node, degree in indegree.items() if degree == 0)
    visited = 0
    while queue:
        node = queue.popleft()
        visited += 1
        for child in graph[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)

    if visited != len(components):
        raise ValueError("MODEL.topology must be acyclic")
