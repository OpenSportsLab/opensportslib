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
    # if model.get("schema_version") != 3:
    #     raise ValueError("MODEL.schema_version must be 3")
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

    for input_cfg in cfg.get("DATA", {}).get("inputs", {}).values():
        if input_cfg.get("representation") == "skeleton_graph":
            from opensportslib.preprocessing.config import validate_preprocessing
            settings = dict(input_cfg.get("params", {}).get("preprocessing", {}))
            settings["sample_fps"] = input_cfg.get("sampling", {}).get("extract_fps", 25)
            validate_preprocessing(settings)
            if input_cfg.get("modality") not in {"video", "skeleton_cache"}:
                raise ValueError("skeleton_graph source modality must be video or skeleton_cache")
            if input_cfg.get("params", {}).get("objects", {}).get("feature_dim", 73) != 73:
                raise ValueError("COCO-17 player features require feature_dim=73")
            encoder = next((c for c in components.values() if c.get("kind") == "encoder"), {})
            if encoder.get("source", {}).get("name") not in {"graph_conv", "graph_conv_seq"}:
                raise ValueError("skeleton_graph requires a graph encoder")
            sampling = input_cfg.get("sampling", {})
            length = sampling.get("clip_len", sampling.get("num_frames", 64))
            if not isinstance(length, int) or length < 1:
                raise ValueError("Skeleton window length must be a positive integer")
            for split in cfg.get("DATA", {}).get("common", {}).get("splits", {}).values():
                if (split.get("nms_window") or 0) < 0:
                    raise ValueError("Skeleton nms_window must be nonnegative")
                if not 0 <= split.get("overlap_len", length//2) < length:
                    raise ValueError("Skeleton overlap_len must be less than clip_len")
            if cfg.get("MODEL", {}).get("policies", {}).get("test_time_adaptation", {}).get("enabled"):
                raise ValueError("Test-time adaptation is not supported for skeleton preprocessing")

    task = str(cfg.get("TASK", "")).lower()
    if task == "vqa":
        data_inputs = cfg.get("DATA", {}).get("inputs", {})
        if "video" not in data_inputs:
            raise ValueError("VQA config must define DATA.inputs.video.")
        if "question" not in data_inputs:
            raise ValueError("VQA config must define DATA.inputs.question.")

        has_decoder = any(
            (component.get("kind") in {"decoder", "head"})
            for component in components.values()
        )
        if not has_decoder:
            raise ValueError("VQA config must define at least one decoder/head component.")


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
