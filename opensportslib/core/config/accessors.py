"""Canonical config access helpers for runtime code."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


def _to_plain(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_plain(v) for v in obj)
    if hasattr(obj, "__dict__"):
        return {k: _to_plain(v) for k, v in vars(obj).items()}
    return obj


def _as_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if hasattr(obj, "__dict__"):
        return {k: _to_plain(v) for k, v in vars(obj).items()}
    return {}


def _ensure_child(container: Any, key: str) -> Any:
    """Ensure a nested child container exists and return it.

    Supports dict and namespace-like objects.
    """
    if isinstance(container, dict):
        child = container.get(key)
        if child is None:
            child = {}
            container[key] = child
        return child

    child = getattr(container, key, None)
    if child is None:
        child = SimpleNamespace()
        setattr(container, key, child)
    return child


def get_loader_backend(cfg: Any) -> str:
    data = _as_dict(getattr(cfg, "DATA", None))
    common = _as_dict(data.get("common"))
    runtime = _as_dict(common.get("runtime"))
    backend = runtime.get("loader_backend", "opencv")
    return str(backend).lower()


def get_system_path(cfg: Any, key: str, default: str | None = None) -> str | None:
    system = _as_dict(getattr(cfg, "SYSTEM", None))
    paths = _as_dict(system.get("paths"))
    value = paths.get(key, system.get(key, default))
    if value is None:
        return None
    return str(value)


def get_system_gpu_count(cfg: Any) -> int:
    system = _as_dict(getattr(cfg, "SYSTEM", None))
    gpu = _as_dict(system.get("gpu"))
    count = gpu.get("count", system.get("GPU", 0))
    return int(count or 0)


def get_system_seed(cfg: Any, default: int = 42) -> int:
    system = _as_dict(getattr(cfg, "SYSTEM", None))
    reproducibility = _as_dict(system.get("reproducibility"))
    seed = reproducibility.get("seed", system.get("seed", default))
    return int(seed)


def get_system_use_seed(cfg: Any, default: bool = False) -> bool:
    system = _as_dict(getattr(cfg, "SYSTEM", None))
    reproducibility = _as_dict(system.get("reproducibility"))
    use_seed = reproducibility.get("use_seed", system.get("use_seed", default))
    return bool(use_seed)


def set_system_path(cfg: Any, key: str, value: str) -> None:
    system = getattr(cfg, "SYSTEM", None)
    if system is None:
        system = SimpleNamespace()
        setattr(cfg, "SYSTEM", system)

    paths = _ensure_child(system, "paths")
    if isinstance(paths, dict):
        paths[key] = value
    else:
        setattr(paths, key, value)


def is_dali_backend(cfg: Any) -> bool:
    return get_loader_backend(cfg) == "dali"


def get_split_cfg(cfg: Any, split: str) -> SimpleNamespace:
    data = _as_dict(getattr(cfg, "DATA", None))
    common = _as_dict(data.get("common"))
    splits = _as_dict(common.get("splits"))
    split_cfg = _as_dict(splits.get(split))

    if not split_cfg and split in data and isinstance(data[split], dict):
        split_cfg = _as_dict(data[split])

    return SimpleNamespace(**split_cfg)


def get_split_dataloader_cfg(cfg: Any, split: str) -> SimpleNamespace:
    split_cfg = get_split_cfg(cfg, split)
    dataloader = getattr(split_cfg, "dataloader", None)
    if dataloader is None:
        return SimpleNamespace()
    if isinstance(dataloader, dict):
        return SimpleNamespace(**dataloader)
    return dataloader


def get_split_annotation_path(cfg: Any, split: str) -> str | None:
    split_cfg = get_split_cfg(cfg, split)
    return getattr(split_cfg, "annotation_path", None)


def get_split_result_name(cfg: Any, split: str) -> str | None:
    split_cfg = get_split_cfg(cfg, split)
    return getattr(split_cfg, "results", None)


def set_split_annotation_path(cfg: Any, split: str, path: str) -> None:
    data = getattr(cfg, "DATA", None)
    if data is None:
        data = SimpleNamespace()
        setattr(cfg, "DATA", data)

    common = _ensure_child(data, "common")
    splits = _ensure_child(common, "splits")
    split_cfg = _ensure_child(splits, split)

    if isinstance(split_cfg, dict):
        split_cfg["annotation_path"] = path
    else:
        setattr(split_cfg, "annotation_path", path)


def get_split_source_path(cfg: Any, split: str) -> str | None:
    split_cfg = get_split_cfg(cfg, split)
    return getattr(split_cfg, "source_path", None)


def get_data_classes(cfg: Any) -> list[str]:
    data = _as_dict(getattr(cfg, "DATA", None))
    common = _as_dict(data.get("common"))
    classes = common.get("classes", [])
    return list(classes) if classes is not None else []


def get_data_num_classes(cfg: Any, default: int = 0) -> int:
    classes = get_data_classes(cfg)
    if classes:
        return len(classes)

    input_cfg = get_input_cfg(cfg)
    params = _as_dict(input_cfg.get("params"))
    num_classes = params.get("num_classes")
    if num_classes is not None:
        return int(num_classes)

    model = _as_dict(getattr(cfg, "MODEL", None))
    components = _as_dict(model.get("components"))
    for component in components.values():
        comp = _as_dict(component)
        if comp.get("kind") != "head":
            continue
        head_params = _as_dict(comp.get("params"))
        if head_params.get("num_classes") is not None:
            return int(head_params["num_classes"])

    return int(default)


def set_data_classes(cfg: Any, classes: list[str]) -> None:
    data = getattr(cfg, "DATA", None)
    if data is None:
        data = SimpleNamespace()
        setattr(cfg, "DATA", data)

    common = _ensure_child(data, "common")
    if isinstance(common, dict):
        common["classes"] = list(classes)
    else:
        setattr(common, "classes", list(classes))


def get_input_cfg(cfg: Any, input_name: str | None = None) -> dict[str, Any]:
    data = _as_dict(getattr(cfg, "DATA", None))
    inputs = _as_dict(data.get("inputs"))
    if not inputs:
        return {}
    if input_name and input_name in inputs:
        return _as_dict(inputs[input_name])
    return _as_dict(next(iter(inputs.values())))


def get_data_modality(cfg: Any) -> str:
    input_cfg = get_input_cfg(cfg)
    modality = input_cfg.get("modality") or input_cfg.get("representation") or "video"
    return str(modality)


def get_video_color_mode(cfg: Any, default: str = "rgb") -> str:
    input_cfg = get_input_cfg(cfg)
    params = _as_dict(input_cfg.get("params"))
    color_mode = params.get("color_mode")
    if color_mode is not None:
        return str(color_mode).lower()
    modality = get_data_modality(cfg).lower()
    if modality in {"rgb", "flow", "bw"}:
        return modality
    return default


def get_runtime_modality(cfg: Any) -> str:
    """Return the modality token expected by legacy runtime/model code."""
    modality = get_data_modality(cfg).lower()
    if modality == "video":
        return get_video_color_mode(cfg)
    return modality


def get_data_sampling(cfg: Any) -> dict[str, Any]:
    return _as_dict(get_input_cfg(cfg).get("sampling"))


def get_data_transform(cfg: Any) -> dict[str, Any]:
    return _as_dict(get_input_cfg(cfg).get("transform"))

def get_data_augmentations(cfg: Any) -> dict[str, Any]:
    return _as_dict(get_input_cfg(cfg).get("augmentations"))


def get_data_params(cfg: Any) -> dict[str, Any]:
    return _as_dict(get_input_cfg(cfg).get("params"))


def get_component_by_kind(cfg: Any, kind: str) -> dict[str, Any] | None:
    model = _as_dict(getattr(cfg, "MODEL", None))
    components = _as_dict(model.get("components"))
    for component in components.values():
        comp = _as_dict(component)
        if comp.get("kind") == kind:
            return comp
    return None


def get_component_name_by_kind(cfg: Any, kind: str) -> str | None:
    comp = get_component_by_kind(cfg, kind)
    if not comp:
        return None
    source = _as_dict(comp.get("source"))
    return source.get("name")


def get_component_provider_by_kind(cfg: Any, kind: str) -> str | None:
    comp = get_component_by_kind(cfg, kind)
    if not comp:
        return None
    source = _as_dict(comp.get("source"))
    provider = source.get("provider")
    return str(provider) if provider is not None else None


def get_component_params_by_kind(cfg: Any, kind: str) -> dict[str, Any]:
    comp = get_component_by_kind(cfg, kind)
    if not comp:
        return {}
    params = _as_dict(comp.get("params"))
    overrides = _as_dict(comp.get("overrides"))
    out = dict(params)
    out.update(overrides)
    name = _as_dict(comp.get("source")).get("name")
    if name is not None and "type" not in out:
        out["type"] = name
    return out


def get_component_load_by_kind(cfg: Any, kind: str) -> dict[str, Any]:
    comp = get_component_by_kind(cfg, kind)
    if not comp:
        return {}
    return _as_dict(comp.get("load"))


def get_model_load(cfg: Any) -> dict[str, Any]:
    model = _as_dict(getattr(cfg, "MODEL", None))
    return _as_dict(model.get("load"))


def get_model_family(cfg: Any) -> str:
    model = _as_dict(getattr(cfg, "MODEL", None))
    metadata = _as_dict(model.get("metadata"))
    family = metadata.get("legacy_type") or metadata.get("family")
    if family:
        return str(family)

    task = str(getattr(cfg, "TASK", "")).lower()
    trainer_type = get_train_trainer_type(cfg).lower()
    if task == "localization":
        if trainer_type == "trainer_e2e":
            return "E2E"
        if trainer_type == "trainer_calf":
            return "ContextAware"
        if trainer_type == "trainer_pooling":
            return "LearnablePooling"

    return "custom"


def get_train_epochs(cfg: Any) -> int:
    train = _as_dict(getattr(cfg, "TRAIN", None))
    return int(train.get("epochs", 1))


def get_train_execution(cfg: Any) -> dict[str, Any]:
    train = _as_dict(getattr(cfg, "TRAIN", None))
    return _as_dict(train.get("execution"))


def get_train_sampling(cfg: Any) -> dict[str, Any]:
    train = _as_dict(getattr(cfg, "TRAIN", None))
    return _as_dict(train.get("sampling"))


def get_train_selection(cfg: Any) -> dict[str, Any]:
    train = _as_dict(getattr(cfg, "TRAIN", None))
    return _as_dict(train.get("selection"))


def get_train_checkpoint(cfg: Any) -> dict[str, Any]:
    train = _as_dict(getattr(cfg, "TRAIN", None))
    return _as_dict(train.get("checkpoint"))


def get_train_trainer_type(cfg: Any) -> str:
    train = _as_dict(getattr(cfg, "TRAIN", None))
    trainer = _as_dict(train.get("trainer"))
    return str(trainer.get("type", "classification"))


def get_runner_type(cfg: Any) -> str:
    model = _as_dict(getattr(cfg, "MODEL", None))
    metadata = _as_dict(model.get("metadata"))
    runner = _as_dict(metadata.get("runner"))
    runner_type = runner.get("type")
    if runner_type:
        return str(runner_type)

    trainer_type = get_train_trainer_type(cfg).lower()
    if trainer_type == "trainer_e2e":
        return "runner_e2e"
    if trainer_type == "trainer_calf":
        return "runner_CALF"
    if trainer_type == "trainer_pooling":
        return "runner_pooling"
    if trainer_type == "vqa":
        return "runner_vqa"
    return "runner_classification"


def get_vqa_prompt_cfg(cfg: Any) -> dict[str, Any]:
    execution = get_train_execution(cfg)
    prompt = execution.get("prompt", {})
    return _as_dict(prompt)


def get_vqa_generation_cfg(cfg: Any) -> dict[str, Any]:
    execution = get_train_execution(cfg)
    generation = execution.get("generation", {})
    return _as_dict(generation)


def get_vqa_backend(cfg: Any) -> str:
    model = _as_dict(getattr(cfg, "MODEL", None))
    metadata = _as_dict(model.get("metadata"))
    backend = metadata.get("backend")
    if backend:
        return str(backend).lower()

    execution = get_train_execution(cfg)
    backend = execution.get("backend")
    if backend:
        return str(backend).lower()
    return "baseline"
