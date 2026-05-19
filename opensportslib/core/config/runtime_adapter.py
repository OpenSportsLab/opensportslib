"""Compatibility adapter from canonical config to the current runtime shape."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any

from .schemas.schema_v3 import is_schema_v3


def dict_to_namespace(d: Any, skip_keys: tuple[str, ...] = ("classes",)) -> Any:
    if isinstance(d, dict):
        out = {}
        for key, value in d.items():
            out[key] = value if key in skip_keys else dict_to_namespace(value, skip_keys)
        return SimpleNamespace(**out)
    if isinstance(d, list):
        return [dict_to_namespace(value, skip_keys) for value in d]
    return d


def namespace_to_plain_dict(ns: Any) -> Any:
    if ns is None or isinstance(ns, (str, int, float, bool)):
        return ns

    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf

        if isinstance(ns, (DictConfig, ListConfig)):
            ns = OmegaConf.to_container(ns, resolve=True)
    except ImportError:
        pass

    if isinstance(ns, dict):
        return {str(key): namespace_to_plain_dict(value) for key, value in ns.items()}
    if isinstance(ns, (list, tuple, set)):
        return [namespace_to_plain_dict(value) for value in ns]
    if hasattr(ns, "__dict__"):
        return {str(key): namespace_to_plain_dict(value) for key, value in vars(ns).items()}
    return ns


def maybe_namespace(payload: Any, *, as_namespace: bool) -> Any:
    return dict_to_namespace(payload) if as_namespace else payload


def adapt_config_to_runtime(config: Any, *, as_namespace: bool = True) -> Any:
    payload = namespace_to_plain_dict(config)
    if not is_schema_v3(payload):
        return maybe_namespace(payload, as_namespace=as_namespace)

    runtime = deepcopy(payload)
    system = runtime.get("SYSTEM", {})
    data = runtime.get("DATA", {})
    model = runtime.get("MODEL", {})
    train = runtime.get("TRAIN", {})
    io_cfg = runtime.get("IO", {})

    runtime["SYSTEM"] = _adapt_system(system)
    runtime["DATA"] = _adapt_data(data, payload.get("TASK", ""), io_cfg)
    runtime["MODEL"] = _adapt_model(model, payload.get("TASK", ""))
    runtime["TRAIN"] = _adapt_train(train)
    runtime["_canonical"] = payload

    return maybe_namespace(runtime, as_namespace=as_namespace)


def _adapt_system(system: dict[str, Any]) -> dict[str, Any]:
    paths = deepcopy(system.get("paths", {}))
    gpu = deepcopy(system.get("gpu", {}))
    reproducibility = deepcopy(system.get("reproducibility", {}))
    return {
        **deepcopy(system),
        "save_dir": paths.get("save_dir"),
        "log_dir": paths.get("log_dir"),
        "work_dir": paths.get("work_dir", paths.get("save_dir")),
        "GPU": gpu.get("count", 0),
        "gpu_id": gpu.get("id", 0),
        "use_seed": reproducibility.get("use_seed", False),
        "seed": reproducibility.get("seed", 42),
    }


def _adapt_data(data: dict[str, Any], task: str, io_cfg: dict[str, Any]) -> dict[str, Any]:
    common = deepcopy(data.get("common", {}))
    inputs = deepcopy(data.get("inputs", {}))
    runtime = {
        **deepcopy(data),
        "data_dir": common.get("data_root"),
        "classes": common.get("classes", data.get("classes", [])),
    }

    splits = common.get("splits", {})
    runtime["annotations"] = {}
    for split_name, split_cfg in splits.items():
        legacy_split = deepcopy(split_cfg)
        if "annotation_path" in split_cfg:
            legacy_split["path"] = split_cfg["annotation_path"]
            runtime["annotations"][split_name] = split_cfg["annotation_path"]
        if "source_path" in split_cfg:
            legacy_split["video_path"] = split_cfg["source_path"]
        runtime[split_name] = legacy_split

    if not inputs:
        return runtime

    primary_name = _select_primary_input(inputs, io_cfg)
    primary_input = inputs[primary_name]
    runtime["data_modality"] = _infer_legacy_data_modality(primary_input)
    runtime["modality"] = primary_input.get("params", {}).get("color_mode", primary_input.get("modality"))

    runtime.update(deepcopy(primary_input.get("sampling", {})))

    transform = deepcopy(primary_input.get("transform", {}))
    resize = transform.get("resize", {})
    normalization = transform.get("normalization", {})
    if "height" in resize:
        runtime["target_height"] = resize["height"]
    if "width" in resize:
        runtime["target_width"] = resize["width"]
    if "height" in resize and "width" in resize:
        runtime["frame_size"] = [resize["height"], resize["width"]]
    if normalization:
        if "mean" in normalization:
            runtime["imagenet_mean"] = normalization["mean"]
        if "std" in normalization:
            runtime["imagenet_std"] = normalization["std"]

    runtime["augmentations"] = deepcopy(primary_input.get("augmentations", {}))
    runtime.update(deepcopy(primary_input.get("augmentations", {})))
    runtime.update(deepcopy(primary_input.get("params", {})))

    if task == "classification" and "num_classes" not in runtime and runtime.get("classes"):
        runtime["num_classes"] = len(runtime["classes"])

    return runtime


def _adapt_model(model: dict[str, Any], task: str) -> dict[str, Any]:
    components = deepcopy(model.get("components", {}))
    metadata = deepcopy(model.get("metadata", {}))
    runtime = deepcopy(model)

    backbone_id = _first_component(components, "encoder")
    neck_id = _first_component(components, "adapter")
    head_id = _first_component(components, "head")
    post_id = _first_component(components, "postprocessor")

    if backbone_id:
        runtime["backbone"] = _legacy_component_config(components[backbone_id])
    if neck_id:
        runtime["neck"] = _legacy_component_config(components[neck_id])
    if head_id:
        runtime["head"] = _legacy_component_config(components[head_id])
    if post_id:
        runtime["post_proc"] = _legacy_component_config(components[post_id])

    runtime["load_weights"] = model.get("load", {}).get("checkpoint_path")
    if "backbone" in runtime:
        runtime["pretrained_model"] = runtime["backbone"].get(
            "pretrained_model",
            runtime["backbone"].get("type"),
        )
    runtime["multi_gpu"] = bool(
        model.get("runtime", {}).get("multi_gpu", False)
        or model.get("runtime", {}).get("device") == "ddp"
    )
    runtime["runner"] = metadata.get("runner", {}) or {"type": _infer_runner_type(runtime, task)}
    runtime["type"] = metadata.get("legacy_type") or _infer_model_type(runtime, task)
    return runtime


def _adapt_train(train: dict[str, Any]) -> dict[str, Any]:
    execution = deepcopy(train.get("execution", {}))
    checkpoint = deepcopy(train.get("checkpoint", {}))
    selection = deepcopy(train.get("selection", {}))
    sampling = deepcopy(train.get("sampling", {}))
    runtime = deepcopy(train)

    runtime["type"] = train.get("trainer", {}).get("type", "classification")
    runtime["epochs"] = train.get("epochs")
    runtime["num_epochs"] = train.get("epochs")
    runtime["max_epochs"] = train.get("epochs")
    runtime["criterion_valid"] = execution.get("criterion_valid", selection.get("monitor", "loss"))
    runtime["evaluation_frequency"] = execution.get("evaluation_frequency")
    runtime["acc_grad_iter"] = execution.get("acc_grad_iter")
    runtime["base_num_valid_epochs"] = execution.get("base_num_valid_epochs")
    runtime["start_valid_epoch"] = execution.get("start_valid_epoch")
    runtime["valid_map_every"] = execution.get("valid_map_every")
    runtime["use_weighted_sampler"] = sampling.get("use_weighted_sampler", False)
    runtime["use_weighted_loss"] = sampling.get("use_weighted_loss", False)
    runtime["save_every"] = checkpoint.get("save_every")
    runtime["save_best"] = checkpoint.get("save_best", False)
    return runtime


def _legacy_component_config(component: dict[str, Any]) -> dict[str, Any]:
    source = deepcopy(component.get("source", {}))
    params = deepcopy(component.get("params", {}))
    overrides = deepcopy(component.get("overrides", {}))

    if source.get("name") is not None and "type" not in params:
        params["type"] = source["name"]
    params.update(overrides)
    return params


def _select_primary_input(inputs: dict[str, Any], io_cfg: dict[str, Any]) -> str:
    io_inputs = io_cfg.get("inputs", {})
    for public_name in ("video", "features", "tracking", "image", "text"):
        if public_name in io_inputs and public_name in inputs:
            return public_name
    return next(iter(inputs))


def _infer_legacy_data_modality(primary_input: dict[str, Any]) -> str:
    modality = primary_input.get("modality")
    representation = primary_input.get("representation")
    source_format = primary_input.get("source", {}).get("format")

    if representation == "frames_npy":
        return "frames_npy"
    if modality == "tracking":
        return "tracking_parquet" if source_format == "parquet" else "tracking"
    if representation == "features":
        return "features"
    return modality or representation


def _first_component(components: dict[str, Any], kind: str) -> str | None:
    for component_id, component_cfg in components.items():
        if component_cfg.get("kind") == kind:
            return component_id
    return None


def _infer_model_type(model: dict[str, Any], task: str) -> str:
    if task == "classification":
        return model.get("type", "custom")

    neck_type = model.get("neck", {}).get("type")
    head_type = model.get("head", {}).get("type")
    if neck_type == "CNN++" or head_type == "SpottingCALF":
        return "ContextAware"
    if neck_type in {"NetVLAD++", "NetVLAD", "NetRVLAD", "NetRVLAD++"}:
        return "LearnablePooling"
    return "E2E"


def _infer_runner_type(model: dict[str, Any], task: str) -> str:
    if task == "classification":
        return "runner_classification"
    model_type = _infer_model_type(model, task)
    if model_type == "E2E":
        return "runner_e2e"
    if model_type == "ContextAware":
        return "runner_CALF"
    return "runner_pooling"
