"""Canonical config access helpers for runtime code."""

from __future__ import annotations

import os
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


def get_model_runtime(cfg: Any) -> dict[str, Any]:
    model = _as_dict(getattr(cfg, "MODEL", None))
    return _as_dict(model.get("runtime"))


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


def get_hf_cuda_device_index(cfg: Any, hf_cfg: dict[str, Any] | None = None) -> int | None:
    """Resolve the CUDA device index used by HuggingFace runtime helpers."""
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return None

    hf_cfg = _as_dict(hf_cfg) if hf_cfg is not None else _as_dict(get_train_execution(cfg).get("hf"))
    explicit = hf_cfg.get("cuda_device_index")
    if explicit is not None:
        try:
            return int(explicit)
        except Exception:
            return None

    system = _as_dict(getattr(cfg, "SYSTEM", None))
    gpu = _as_dict(system.get("gpu"))
    gid = gpu.get("id")
    if gid is not None:
        try:
            return int(gid)
        except Exception:
            return None

    return None


def get_train_optimizer(cfg: Any) -> dict[str, Any]:
    train = _as_dict(getattr(cfg, "TRAIN", None))
    return _as_dict(train.get("optimizer"))


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
    generation = _as_dict(execution.get("generation", {}))
    production = _as_dict(execution.get("production", {}))
    out = dict(generation)
    if production:
        out.setdefault("max_new_tokens_cap", production.get("max_new_tokens_cap"))
        out.setdefault("retry_count", production.get("retry_count"))
        out.setdefault("retry_backoff_s", production.get("retry_backoff_s"))
        out.setdefault("timeout_s", production.get("timeout_s"))
        out.setdefault("fallback_policy", production.get("fallback_policy"))
    return out


def get_vqa_eval_profile_cfg(cfg: Any) -> dict[str, Any]:
    execution = get_train_execution(cfg)
    profile = execution.get("eval_profile", {})
    return _as_dict(profile)


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


def is_xvars_videochatgpt_backend(cfg: Any) -> bool:
    return get_vqa_backend(cfg) == "xvars_videochatgpt"


def is_qwen_vl_native_backend(cfg: Any) -> bool:
    return get_vqa_backend(cfg) == "qwen_vl_native_infer"


def normalize_xvars_feature_mode(mode: Any, default: str = "strict_xvars") -> str:
    value = str(mode or "").strip().lower()
    if not value:
        return str(default)
    aliases = {
        "strict": "strict_xvars",
        "strict_xvars": "strict_xvars",
        "original": "strict_xvars",
        "original_xvars": "strict_xvars",
        "clip": "clip_compat",
        "clip_compat": "clip_compat",
        "compat": "clip_compat",
        "compatibility": "clip_compat",
    }
    if value in aliases:
        return aliases[value]
    raise ValueError(f"Unsupported X-VARS feature mode '{mode}'. Expected 'strict_xvars' or 'clip_compat'.")


def get_xvars_feature_token_len_for_mode(mode: Any) -> int:
    normalized = normalize_xvars_feature_mode(mode)
    if normalized == "strict_xvars":
        return 300
    if normalized == "clip_compat":
        return 356
    raise ValueError(f"Unsupported X-VARS feature mode '{mode}'.")


def get_vqa_xvars_feature_mode(cfg: Any, default: str = "strict_xvars") -> str:
    execution = get_train_execution(cfg)
    xvars_cfg = _as_dict(execution.get("xvars"))
    mode = xvars_cfg.get("feature_mode")
    if mode:
        return normalize_xvars_feature_mode(mode, default=default)

    token_len = None
    try:
        token_len = get_vqa_prompt_video_token_len(cfg, default=get_xvars_feature_token_len_for_mode(default))
    except Exception:
        token_len = None
    if int(token_len or 0) == 356:
        return "clip_compat"
    return normalize_xvars_feature_mode(default, default=default)


def has_explicit_xvars_feature_mode(cfg: Any) -> bool:
    execution = get_train_execution(cfg)
    xvars_cfg = _as_dict(execution.get("xvars"))
    return xvars_cfg.get("feature_mode") is not None


def get_vqa_prompt_video_token_len(cfg: Any, default: int = 300) -> int:
    prompt_cfg = get_vqa_prompt_cfg(cfg)
    token_len = prompt_cfg.get("video_token_len")
    if token_len is not None:
        return int(token_len)

    execution = get_train_execution(cfg)
    sft_cfg = _as_dict(execution.get("sft"))
    if sft_cfg.get("video_token_len") is not None:
        return int(sft_cfg["video_token_len"])

    xvars_cfg = _as_dict(execution.get("xvars"))
    if xvars_cfg.get("video_token_len") is not None:
        return int(xvars_cfg["video_token_len"])
    if xvars_cfg.get("feature_mode") is not None:
        return get_xvars_feature_token_len_for_mode(xvars_cfg.get("feature_mode"))

    return int(default)


def get_xvars_train_video_token_len(cfg: Any) -> int:
    return get_xvars_feature_token_len_for_mode(get_vqa_xvars_feature_mode(cfg, default="strict_xvars"))


def get_xvars_infer_video_token_len(cfg: Any) -> int:
    return get_xvars_feature_token_len_for_mode(get_vqa_xvars_feature_mode(cfg, default="strict_xvars"))


def get_vqa_decoder_model_id(cfg: Any, default: str = "distilgpt2") -> str:
    decoder = get_component_by_kind(cfg, "decoder") or {}
    decoder = _as_dict(decoder)
    source = _as_dict(decoder.get("source"))
    params = get_component_params_by_kind(cfg, "decoder")
    model_id = params.get("repo_id") or source.get("repo_id")
    if model_id:
        return str(model_id)

    execution = get_train_execution(cfg)
    xvars_cfg = _as_dict(execution.get("xvars"))
    if xvars_cfg.get("base_model"):
        return str(xvars_cfg["base_model"])
    hf_cfg = _as_dict(execution.get("hf"))
    if hf_cfg.get("model_id"):
        return str(hf_cfg["model_id"])
    if str(source.get("provider", "")).lower() == "huggingface" and source.get("name"):
        return str(source["name"])
    if source.get("name"):
        return str(source["name"])
    return str(default)


def get_xvars_train_model_id(cfg: Any, default: str = "base_model_videoChatGPT") -> str:
    model_id = get_vqa_decoder_model_id(cfg, default=default)
    return str(model_id or default)


def get_xvars_train_tokenizer_id(cfg: Any) -> str:
    return get_xvars_train_model_id(cfg, default="base_model_videoChatGPT")


def get_xvars_infer_tokenizer_id(cfg: Any, default: str = "base_model_videoChatGPT") -> str:
    execution = get_train_execution(cfg)
    hf_cfg = _as_dict(execution.get("hf"))
    tokenizer_id = hf_cfg.get("tokenizer_id")
    if tokenizer_id:
        return str(tokenizer_id)
    return str(default)


def get_vqa_feature_source(cfg: Any, default: str = "indexed") -> str:
    encoder_params = get_component_params_by_kind(cfg, "encoder")
    feature_source = encoder_params.get("feature_source")
    if feature_source is not None:
        return str(feature_source).lower()

    execution = get_train_execution(cfg)
    xvars_cfg = _as_dict(execution.get("xvars"))
    if xvars_cfg.get("feature_source") is not None:
        return str(xvars_cfg["feature_source"]).lower()
    return str(default).lower()


def get_vqa_mm_hidden_size(cfg: Any, default: int = 1024) -> int:
    projector_params = get_component_params_by_kind(cfg, "projector")
    hidden_size = projector_params.get("input_dim") or projector_params.get("mm_hidden_size")
    if hidden_size is not None:
        return int(hidden_size)

    execution = get_train_execution(cfg)
    xvars_cfg = _as_dict(execution.get("xvars"))
    if xvars_cfg.get("mm_hidden_size") is not None:
        return int(xvars_cfg["mm_hidden_size"])
    return int(default)


def get_vqa_native_visual_cfg(cfg: Any) -> dict[str, Any]:
    execution = get_train_execution(cfg)
    native_cfg = _as_dict(execution.get("native_vl"))
    if native_cfg:
        return native_cfg

    encoder_params = get_component_params_by_kind(cfg, "encoder")
    return _as_dict(encoder_params.get("native_vl"))


def get_vqa_native_visual_input_mode(cfg: Any, default: str = "frames") -> str:
    native_cfg = get_vqa_native_visual_cfg(cfg)
    mode = str(native_cfg.get("visual_input_mode", default) or default).strip().lower()
    if mode not in {"frames", "video_with_frames_fallback"}:
        raise ValueError(
            f"Unsupported native VL visual_input_mode '{mode}'. "
            "Expected 'frames' or 'video_with_frames_fallback'."
        )
    return mode


def get_vqa_native_num_frames(cfg: Any, default: int = 8) -> int:
    native_cfg = get_vqa_native_visual_cfg(cfg)
    if native_cfg.get("num_frames") is not None:
        return max(1, int(native_cfg["num_frames"]))

    sampling = get_data_sampling(cfg)
    if sampling.get("num_frames") is not None:
        return max(1, int(sampling["num_frames"]))
    return int(default)


def get_vqa_native_min_pixels(cfg: Any) -> int | None:
    native_cfg = get_vqa_native_visual_cfg(cfg)
    value = native_cfg.get("min_pixels")
    return int(value) if value is not None else None


def get_vqa_native_max_pixels(cfg: Any) -> int | None:
    native_cfg = get_vqa_native_visual_cfg(cfg)
    value = native_cfg.get("max_pixels")
    return int(value) if value is not None else None


def get_model_runtime_dtype(cfg: Any, default: str = "fp32") -> str:
    runtime = get_model_runtime(cfg)
    dtype = runtime.get("dtype", default)
    return str(dtype).lower()
