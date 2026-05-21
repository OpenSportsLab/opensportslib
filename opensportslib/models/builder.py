"""Model builder entrypoints."""
from types import SimpleNamespace

from opensportslib.core.config.accessors import (
    get_component_load_by_kind,
    get_component_name_by_kind,
    get_component_params_by_kind,
    get_data_classes,
    get_data_num_classes,
    get_data_modality,
    get_runtime_modality,
    get_data_sampling,
    get_model_family,
    get_model_load,
    get_runner_type,
)


def _to_ns(d):
    return SimpleNamespace(**d)


def build_model_canonical(config, device):
    """Build model directly from canonical config."""
    task = config.TASK.lower()

    encoder_type = get_component_name_by_kind(config, "encoder")
    backbone = _to_ns(get_component_params_by_kind(config, "encoder"))
    neck = _to_ns(get_component_params_by_kind(config, "adapter"))
    head = _to_ns(get_component_params_by_kind(config, "head"))
    post_proc = _to_ns(get_component_params_by_kind(config, "postprocessor"))
    sampling = get_data_sampling(config)
    model_family = get_model_family(config)

    if task == "classification":
        if encoder_type == "video_mae":
            from opensportslib.models.base.video import build_video_mae_backbone
            return build_video_mae_backbone(config, device)

        elif encoder_type in ["r3d_18", "mc3_18", "r2plus1d_18", "s3d", "mvit_v2_s"]:
            from opensportslib.models.base.vars import MVNetwork
            return MVNetwork(config, backbone, neck, head), None

        elif encoder_type == "graph_conv":
            from opensportslib.models.base.tracking import TrackingModel
            return TrackingModel(config, device), None

        elif encoder_type in ("dinov3", "clip", "videomae", "videomae2"):
            from opensportslib.models.base.video import VideoModel
            return VideoModel(config, device), None

        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")

    if task == "localization":
        from opensportslib.models.base.e2e import E2EModel
        from opensportslib.models.base.contextaware import LiteContextAwareModel
        from opensportslib.models.base.learnablepooling import LiteLearnablePoolingModel

        model_weights = (
            get_model_load(config).get("checkpoint_path")
            or get_component_load_by_kind(config, "encoder").get("weights_path")
        )
        runner = get_runner_type(config)
        normalized_family = str(model_family or "").strip().lower()

        if normalized_family == "learnablepooling":
            return LiteLearnablePoolingModel(
                cfg=config,
                weights=model_weights,
                backbone=backbone,
                head=head,
                neck=neck,
                post_proc=post_proc,
                runner=runner,
            )
        if normalized_family == "contextaware":
            return LiteContextAwareModel(
                cfg=config,
                weights=model_weights,
                backbone=backbone,
                head=head,
                neck=neck,
                runner=runner,
            )

        if normalized_family == "e2e":
            return E2EModel(
                config,
                get_data_num_classes(config) + 1,
                backbone,
                head,
                clip_len=sampling.get("clip_len"),
                modality=get_runtime_modality(config),
                device=device,
                multi_gpu=getattr(config.TRAIN.execution, "multi_gpu", False),
            )

        raise ValueError(
            f"Unsupported localization model family: {model_family!r}. "
            "Expected one of: E2E, ContextAware, LearnablePooling."
        )
    else:
        raise ValueError(f"Unsupported model family for task: {task}")


def build_model_from_config(config, device):
    """Version-neutral public dispatcher for model construction."""
    model_cfg = getattr(config, "MODEL", None)
    if hasattr(model_cfg, "components"):
        return build_model_canonical(config, device)
    raise ValueError("Only canonical model config is supported at runtime.")


def build_model(config, device):
    """Backward-compatible alias for the public dispatcher."""
    return build_model_from_config(config, device)


def _resolve_model_route(payload):
    """Validate canonical source/provider metadata for routing."""
    model = payload.get("MODEL", {})
    task = str(payload.get("TASK", "")).lower()
    components = model.get("components", {})

    if task == "classification":
        encoders = [c for c in components.values() if c.get("kind") == "encoder"]
        if not encoders:
            raise ValueError("Canonical classification config must define an encoder component.")
        source = encoders[0].get("source", {})
        if not source.get("provider"):
            raise ValueError("Encoder component must define source.provider.")
        return

    if task == "localization":
        if not model.get("metadata", {}).get("family"):
            # Family inference fallback remains in runtime adapter.
            return
