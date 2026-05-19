# opensportslib/models/builder.py

def build_model_legacy(config, device):
    """
    Dispatch model builder for the current legacy runtime shape.
    """
    task = config.TASK.lower()
    
    if task == "classification":
        # return model, processor
        if config.MODEL.backbone.type == "video_mae":
            from opensportslib.models.base.video import build_video_mae_backbone
            return build_video_mae_backbone(config, device)
        
        elif config.MODEL.backbone.type in ["r3d_18", "mc3_18", "r2plus1d_18", "s3d", "mvit_v2_s"]:
            from opensportslib.models.base.vars import MVNetwork
            return MVNetwork(config, config.MODEL.backbone, config.MODEL.neck, config.MODEL.head), None
        
        elif config.MODEL.backbone.type == "graph_conv":
            from opensportslib.models.base.tracking import TrackingModel
            return TrackingModel(config, device), None

        elif config.MODEL.backbone.type in ("dinov3", "clip", "videomae", "videomae2"):
            from opensportslib.models.base.video import VideoModel
            return VideoModel(config, device), None
        
        else:
            raise ValueError(f"Unsupported backbone type: {config.MODEL.backbone.type}")
    
    if task == "localization":
        from opensportslib.models.base.e2e import E2EModel
        from opensportslib.models.base.contextaware import LiteContextAwareModel
        from opensportslib.models.base.learnablepooling import LiteLearnablePoolingModel
        
        if config.MODEL.type == "LearnablePooling":
            model = LiteLearnablePoolingModel(
                cfg=config,
                weights=config.MODEL.load_weights,
                backbone=config.MODEL.backbone,
                head=config.MODEL.head,
                neck=config.MODEL.neck,
                post_proc=config.MODEL.post_proc,
                runner=config.MODEL.runner.type,
            )
        elif config.MODEL.type == "ContextAware":
            model = LiteContextAwareModel(
                cfg=config,
                weights=config.MODEL.load_weights,
                backbone=config.MODEL.backbone,
                head=config.MODEL.head,
                neck=config.MODEL.neck,
                runner=config.MODEL.runner.type,
            )
            
        elif config.MODEL.type == "E2E":
            model = E2EModel(config, 
                            len(config.DATA.classes)+1,
                            config.MODEL.backbone,
                            config.MODEL.head,
                            clip_len=config.DATA.clip_len,
                            modality=config.DATA.modality,
                            device=device,
                            multi_gpu=config.MODEL.multi_gpu)
        return model
    else:
        raise ValueError(f"Unsupported model type: {config.MODEL.backbone} for task: {task}")


def build_model_canonical(config, device):
    """Build from canonical config by adapting to the current runtime."""
    from opensportslib.core.config import adapt_config_to_runtime

    runtime_config = adapt_config_to_runtime(config, as_namespace=True)
    return build_model_legacy(runtime_config, device)


def build_model_from_config(config, device):
    """Version-neutral public dispatcher for model construction."""
    model_cfg = getattr(config, "MODEL", None)
    if getattr(config, "VERSION", None) == 3 and hasattr(model_cfg, "components"):
        return build_model_canonical(config, device)
    return build_model_legacy(config, device)


def build_model(config, device):
    """Backward-compatible alias for the public dispatcher."""
    return build_model_from_config(config, device)
