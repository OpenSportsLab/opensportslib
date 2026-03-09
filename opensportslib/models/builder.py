# opensportslib/models/builder.py

def build_model(config, device):
    """
    Dispatch model builder based on cfg.MODEL.task
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
        if config.MODEL.type == "E2E":
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
