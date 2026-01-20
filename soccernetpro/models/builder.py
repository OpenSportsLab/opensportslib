# soccernetpro/models/builder.py

from soccernetpro.models.base.video_mae import build_video_mae_backbone
from soccernetpro.models.base.e2e import E2EModel
from soccernetpro.models.base.vars import MVNetwork

def build_model(config, device):
    """
    Dispatch model builder based on cfg.MODEL.task
    """
    task = config.TASK.lower()
    
    if task == "classification":
        # return model, processor
        if config.MODEL.backbone.type == "video_mae":
            return build_video_mae_backbone(config, device)
        if config.MODEL.backbone.type in ["r3d_18", "mc3_18", "r2plus1d_18", "s3d", "mvit_v2_s"]:
            return MVNetwork(config, config.MODEL.backbone, config.MODEL.neck, config.MODEL.head), None
    if task == "localization":
        if config.MODEL.type == "E2E":
            model =  E2EModel(config, 
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
