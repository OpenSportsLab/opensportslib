# soccernetpro/models/builder.py

from soccernetpro.models.base.video_mae import build_video_mae_backbone
from soccernetpro.models.base.e2e import E2EModel

def build_model(config, device):
    """
    Dispatch model builder based on cfg.MODEL.task
    """
    task = config.TASK.lower()
    
    if task == "classification":
        if config.MODEL.backbone == "video_mae":
            return build_video_mae_backbone(config, device)
    
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
