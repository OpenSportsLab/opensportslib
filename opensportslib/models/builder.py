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
    
    elif task == "localization":
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

    elif task == "pretraining":
        return _build_ssl_model(config, device)

    else:
        raise ValueError(f"Unsupported task: {task}")


def _build_ssl_model(config, device):
    """Build an SSL model based on the configured method."""
    method = config.SSL.method.lower()
    encoder_cfg = config.MODEL.encoder
    data_cfg = config.DATA

    img_size = tuple(data_cfg.frame_size)
    patch_size = data_cfg.patch_size
    num_frames = data_cfg.num_frames
    tubelet_size = getattr(data_cfg, "tubelet_size", 2)

    if method == "mae":
        from opensportslib.models.ssl.mae import VideoMAE
        mae_cfg = config.SSL.mae
        decoder_cfg = config.MODEL.decoder
        model = VideoMAE(
            img_size=img_size,
            patch_size=patch_size,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            in_channels=3,
            encoder_embed_dim=encoder_cfg.embed_dim,
            encoder_depth=encoder_cfg.depth,
            encoder_num_heads=encoder_cfg.num_heads,
            decoder_embed_dim=decoder_cfg.embed_dim,
            decoder_depth=decoder_cfg.depth,
            decoder_num_heads=getattr(decoder_cfg, "num_heads", 8),
            mask_ratio=mae_cfg.mask_ratio,
            mask_type=mae_cfg.mask_type,
            norm_pix_loss=mae_cfg.norm_pix_loss,
        )

    elif method == "dino":
        from opensportslib.models.ssl.dino import DINO
        dino_cfg = config.SSL.dino
        model = DINO(
            img_size=img_size,
            patch_size=patch_size,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            in_channels=3,
            embed_dim=encoder_cfg.embed_dim,
            depth=encoder_cfg.depth,
            num_heads=encoder_cfg.num_heads,
            head_hidden_dim=getattr(dino_cfg, "head_hidden_dim", 2048),
            head_bottleneck_dim=getattr(dino_cfg, "head_bottleneck_dim", 256),
            head_out_dim=getattr(dino_cfg, "head_out_dim", 65536),
            momentum=dino_cfg.momentum,
            teacher_temp=dino_cfg.teacher_temp,
            student_temp=dino_cfg.student_temp,
            center_momentum=dino_cfg.center_momentum,
        )

    elif method in ("simclr", "contrastive"):
        from opensportslib.models.ssl.contrastive import SimCLR
        simclr_cfg = config.SSL.simclr
        model = SimCLR(
            img_size=img_size,
            patch_size=patch_size,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            in_channels=3,
            embed_dim=encoder_cfg.embed_dim,
            depth=encoder_cfg.depth,
            num_heads=encoder_cfg.num_heads,
            proj_hidden_dim=getattr(simclr_cfg, "proj_hidden_dim", 2048),
            proj_out_dim=getattr(simclr_cfg, "proj_out_dim", 128),
            temperature=simclr_cfg.temperature,
        )

    else:
        raise ValueError(f"Unsupported SSL method: {method}")

    return model
