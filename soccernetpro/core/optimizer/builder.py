import torch


def build_optimizer(parameters, cfg, default_args=None):
    """Build a optimizer from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        default_args (dict | None, optional): Default initialization arguments.
            Default: None.

    Returns:
        optimizer: The constructed optimizer.
    """
    if cfg.type == "Adam":
        optimizer = torch.optim.Adam(
            parameters,
            lr=cfg.lr,
            betas=tuple(cfg.betas),
            eps=cfg.eps,
            weight_decay=cfg.weight_decay,
            amsgrad=cfg.amsgrad,
        )
    elif cfg.type == "AdamWithScaler":
        optimizer = (
            torch.optim.AdamW(parameters, lr=cfg.lr),
            torch.cuda.amp.GradScaler(),
        )
    elif cfg.type == "AdamW":
        optimizer = torch.optim.AdamW(
            parameters,
            lr=cfg.lr,
            betas=tuple(cfg.betas),
            eps=cfg.eps,
            weight_decay=cfg.weight_decay,
            amsgrad=cfg.amsgrad,
        )
    return optimizer
