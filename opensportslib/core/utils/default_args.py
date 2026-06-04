from opensportslib.core.config.accessors import (
    get_data_classes,
    get_loader_backend,
    get_model_family,
    get_runner_type,
    get_system_gpu_count,
    get_system_path,
    get_train_execution,
    get_train_epochs,
)
from opensportslib.core.utils.load_annotations import get_repartition_gpu

def _runner_type(cfg):
    return get_runner_type(cfg) or "classification"


def _execution_cfg(cfg):
    return get_train_execution(cfg)


def _execution_value(cfg, key, default=None):
    return _execution_cfg(cfg).get(key, default)


def _repartitions(cfg):
    execution = _execution_cfg(cfg)
    repartitions = execution.get("repartitions")
    if repartitions is not None:
        return repartitions
    return get_repartition_gpu(get_system_gpu_count(cfg))


def get_default_args_data_train_e2e_dali(cfg):
    return {
        "classes": get_data_classes(cfg),
        "train": True,
        "acc_grad_iter": _execution_value(cfg, "acc_grad_iter", 1),
        "num_epochs": get_train_epochs(cfg),
        "repartitions": _repartitions(cfg),
    }


def get_default_args_data_valid_e2e_dali(cfg):
    return {
        "classes": get_data_classes(cfg),
        "train": False,
        "acc_grad_iter": _execution_value(cfg, "acc_grad_iter", 1),
        "num_epochs": get_train_epochs(cfg),
        "repartitions": _repartitions(cfg),
    }


def get_default_args_data_train_e2e_opencv(cfg):
    return {"classes": get_data_classes(cfg), "train": True}


def get_default_args_data_valid_e2e_opencv(cfg):
    return {"classes": get_data_classes(cfg), "train": False}


def get_default_args_data_train():
    return {"train": True}


def get_default_args_data_valid():
    return {"train": False}


def get_default_args_data_valid_data_frames_e2e_dali(cfg):
    return {"classes": get_data_classes(cfg), "repartitions": _repartitions(cfg)}


def get_default_args_data_valid_data_frames_e2e_opencv(cfg):
    return {"classes": get_data_classes(cfg)}


def get_default_args_dataset(split, cfg):
    if split == "train":
        if _runner_type(cfg) == "runner_e2e":
            if get_loader_backend(cfg) == "dali":
                return get_default_args_data_train_e2e_dali(cfg)
            else:
                return get_default_args_data_train_e2e_opencv(cfg)
        else:
            return get_default_args_data_train()

    elif split == "valid":
        if _runner_type(cfg) == "runner_e2e":
            if get_loader_backend(cfg) == "dali":
                return get_default_args_data_valid_e2e_dali(cfg)
            else:
                return get_default_args_data_valid_e2e_opencv(cfg)
        else:
            return get_default_args_data_valid()

    elif split == "valid_data_frames" or split == "test" or split == "challenge":
        if _runner_type(cfg) == "runner_e2e":
            if get_loader_backend(cfg) == "dali":
                return get_default_args_data_valid_data_frames_e2e_dali(cfg)
            else:
                return get_default_args_data_valid_data_frames_e2e_opencv(cfg)
        else:
            return
    else:
        return None


def get_default_args_model(cfg):
    if get_model_family(cfg) == "E2E":
        return {"classes": get_data_classes(cfg)}
    else:
        return None


def get_default_args_trainer(cfg, len_train_loader):
    work_dir = get_system_path(cfg, "work_dir", "./checkpoints")
    if cfg.TRAIN.trainer.type == "trainer_e2e":
        return {
            "len_train_loader": len_train_loader,
            "work_dir": work_dir,
            "dali": get_loader_backend(cfg) == "dali",
            "repartitions": _repartitions(cfg) if get_loader_backend(cfg) == "dali" else None,
            "cfg_test": cfg.DATA.common.splits.test,
            #"cfg_challenge": cfg.DATA.challenge,
            "cfg_valid_data_frames": cfg.DATA.common.splits.valid_data_frames,
        }
    else:
        return {"work_dir": work_dir}


def get_default_args_train(model, train_loader, valid_loader, classes, trainer_type):
    if trainer_type == "trainer_CALF" or trainer_type == "trainer_pooling":
        return {
            "model": model,
            "train_dataloaders": train_loader,
            "val_dataloaders": valid_loader,
        }
    elif trainer_type == "trainer_e2e":
        return {
            "train_loader": train_loader,
            "valid_loader": valid_loader,
            "classes": classes,
        }
