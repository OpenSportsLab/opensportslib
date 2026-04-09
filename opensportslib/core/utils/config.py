
import os
import re
import json
import gzip
import yaml
from types import SimpleNamespace

def dict_to_namespace(d, skip_keys=("classes",)):
    """
    Recursively convert dict to namespace for easy access,
    but keep certain keys (like 'classes') as raw dict/list.
    """
    from types import SimpleNamespace

    if isinstance(d, dict):
        out = {}
        for k, v in d.items():
            if k in skip_keys:
                out[k] = v  # leave as-is
            else:
                out[k] = dict_to_namespace(v, skip_keys)
        return SimpleNamespace(**out)
    elif isinstance(d, list):
        return [dict_to_namespace(v, skip_keys) for v in d]
    else:
        return d

def namespace_to_dict(ns):
    """
    Recursively convert namespace/dict/list containers into plain Python types.
    """
    if ns is None or isinstance(ns, (str, int, float, bool)):
        return ns

    if isinstance(ns, dict):
        return {str(k): namespace_to_dict(v) for k, v in ns.items()}

    if isinstance(ns, (list, tuple, set)):
        return [namespace_to_dict(v) for v in ns]

    if hasattr(ns, "__dict__"):
        return {str(k): namespace_to_dict(v) for k, v in vars(ns).items()}

    return ns

def namespace_to_omegaconf(ns):
    """
    Recursively convert SimpleNamespace (or dict/list) back to OmegaConf
    """
    from omegaconf import OmegaConf
    from types import SimpleNamespace

    def to_dict(obj):
        if isinstance(obj, SimpleNamespace):
            return {k: to_dict(v) for k, v in vars(obj).items()}
        elif isinstance(obj, dict):
            return {k: to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_dict(v) for v in obj]
        else:
            return obj

    return OmegaConf.create(to_dict(ns))
    
def load_config(config_path):
    """
    Loading configurations
    """
    print(config_path)
    if config_path.endswith(".yaml") or config_path.endswith(".yml"):
        with open(config_path, "r") as f:
            cfg_dict = yaml.safe_load(f)
    elif config_path.endswith(".json"):
        with open(config_path, "r") as f:
            cfg_dict = json.load(f)
    else:
        raise ValueError("Unsupported config format. Use YAML or JSON.")
    return dict_to_namespace(cfg_dict)



def load_config_omega(path):
    
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(path)
    # OmegaConf.resolve(cfg)
    # cfg = OmegaConf.to_container(cfg, resolve=True)
    return normalize_v2_config(dict_to_namespace(cfg))

def resolve_config_omega(cfg):
    from omegaconf import OmegaConf, DictConfig
    #cfg = namespace_to_omegaconf(cfg)
    #cfg = namespace_to_dict(cfg)
    #print(type(cfg))
    #cfg = OmegaConf.create(cfg)
    if not isinstance(cfg, DictConfig):
        return normalize_v2_config(cfg)
    OmegaConf.resolve(cfg)
    cfg = dict_to_namespace(OmegaConf.to_container(cfg, resolve=True))
    return normalize_v2_config(cfg)


def normalize_v2_config(cfg):
    """Normalize V2 schema into runtime-compatible keys.

    The runtime currently expects a legacy flattened shape (e.g. DATA.train.path,
    MODEL.backbone, TRAIN.num_epochs, SYSTEM.GPU). V2 configs are normalized
    into these expected aliases so the rest of the code can run unchanged.
    """
    if getattr(cfg, "VERSION", None) != 2:
        return cfg

    def _set_if_missing(ns, key, value):
        if value is None:
            return
        if not hasattr(ns, key):
            setattr(ns, key, value)

    # ----------------------------
    # SYSTEM aliases
    # ----------------------------
    if hasattr(cfg, "SYSTEM"):
        system = cfg.SYSTEM
        paths = getattr(system, "paths", None)
        if paths:
            _set_if_missing(system, "log_dir", getattr(paths, "log_dir", None))
            _set_if_missing(system, "save_dir", getattr(paths, "save_dir", None))
            _set_if_missing(system, "work_dir", getattr(paths, "work_dir", None))

        gpu = getattr(system, "gpu", None)
        if gpu:
            _set_if_missing(system, "GPU", getattr(gpu, "count", None))
            _set_if_missing(system, "gpu_id", getattr(gpu, "id", None))

        repro = getattr(system, "reproducibility", None)
        if repro:
            _set_if_missing(system, "use_seed", getattr(repro, "use_seed", None))
            _set_if_missing(system, "seed", getattr(repro, "seed", None))

    # ----------------------------
    # DATA aliases
    # ----------------------------
    if hasattr(cfg, "DATA"):
        data = cfg.DATA
        common = getattr(data, "common", None)
        if common:
            _set_if_missing(data, "dataset_name", getattr(common, "dataset_name", None))
            _set_if_missing(data, "data_dir", getattr(common, "data_dir", None))
            _set_if_missing(data, "classes", getattr(common, "classes", None))

            runtime = getattr(common, "runtime", None)
            backend = getattr(runtime, "loader_backend", None) if runtime else None
            if backend is not None and not hasattr(cfg, "dali"):
                cfg.dali = str(backend).lower() == "dali"

            splits = getattr(common, "splits", None)
            if splits:
                annotations = SimpleNamespace()
                for split_name in ["train", "valid", "test", "valid_data_frames", "challenge"]:
                    if not hasattr(splits, split_name):
                        continue
                    split_cfg = getattr(splits, split_name)
                    setattr(data, split_name, split_cfg)

                    _set_if_missing(split_cfg, "path", getattr(split_cfg, "annotation_path", None))
                    _set_if_missing(split_cfg, "video_path", getattr(split_cfg, "source_path", None))

                    if hasattr(split_cfg, "path"):
                        setattr(annotations, split_name, split_cfg.path)
                data.annotations = annotations

        inputs = getattr(data, "inputs", None)
        primary_input = inputs[0] if isinstance(inputs, list) and len(inputs) > 0 else None
        if primary_input:
            modality = getattr(primary_input, "modality", None)
            representation = getattr(primary_input, "representation", None)
            params = getattr(primary_input, "params", None)
            sampling = getattr(primary_input, "sampling", None)
            transform = getattr(primary_input, "transform", None)

            if modality == "tracking":
                data.data_modality = "tracking_parquet"
            elif modality == "video" and representation == "frames":
                data.data_modality = "frames_npy"
            elif modality == "video":
                data.data_modality = "video"
            else:
                data.data_modality = modality

            if modality == "video":
                data.modality = getattr(params, "color_mode", "rgb") if params else "rgb"
            elif modality:
                data.modality = modality

            if sampling:
                for key in [
                    "num_frames",
                    "clip_len",
                    "frame_interval",
                    "input_fps",
                    "target_fps",
                    "extract_fps",
                    "start_frame",
                    "end_frame",
                    "epoch_num_frames",
                    "framerate",
                    "window_size",
                    "chunk_size",
                    "receptive_field",
                    "chunks_per_epoch",
                ]:
                    _set_if_missing(data, key, getattr(sampling, key, None))

            if transform:
                resize = getattr(transform, "resize", None)
                if resize:
                    _set_if_missing(data, "target_height", getattr(resize, "height", None))
                    _set_if_missing(data, "target_width", getattr(resize, "width", None))
                    if hasattr(resize, "height") and hasattr(resize, "width"):
                        _set_if_missing(data, "frame_size", [resize.height, resize.width])
                normalization = getattr(transform, "normalization", None)
                if normalization:
                    _set_if_missing(data, "imagenet_mean", getattr(normalization, "mean", None))
                    _set_if_missing(data, "imagenet_std", getattr(normalization, "std", None))

            if params:
                for key in ["crop_dim", "mixup", "dilate_len", "view_type", "preload_data", "normalize"]:
                    _set_if_missing(data, key, getattr(params, key, None))

                objects = getattr(params, "objects", None)
                if objects:
                    for key in [
                        "num_objects",
                        "feature_dim",
                        "pitch_half_length",
                        "pitch_half_width",
                        "max_displacement",
                        "max_ball_height",
                    ]:
                        _set_if_missing(data, key, getattr(objects, key, None))

                _set_if_missing(data, "data_slicing", getattr(params, "data_slicing", None))

            # Split-level aliases expected by localization dataset classes
            for split_name in ["train", "valid", "test"]:
                if not hasattr(data, split_name):
                    continue
                split_cfg = getattr(data, split_name)
                _set_if_missing(split_cfg, "classes", getattr(data, "classes", None))
                for key in ["framerate", "window_size", "chunk_size", "receptive_field", "chunks_per_epoch"]:
                    val = getattr(sampling, key, None) if sampling else None
                    if val is None and params:
                        val = getattr(params, key, None)
                    _set_if_missing(split_cfg, key, val)

    # ----------------------------
    # MODEL aliases
    # ----------------------------
    if hasattr(cfg, "MODEL"):
        model = cfg.MODEL
        family = str(getattr(model, "family", "")).lower()
        family_to_type = {
            "e2e": "E2E",
            "contextaware": "ContextAware",
            "learnable_pooling": "LearnablePooling",
            "custom": "custom",
        }
        _set_if_missing(model, "type", family_to_type.get(family, getattr(model, "family", None)))

        model_runtime = getattr(model, "runtime", None)
        if model_runtime:
            _set_if_missing(model, "multi_gpu", getattr(model_runtime, "multi_gpu", None))

        model_training = getattr(model, "training", None)
        if model_training:
            _set_if_missing(model, "load_weights", getattr(model_training, "load_weights", None))
            _set_if_missing(model, "pretrained_model", getattr(model_training, "pretrained", None))
            _set_if_missing(model, "unfreeze_head", getattr(model_training, "unfreeze_head", None))
            _set_if_missing(model, "unfreeze_last_n_layers", getattr(model_training, "unfreeze_last_n_layers", None))

        streams = getattr(model, "streams", None)
        if streams:
            active_stream = None
            for name in ["video", "tracking", "video_features", "audio", "text"]:
                stream_cfg = getattr(streams, name, None)
                if stream_cfg and getattr(stream_cfg, "enabled", False):
                    active_stream = stream_cfg
                    break

            if active_stream is None:
                stream_values = list(vars(streams).values())
                if stream_values:
                    active_stream = stream_values[0]

            if active_stream:
                _set_if_missing(model, "backbone", getattr(active_stream, "backbone", None))
                _set_if_missing(model, "neck", getattr(active_stream, "neck", None))
                _set_if_missing(model, "preprocessor", getattr(active_stream, "preprocessor", None))

    # ----------------------------
    # TRAIN aliases
    # ----------------------------
    if hasattr(cfg, "TRAIN"):
        train = cfg.TRAIN
        trainer = getattr(train, "trainer", None)
        if trainer:
            _set_if_missing(train, "type", getattr(trainer, "type", None))

        epochs = getattr(train, "epochs", None)
        _set_if_missing(train, "num_epochs", epochs)
        _set_if_missing(train, "max_epochs", epochs)

        execution = getattr(train, "execution", None)
        if execution:
            for key in [
                "enabled",
                "log_interval",
                "use_amp",
                "mixup_alpha",
                "acc_grad_iter",
                "criterion_valid",
                "valid_map_every",
                "base_num_valid_epochs",
                "start_valid_epoch",
                "evaluation_frequency",
                "detailed_results",
            ]:
                _set_if_missing(train, key, getattr(execution, key, None))

        sampling = getattr(train, "sampling", None)
        if sampling:
            for key in ["use_weighted_sampler", "use_weighted_loss", "samples_per_class"]:
                _set_if_missing(train, key, getattr(sampling, key, None))

        selection = getattr(train, "selection", None)
        if selection:
            for key in ["monitor", "mode", "patience"]:
                _set_if_missing(train, key, getattr(selection, key, None))

        checkpoint = getattr(train, "checkpoint", None)
        if checkpoint:
            for key in ["save_every", "save_best", "resume_from", "keep_last"]:
                _set_if_missing(train, key, getattr(checkpoint, key, None))

    return cfg


def expand(path):
    return os.path.abspath(os.path.expanduser(path))


def load_json(fpath):
    with open(fpath) as fp:
        return json.load(fp)

def load_gz_json(fpath):
    with gzip.open(fpath, "rt", encoding="ascii") as fp:
        return json.load(fp)


def store_json(fpath, obj, pretty=False):
    kwargs = {}
    if pretty:
        kwargs["indent"] = 4
        kwargs["sort_keys"] = False
    with open(fpath, "w") as fp:
        json.dump(obj, fp, **kwargs)


def store_gz_json(fpath, obj):
    with gzip.open(fpath, "wt", encoding="ascii") as fp:
        json.dump(obj, fp)


def load_text(fpath):
    """Load text from a given file.

    Args:
        fpath (string): The path of the file.

    Returns:
        lines (List): List in which element is a line of the file.

    """
    lines = []
    with open(fpath, "r") as fp:
        for l in fp:
            l = l.strip()
            if l:
                lines.append(l)
    return lines

def load_classes(input):
    """Load classes from either list or txt file.

    Args:
        input (string): Path of the file that contains one class per line or list of classes.

    Returns:
        Dictionnary with classes associated to indexes.
    """
    from omegaconf import ListConfig
    if isinstance(input, (list, ListConfig)):
        return {x: i + 1 for i, x in enumerate(input)}
    return {x: i + 1 for i, x in enumerate(load_text(input))}

def clear_files(dir_name, re_str, exclude=[]):
    for file_name in os.listdir(dir_name):
        if re.match(re_str, file_name):
            if file_name not in exclude:
                file_path = os.path.join(dir_name, file_name)
                os.remove(file_path)


def _print_info_helper(src_file, labels):
    """Print informations about videos contained in a json file.

    Args:
        src_file (string): The source file.
        labels (list(dict)): List containing a dict fro each video.
    """
    num_frames = sum([x["num_frames"] for x in labels])
    num_events = sum([len(x["events"]) for x in labels])
    print(
        "{} : {} videos, {} frames, {:0.5f}% non-bg".format(
            src_file, len(labels), num_frames, num_events / num_frames * 100
        )
    )

def select_device(config):
    import torch
    mode = config.device.lower()

    if mode == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    elif mode == "cuda":
        assert torch.cuda.is_available(), "CUDA requested but not available"
        gpu_id = getattr(config, "gpu_id", 0)
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")

    elif mode == "cpu":
        device = torch.device("cpu")

    else:
        raise ValueError(f"Unknown device mode: {mode}")

    print(f"Using device: {device}")
    if device.type == "cuda" or device.type == "auto":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    return device

def is_local_path(p):
    return p and (
        os.path.exists(p) or
        p.endswith((".pt", ".pth", ".tar"))
    )
