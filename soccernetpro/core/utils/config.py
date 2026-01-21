
import os
import re
import json
import gzip
import yaml

def dict_to_namespace(d):
    """
    Recursively convert dict to namespace for easy access
    """
    from types import SimpleNamespace

    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    return d
    
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
    OmegaConf.resolve(cfg)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    #cfg.DATA.classes = OmegaConf.to_container(cfg.DATA.classes, resolve=True)
    return dict_to_namespace(cfg)


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
        return {x: i + 1 for i, x in enumerate(sorted(input))}
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
