import os
import re
import json
import gzip
import logging
try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - runtime compatibility
    import yaml_compat as yaml

from opensportslib.core.config.accessors import (
    get_data_classes,
    get_data_num_classes,
    get_data_runtime,
    set_data_classes,
    set_data_num_classes,
    set_data_runtime_value,
)
from opensportslib.core.config import (
    load_config as _load_config,
    load_config_omega as _load_config_omega,
    migrate_config,
    resolve_config as _resolve_config,
    validate_config,
)


def _nested_get(mapping, path, default=None):
    current = mapping
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current

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

    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf
        if isinstance(ns, (DictConfig, ListConfig)):
            ns = OmegaConf.to_container(ns, resolve=True)
    except ImportError:
        pass

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
    return _load_config(config_path, validate=True, as_namespace=True)



def load_config_omega(path):
    return _load_config_omega(path, validate=True, as_namespace=True)

def resolve_config_omega(cfg, weights=None):
    if weights is not None:
        cfg = fetch_and_merge_config_from_HF(cfg, weights, merge_policy="compatibility")
    return _resolve_config(cfg, as_namespace=True)


def expand(path):
    return os.path.abspath(os.path.expanduser(path))


def load_json(fpath):
    with open(fpath, encoding="utf-8") as fp:
        return json.load(fp)

def load_gz_json(fpath):
    with gzip.open(fpath, "rt", encoding="utf-8") as fp:
        return json.load(fp)


def store_json(fpath, obj, pretty=False):
    kwargs = {}
    if pretty:
        kwargs["indent"] = 4
        kwargs["sort_keys"] = False
    with open(fpath, "w", encoding="utf-8") as fp:
        json.dump(obj, fp, **kwargs)


def store_gz_json(fpath, obj):
    with gzip.open(fpath, "wt", encoding="utf-8") as fp:
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

    cfg_dict = namespace_to_dict(config)
    mode = str(cfg_dict.get("device", "auto")).lower()
    gpu_cfg = cfg_dict.get("gpu", {}) if isinstance(cfg_dict, dict) else {}
    gpu_id = int(gpu_cfg.get("id", cfg_dict.get("gpu_id", 0)) or 0)

    if mode == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.set_device(gpu_id)
            device = torch.device(f"cuda:{gpu_id}")

    elif mode == "cuda":
        assert torch.cuda.is_available(), "CUDA requested but not available"
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")

    elif mode == "cpu":
        device = torch.device("cpu")

    else:
        raise ValueError(f"Unknown device mode: {mode}")

    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    return device

def resolve_config_path(config, hf_token=None):
    """Return a local path for a config given as a path or a HF repo id."""
    path = expand(config)
    if os.path.exists(path):
        return path

    from huggingface_hub import hf_hub_download

    resolved = hf_hub_download(repo_id=config, filename="config.yaml", token=hf_token)
    logging.info(f"Loaded config.yaml from HF repo {config}")
    return resolved


def is_local_path(p):
    return p and (
        os.path.exists(p) or
        p.endswith((".pt", ".pth", ".tar"))
    )


def _extract_class_metadata(data_section):
    if not isinstance(data_section, dict):
        return None, None

    from opensportslib.core.config.accessors import classes_to_ordered_list

    common = data_section.get("common", {}) if isinstance(data_section.get("common", {}), dict) else {}
    classes = common.get("classes")
    if classes is not None:
        # Order by class index: a saved config stores {name: index}, whose
        # YAML key order is alphabetical and would otherwise permute labels.
        classes = classes_to_ordered_list(classes)

    num_classes = common.get("num_classes")
    if num_classes is None:
        inputs = data_section.get("inputs", {})
        if isinstance(inputs, dict):
            for input_cfg in inputs.values():
                if not isinstance(input_cfg, dict):
                    continue
                params = input_cfg.get("params", {})
                if isinstance(params, dict) and params.get("num_classes") is not None:
                    num_classes = params.get("num_classes")
                    break

    if num_classes is None and classes is not None:
        num_classes = len(classes)

    return classes, num_classes


def _cache_pretrained_class_metadata(target_dict, loaded_dict):
    if not isinstance(target_dict, dict) or not isinstance(loaded_dict, dict):
        return

    loaded_data = loaded_dict.get("DATA", {})
    pretrained_classes, pretrained_num_classes = _extract_class_metadata(loaded_data)
    if pretrained_classes is None and pretrained_num_classes is None:
        return

    data = target_dict.setdefault("DATA", {})
    if not isinstance(data, dict):
        return
    common = data.setdefault("common", {})
    if not isinstance(common, dict):
        return
    runtime = common.setdefault("runtime", {})
    if not isinstance(runtime, dict):
        return

    if pretrained_classes is not None:
        runtime["pretrained_classes"] = list(pretrained_classes)
    if pretrained_num_classes is not None:
        runtime["pretrained_num_classes"] = int(pretrained_num_classes)


def resolve_inference_class_metadata(cfg):
    runtime = get_data_runtime(cfg)
    pretrained_classes = runtime.get("pretrained_classes")
    pretrained_num_classes = runtime.get("pretrained_num_classes")

    local_classes = get_data_classes(cfg)
    local_num_classes = get_data_num_classes(cfg, default=0)

    chosen_classes = None
    chosen_num_classes = None
    source = None

    if pretrained_classes:
        chosen_classes = list(pretrained_classes)
        chosen_num_classes = len(chosen_classes)
        source = "model"
        if local_classes and list(local_classes) != chosen_classes:
            logging.warning(
                "Inference class mismatch: local config classes differ from pretrained model classes. "
                "Using pretrained model classes."
            )
    elif pretrained_num_classes is not None:
        chosen_num_classes = int(pretrained_num_classes)
        source = "model"
        if local_num_classes and int(local_num_classes) != chosen_num_classes:
            logging.warning(
                "Inference class-count mismatch: local config num_classes=%s, pretrained model num_classes=%s. "
                "Using pretrained model class count.",
                local_num_classes,
                chosen_num_classes,
            )
    elif local_classes:
        chosen_classes = list(local_classes)
        chosen_num_classes = len(chosen_classes)
        source = "local"
    elif local_num_classes:
        chosen_num_classes = int(local_num_classes)
        source = "local"
    else:
        source = "annotation"

    if chosen_classes is not None:
        set_data_classes(cfg, chosen_classes)
    if chosen_num_classes is not None:
        set_data_num_classes(cfg, chosen_num_classes)
    set_data_runtime_value(cfg, "inference_class_source", source)
    set_data_runtime_value(cfg, "inference_model_classes_authoritative", source == "model")
    return cfg


def fetch_and_merge_config_from_HF(
    target_config, weights, hf_token=None, merge_policy="full"
):
    """
    Fetch config from a local path or HF repo and merge it with the local config.

    merge_policy:
      - "full": backward-compat behavior; local config overrides loaded config
        for TASK/MODEL/SYSTEM/TRAIN/DATA.
      - "compatibility": used for inference; only TASK/MODEL are updated from
        pretrained config while runtime/system/data settings remain local.
    """
    import os
    import logging
    from omegaconf import OmegaConf
    
    loaded_cfg = None

    if is_local_path(weights):
        abs_weights = os.path.abspath(weights)
        dir_name = abs_weights if os.path.isdir(abs_weights) else os.path.dirname(abs_weights)
        yaml_path = os.path.join(dir_name, "config.yaml")
        json_path = os.path.join(dir_name, "config.json")
        if os.path.exists(yaml_path):
            loaded_cfg = load_config_omega(yaml_path)
            logging.info(f"Loaded config from {yaml_path}")
        elif os.path.exists(json_path):
            loaded_cfg = load_config_omega(json_path)
            logging.info(f"Loaded config from {json_path}")
    else:
        try:
            from huggingface_hub import hf_hub_download
            try:
                config_path = hf_hub_download(repo_id=weights, filename="config.yaml", token=hf_token)
                loaded_cfg = load_config_omega(config_path)
                logging.info(f"Loaded config.yaml from HF repo {weights}")
            except Exception:
                config_path = hf_hub_download(repo_id=weights, filename="config.json", token=hf_token)
                loaded_cfg = load_config_omega(config_path)
                logging.info(f"Loaded config.json from HF repo {weights}")
        except Exception as e:
            logging.warning(f"Could not load config from HF repo {weights}: {e}")

    if loaded_cfg is not None:
        logging.info(f"Merging pretrained config from {weights}")
        
        target_dict = namespace_to_dict(target_config)
        loaded_dict = namespace_to_dict(loaded_cfg)

        _warn_critical_config_conflicts(target_dict, loaded_dict)
        _cache_pretrained_class_metadata(target_dict, loaded_dict)

        if merge_policy == "compatibility":
            # Keep local runtime config as source of truth. Pull only compatibility-
            # critical sections from the pretrained config.
            for section in ["TASK", "MODEL"]:
                if section in loaded_dict:
                    if isinstance(loaded_dict[section], dict):
                        target_oc = OmegaConf.create(target_dict.get(section, {}))
                        loaded_oc = OmegaConf.create(loaded_dict[section])
                        merged_oc = OmegaConf.merge(target_oc, loaded_oc)
                        target_dict[section] = OmegaConf.to_container(merged_oc, resolve=False)
                    else:
                        target_dict[section] = loaded_dict[section]
        elif merge_policy == "full":
            for section in ["TASK", "MODEL", "SYSTEM", "TRAIN", "DATA"]:
                if section in loaded_dict:
                    # Sanitize the DATA block to strip out remote machine-specific paths
                    if section == "DATA" and isinstance(loaded_dict[section], dict):
                        keys_to_remove = ["data_dir", "train", "valid", "test"]
                        for k in keys_to_remove:
                            loaded_dict[section].pop(k, None)

                    # Legacy merge logic: pretrained config as base, local config overrides it.
                    if isinstance(loaded_dict[section], dict):
                        loaded_oc = OmegaConf.create(loaded_dict[section])
                        target_oc = OmegaConf.create(target_dict.get(section, {}))
                        merged_oc = OmegaConf.merge(loaded_oc, target_oc)
                        target_dict[section] = OmegaConf.to_container(merged_oc, resolve=False)
                    else:
                        target_dict[section] = target_dict.get(section, loaded_dict[section])
        else:
            raise ValueError(f"Unknown merge_policy: {merge_policy}")

        validate_config(target_dict)
        return dict_to_namespace(target_dict)
    
    return target_config


def _warn_critical_config_conflicts(target_dict, loaded_dict):
    local_data = target_dict.get("DATA", {}) if isinstance(target_dict, dict) else {}
    hf_data = loaded_dict.get("DATA", {}) if isinstance(loaded_dict, dict) else {}

    local_num_classes = (
        _nested_get(local_data, ["common", "num_classes"])
        or local_data.get("num_classes")
    )
    hf_num_classes = (
        _nested_get(hf_data, ["common", "num_classes"])
        or hf_data.get("num_classes")
    )
    if (
        local_num_classes is not None
        and hf_num_classes is not None
        and local_num_classes != hf_num_classes
    ):
        logging.warning(
            "Config mismatch: DATA.num_classes local=%s hf=%s. "
            "Inference may use pretrained model class metadata.",
            local_num_classes,
            hf_num_classes,
        )

    local_classes = (
        _nested_get(local_data, ["common", "classes"])
        or local_data.get("classes")
    )
    hf_classes = (
        _nested_get(hf_data, ["common", "classes"])
        or hf_data.get("classes")
    )
    if (
        local_classes is not None
        and hf_classes is not None
        and local_classes != hf_classes
    ):
        logging.warning(
            "Config mismatch: DATA.classes differs between local and HF config. "
            "Inference may use pretrained model classes.",
        )


fetch_and_merge_pretrained_config = fetch_and_merge_config_from_HF

def save_config(config_obj, path):
    """Save the configuration object to a YAML file."""
    from omegaconf import DictConfig, OmegaConf

    if isinstance(config_obj, DictConfig):
        cfg_dict = OmegaConf.to_container(config_obj, resolve=True)
    else:
        cfg_dict = namespace_to_dict(config_obj)

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)
