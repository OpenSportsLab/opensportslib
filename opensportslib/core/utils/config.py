
import os
import re
import json
import gzip
import yaml

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
    try:
        from omegaconf import OmegaConf
    except ImportError:
        return load_config(path)

    cfg = OmegaConf.load(path)
    # OmegaConf.resolve(cfg)
    # cfg = OmegaConf.to_container(cfg, resolve=True)
    return dict_to_namespace(cfg)

def resolve_config_omega(cfg):
    try:
        from omegaconf import OmegaConf, DictConfig
    except ImportError:
        return cfg

    #cfg = namespace_to_omegaconf(cfg)
    #cfg = namespace_to_dict(cfg)
    #print(type(cfg))
    #cfg = OmegaConf.create(cfg)
    if not isinstance(cfg, DictConfig):
        return cfg 
    OmegaConf.resolve(cfg)
    cfg = dict_to_namespace(OmegaConf.to_container(cfg, resolve=True))
    return cfg


def resolve_config_references(cfg):
    """Resolve OmegaConf-style interpolations on namespace-backed configs."""
    try:
        from omegaconf import OmegaConf
    except ImportError:
        return _resolve_config_references_without_omegaconf(cfg)

    cfg_oc = namespace_to_omegaconf(cfg)
    OmegaConf.resolve(cfg_oc)
    return dict_to_namespace(OmegaConf.to_container(cfg_oc, resolve=True))


def _resolve_config_references_without_omegaconf(cfg):
    data = namespace_to_dict(cfg)

    def lookup(root, path):
        current = root
        for part in path.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current

    def resolve_obj(obj, root):
        if isinstance(obj, dict):
            return {key: resolve_obj(value, root) for key, value in obj.items()}
        if isinstance(obj, list):
            return [resolve_obj(value, root) for value in obj]
        if isinstance(obj, str):
            resolved = obj
            for match in re.findall(r"\$\{([^}]+)\}", resolved):
                value = lookup(root, match)
                if value is not None:
                    resolved = resolved.replace("${" + match + "}", str(value))
            return resolved
        return obj

    for _ in range(10):
        resolved_data = resolve_obj(data, data)
        if resolved_data == data:
            break
        data = resolved_data

    return dict_to_namespace(data)


def _replace_data_dir_references(cfg, old_data_dir, new_data_dir):
    if not old_data_dir:
        return cfg

    old_data_dir = expand(old_data_dir)
    new_data_dir = expand(new_data_dir)
    data = namespace_to_dict(cfg)

    def replace_obj(obj):
        if isinstance(obj, dict):
            return {key: replace_obj(value) for key, value in obj.items()}
        if isinstance(obj, list):
            return [replace_obj(value) for value in obj]
        if isinstance(obj, str) and obj.startswith(old_data_dir):
            return new_data_dir + obj[len(old_data_dir):]
        return obj

    return dict_to_namespace(replace_obj(data))


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
    try:
        from omegaconf import ListConfig
    except ImportError:
        ListConfig = ()

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


def download_hf_dataset(repo_id, revision=None, hf_token=None):
    """Download an HF dataset repo snapshot and return its local cache path."""
    import logging

    if not repo_id:
        raise ValueError("dataset_repo_id is required to download a Hugging Face dataset.")

    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required to download datasets from Hugging Face. "
            "Install it with `pip install huggingface_hub`."
        ) from e

    repo_type = "dataset"
    normalized_repo_id = re.sub(r"[^a-zA-Z0-9._-]+", "__", repo_id)
    normalized_revision = re.sub(
        r"[^a-zA-Z0-9._-]+",
        "__",
        revision if revision is not None else "main",
    )
    local_dir = os.path.join(
        os.path.expanduser("~"),
        ".cache",
        "opensportslib",
        "huggingface",
        repo_type,
        normalized_repo_id,
        normalized_revision,
    )
    logging.info(
        "Downloading Hugging Face dataset repo=%s revision=%s into %s",
        repo_id,
        revision,
        local_dir,
    )

    try:
        dataset_path = snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            token=hf_token,
            local_dir=local_dir,
        )
        logging.info("Hugging Face dataset available at %s", dataset_path)
        return dataset_path
    except Exception as e:
        revision_msg = f" at revision {revision!r}" if revision is not None else ""
        raise RuntimeError(
            f"Could not download Hugging Face dataset repo {repo_id!r}{revision_msg}."
        ) from e


def resolve_hf_dataset(
    config,
    dataset_repo_id=None,
    dataset_revision=None,
    hf_token=None,
    skip_download=False,
):
    """Resolve a runtime HF dataset repo into DATA.data_dir."""
    import logging

    if not dataset_repo_id:
        return config

    if skip_download:
        logging.info(
            "Skipping Hugging Face dataset download because data_dir was provided."
        )
        return config

    old_data_dir = getattr(config.DATA, "data_dir", None)
    dataset_path = download_hf_dataset(
        repo_id=dataset_repo_id,
        revision=dataset_revision,
        hf_token=hf_token,
    )
    config.DATA.data_dir = expand(dataset_path)
    logging.info(
        "Downloaded Hugging Face dataset %s revision=%s to %s",
        dataset_repo_id,
        dataset_revision,
        config.DATA.data_dir,
    )
    config = resolve_config_references(config)
    return _replace_data_dir_references(config, old_data_dir, config.DATA.data_dir)

def save_config(config_obj, path):
    """Save the configuration object to a YAML file."""
    from omegaconf import OmegaConf, DictConfig
    import yaml
    
    if isinstance(config_obj, DictConfig):
        cfg_dict = OmegaConf.to_container(config_obj, resolve=True)
    else:
        cfg_dict = namespace_to_dict(config_obj)

    with open(path, "w") as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)

def _warn_critical_config_conflicts(target_dict, loaded_dict):
    import logging

    local_data = target_dict.get("DATA", {}) if isinstance(target_dict, dict) else {}
    hf_data = loaded_dict.get("DATA", {}) if isinstance(loaded_dict, dict) else {}

    local_num_classes = local_data.get("num_classes")
    hf_num_classes = hf_data.get("num_classes")
    if (
        local_num_classes is not None
        and hf_num_classes is not None
        and local_num_classes != hf_num_classes
    ):
        logging.warning(
            "Config mismatch: DATA.num_classes local=%s hf=%s. "
            "Keeping local runtime config values.",
            local_num_classes,
            hf_num_classes,
        )

    local_classes = local_data.get("classes")
    hf_classes = hf_data.get("classes")
    if (
        local_classes is not None
        and hf_classes is not None
        and local_classes != hf_classes
    ):
        logging.warning(
            "Config mismatch: DATA.classes differs between local and HF config. "
            "Keeping local runtime config values.",
        )


def fetch_and_merge_pretrained_config(
    target_config, pretrained, hf_token=None, merge_policy="full"
):
    """
    Fetch config from a local path or HF repo and merge it with the local config.

    merge_policy:
      - "full": legacy behavior; local config overrides loaded config for
        TASK/MODEL/SYSTEM/TRAIN/DATA.
      - "compatibility": used for inference; only TASK/MODEL are updated from
        pretrained config while runtime/system/data settings remain local.
    """
    import os
    import logging
    from omegaconf import OmegaConf
    
    loaded_cfg = None

    if is_local_path(pretrained):
        dir_name = os.path.dirname(os.path.abspath(pretrained))
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
                config_path = hf_hub_download(repo_id=pretrained, filename="config.yaml", token=hf_token)
                loaded_cfg = load_config_omega(config_path)
                logging.info(f"Loaded config.yaml from HF repo {pretrained}")
            except Exception:
                config_path = hf_hub_download(repo_id=pretrained, filename="config.json", token=hf_token)
                loaded_cfg = load_config_omega(config_path)
                logging.info(f"Loaded config.json from HF repo {pretrained}")
        except Exception as e:
            logging.warning(f"Could not load config from HF repo {pretrained}: {e}")

    if loaded_cfg is not None:
        logging.info(f"Merging pretrained config from {pretrained}")
        
        target_dict = namespace_to_dict(target_config)
        loaded_dict = namespace_to_dict(loaded_cfg)

        _warn_critical_config_conflicts(target_dict, loaded_dict)

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

        # Convert back using DictConfig or SimpleNamespace
        return dict_to_namespace(target_dict)
    
    return target_config
