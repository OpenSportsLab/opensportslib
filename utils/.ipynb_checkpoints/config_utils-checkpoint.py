import yaml
import json
from easydict import EasyDict as edict

def load_config(config_path):
    if config_path.endswith(".yaml") or config_path.endswith(".yml"):
        with open(config_path, "r") as f:
            cfg_dict = yaml.safe_load(f)
    elif config_path.endswith(".json"):
        with open(config_path, "r") as f:
            cfg_dict = json.load(f)
    else:
        raise ValueError("Unsupported config format. Use YAML or JSON.")
    return edict(cfg_dict)