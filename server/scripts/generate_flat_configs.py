from __future__ import annotations

import copy
from pathlib import Path

import yaml

CLASSIFICATION_ACTION_CLASSES = [
    "Challenge",
    "Dive",
    "Elbowing",
    "High leg",
    "Holding",
    "Pushing",
    "Standing tackling",
    "Tackling",
]
CLASSIFICATION_DATA_ROOT = "./datasets/OSL-XFoul"
LOCALIZATION_DATA_ROOT = "./datasets/BAS"


def deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def dump_yaml(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def compose_task_config(configs_root: Path, task: str, name: str) -> dict:
    root_default = load_yaml(configs_root / "default.yaml")
    task_default = load_yaml(configs_root / task / "default.yaml")
    task_config = load_yaml(configs_root / task / name)
    return deep_merge(deep_merge(root_default, task_default), task_config)


def resolve_xvars_config_source(configs_root: Path) -> Path:
    package_candidate = configs_root / "vqa" / "xvars.yaml"
    if package_candidate.is_file():
        return package_candidate
    raise FileNotFoundError(f"Could not locate xvars.yaml under {configs_root / 'vqa'}")


def main() -> None:
    import opensportslib

    package_root = Path(opensportslib.__file__).resolve().parent
    configs_root = package_root / "configs"
    target_root = Path(__file__).resolve().parent.parent / "config"
    target_root.mkdir(parents=True, exist_ok=True)

    classification = compose_task_config(
        configs_root, "classification", "video.yaml"
    )
    classification.setdefault("DATA", {}).setdefault("common", {})["classes"] = CLASSIFICATION_ACTION_CLASSES
    classification["DATA"]["common"]["data_root"] = CLASSIFICATION_DATA_ROOT
    localization = compose_task_config(
        configs_root, "localization", "video_dali.yaml"
    )
    localization.setdefault("DATA", {}).setdefault("common", {})["data_root"] = LOCALIZATION_DATA_ROOT
    vqa_qwen3 = compose_task_config(
        configs_root, "vqa", "qwen3_vl_native.yaml"
    )
    vqa_xvars = deep_merge(
        deep_merge(
            load_yaml(configs_root / "default.yaml"),
            load_yaml(configs_root / "vqa" / "default.yaml"),
        ),
        load_yaml(resolve_xvars_config_source(configs_root)),
    )

    vqa_qwen25 = copy.deepcopy(vqa_qwen3)
    vqa_qwen25["SYSTEM"]["paths"]["save_dir"] = "./checkpoints_vqa_qwen2_5_vl_native"
    llm = vqa_qwen25["MODEL"]["components"]["llm_decoder"]
    llm["source"]["name"] = "Qwen/Qwen2.5-VL-7B-Instruct"
    llm["params"]["repo_id"] = "Qwen/Qwen2.5-VL-7B-Instruct"

    dump_yaml(target_root / "classification_video.standalone.yaml", classification)
    dump_yaml(target_root / "localization_video_dali.standalone.yaml", localization)
    dump_yaml(target_root / "vqa_qwen3_vl_native.standalone.yaml", vqa_qwen3)
    dump_yaml(target_root / "vqa_qwen2_5_vl_native.standalone.yaml", vqa_qwen25)
    dump_yaml(target_root / "vqa_xvars.standalone.yaml", vqa_xvars)

    print(f"Wrote flat standalone configs to {target_root}")


if __name__ == "__main__":
    main()
