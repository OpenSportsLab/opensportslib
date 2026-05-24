"""Migration from legacy configs into the canonical schema."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from ..conflicts import detect_legacy_conflicts, emit_conflict_warnings


def migrate_legacy_to_canonical(payload: dict[str, Any]) -> dict[str, Any]:
    cfg = deepcopy(payload or {})
    emit_conflict_warnings(detect_legacy_conflicts(cfg))

    task = _infer_task(cfg)
    system = _migrate_system(cfg.get("SYSTEM", {}))
    data = _migrate_data(cfg.get("DATA", {}), task, cfg.get("dali"))
    model = _migrate_model(cfg.get("MODEL", {}), data, task)
    train = _migrate_train(cfg.get("TRAIN", {}), task, cfg.get("MODEL", {}))
    io = _build_io(model, data)

    migrated = {
        "TASK": task,
        "VERSION": 3,
        "SYSTEM": system,
        "DATA": data,
        "MODEL": model,
        "TRAIN": train,
        "IO": io,
    }
    # Preserve any unknown top-level legacy keys so migration is lossless.
    top_level_unmapped = _pick_unmapped_keys(
        cfg, {"TASK", "VERSION", "SYSTEM", "DATA", "MODEL", "TRAIN", "IO", "dali"}
    )
    if top_level_unmapped:
        migrated["LEGACY_UNMAPPED"] = top_level_unmapped
    return migrated


def _infer_task(cfg: dict[str, Any]) -> str:
    task = cfg.get("TASK")
    if task:
        return str(task).lower()

    trainer_type = str(cfg.get("TRAIN", {}).get("type", "")).lower()
    model_type = str(cfg.get("MODEL", {}).get("type", "")).lower()
    if "local" in trainer_type or model_type in {"learnablepooling", "contextaware", "e2e"}:
        return "localization"
    return "classification"


def _migrate_system(system: dict[str, Any]) -> dict[str, Any]:
    migrated = {
        "paths": {
            "log_dir": system.get("log_dir", "./logs"),
            "save_dir": system.get("save_dir", "./checkpoints"),
            "work_dir": system.get("work_dir", system.get("save_dir", "./checkpoints")),
        },
        "device": system.get("device", "auto"),
        "gpu": {
            "count": system.get("GPU", 0),
            "id": system.get("gpu_id", 0),
        },
        "reproducibility": {
            "use_seed": system.get("use_seed", False),
            "seed": system.get("seed", 42),
        },
    }
    unmapped = _pick_unmapped_keys(
        system, {"log_dir", "save_dir", "work_dir", "device", "GPU", "gpu_id", "use_seed", "seed"}
    )
    if unmapped:
        migrated["legacy_unmapped"] = unmapped
    return migrated


def _migrate_data(data: dict[str, Any], task: str, legacy_dali: Any = None) -> dict[str, Any]:
    split_names = ["train", "valid", "test", "valid_data_frames", "challenge", "infer"]
    annotations = data.get("annotations", {})
    # Legacy configs usually carry `dali` at the top level, but some variants
    # place it under DATA. Accept both to avoid silent backend drift.
    dali_flag = legacy_dali
    if dali_flag is None:
        dali_flag = data.get("dali")
    runtime = {"loader_backend": "dali" if bool(dali_flag) else "opencv"}
    splits: dict[str, Any] = {}
    for split in split_names:
        split_cfg = deepcopy(data.get(split, {}))
        annotation_path = split_cfg.get("path") or annotations.get(split)
        source_path = split_cfg.get("video_path") or data.get("data_dir")
        if annotation_path is None and source_path is None and not split_cfg:
            continue

        migrated = {k: deepcopy(v) for k, v in split_cfg.items() if k not in {"path", "video_path"}}
        if annotation_path is not None:
            migrated["annotation_path"] = annotation_path
        if source_path is not None:
            migrated["source_path"] = source_path
        splits[split] = migrated

    input_name = _infer_input_name(data, task)
    input_cfg = {
        "modality": _infer_input_modality(data, input_name),
        "representation": _infer_representation(data, task),
        "source": {"format": _infer_source_format(data, input_name)},
        "sampling": _pick_keys(
            data,
            "num_frames",
            "clip_len",
            "input_fps",
            "target_fps",
            "extract_fps",
            "framerate",
            "window_size",
            "chunk_size",
            "receptive_field",
            "epoch_num_frames",
            "start_frame",
            "end_frame",
            "overlap_len",
        ),
        "transform": {
            "resize": _pick_renamed(
                data,
                {
                    "target_height": "height",
                    "target_width": "width",
                    "frame_height": "height",
                    "frame_width": "width",
                },
            ),
            # Legacy tracking configs commonly use a boolean `normalize` flag.
            # Preserve it so inference/training preprocessing remains identical.
            "normalize": data.get("normalize"),
            "normalization": {
                "mean": data.get("imagenet_mean"),
                "std": data.get("imagenet_std"),
            },
        },
        "augmentations": _pick_keys(
            data,
            "random_affine",
            "translate",
            "affine_scale",
            "random_perspective",
            "distortion_scale",
            "perspective_prob",
            "random_rotation",
            "rotation_degrees",
            "color_jitter",
            "jitter_params",
            "random_horizontal_flip",
            "flip_prob",
            "random_crop",
            # Legacy tracking augmentation flags.
            "horizontal_flip",
            "vertical_flip",
            "team_flip",
        ),
        "params": _pick_keys(
            data,
            "view_type",
            "num_classes",
            "crop_dim",
            "mixup",
            "dilate_len",
            "max_samples",
            "preload_data",
        ),
    }
    # Legacy tracking geometry/object params -> canonical params.objects.
    objects_cfg = _pick_keys(
        data,
        "num_objects",
        "feature_dim",
        "pitch_half_length",
        "pitch_half_width",
        "max_displacement",
        "max_ball_height",
    )
    if objects_cfg:
        input_cfg["params"]["objects"] = objects_cfg

    # Legacy data slicing knob -> canonical params.data_slicing.
    if isinstance(data.get("data_slicing"), dict):
        input_cfg["params"]["data_slicing"] = deepcopy(data["data_slicing"])

    # Preserve full legacy augmentations payload when present.
    if isinstance(data.get("augmentations"), dict):
        aug = deepcopy(data["augmentations"])
        aug.update(input_cfg["augmentations"])
        input_cfg["augmentations"] = aug

    color_mode = _infer_color_mode(data, input_name)
    if color_mode is not None:
        input_cfg["params"]["color_mode"] = color_mode

    _drop_nones(input_cfg["transform"]["normalization"])
    _drop_empty(input_cfg["transform"])
    _drop_empty(input_cfg["augmentations"])
    _drop_empty(input_cfg["params"])

    common = {
        "dataset_name": data.get("dataset_name", data.get("type", task)),
        "data_root": data.get("data_dir"),
        "classes": deepcopy(data.get("classes", [])),
        "runtime": runtime,
        "splits": splits,
    }
    consumed_data_keys = {
        "annotations",
        "dali",
        "dataset_name",
        "type",
        "data_dir",
        "classes",
        "data_modality",
        "modality",
        "normalize",
        "imagenet_mean",
        "imagenet_std",
        "augmentations",
        "data_slicing",
        "train",
        "valid",
        "test",
        "valid_data_frames",
        "challenge",
        "infer",
        "num_frames",
        "clip_len",
        "input_fps",
        "target_fps",
        "extract_fps",
        "framerate",
        "window_size",
        "chunk_size",
        "receptive_field",
        "epoch_num_frames",
        "start_frame",
        "end_frame",
        "overlap_len",
        "target_height",
        "target_width",
        "frame_height",
        "frame_width",
        "random_affine",
        "translate",
        "affine_scale",
        "random_perspective",
        "distortion_scale",
        "perspective_prob",
        "random_rotation",
        "rotation_degrees",
        "color_jitter",
        "jitter_params",
        "random_horizontal_flip",
        "flip_prob",
        "random_crop",
        "horizontal_flip",
        "vertical_flip",
        "team_flip",
        "view_type",
        "num_classes",
        "crop_dim",
        "mixup",
        "dilate_len",
        "max_samples",
        "preload_data",
        "num_objects",
        "feature_dim",
        "pitch_half_length",
        "pitch_half_width",
        "max_displacement",
        "max_ball_height",
    }
    data_unmapped = _pick_unmapped_keys(data, consumed_data_keys)
    if data_unmapped:
        common["legacy_unmapped"] = data_unmapped
    return {"common": common, "inputs": {input_name: input_cfg}}


def _migrate_model(model: dict[str, Any], data: dict[str, Any], task: str) -> dict[str, Any]:
    components: dict[str, Any] = {}
    topology: list[dict[str, Any]] = []
    input_cfgs = data.get("inputs", {})
    primary_input = next(iter(input_cfgs.values()), {})
    primary_input_params = primary_input.get("params", {})
    common_classes = data.get("common", {}).get("classes", [])
    inferred_num_classes = primary_input_params.get("num_classes") or (
        len(common_classes) if common_classes else None
    )

    encoder_name = _infer_encoder_component_name(model, data, task)
    backbone = deepcopy(model.get("backbone", {}))

    # Legacy tracking configs often store graph edge settings at MODEL level
    # (edge/k/r) rather than inside MODEL.backbone. Preserve those values by
    # projecting them into encoder params during migration.
    if encoder_name == "tracking_encoder":
        if backbone.get("edge_type") is None and model.get("edge") is not None:
            backbone["edge_type"] = model.get("edge")
        if backbone.get("k") is None and model.get("k") is not None:
            backbone["k"] = model.get("k")
        if (
            backbone.get("radius") is None
            and backbone.get("r") is None
            and model.get("r") is not None
        ):
            backbone["radius"] = model.get("r")

    if backbone:
        components[encoder_name] = {
            "kind": "encoder",
            "source": {
                "provider": "opensportslib",
                "registry": "backbone",
                "name": backbone.get("type"),
            },
            "params": backbone,
            "overrides": _pick_keys(backbone, "unfreeze_head", "unfreeze_last_n_layers"),
        }

    prev_name = encoder_name
    neck = deepcopy(model.get("neck", {}))
    if neck:
        adapter_name = f"{encoder_name.rsplit('_', 1)[0]}_adapter"
        components[adapter_name] = {
            "kind": "adapter",
            "source": {
                "provider": "opensportslib",
                "registry": "neck",
                "name": neck.get("type"),
            },
            "params": neck,
            "overrides": {},
        }
        topology.append({"from": prev_name, "to": adapter_name})
        prev_name = adapter_name

    head = deepcopy(model.get("head", {}))
    if head:
        if task == "classification" and head.get("num_classes") is None and inferred_num_classes is not None:
            head["num_classes"] = inferred_num_classes
        components["task_head"] = {
            "kind": "head",
            "source": {
                "provider": "opensportslib",
                "registry": "head",
                "name": head.get("type"),
            },
            "params": head,
            "overrides": {},
        }
        topology.append({"from": prev_name, "to": "task_head"})
        prev_name = "task_head"

    post_proc = deepcopy(model.get("post_proc", {}))
    if post_proc:
        components["event_postprocessor"] = {
            "kind": "postprocessor",
            "source": {
                "provider": "opensportslib",
                "registry": "post_proc",
                "name": post_proc.get("type", "NMS"),
            },
            "params": post_proc,
            "overrides": {},
        }
        topology.append({"from": prev_name, "to": "event_postprocessor"})

    load_cfg = {
        "checkpoint_path": model.get("load_weights"),
        "pretrained": bool(model.get("pretrained", False)),
        "strict": True,
        "map_location": None,
        "format": "auto",
    }

    runtime = {
        "dtype": "fp32",
        "device": "auto",
        "compile": False,
        "freeze": False,
        "multi_gpu": bool(model.get("multi_gpu", False)),
    }

    metadata = {
        "family": model.get("type"),
        "runner": deepcopy(model.get("runner", {})),
        "legacy_type": model.get("type"),
    }
    model_unmapped = _pick_unmapped_keys(
        model,
        {
            "type",
            "runner",
            "backbone",
            "neck",
            "head",
            "post_proc",
            "edge",
            "k",
            "r",
            "load_weights",
            "pretrained",
            "multi_gpu",
        },
    )
    if model_unmapped:
        metadata["legacy_unmapped"] = model_unmapped

    return {
        "schema_version": 3,
        "task": task,
        "runtime": runtime,
        "load": load_cfg,
        "components": components,
        "topology": topology,
        "policies": {},
        "metadata": metadata,
    }


def _migrate_train(
    train: dict[str, Any], task: str, model: dict[str, Any] | None = None
) -> dict[str, Any]:
    epochs = train.get("epochs", train.get("num_epochs", train.get("max_epochs", 1)))
    execution = _pick_keys(
        train,
        "enabled",
        "log_interval",
        "multi_gpu",
        "acc_grad_iter",
        "evaluation_frequency",
        "base_num_valid_epochs",
        "start_valid_epoch",
        "valid_map_every",
        "criterion_valid",
    )
    if "multi_gpu" not in execution and isinstance(model, dict) and "multi_gpu" in model:
        execution["multi_gpu"] = bool(model.get("multi_gpu"))

    migrated = {
        "trainer": {"type": train.get("type", task)},
        "epochs": epochs,
        "criterion": deepcopy(train.get("criterion", {"type": "CrossEntropyLoss"})),
        "optimizer": deepcopy(train.get("optimizer", {})),
        "scheduler": deepcopy(train.get("scheduler", {})),
        "execution": execution,
        "sampling": _pick_keys(train, "use_weighted_sampler", "use_weighted_loss", "batch_size"),
        "selection": {
            "monitor": train.get("criterion_valid", "loss"),
            "mode": "max" if train.get("criterion_valid") == "map" else "min",
        },
        "checkpoint": _pick_keys(train, "save_every", "save_best"),
    }
    train_unmapped = _pick_unmapped_keys(
        train,
        {
            "type",
            "epochs",
            "num_epochs",
            "max_epochs",
            "criterion",
            "optimizer",
            "scheduler",
            "enabled",
            "log_interval",
            "multi_gpu",
            "acc_grad_iter",
            "evaluation_frequency",
            "base_num_valid_epochs",
            "start_valid_epoch",
            "valid_map_every",
            "criterion_valid",
            "use_weighted_sampler",
            "use_weighted_loss",
            "batch_size",
            "save_every",
            "save_best",
        },
    )
    if train_unmapped:
        migrated["legacy_unmapped"] = train_unmapped
    return migrated


def _build_io(model: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    inputs = {}
    data_inputs = data.get("inputs", {})
    components = model.get("components", {})
    for input_name in data_inputs:
        if input_name == "tracking":
            target = "tracking_encoder"
        elif input_name == "features":
            target = "feature_encoder"
        else:
            target = "video_encoder"
        if target not in components and components:
            target = next(iter(components))
        inputs[input_name] = target
    if "task_head" in components:
        inputs["label"] = "task_head"

    outputs = {}
    if "event_postprocessor" in components:
        outputs["events"] = "event_postprocessor"
    if "task_head" in components:
        outputs["logits"] = "task_head"
    return {"inputs": inputs, "outputs": outputs}


def _infer_input_name(data: dict[str, Any], task: str) -> str:
    modality = str(data.get("data_modality", data.get("modality", ""))).lower()
    if "track" in modality:
        return "tracking"
    if task == "localization" and any("features" in str(v.get("type", "")).lower() for v in data.values() if isinstance(v, dict)):
        return "features"
    if task == "localization" and any("Feature" in str(v.get("type", "")) for v in data.values() if isinstance(v, dict)):
        return "features"
    return "video"


def _infer_input_modality(data: dict[str, Any], input_name: str) -> str:
    if input_name == "tracking":
        return "tracking"
    legacy_modality = str(data.get("modality", data.get("data_modality", "video"))).lower()
    if legacy_modality in {"rgb", "flow", "bw"}:
        return "video"
    return str(data.get("modality", data.get("data_modality", "video")))


def _infer_color_mode(data: dict[str, Any], input_name: str) -> str | None:
    if input_name != "video":
        return None
    legacy_modality = str(data.get("modality", "")).lower()
    if legacy_modality in {"rgb", "flow", "bw"}:
        return legacy_modality
    return None


def _infer_representation(data: dict[str, Any], task: str) -> str:
    input_name = _infer_input_name(data, task)
    if input_name == "tracking":
        return "graph"
    if input_name == "features":
        return "features"
    if str(data.get("data_modality", "")).lower() == "frames_npy":
        return "frames_npy"
    return "raw"


def _infer_source_format(data: dict[str, Any], input_name: str) -> str:
    if input_name == "tracking":
        return "parquet"
    if input_name == "features":
        return "npy"
    if str(data.get("data_modality", "")).lower() == "frames_npy":
        return "npy"
    return "mp4"


def _infer_encoder_component_name(model: dict[str, Any], data: dict[str, Any], task: str) -> str:
    # When called from _migrate_model, `data` is already canonical and carries
    # modality under DATA.inputs. Handle that shape first.
    inputs = data.get("inputs", {})
    if isinstance(inputs, dict):
        if "tracking" in inputs:
            return "tracking_encoder"
        if "features" in inputs:
            return "feature_encoder"
        if inputs:
            return "video_encoder"

    input_name = _infer_input_name(data, task)
    if input_name == "tracking":
        return "tracking_encoder"
    if input_name == "features":
        return "feature_encoder"
    return "video_encoder"


def _pick_keys(source: dict[str, Any], *keys: str) -> dict[str, Any]:
    return {key: deepcopy(source[key]) for key in keys if key in source}


def _pick_renamed(source: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
    picked = {}
    for old_key, new_key in mapping.items():
        if old_key in source:
            picked[new_key] = deepcopy(source[old_key])
    return picked


def _drop_nones(target: dict[str, Any]) -> None:
    for key in list(target.keys()):
        if target[key] is None:
            target.pop(key)


def _drop_empty(target: dict[str, Any]) -> None:
    for key in list(target.keys()):
        value = target[key]
        if value in ({}, [], None):
            target.pop(key)


def _pick_unmapped_keys(source: dict[str, Any], consumed_keys: set[str]) -> dict[str, Any]:
    return {
        key: deepcopy(value)
        for key, value in source.items()
        if key not in consumed_keys
    }
