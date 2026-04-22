# Configuration Guide

This guide explains how OpenSportsLib YAML configs are structured, what each key does, and how to run many experiments without creating a new YAML file every time.

## Source Config Files

Main config files in the repo:

- `opensportslib/config/classification.yaml`
- `opensportslib/config/localization.yaml`
- `opensportslib/config/localization-json_netvlad++_resnetpca512.yaml`
- `opensportslib/config/localization-json_calf_resnetpca512.yaml`
- `opensportslib/config/sngar-tracking.yaml`
- `opensportslib/config/sngar-frames.yaml`

## Base YAML Overview

Most task configs follow this high-level structure:

```yaml
TASK: classification | localization

DATA:
  ...

MODEL:
  ...

TRAIN:
  ...

SYSTEM:
  ...
```

### `TASK`

Defines which task pipeline is used.

- `classification`: clip-level classification pipeline.
- `localization`: spotting/localization pipeline.

If `TASK` does not match the selected API (`model.classification` / `model.localization`), behavior can be incorrect or fail.

### `DATA`

Defines data source, annotation paths, sampling, preprocessing, and dataloader behavior.

This block usually contains:

- Dataset identity and root path
- Class list or class count
- Split definitions (`train`, `valid`, `test`)
- Temporal sampling settings (`num_frames`, `clip_len`, `input_fps`, `extract_fps`, etc.)
- Spatial settings (`frame_size`, `target_height`, `target_width`)
- Augmentation switches
- Dataloader options

### `MODEL`

Defines model family and submodules.

Typical nested blocks:

- `backbone`: feature extractor
- `neck`: temporal/multi-view aggregation
- `head`: classifier or spotting head
- optional post-processing (`post_proc`)
- optional checkpoint loading (`load_weights`, `pretrained_model`)

### `TRAIN`

Defines optimization logic.

Typical nested blocks:

- Monitoring and early-stop criteria
- Epoch controls (`epochs` or `num_epochs` or `max_epochs` depending on trainer)
- `criterion` (loss)
- `optimizer`
- `scheduler`
- Sampling and weighting controls

### `SYSTEM`

Defines runtime environment and output paths.

Typical keys:

- `log_dir`, `save_dir`, `work_dir`
- seed control
- device/GPU settings

## Interpolation and Derived Keys

OpenSportsLib configs use OmegaConf interpolation syntax:

- `${DATA.data_dir}`
- `${TRAIN.num_epochs}`
- `${SYSTEM.save_dir}`

This avoids duplication and keeps paths consistent.

## Key-by-Key Reference

## Common Keys Across Most Configs

| Key | Type | Example | What it controls | Tuning guidance |
|---|---|---|---|---|
| `TASK` | string | `classification` | Chooses pipeline entrypoint | Keep aligned with model API |
| `SYSTEM.device` | string | `cuda` | Device selection (`auto/cuda/cpu`) | Use `cpu` for smoke/local checks |
| `SYSTEM.gpu_id` | int | `0` | GPU index for single-GPU runs | Set only if multiple GPUs |
| `SYSTEM.seed` | int | `42` | Random seed value | Use fixed seed for reproducibility |
| `SYSTEM.log_dir` | path | `./logs` | Training logs location | Separate per experiment when needed |
| `SYSTEM.save_dir` | path | `./checkpoints` | Checkpoint output root | Keep enough disk space |

## Classification (`classification.yaml`) Reference

### DATA block

| Key | Type | Example | Meaning |
|---|---|---|---|
| `DATA.dataset_name` | string | `mvfouls` | Dataset identifier |
| `DATA.data_dir` | path | `/.../SoccerNet/mvfouls` | Dataset root directory |
| `DATA.data_modality` | string | `video` | Input modality for loader |
| `DATA.view_type` | string | `multi` | Single-view or multi-view processing |
| `DATA.num_classes` | int | `8` | Number of target classes |
| `DATA.num_frames` | int | `16` | Number of sampled frames per clip |
| `DATA.input_fps` | int | `25` | Original video FPS |
| `DATA.target_fps` | int | `17` | Downsampled FPS used by loader |
| `DATA.start_frame` | int | `63` | Clip start frame (relative) |
| `DATA.end_frame` | int | `87` | Clip end frame (relative) |
| `DATA.frame_size` | list[int,int] | `[224, 224]` | Spatial resize `(H, W)` |

### DATA split sub-blocks

Each split (`train`, `valid`, `test`) has:

| Key | Type | Example | Meaning |
|---|---|---|---|
| `DATA.<split>.video_path` | path | `${DATA.data_dir}/train` | Video root for split |
| `DATA.<split>.path` | path | `.../annotations-train.json` | Annotation file |
| `DATA.<split>.dataloader.batch_size` | int | `8` | Batch size |
| `DATA.<split>.dataloader.shuffle` | bool | `true` | Shuffle data each epoch |
| `DATA.<split>.dataloader.num_workers` | int | `4` | Data loading worker count |
| `DATA.<split>.dataloader.pin_memory` | bool | `true` | Host-to-device transfer optimization |

### DATA augmentation keys

| Key | Type | Example | Meaning |
|---|---|---|---|
| `random_affine` | bool | `true` | Enable affine transform |
| `translate` | list[float,float] | `[0.1, 0.1]` | Affine translation range |
| `affine_scale` | list[float,float] | `[0.9, 1.0]` | Affine zoom range |
| `random_perspective` | bool | `true` | Enable perspective distortion |
| `distortion_scale` | float | `0.3` | Strength of perspective distortion |
| `perspective_prob` | float | `0.5` | Probability for perspective augmentation |
| `random_rotation` | bool | `true` | Enable random rotation |
| `rotation_degrees` | float/int | `5` | Rotation limit |
| `color_jitter` | bool | `true` | Enable color jitter |
| `jitter_params` | list[float,float,float,float] | `[0.2, 0.2, 0.2, 0.1]` | Brightness/contrast/saturation/hue |
| `random_horizontal_flip` | bool | `true` | Enable horizontal flip |
| `flip_prob` | float | `0.5` | Horizontal flip probability |
| `random_crop` | bool | `false` | Enable random crop |

### MODEL block

| Key | Type | Example | Meaning |
|---|---|---|---|
| `MODEL.type` | string | `custom` | Model family implementation |
| `MODEL.backbone.type` | string | `mvit_v2_s` | Backbone architecture |
| `MODEL.neck.type` | string | `MV_Aggregate` | Feature aggregation module |
| `MODEL.neck.agr_type` | string | `max` | Aggregation mode (`max/mean/attention`) |
| `MODEL.head.type` | string | `MV_LinearLayer` | Classification head |
| `MODEL.pretrained_model` | string | `mvit_v2_s` | Pretrained source identifier |
| `MODEL.unfreeze_head` | bool | `true` | Whether to train head |
| `MODEL.unfreeze_last_n_layers` | int | `3` | Last N backbone layers to unfreeze |

### TRAIN block

| Key | Type | Example | Meaning |
|---|---|---|---|
| `TRAIN.enabled` | bool | `true` | Enable training mode |
| `TRAIN.epochs` | int | `20` | Total epochs |
| `TRAIN.monitor` | string | `balanced_accuracy` | Metric to monitor |
| `TRAIN.mode` | string | `max` | Monitor direction (`max/min`) |
| `TRAIN.log_interval` | int | `10` | Logging interval |
| `TRAIN.save_every` | int | `2` | Checkpoint interval |
| `TRAIN.use_weighted_sampler` | bool | `false` | Class balancing via sampler |
| `TRAIN.use_weighted_loss` | bool | `true` | Class balancing via loss weighting |
| `TRAIN.criterion.type` | string | `CrossEntropyLoss` | Loss function |
| `TRAIN.optimizer.type` | string | `AdamW` | Optimizer |
| `TRAIN.optimizer.lr` | float | `1e-4` | Global learning rate |
| `TRAIN.optimizer.backbone_lr` | float | `5e-5` | Backbone learning rate |
| `TRAIN.optimizer.head_lr` | float | `1e-3` | Head learning rate |
| `TRAIN.optimizer.weight_decay` | float | `1e-3` | L2 regularization |
| `TRAIN.scheduler.type` | string | `StepLR` | LR scheduler |
| `TRAIN.scheduler.step_size` | int | `3` | Epoch step before LR decay |
| `TRAIN.scheduler.gamma` | float | `0.1` | LR decay factor |

## Localization (`localization.yaml`, RGB end-to-end) Reference

### Additional global key

| Key | Type | Example | Meaning |
|---|---|---|---|
| `dali` | bool | `true` | Use DALI-based video data pipeline |

### DATA block

| Key | Type | Example | Meaning |
|---|---|---|---|
| `DATA.dataset_name` | string | `SoccerNet` | Dataset identity |
| `DATA.data_dir` | path | `/.../annotations` | Data root |
| `DATA.classes` | list[string] | `PASS, DRIVE, ...` | Event class set |
| `DATA.epoch_num_frames` | int | `500000` | Frames sampled per epoch |
| `DATA.mixup` | bool | `true` | Mixup augmentation |
| `DATA.modality` | string | `rgb` | Input modality |
| `DATA.crop_dim` | int | `-1` | Crop configuration |
| `DATA.dilate_len` | int | `0` | Label dilation for events |
| `DATA.clip_len` | int | `100` | Temporal clip length |
| `DATA.input_fps` | int | `25` | Source FPS |
| `DATA.extract_fps` | int | `2` | Effective FPS after sampling |
| `DATA.imagenet_mean` | list[float] | `[0.485, 0.456, 0.406]` | Input normalization mean |
| `DATA.imagenet_std` | list[float] | `[0.229, 0.224, 0.225]` | Input normalization std |
| `DATA.target_height` | int | `224` | Resize height |
| `DATA.target_width` | int | `398` | Resize width |

### DATA split-specific keys for localization

| Key | Type | Example | Meaning |
|---|---|---|---|
| `DATA.train.type` | string | `VideoGameWithDali` | Training dataset loader class |
| `DATA.valid.type` | string | `VideoGameWithDali` | Validation dataset loader class |
| `DATA.valid_data_frames.type` | string | `VideoGameWithDaliVideo` | Frame-level validation/eval loader |
| `DATA.test.type` | string | `VideoGameWithDaliVideo` | Test loader |
| `DATA.test.results` | string | `results_spotting_test` | Output results folder name |
| `DATA.test.nms_window` | int | `2` | NMS window for postprocessing |
| `DATA.test.metric` | string | `tight` | Spotting metric mode |
| `DATA.<split>.overlap_len` | int | `0` / `50` | Sliding-window overlap for inference |
| `DATA.challenge.type` | string | `VideoGameWithDaliVideo` | Challenge split loader |

### MODEL block

| Key | Type | Example | Meaning |
|---|---|---|---|
| `MODEL.type` | string | `E2E` | End-to-end localization model |
| `MODEL.runner.type` | string | `runner_e2e` | Runner implementation |
| `MODEL.backbone.type` | string | `rny008_gsm` | Backbone network |
| `MODEL.head.type` | string | `gru` | Temporal head |
| `MODEL.multi_gpu` | bool | `true` | Distributed/multi-GPU flag |
| `MODEL.load_weights` | string/null | `null` | Optional checkpoint path |

### TRAIN block

| Key | Type | Example | Meaning |
|---|---|---|---|
| `TRAIN.type` | string | `trainer_e2e` | Trainer implementation |
| `TRAIN.num_epochs` | int | `10` | Epoch count |
| `TRAIN.acc_grad_iter` | int | `1` | Gradient accumulation steps |
| `TRAIN.base_num_valid_epochs` | int | `30` | Base validation schedule |
| `TRAIN.start_valid_epoch` | int | `4` | First validation epoch |
| `TRAIN.valid_map_every` | int | `1` | mAP validation interval |
| `TRAIN.criterion_valid` | string | `map` | Validation criterion |
| `TRAIN.criterion.type` | string | `CrossEntropyLoss` | Loss function |
| `TRAIN.optimizer.type` | string | `AdamWithScaler` | Optimizer |
| `TRAIN.optimizer.lr` | float | `0.01` | Learning rate |
| `TRAIN.scheduler.type` | string | `ChainedSchedulerE2E` | Scheduler strategy |
| `TRAIN.scheduler.warm_up_epochs` | int | `3` | Warmup duration |

## Localization JSON Feature Configs (`localization-json_*`) Reference

These configs target pre-extracted feature pipelines (for example ResNET PCA512 features) instead of raw RGB clip ingestion.

### Data and model differences vs RGB E2E

| Key family | NetVLAD++ config | CALF config | Meaning |
|---|---|---|---|
| Train dataset `type` | `FeatureClipsfromJSON` | `FeatureClipChunksfromJson` | Feature clip/chunk loader |
| Test dataset `type` | `FeatureVideosfromJSON` | `FeatureVideosChunksfromJson` | Feature video/chunked video loader |
| Temporal key | `window_size` | `chunk_size` + `receptive_field` | Temporal context strategy |
| Model `type` | `LearnablePooling` | `ContextAware` | Localization architecture |
| Neck | `NetVLAD++` | `CNN++` | Feature aggregation style |
| Head | `LinearLayer` | `SpottingCALF` | Event prediction head |
| Trainer | `trainer_pooling` | `trainer_CALF` | Training loop implementation |
| Criterion | `NLLLoss` | `Combined2x` (ContextAwareLoss + SpottingLoss) | Loss setup |

### Key tuning points for JSON feature configs

| Key | Recommendation |
|---|---|
| `DATA.train.dataloader.batch_size` | Start high for feature pipelines, reduce if OOM |
| `TRAIN.optimizer.lr` | Tune first; often most sensitive parameter |
| `TRAIN.max_epochs` | Keep smaller for quick iteration, larger for final runs |
| `MODEL.neck.vocab_size` (NetVLAD++) | Higher can improve capacity but increases compute |
| `DATA.train.chunk_size/window_size` | Controls temporal context and memory footprint |

## SN-GAR Config Highlights (`sngar-tracking.yaml`, `sngar-frames.yaml`)

Use these when working on group activity recognition.

### Tracking config highlights

- `DATA.data_modality: tracking_parquet`
- Graph backbone keys:
  - `MODEL.backbone.encoder` (`gin` etc.)
  - `MODEL.edge`, `MODEL.k`, `MODEL.r` for graph connectivity
- Motion/object normalization keys:
  - `DATA.num_objects`, `DATA.feature_dim`, `DATA.max_displacement`, `DATA.max_ball_height`

### Frames config highlights

- `DATA.data_modality: frames_npy`
- Video transformer backbone keys:
  - `MODEL.backbone.type: videomae2`
  - `MODEL.backbone.pretrained_model`
  - `MODEL.backbone.freeze`, `MODEL.backbone.unfreeze_last_n_layers`
- Mixed precision support:
  - `TRAIN.use_amp: true`

## Running Many Experiments Without Creating New YAML Files

If your API call expects a YAML path, keep one base YAML and generate a temporary YAML per run from Python overrides.

```python
import tempfile
import yaml
from copy import deepcopy

from opensportslib import model
from opensportslib.core.utils.config import load_config, namespace_to_dict


def set_nested(d, path, value):
    keys = path.split('.')
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            raise KeyError(f"Invalid key path: {path} (failed at {k})")
        cur = cur[k]
    if keys[-1] not in cur:
        raise KeyError(f"Invalid key path: {path} (missing leaf {keys[-1]})")
    cur[keys[-1]] = value


def make_temp_config(base_yaml_path, overrides):
    cfg_ns = load_config(base_yaml_path)
    cfg = deepcopy(namespace_to_dict(cfg_ns))

    for key_path, value in overrides.items():
        set_nested(cfg, key_path, value)

    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.safe_dump(cfg, tmp, sort_keys=False)
    tmp.flush()
    tmp.close()
    return tmp.name


config_path = make_temp_config(
    "/home/vorajv/opensportslib/opensportslib/config/classification.yaml",
    {
        "TRAIN.optimizer.lr": 1e-4,
        "TRAIN.epochs": 30,
        "DATA.train.dataloader.batch_size": 4,
        "MODEL.backbone.type": "mvit_v2_s",
    },
)

m = model.classification(
    config=config_path,
    overrides={"DATA.data_dir": "$HOME/opensportslib/SoccerNet/mvfouls"},
)
```

## Recommended Tuning Order

When starting a new experiment, tune in this order:

1. `TRAIN.optimizer.lr`
2. `DATA.train.dataloader.batch_size`
3. `TRAIN.epochs` or `TRAIN.num_epochs` / `TRAIN.max_epochs`
4. `MODEL.backbone.type` and unfreeze settings
5. Scheduler settings

This gives the highest return with minimal config churn.

## Reproducibility Checklist

- Keep a stable base YAML per task.
- Store all per-run changes in an explicit Python `overrides` dict.
- Save/log the resolved YAML used for the run.
- Fix seed when comparing experiments (`SYSTEM.seed`, plus deterministic settings if needed).
