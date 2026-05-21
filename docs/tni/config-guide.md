# OpenSportsLib Canonical Config Guide

This guide is the single source of truth for canonical config authoring.

## 1) Canonical Contract

- Runtime consumes canonical config only.
- Legacy config is accepted only at ingestion and migrated once.
- Canonical payloads containing legacy aliases are rejected.

## 2) Top-Level Schema

```yaml
TASK: <classification|localization|retrieval|captioning|reasoning>
VERSION: 3

SYSTEM: <SystemSchema>
DATA: <DataSchema>
MODEL: <ModelSchema>
TRAIN: <TrainSchema>
IO: <IoSchema>
```

### Required top-level keys
- `TASK`, `VERSION`, `SYSTEM`, `DATA`, `MODEL`, `TRAIN`

### Optional top-level keys
- `IO`

## 3) SYSTEM Schema

```yaml
SYSTEM:
  paths:
    log_dir: ./logs
    save_dir: ./checkpoints
    work_dir: ./checkpoints
  device: auto
  gpu:
    count: 0
    id: 0
  reproducibility:
    use_seed: false
    seed: 42
```

| Key | Type | Allowed values | Default | Notes |
|---|---|---|---|---|
| `SYSTEM.paths.log_dir` | string | any path | `./logs` | logs/artifacts |
| `SYSTEM.paths.save_dir` | string | any path | `./checkpoints` | checkpoints |
| `SYSTEM.paths.work_dir` | string | any path | `save_dir` | eval outputs/temp |
| `SYSTEM.device` | string | `auto`, `cpu`, `cuda` | `auto` | runtime device mode |
| `SYSTEM.gpu.count` | int | `>=0` | `0` | device count hint |
| `SYSTEM.gpu.id` | int | `>=0` | `0` | single-device index |
| `SYSTEM.reproducibility.use_seed` | bool | `true/false` | `false` | deterministic mode |
| `SYSTEM.reproducibility.seed` | int | any int | `42` | seed value |

## 4) DATA Schema

```yaml
DATA:
  common:
    dataset_name: <string>
    data_root: <path|null>
    classes: [<label>, ...]
    runtime:
      loader_backend: <opencv|dali>
    splits:
      train:
        annotation_path: <path>
        source_path: <path>
      valid:
        annotation_path: <path>
        source_path: <path>
      test:
        annotation_path: <path>
        source_path: <path>

  inputs:
    <input_name>:
      modality: <video|tracking|text|audio|custom>
      representation: <raw|features|frames_npy|graph|custom>
      source:
        format: <mp4|npy|parquet|json|custom>
      sampling: {}
      transform: {}
      augmentations: {}
      params: {}
```

### 4.1 `DATA.common`

| Key | Type | Allowed values | Required | Notes |
|---|---|---|---|---|
| `dataset_name` | string | any | yes | logical dataset id |
| `data_root` | string/null | path or null | no | optional global root |
| `classes` | list[string] | label names | no | classification/localization labels |
| `runtime.loader_backend` | string | `opencv`, `dali` | yes | loader implementation |

### 4.2 `DATA.common.splits.<split>`

| Key | Type | Required | Notes |
|---|---|---|---|
| `annotation_path` | string | no | split annotation source |
| `source_path` | string | no | split media/feature base path |
| `type` | string | no | dataset-class selector for legacy dataset builders |
| `dataloader` | object | no | split dataloader config |
| `results` | string | no | output name for inference/evaluation |
| `metric` | string | no | eval mode (task-specific) |
| `nms_window` | int | no | localization-specific |
| `overlap_len` | int | no | clip overlap for inference |

### 4.3 `DATA.inputs.<input_name>`

| Key | Type | Allowed values | Required | Notes |
|---|---|---|---|---|
| `modality` | string | `video`, `tracking`, `text`, ... | yes | semantic modality |
| `representation` | string | `raw`, `features`, `frames_npy`, `graph`, ... | yes | storage/feature style |
| `source.format` | string | `mp4`, `npy`, `parquet`, ... | yes | file payload format |
| `sampling` | object | free-form numeric fields | no | temporal sampling knobs |
| `transform` | object | resize/norm etc | no | deterministic transforms |
| `augmentations` | object | augmentation toggles/params | no | train-time aug |
| `params` | object | task-specific | no | extra input metadata |

### 4.4 Common sampling keys (convention)

| Key | Type | Typical range |
|---|---|---|
| `num_frames` | int | 1..512 |
| `clip_len` | int | 1..512 |
| `input_fps` | int/float | >0 |
| `target_fps` | int/float | >0 |
| `extract_fps` | int/float | >0 |
| `window_size` | int | >0 |
| `chunk_size` | int | >0 |
| `receptive_field` | int | >=0 |
| `start_frame` | int | >=0 |
| `end_frame` | int | > start_frame |
| `overlap_len` | int | 0..clip_len-1 |

### 4.5 Common transform keys

```yaml
transform:
  resize:
    height: 224
    width: 224
  normalization:
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
```

### 4.6 Common dataloader keys (split-level)

```yaml
dataloader:
  batch_size: 8
  shuffle: true
  num_workers: 4
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 4
  mp_context: spawn
```

## 5) MODEL Schema

```yaml
MODEL:
  schema_version: 3
  task: <same as TASK>

  runtime:
    dtype: <fp32|fp16|bf16>
    device: <auto|cpu|cuda|ddp>
    compile: <bool>
    freeze: <bool>
    multi_gpu: <bool>   # compatibility marker; policy owner is TRAIN.execution.multi_gpu

  load:
    checkpoint_path: <path|null>
    pretrained: <bool>
    strict: <bool>
    map_location: <cpu|cuda|null>
    format: <auto|custom>

  components:
    <component_id>:
      kind: <encoder|decoder|fusion|adapter|projector|head|postprocessor|custom>
      source:
        provider: <opensportslib|huggingface|torchvision|timm|torch|custom>
        registry: <optional-string>
        name: <optional-string>
        repo_id: <optional-string>
        revision: <optional-string>
        entrypoint: <optional-string>
      params: {}
      overrides: {}

  topology:
    - from: <component_id>
      to: <component_id>
      map: {<src_key>: <dst_key>}   # optional
      merge: <none|concat|sum|cross_attn|custom>   # optional

  policies: {}
  metadata: {}
```

### 5.1 Component naming
- `component_id` should be lowercase snake_case.
- Prefer semantic IDs: `video_encoder`, `task_head`, `event_postprocessor`.
- Do not encode vendor/model names in `component_id`.

### 5.2 `kind` values
- `encoder`, `decoder`, `fusion`, `adapter`, `projector`, `head`, `postprocessor`, `custom`

### 5.3 Provider rules
- `huggingface`: require at least one of `repo_id` or `name`.
- `custom`: require `entrypoint`.
- `opensportslib`: prefer `registry + name`.
- `torchvision` / `timm` / `torch`: require `name`.

### 5.4 Topology rules
- Every `from`/`to` node must exist in `components`.
- Graph must be acyclic.
- For multi-root or ambiguous routing, define `IO` explicitly.

## 6) TRAIN Schema

```yaml
TRAIN:
  trainer:
    type: <classification|trainer_e2e|trainer_pooling|trainer_calf|custom>

  epochs: 20

  criterion:
    type: CrossEntropyLoss

  optimizer:
    type: AdamW
    lr: 0.0001

  scheduler:
    type: StepLR
    step_size: 3
    gamma: 0.1

  execution:
    multi_gpu: false
    log_interval: 10
    acc_grad_iter: 1
    evaluation_frequency: 1
    base_num_valid_epochs: 30
    start_valid_epoch: 4
    valid_map_every: 1
    criterion_valid: loss

  sampling:
    batch_size: 8
    use_weighted_sampler: false
    use_weighted_loss: false

  selection:
    monitor: loss
    mode: min

  checkpoint:
    save_every: 2
    save_best: true
```

### Important ownership
- `TRAIN.execution.multi_gpu` is the canonical execution owner.

### Key options
| Key | Type | Common values |
|---|---|---|
| `trainer.type` | string | `classification`, `trainer_e2e`, `trainer_pooling`, `trainer_CALF` |
| `selection.monitor` | string | `loss`, `balanced_accuracy`, `map` |
| `selection.mode` | string | `min`, `max` |
| `criterion_valid` | string | `loss`, `map` |

## 7) IO Schema

```yaml
IO:
  inputs:
    video: video_encoder
    text: text_encoder
  outputs:
    logits: task_head
    events: event_postprocessor
```

Use `IO` when:
- there are multiple roots,
- multiple exposed outputs,
- custom component signatures require explicit routing.

## 8) Validation and Rejection Rules

Canonical validation enforces:
- required sections exist,
- `MODEL.task` equals `TASK`,
- component graph validity,
- no legacy aliases in canonical payload.

Forbidden in canonical payload (examples):
- top-level `dali`
- `DATA.annotations.*`
- `DATA.<split>.path` / `video_path`
- `MODEL.backbone` / `neck` / `head` / `post_proc`
- `TRAIN.num_epochs` / `TRAIN.max_epochs`

## 9) Migration Mapping Quick Reference

| Legacy | Canonical |
|---|---|
| `dali` | `DATA.common.runtime.loader_backend` |
| `DATA.<split>.path` | `DATA.common.splits.<split>.annotation_path` |
| `DATA.<split>.video_path` | `DATA.common.splits.<split>.source_path` |
| `MODEL.backbone` | `MODEL.components.*(kind=encoder)` |
| `MODEL.neck` | `MODEL.components.*(kind=adapter)` |
| `MODEL.head` | `MODEL.components.*(kind=head)` |
| `MODEL.post_proc` | `MODEL.components.*(kind=postprocessor)` |
| `TRAIN.num_epochs` / `TRAIN.max_epochs` | `TRAIN.epochs` |

## 10) Practical Templates

### Classification (minimal)

```yaml
TASK: classification
VERSION: 3

SYSTEM:
  paths: {log_dir: ./logs, save_dir: ./checkpoints, work_dir: ./checkpoints}
  device: auto
  gpu: {count: 1, id: 0}
  reproducibility: {use_seed: true, seed: 42}

DATA:
  common:
    dataset_name: mvfouls
    data_root: /data
    classes: [A, B]
    runtime: {loader_backend: opencv}
    splits:
      train: {annotation_path: /data/train.json, source_path: /data/train}
      valid: {annotation_path: /data/valid.json, source_path: /data/valid}
      test: {annotation_path: /data/test.json, source_path: /data/test}
  inputs:
    video:
      modality: video
      representation: raw
      source: {format: mp4}
      sampling: {num_frames: 16, input_fps: 25, target_fps: 17}
      transform:
        resize: {height: 224, width: 224}

MODEL:
  schema_version: 3
  task: classification
  runtime: {dtype: fp32, device: auto, compile: false, freeze: false, multi_gpu: false}
  load: {checkpoint_path: null, pretrained: false, strict: true, map_location: null, format: auto}
  components:
    video_encoder:
      kind: encoder
      source: {provider: opensportslib, registry: backbone, name: mvit_v2_s}
      params: {}
      overrides: {}
    task_head:
      kind: head
      source: {provider: opensportslib, registry: head, name: MV_LinearLayer}
      params: {num_classes: 2}
      overrides: {}
  topology:
    - {from: video_encoder, to: task_head}
  policies: {}
  metadata: {family: custom, runner: {type: classification}}

TRAIN:
  trainer: {type: classification}
  epochs: 20
  criterion: {type: CrossEntropyLoss}
  optimizer: {type: AdamW, lr: 0.0001}
  scheduler: {type: StepLR, step_size: 3, gamma: 0.1}
  execution: {multi_gpu: false, acc_grad_iter: 1, log_interval: 10, criterion_valid: loss}
  sampling: {batch_size: 8, use_weighted_sampler: false, use_weighted_loss: false}
  selection: {monitor: balanced_accuracy, mode: max}
  checkpoint: {save_every: 2, save_best: true}
```

### Localization (feature/video spotting)

Use same top-level structure with:
- `DATA.common.runtime.loader_backend: dali|opencv`
- split-level fields such as `results`, `metric`, `nms_window`, `overlap_len`
- `MODEL` components including optional `postprocessor`
- `TRAIN.trainer.type` aligned with localization pipeline (`trainer_e2e`, `trainer_pooling`, etc.)

## 11) Authoring Checklist

Before running:
1. `TASK` matches `MODEL.task`.
2. `TRAIN.epochs` is set (no `num_epochs`/`max_epochs`).
3. Split paths are canonical (`annotation_path`, `source_path`).
4. Model graph (`components` + `topology`) is valid.
5. `TRAIN.execution.multi_gpu` is set as needed.
6. No forbidden legacy keys in canonical files.

## 12) Supported Type Catalogs (Concrete)

This section lists concrete `type` values currently wired in the codebase.

### 12.1 Runner Types (`MODEL.metadata.runner.type`)

Used mainly by localization trainer/inference routing.

| Runner type | Primary behavior |
|---|---|
| `runner_classification` | Classification pipeline |
| `runner_JSON` | Localization from JSON feature/video sources |
| `runner_e2e` | End-to-end localization pipeline |
| `runner_CALF` | CALF-style localization pipeline |
| `runner_pooling` | Learnable pooling localization pipeline |

### 12.2 Dataset Split `type` Values (Localization)

Configured under `DATA.common.splits.<split>.type`.

| Dataset type | Typical use |
|---|---|
| `SoccerNetClips` | SoccerNet clip training |
| `SoccerNetGames` | SoccerNet game-level features/infer |
| `SoccerNetClipsCALF` | CALF training |
| `SoccerNetClipsTestingCALF` | CALF testing |
| `FeatureClipsfromJSON` | Feature clips from JSON |
| `FeatureVideosfromJSON` | Feature videos from JSON |
| `FeatureClipChunksfromJson` | Chunked feature clips |
| `FeatureVideosChunksfromJson` | Chunked feature videos |
| `VideoGameWithOpencv` | Video pipeline using OpenCV |
| `VideoGameWithOpencvVideo` | Video inference/eval using OpenCV |
| `VideoGameWithDali` | Video pipeline using NVIDIA DALI |
| `VideoGameWithDaliVideo` | Video inference/eval using NVIDIA DALI |

### 12.3 Backbone Types (`MODEL.components.* kind=encoder`)

Use as `source.name` for OpenSportsLib backbones (with component params as needed).

| Backbone type | Family |
|---|---|
| `graph_conv` | Tracking/graph encoder |
| `PreExtactedFeatures` | Pre-extracted feature passthrough |
| `rn18`, `rn18_tsm`, `rn18_gsm`, `rn50`, `rn50_tsm`, `rn50_gsm` | ResNet family |
| `rny002`, `rny002_tsm`, `rny002_gsm`, `rny008`, `rny008_tsm`, `rny008_gsm` | RegNetY family |
| `convnextt`, `convnextt_tsm`, `convnextt_gsm` | ConvNeXt-Tiny family |
| `r3d_18`, `mc3_18`, `r2plus1d_18`, `s3d`, `mvit_v2_s` | Torchvision video models |
| `dinov3`, `clip`, `videomae`, `videomae2` | Feature-extractor wrappers |
| `video_mae` | HuggingFace VideoMAE classification builder |

### 12.4 Neck Types (`MODEL.components.* kind=adapter`)

| Neck type | Purpose |
|---|---|
| `MV_Aggregate` | Multi-view aggregation |
| `TemporalAggregation` | Temporal aggregation (max/avg/attention/lstm/tcn) |
| `MaxPool`, `MaxPool++` | Pooling adapters |
| `AvgPool`, `AvgPool++` | Pooling adapters |
| `NetRVLAD`, `NetRVLAD++` | Learnable pooling |
| `NetVLAD`, `NetVLAD++` | Learnable pooling |
| `CNN++` | CALF-style temporal module |

### 12.5 Head Types (`MODEL.components.* kind=head`)

| Head type | Purpose |
|---|---|
| `TrackingClassifier` | Tracking classification head |
| `LinearLayer` | Linear spotting/classification head |
| `SpottingCALF` | CALF spotting head |
| `MV_LinearLayer` | Multi-view classification head |
| `gru`, `deeper_gru`, `mstcn`, `asformer`, `` (empty) | Temporal E2E head variants |

### 12.6 Postprocessor Types (`MODEL.components.* kind=postprocessor`)

| Type | Purpose |
|---|---|
| `NMS` | Non-maximum suppression for spotting events |

## 13) Type-Specific Parameter Hints

### 13.1 `TemporalAggregation` params
- `agr_type`: `maxpool`, `avgpool`, `attention`, `bilstm`, `tcn`
- `hidden_dim`: int
- `dropout`: float
- `use_position_encoding`: bool
- `num_attention_heads`: int (attention mode)
- `lstm_dropout`: float (bilstm mode)

### 13.2 `graph_conv` params
- `encoder`: graph conv variant (commonly `gin`)
- `hidden_dim`: int
- `num_layers`: int
- `dropout`: float
- graph extras often provided via params: `edge_type`, `k`, `r`

### 13.3 `SpottingCALF` params
- `dim_capsule`: int
- `num_detections`: int
- `chunk_size`: int
- `num_classes`: int

### 13.4 `NMS` postprocessing params
- `NMS_window`: int
- `NMS_threshold`: float

## 14) Recommended Authoring Pattern for Type Fields

In canonical authoring:
- put implementation choice in `MODEL.components.<id>.source.name`
- keep component role in `kind`
- keep algorithm hyperparameters under `params`
- keep runtime toggles under `MODEL.runtime` or `TRAIN.execution`

Example:

```yaml
MODEL:
  components:
    video_encoder:
      kind: encoder
      source:
        provider: opensportslib
        registry: backbone
        name: mvit_v2_s
      params: {}
    video_adapter:
      kind: adapter
      source:
        provider: opensportslib
        registry: neck
        name: TemporalAggregation
      params:
        agr_type: attention
        hidden_dim: 768
        num_attention_heads: 8
    task_head:
      kind: head
      source:
        provider: opensportslib
        registry: head
        name: MV_LinearLayer
      params:
        num_classes: 8
```
