# OpenSportsLib Configuration Guide

Canonical configuration authoring reference for training, inference, and evaluation.

See also:
- [Legacy to Canonical Mapping](legacy-to-canonical-mapping.md)
- [Configuration Developer Guide](developer-guide.md)

## Canonical Config Examples

Use production-ready canonical templates from:
- [opensportslib/configs/](../../opensportslib/configs/)
- [opensportslib/configs/classification/](../../opensportslib/configs/classification/)
- [opensportslib/configs/localization/](../../opensportslib/configs/localization/)

## 1) Canonical Contract

- Runtime consumes canonical config only.
- Legacy config is accepted only at ingestion and migrated once.
- Canonical payloads containing legacy aliases are rejected.

## 2) Top-Level Schema

```yaml
TASK: <classification|localization|retrieval|captioning|reasoning>
VERSION: 2

SYSTEM: <SystemSchema>
DATA: <DataSchema>
MODEL: <ModelSchema>
TRAIN: <TrainSchema>
IO: <IoSchema>
```

### 2.1 Top-level key matrix

| Key | Type | Required | Default | Allowed values | Owner | Runtime consumer / validator notes |
|---|---|---|---|---|---|---|
| `TASK` | string | yes | none | `classification`, `localization`, `retrieval`, `captioning`, `reasoning` | config author | Used for task routing and migration decisions. |
| `VERSION` | int | yes | none | currently canonical payloads use `2` | config policy | Required section by validator; compatibility marker retained. |
| `SYSTEM` | object | yes | none | see SYSTEM section | platform/runtime | Required section by validator. |
| `DATA` | object | yes | none | see DATA section | data pipeline | Required section by validator. |
| `MODEL` | object | yes | none | see MODEL section | model/runtime | Required section; `MODEL.components` must be non-empty. |
| `TRAIN` | object | yes | none | see TRAIN section | trainer/runtime | Required section by validator. |
| `IO` | object | no | omitted | see IO section | model interface | If provided, mappings must reference existing components. |

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

| Key | Type | Required | Default | Allowed values | Owner | Runtime consumer / validator notes |
|---|---|---|---|---|---|---|
| `SYSTEM.paths.log_dir` | string | no | `./logs` | any path | platform/runtime | Accessed by system path helpers. |
| `SYSTEM.paths.save_dir` | string | no | `./checkpoints` | any path | platform/runtime | Accessed by system path helpers. |
| `SYSTEM.paths.work_dir` | string | no | `save_dir` | any path | platform/runtime | Accessed by system path helpers. |
| `SYSTEM.device` | string | no | `auto` | `auto`, `cpu`, `cuda` | runtime | Preferred device mode token. |
| `SYSTEM.gpu.count` | int | no | `0` | `>=0` | runtime | Read by GPU-count accessor. |
| `SYSTEM.gpu.id` | int | no | `0` | `>=0` | runtime | Single-device selector. |
| `SYSTEM.reproducibility.use_seed` | bool | no | `false` | `true`, `false` | runtime | Read by deterministic setup helper. |
| `SYSTEM.reproducibility.seed` | int | no | `42` | any int | runtime | Read by seed accessor. |

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

| Key | Type | Required | Default | Allowed values | Owner | Runtime consumer / validator notes |
|---|---|---|---|---|---|---|
| `dataset_name` | string | yes | none | any | data pipeline | Dataset identity token. |
| `data_root` | string/null | no | `null` | path or `null` | data pipeline | Optional base path. |
| `classes` | list[string] | no | `[]` | label names | task owner | Used to derive `num_classes` when present. |
| `runtime.loader_backend` | string | yes | `opencv` | `opencv`, `dali` | runtime | Read by loader backend accessor (`get_loader_backend`). |

### 4.2 `DATA.common.splits.<split>`

| Key | Type | Required | Default | Allowed values | Owner | Runtime consumer / validator notes |
|---|---|---|---|---|---|---|
| `annotation_path` | string | no | none | path | data pipeline | Read by split annotation accessor. |
| `source_path` | string | no | none | path | data pipeline | Read by split source accessor. |
| `type` | string | no | none | task-specific dataset type token | task owner | Used by dataset builder selection in some flows. |
| `dataloader` | object | no | `{}` | dataloader options | trainer/data | Read by split dataloader accessor. |
| `results` | string | no | none | output tag | eval/infer | Read by split result-name accessor. |
| `metric` | string | no | none | task-specific metric mode | eval owner | Common in localization eval configs. |
| `nms_window` | int | no | none | `>=0` | localization owner | Localization post-processing control. |
| `overlap_len` | int | no | none | `>=0` | localization owner | Sliding window overlap control. |

### 4.3 `DATA.inputs.<input_name>`

| Key | Type | Required | Default | Allowed values | Owner | Runtime consumer / validator notes |
|---|---|---|---|---|---|---|
| `modality` | string | yes | none | `video`, `tracking`, `text`, `audio`, `custom` | model/data | Used by modality accessors and runtime adaptation. |
| `representation` | string | yes | none | `raw`, `features`, `frames_npy`, `graph`, `custom` | model/data | Secondary modality/format signal. |
| `source.format` | string | yes | none | `mp4`, `npy`, `parquet`, `json`, `custom` | data pipeline | Payload format contract for loaders. |
| `sampling` | object | no | `{}` | numeric/time controls | task owner | Read by data sampling accessor. |
| `transform` | object | no | `{}` | transform knobs | task owner | Read by transform accessor. |
| `augmentations` | object | no | `{}` | augmentation knobs | task owner | Read by augmentation accessor. |
| `params` | object | no | `{}` | task-specific metadata | task owner | Read by params accessor (`get_data_params`). |

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
      load: {}

  topology:
    - from: <component_id>
      to: <component_id>
      map: {<src_key>: <dst_key>}   # optional
      merge: <none|concat|sum|cross_attn|custom>   # optional

  policies: {}
  metadata: {}
```

### 5.1 `MODEL` key matrix

| Key | Type | Required | Default | Allowed values | Owner | Runtime consumer / validator notes |
|---|---|---|---|---|---|---|
| `runtime.dtype` | string | no | `fp32` | `fp32`, `fp16`, `bf16` | runtime/model | Runtime precision hint. |
| `runtime.device` | string | no | `auto` | `auto`, `cpu`, `cuda`, `ddp` | runtime/model | Device override hint. |
| `runtime.compile` | bool | no | `false` | `true`, `false` | runtime/model | Compile toggle hint. |
| `runtime.freeze` | bool | no | `false` | `true`, `false` | model owner | Freezing policy hint. |
| `runtime.multi_gpu` | bool | no | `false` | `true`, `false` | compatibility | Compatibility marker; `TRAIN.execution.multi_gpu` is canonical owner. |
| `load.checkpoint_path` | string/null | no | `null` | path or `null` | runtime/model | Global checkpoint load source. |
| `load.pretrained` | bool | no | `false` | `true`, `false` | model owner | Pretrained toggle hint. |
| `load.strict` | bool | no | `true` | `true`, `false` | runtime/model | State dict strictness hint. |
| `load.map_location` | string/null | no | `null` | `cpu`, `cuda`, `null` | runtime/model | Device remap hint for checkpoint loading. |
| `load.format` | string | no | `auto` | `auto`, `custom` | runtime/model | Loader format hint. |
| `components` | mapping | yes | none | non-empty mapping | model owner | Required by validator; each component requires `source.provider`. |
| `topology` | list[edge] | no | `[]` | DAG edges | model owner | Validator ensures edges reference known components and graph is acyclic. |
| `policies` | object | no | `{}` | free-form | model owner | Policy hooks for advanced runtimes. |
| `metadata` | object | no | `{}` | free-form with common fields below | model owner | Used by helper accessors for family/runner fallbacks. |

### 5.2 `MODEL.components.<component_id>` matrix

| Key | Type | Required | Default | Allowed values | Owner | Runtime consumer / validator notes |
|---|---|---|---|---|---|---|
| `<component_id>` | key token | yes | none | lowercase snake_case-like token | model owner | Validator enforces alnum/underscore and no leading underscore. |
| `kind` | string | yes | none | `encoder`, `decoder`, `fusion`, `adapter`, `projector`, `head`, `postprocessor`, `custom` | model owner | Used by kind-based accessors. |
| `source.provider` | string | yes | none | `opensportslib`, `huggingface`, `torchvision`, `timm`, `torch`, `custom` | model owner | Validator enforces existence. |
| `source.registry` | string | no | none | provider-specific | model owner | Commonly used with `opensportslib`. |
| `source.name` | string | provider-dependent | none | provider-specific | model owner | Used by name accessors and type fallback. |
| `source.repo_id` | string | provider-dependent | none | repository ID | model owner | Typical for `huggingface`. |
| `source.revision` | string | no | none | revision ref | model owner | Optional provider revision pin. |
| `source.entrypoint` | string | provider-dependent | none | Python entrypoint path | model owner | Required for `custom` provider policy. |
| `params` | object | no | `{}` | free-form | model owner | Read by parameter accessors. |
| `overrides` | object | no | `{}` | free-form | model owner | Merged over `params` by helper accessors. |
| `load` | object | no | `{}` | free-form | runtime/model | Optional component-level load overrides. |

### 5.3 `MODEL.metadata` commonly used fields

| Key | Type | Required | Default / fallback | Notes |
|---|---|---|---|---|
| `metadata.family` | string | no | inferred from task/trainer when absent | Used by model-family helper. |
| `metadata.legacy_type` | string | no | takes precedence over `family` when present | Compatibility marker for legacy family naming. |
| `metadata.runner.type` | string | no | inferred from `TRAIN.trainer.type` | Used by runner-type helper. |

### 5.4 Provider guidance

- `huggingface`: provide at least one of `repo_id` or `name`.
- `custom`: provide `entrypoint`.
- `opensportslib`: prefer `registry + name`.
- `torchvision` / `timm` / `torch`: provide `name`.

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

### 6.1 `TRAIN` key matrix

| Key | Type | Required | Default | Allowed values | Owner | Runtime consumer / validator notes |
|---|---|---|---|---|---|---|
| `trainer.type` | string | yes | `classification` | `classification`, `trainer_e2e`, `trainer_pooling`, `trainer_calf`, `custom` | trainer owner | Read by trainer-type accessor; influences runner fallback. |
| `epochs` | int | yes | `1` runtime fallback | `>=1` | trainer owner | Read by `get_train_epochs`. |
| `criterion` | object | no | `{}` | loss config | trainer owner | Training loss config block. |
| `optimizer` | object | no | `{}` | optimizer config | trainer owner | Optimizer config block. |
| `scheduler` | object | no | `{}` | scheduler config | trainer owner | Scheduler config block. |
| `execution` | object | no | `{}` | execution knobs below | trainer/runtime | Read by execution accessor. |
| `sampling` | object | no | `{}` | sampling knobs below | trainer/runtime | Read by sampling accessor. |
| `selection` | object | no | `{}` | selection knobs below | trainer/runtime | Read by selection accessor. |
| `checkpoint` | object | no | `{}` | checkpoint knobs below | trainer/runtime | Read by checkpoint accessor. |

### 6.2 `TRAIN.execution` commonly used keys

| Key | Type | Default | Allowed values | Notes |
|---|---|---|---|---|
| `multi_gpu` | bool | `false` | `true`, `false` | Canonical owner for multi-GPU execution policy. |
| `log_interval` | int | task-dependent | `>=1` | Training log cadence. |
| `acc_grad_iter` | int | `1` | `>=1` | Gradient accumulation steps. |
| `evaluation_frequency` | int | task-dependent | `>=1` | Validation cadence in epochs. |
| `start_valid_epoch` | int | task-dependent | `>=0` | First epoch to run validation. |
| `valid_map_every` | int | task-dependent | `>=1` | Localization mAP evaluation cadence. |
| `criterion_valid` | string | `loss` | `loss`, `map` | Validation criterion selector. |

Authoring rule:
- Keep general hyperparameters in canonical `TRAIN` keys such as `TRAIN.epochs`, `TRAIN.optimizer`, and `TRAIN.scheduler`.
- Keep model identity and architecture values in `MODEL.components.*` and `MODEL.runtime`.
- Use `TRAIN.execution` only for runtime/backend-specific controls; do not duplicate learning rate, epoch count, model id, hidden size, or precision here when a canonical owner already exists.
- Backend adapters may translate canonical values into backend-specific runtime defaults. For the X-VARS VideoChatGPT backend, canonical prompt/video-token authoring can remain single-source while train and infer adapters preserve the original upstream defaults where they differ.

CPU behavior note:
`TRAIN.execution.multi_gpu` requires CUDA. If effective runtime device is CPU,
localization runtime automatically forces `multi_gpu=false`.

### 6.3 `TRAIN.selection.mode` values

| Value | Meaning |
|---|---|
| `min` | lower metric is better |
| `max` | higher metric is better |

### 6.4 Trainer-to-runner expectation

| `TRAIN.trainer.type` | Default runner fallback |
|---|---|
| `classification` | `runner_classification` |
| `trainer_e2e` | `runner_e2e` |
| `trainer_calf` | `runner_JSON` |
| `trainer_pooling` | `runner_JSON` |
| `custom` | `runner_classification` (unless `MODEL.metadata.runner.type` is explicitly set) |

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

### 7.1 IO key matrix

| Key | Type | Required | Default | Allowed values | Owner | Runtime consumer / validator notes |
|---|---|---|---|---|---|---|
| `IO.inputs` | mapping | no | `{}` | `<public_input_name> -> <component_id>` | model interface owner | Validator checks each component ID exists in `MODEL.components`. |
| `IO.outputs` | mapping | no | `{}` | `<public_output_name> -> <component_id>` | model interface owner | Validator checks each component ID exists in `MODEL.components`. |

Use `IO` when:
- there are multiple roots,
- multiple exposed outputs,
- custom component signatures require explicit routing.

## 8) Validation and Rejection Rules

### 8.1 Enforced by validator/runtime code

- Top-level required sections must exist: `TASK`, `VERSION`, `SYSTEM`, `DATA`, `MODEL`, `TRAIN`.
- Payload must resolve to canonical schema.
- Legacy aliases are rejected in strict canonical runtime.
- `MODEL.components` must be a non-empty mapping.
- Every component must define `source.provider`.
- Component IDs must be valid canonical tokens (alnum/underscore, no leading underscore).
- Every topology edge must reference existing components.
- `MODEL.topology` must be acyclic.
- If `IO.inputs`/`IO.outputs` are present, each mapped component must exist.

### 8.2 Policy-level recommendations (not strictly enforced everywhere)

- Set `VERSION: 3` and `MODEL.schema_version: 3` for canonical authoring.
- Keep `MODEL.task` semantically equal to `TASK`.
- Prefer `TRAIN.execution.multi_gpu` as single owner for execution policy.

### 8.3 Compatibility caveat

- Migration can ingest legacy split-path aliases at the ingestion boundary.
- Strict canonical payloads must use only canonical split keys.
- For explicit legacy key spellings, see [Legacy to Canonical Mapping](legacy-to-canonical-mapping.md).

## 9) Practical Templates

### 9.1 Classification (minimal canonical)

```yaml
TASK: classification
VERSION: 2

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
  metadata: {family: custom, runner: {type: runner_classification}}

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

### 9.2 Localization (expanded canonical)

```yaml
TASK: localization
VERSION: 2

SYSTEM:
  paths: {log_dir: ./logs, save_dir: ./checkpoints, work_dir: ./checkpoints}
  device: cuda
  gpu: {count: 4, id: 0}
  reproducibility: {use_seed: true, seed: 42}

DATA:
  common:
    dataset_name: SoccerNet
    classes: [PASS, DRIVE, HEADER, HIGH_PASS, OUT, CROSS, THROW_IN, SHOT, BALL_PLAYER_BLOCK, PLAYER_SUCCESSFUL_TACKLE, FREE_KICK, GOAL]
    runtime: {loader_backend: dali}
    splits:
      train:
        annotation_path: /data/train.json
        source_path: /data/train
        type: VideoGameWithDali
        dataloader: {batch_size: 8, shuffle: true, num_workers: 4}
      valid:
        annotation_path: /data/valid.json
        source_path: /data/valid
        type: VideoGameWithDali
        dataloader: {batch_size: 8, shuffle: false, num_workers: 4}
      test:
        annotation_path: /data/test.json
        source_path: /data/test
        type: VideoGameWithDaliVideo
        results: results_spotting_test
        metric: tight
        nms_window: 2
        overlap_len: 50
  inputs:
    video:
      modality: video
      representation: raw
      source: {format: mp4}
      sampling: {clip_len: 100, input_fps: 25, extract_fps: 2}
      transform: {resize: {height: 224, width: 398}}

MODEL:
  runtime: {dtype: fp32, device: cuda, compile: false, freeze: false, multi_gpu: true}
  load: {checkpoint_path: null, pretrained: false, strict: true, map_location: null, format: auto}
  components:
    video_encoder:
      kind: encoder
      source: {provider: opensportslib, registry: backbone, name: rny008_gsm}
      params: {}
    task_head:
      kind: head
      source: {provider: opensportslib, registry: head, name: gru}
      params: {}
    event_postprocessor:
      kind: postprocessor
      source: {provider: opensportslib, registry: post_proc, name: nms}
      params: {window: 2}
  topology:
    - {from: video_encoder, to: task_head}
    - {from: task_head, to: event_postprocessor}
  metadata: {family: E2E, runner: {type: runner_e2e}}

IO:
  inputs: {video: video_encoder}
  outputs: {logits: task_head, events: event_postprocessor}

TRAIN:
  trainer: {type: trainer_e2e}
  epochs: 10
  criterion: {type: CrossEntropyLoss}
  optimizer: {type: AdamWithScaler, lr: 0.01}
  scheduler: {type: ChainedSchedulerE2E, acc_grad_iter: 1, num_epochs: 10, warm_up_epochs: 3}
  execution:
    multi_gpu: true
    acc_grad_iter: 1
    log_interval: 10
    evaluation_frequency: 1
    start_valid_epoch: 4
    valid_map_every: 1
    criterion_valid: map
  sampling: {batch_size: 8, use_weighted_sampler: false, use_weighted_loss: false}
  selection: {monitor: map, mode: max}
  checkpoint: {save_every: 2, save_best: true}
```
