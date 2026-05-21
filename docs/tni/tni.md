# Training & Inference

This section explains how to:

- Configure experiments
- Train models (single & multi-GPU)
- Run inference
- Use pretrained weights from HuggingFace

For full key-by-key config documentation and Python-only override workflow, see [Configuration Guide](config-guide.md).

---
## Configuration Sample (.yaml) file

The snippets below show the main structure of the runnable configs in
`opensportslib/config/`. Use the source files when you need the complete
experiment defaults.

### 1. Classification

```yaml
TASK: classification

DATA:
  dataset_name: mvfouls
  data_dir: /path/to/OSL-XFoul/224p
  data_modality: video
  view_type: multi
  train:
    video_path: ${DATA.data_dir}/train
    path: ${DATA.train.video_path}/train.json
    dataloader:
      batch_size: 8
      shuffle: true
      num_workers: 4
  valid:
    video_path: ${DATA.data_dir}/valid
    path: ${DATA.valid.video_path}/valid.json
    dataloader:
      batch_size: 1
      shuffle: false
  test:
    video_path: ${DATA.data_dir}/test
    path: ${DATA.test.video_path}/test.json
    dataloader:
      batch_size: 1
      shuffle: false
  num_frames: 16
  input_fps: 25
  target_fps: 17
  frame_size: [224, 224]

MODEL:
  type: custom
  backbone:
    type: mvit_v2_s
  neck:
    type: MV_Aggregate
    agr_type: max
  head:
    type: MV_LinearLayer

TRAIN:
  monitor: balanced_accuracy
  mode: max
  epochs: 20
  criterion:
    type: CrossEntropyLoss
  optimizer:
    type: AdamW
    lr: 0.0001

SYSTEM:
  save_dir: ./checkpoints
  device: cuda
  GPU: 4
```

### 2. Classification (Tracking)

```yaml
TASK: classification

DATA:
  dataset_name: sngar
  data_modality: tracking_parquet
  data_dir: /path/to/soccernetpro-classification-GAR/tracking-parquet
  train:
    video_path: ${DATA.data_dir}/train
    path: ${DATA.train.video_path}/train.json
    dataloader:
      batch_size: 32
      shuffle: true
  valid:
    video_path: ${DATA.data_dir}/valid
    path: ${DATA.valid.video_path}/valid.json
    dataloader:
      batch_size: 32
      shuffle: false
  test:
    video_path: ${DATA.data_dir}/test
    path: ${DATA.test.video_path}/test.json
    dataloader:
      batch_size: 32
      shuffle: false
  num_frames: 16
  frame_interval: 9
  normalize: true
  num_objects: 23
  feature_dim: 8

MODEL:
  type: custom
  backbone:
    type: graph_conv
    encoder: gin
    hidden_dim: 64
    num_layers: 20
  neck:
    type: TemporalAggregation
    agr_type: maxpool
  head:
    type: TrackingClassifier
    num_classes: 10
  edge: positional
  k: 8
  r: 15.0

TRAIN:
  monitor: loss
  mode: min
  epochs: 100
  optimizer:
    type: Adam
    lr: 0.001

SYSTEM:
  save_dir: ./checkpoints_tracking
  device: cuda
  GPU: 1
```

### 3. Localization

```yaml
TASK: localization
dali: true

DATA:
  dataset_name: SoccerNet
  data_dir: /path/to/OSL-SNBAS/224p-2024
  classes:
    - PASS
    - DRIVE
    - HEADER
    - HIGH PASS
    - OUT
    - CROSS
    - THROW IN
    - SHOT
    - BALL PLAYER BLOCK
    - PLAYER SUCCESSFUL TACKLE
    - FREE KICK
    - GOAL
  modality: rgb
  clip_len: 100
  input_fps: 25
  extract_fps: 2
  target_height: 224
  target_width: 398
  train:
    type: VideoGameWithDali
    video_path: ${DATA.data_dir}/train
    path: ${DATA.train.video_path}/train.json
    dataloader:
      batch_size: 8
      shuffle: true
  valid:
    type: VideoGameWithDali
    video_path: ${DATA.data_dir}/valid
    path: ${DATA.valid.video_path}/valid.json
    dataloader:
      batch_size: 8
      shuffle: true
  test:
    type: VideoGameWithDaliVideo
    video_path: ${DATA.data_dir}/test
    path: ${DATA.test.video_path}/test.json
    results: results_spotting_test
    nms_window: 2
    metric: tight
    overlap_len: 50

MODEL:
  type: E2E
  runner:
    type: runner_e2e
  backbone:
    type: rny008_gsm
  head:
    type: gru
  multi_gpu: true

TRAIN:
  type: trainer_e2e
  num_epochs: 10
  criterion_valid: map
  criterion:
    type: CrossEntropyLoss
  optimizer:
    type: AdamWithScaler
    lr: 0.01

SYSTEM:
  save_dir: ./checkpoints
  work_dir: ${SYSTEM.save_dir}
  device: cuda
  GPU: 4
```

## Annotations (train/valid/test) JSON Format

OpenSportsLib uses the OSL JSON v2.0 format for annotation files. Each split
file is a JSON object with a root `labels` schema and a `data` array of samples.
For the full schema, supported input types, multi-modal examples, and prediction
payloads, see [OSL JSON Format](../data/osl-json-format.md).

### Classification annotations

Classification samples use `data[].labels.action.label` by default. The label
must be present in the root `labels.action.labels` list.

```json
{
  "version": "2.0",
  "task": "action_classification",
  "labels": {
    "action": {
      "type": "single_label",
      "labels": ["pass", "shot"]
    }
  },
  "data": [
    {
      "id": "clip_0001",
      "inputs": [
        {
          "type": "video",
          "path": "clips/clip_0001.mp4",
          "fps": 25.0
        }
      ],
      "labels": {
        "action": {
          "label": "shot"
        }
      }
    }
  ]
}
```

For video classification, `inputs[].path` is resolved from the split media root
in the YAML config, such as `DATA.train.video_path`. For tracking
classification, use `type: tracking_parquet` and set
`DATA.data_modality: tracking_parquet`.

### Localization annotations

Localization samples use `data[].events[]`. OpenSportsLib prefers
`position_ms` and falls back to `gameTime` in feature-based JSON loaders.

```json
{
  "version": "2.0",
  "task": "action_spotting",
  "labels": {
    "action": {
      "type": "single_label",
      "labels": ["pass", "shot"]
    }
  },
  "data": [
    {
      "id": "game_0001",
      "inputs": [
        {
          "type": "video",
          "path": "games/game_0001.mp4",
          "fps": 25.0
        }
      ],
      "events": [
        {
          "head": "action",
          "label": "pass",
          "position_ms": 1240,
          "gameTime": "1 - 00:01"
        }
      ]
    }
  ]
}
```

### Public example datasets

Download or inspect annotation files from:

- **Classification: MVFouls and SVFouls**<br>
  https://huggingface.co/datasets/OpenSportsLab/opensportslib-classification-vars
- **Localization: Ball Action Spotting**<br>
  https://huggingface.co/datasets/OpenSportsLab/opensportslib-localization-snbas


---

## Download Weights from HuggingFace

For a comparison table with datasets, reported scores, and model links, see the
[Model Zoo](../model-zoo.md).

### 1. Classification (MViT)

**MVFoul Classification (MViT backbone)**  
https://huggingface.co/OpenSportsLab/OSL-cls-action-mvitv2


### 2. Localization (E2E Spotting)

- **2023 Ball Action Spotting (2 classes)**  
  https://huggingface.co/OpenSportsLab/OSL-loc-snbas-2023-e2e  

- **2024 Ball Action Spotting (12 classes)**  
  https://huggingface.co/OpenSportsLab/OSL-loc-snbas-2025-e2e 

Usage:
```bash
### Load weights from HF ###

#### For Classification ####
myModel.load_weights(weights="OpenSportsLab/OSL-cls-action-mvitv2")

#### For Localization ####
weights = "OpenSportsLab/OSL-loc-snbas-2023-e2e" # SNBAS - 2 classes (E2E spot)
weights = "OpenSportsLab/OSL-loc-snbas-2025-e2e" # SNBAS - 12 classes (E2E spot)
myModel.load_weights(weights=weights)
```

## Train on SINGLE GPU
```python
from opensportslib import model
import wandb

# Initialize model with config
myModel = model.ClassificationModel(
    config="/path/to/classification.yaml",
    weights=None,  # optional: path or Hugging Face model ID
)

## Localization ##
# myModel = model.LocalizationModel(
#     config="/path/to/localization.yaml"
# )

# Train on your dataset
myModel.train(
    train_set="/path/to/train_annotations.json",
    valid_set="/path/to/valid_annotations.json",
)
```

## Train on Multiple GPU (DDP)
```python
from opensportslib import model

def main():
    myModel = model.ClassificationModel(
        config="/path/to/classification.yaml",
        weights=None,  # optional: path or Hugging Face model ID
    )

    ## Localization ##
    # myModel = model.LocalizationModel(
    #     config="/path/to/classification.yaml"
    # )

    myModel.train(
        train_set="/path/to/train_annotations.json",
        valid_set="/path/to/valid_annotations.json",
        use_ddp=True,  # IMPORTANT
    )

if __name__ == "__main__":
    main()
```


## Test / Inference on SINGLE GPU
```python
from opensportslib import model

# Load trained model
myModel = model.ClassificationModel(
    config="/path/to/classification.yaml",
    weights=None,  # optional: path or Hugging Face model ID
)

## Localization ##
# myModel = model.LocalizationModel(
#     config="/path/to/classification.yaml"
# )

# Run inference on test set
predictions = myModel.infer(
    test_set="/path/to/test_annotations.json",
)

saved_predictions = myModel.save_predictions(
    output_path="/path/to/predictions.json",
    predictions=predictions,
)

metrics = myModel.evaluate(
    test_set="/path/to/test_annotations.json",
)

metrics_from_saved_predictions = myModel.evaluate(
    test_set="/path/to/test_annotations.json",
    predictions=saved_predictions,
)
```

`infer()` returns an in-memory OSL JSON-style prediction payload. It does not
require an output path. `save_predictions(...)` is the explicit API for writing
that payload to disk.

## Test / Inference on Multiple GPU (DDP)
```python
from opensportslib import model

def main():
    myModel = model.ClassificationModel(
        config="/path/to/classification.yaml",
        weights=None,  # optional: path or Hugging Face model ID
    )

    ## Localization ##
    # myModel = model.LocalizationModel(
    #     config="/path/to/classification.yaml"
    # )

    predictions = myModel.infer(
        test_set="/path/to/test_annotations.json",
        use_ddp=True,   # optional (usually not needed)
    )

    metrics = myModel.evaluate(
        test_set="/path/to/test_annotations.json",
    )

    print(metrics)

if __name__ == "__main__":
    main()
```
