# Training & Inference

This section explains how to:

- Configure experiments
- Train models (single & multi-GPU)
- Run inference
- Use pretrained weights from HuggingFace

For full key-by-key config documentation and Python-only override workflow, see [Configuration Guide](../config/configuration-guide.md).

---
## Configuration Sample (.yaml) file

The snippets below show the main structure of the runnable configs in
`opensportslib/config/`. Use the source files when you need the complete
experiment defaults.

### 1. Classification

```yaml
TASK: classification
VERSION: 3

SYSTEM:
  paths:
    save_dir: ./checkpoints
    work_dir: ./checkpoints
  device: cuda
  gpu:
    count: 1
    id: 0

DATA:
  common:
    dataset_name: mvfouls
    runtime:
      loader_backend: opencv
    splits:
      train:
        source_path: /path/to/OSL-XFoul/224p/train
        annotation_path: /path/to/OSL-XFoul/224p/train/train.json
        dataloader: {batch_size: 8, shuffle: true, num_workers: 4}
      valid:
        source_path: /path/to/OSL-XFoul/224p/valid
        annotation_path: /path/to/OSL-XFoul/224p/valid/valid.json
      test:
        source_path: /path/to/OSL-XFoul/224p/test
        annotation_path: /path/to/OSL-XFoul/224p/test/test.json
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
  components:
    video_encoder:
      kind: encoder
      source: {provider: opensportslib, name: mvit_v2_s}
      params: {}
    task_head:
      kind: head
      source: {provider: opensportslib, name: MV_LinearLayer}
      params: {}
  topology:
    - from: video_encoder
      to: task_head

TRAIN:
  epochs: 20
  criterion:
    type: CrossEntropyLoss
  optimizer:
    type: AdamW
    lr: 0.0001
  selection:
    monitor: balanced_accuracy
    mode: max
```

### 2. Classification (Tracking)

```yaml
TASK: classification
VERSION: 3

SYSTEM:
  paths:
    save_dir: ./checkpoints_tracking
    work_dir: ./checkpoints_tracking
  device: cuda
  gpu:
    count: 1
    id: 0

DATA:
  common:
    dataset_name: sngar
    runtime:
      loader_backend: opencv
    splits:
      train:
        source_path: /path/to/soccernetpro-classification-GAR/tracking-parquet/train
        annotation_path: /path/to/soccernetpro-classification-GAR/tracking-parquet/train/train.json
      valid:
        source_path: /path/to/soccernetpro-classification-GAR/tracking-parquet/valid
        annotation_path: /path/to/soccernetpro-classification-GAR/tracking-parquet/valid/valid.json
      test:
        source_path: /path/to/soccernetpro-classification-GAR/tracking-parquet/test
        annotation_path: /path/to/soccernetpro-classification-GAR/tracking-parquet/test/test.json
  inputs:
    tracking:
      modality: tracking
      representation: features
      source: {format: parquet}
      sampling: {num_frames: 16}
      params:
        normalize: true
        feature_dim: 8

MODEL:
  schema_version: 3
  task: classification
  components:
    tracking_encoder:
      kind: encoder
      source: {provider: opensportslib, name: graph_conv}
      params: {encoder: gin, hidden_dim: 64, num_layers: 20, edge: positional, k: 8, r: 15.0}
    task_head:
      kind: head
      source: {provider: opensportslib, name: TrackingClassifier}
      params: {num_classes: 10}
  topology:
    - from: tracking_encoder
      to: task_head

TRAIN:
  epochs: 100
  optimizer:
    type: Adam
    lr: 0.001
  selection:
    monitor: loss
    mode: min
```

### 3. Localization

```yaml
TASK: localization
VERSION: 3

SYSTEM:
  paths:
    save_dir: ./checkpoints
    work_dir: ./checkpoints
  device: cuda
  gpu:
    count: 4
    id: 0

DATA:
  common:
    dataset_name: SoccerNet
    runtime:
      loader_backend: dali
    splits:
      train:
        annotation_path: /path/to/OSL-SNBAS/224p-2024/train/train.json
        source_path: /path/to/OSL-SNBAS/224p-2024/train
        type: VideoGameWithDali
      valid:
        annotation_path: /path/to/OSL-SNBAS/224p-2024/valid/valid.json
        source_path: /path/to/OSL-SNBAS/224p-2024/valid
        type: VideoGameWithDali
      test:
        annotation_path: /path/to/OSL-SNBAS/224p-2024/test/test.json
        source_path: /path/to/OSL-SNBAS/224p-2024/test
        type: VideoGameWithDaliVideo
        results: results_spotting_test
        nms_window: 2
        metric: tight
        overlap_len: 50
  inputs:
    video:
      modality: video
      representation: raw
      source: {format: mp4}
      sampling: {clip_len: 100, input_fps: 25, extract_fps: 2}
      transform:
        resize: {height: 224, width: 398}

MODEL:
  schema_version: 3
  task: localization
  components:
    video_encoder:
      kind: encoder
      source: {provider: opensportslib, name: rny008_gsm}
      params: {}
    task_head:
      kind: head
      source: {provider: opensportslib, name: gru}
      params: {}
  topology:
    - from: video_encoder
      to: task_head

TRAIN:
  trainer:
    type: trainer_e2e
  epochs: 10
  criterion:
    type: CrossEntropyLoss
  optimizer:
    type: AdamWithScaler
    lr: 0.01
  execution:
    multi_gpu: true
    criterion_valid: map
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
in the YAML config, such as `DATA.common.splits.train.source_path`. For tracking
classification, use a tracking input block under `DATA.inputs` with
`source.format: parquet`.

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
