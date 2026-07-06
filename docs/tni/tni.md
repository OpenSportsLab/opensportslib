# Training & Inference

This section explains how to:

- Configure experiments
- Train models (single & multi-GPU)
- Run inference
- Use pretrained weights from HuggingFace

For full key-by-key config documentation and Python-only override workflow, see [Configuration Guide](../config/configuration-guide.md).

---
## Configuration Sample (.yaml) file

Use source-of-truth runnable configs from `opensportslib/configs/`.
`examples/configs/` mirrors these files.

### 1. Classification (Video)

- Source: [`opensportslib/configs/classification/video.yaml`](../../opensportslib/configs/classification/video.yaml)
- Example mirror: [`examples/configs/classification_video.yaml`](../../examples/configs/classification_video.yaml)

### 2. Classification (Tracking)

- Source: [`opensportslib/configs/classification/sngar_tracking.yaml`](../../opensportslib/configs/classification/sngar_tracking.yaml)
- Example mirror: [`examples/configs/classification_sngar_tracking.yaml`](../../examples/configs/classification_sngar_tracking.yaml)

### 3. Localization (DALI)

- Source: [`opensportslib/configs/localization/video_dali.yaml`](../../opensportslib/configs/localization/video_dali.yaml)
- Example mirror: [`examples/configs/localization_video_dali.yaml`](../../examples/configs/localization_video_dali.yaml)

### 4. VQA

- Source configs:
  - [`opensportslib/configs/vqa/xvars.yaml`](../../opensportslib/configs/vqa/xvars.yaml)
  - [`opensportslib/configs/vqa/qwen.yaml`](../../opensportslib/configs/vqa/qwen.yaml)
- Example mirrors:
  - [`examples/configs/vqa_xvars.yaml`](../../examples/configs/vqa_xvars.yaml)
  - [`examples/configs/vqa_qwen.yaml`](../../examples/configs/vqa_qwen.yaml)

For canonical key definitions and migration-safe authoring rules, use the
[Configuration Guide](../config/configuration-guide.md).

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

### VQA annotations

VQA samples use `data[].answers[]` with a question and one or more reference
answers.

```json
{
  "version": "2.0",
  "task": "vqa",
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
      "answers": [
        {
          "question": "What card would you give? Why?",
          "answers": ["No card, because this is a fair challenge."]
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
#     config="/path/to/localization_video_dali.yaml"
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

## VQA Inference and Evaluation

```python
from opensportslib.apis import VQAModel

myModel = VQAModel(
    config="opensportslib/configs/vqa/qwen.yaml",
    weights=None,  # optional: path or Hugging Face model ID
)

predictions = myModel.infer(
    test_set="/path/to/test_annotations.json",
)

```


Use `opensportslib/configs/vqa/xvars.yaml` with `opensportslib setup --vqa_xvars`
for the X-VARS-compatible backend, or `opensportslib/configs/vqa/qwen.yaml` with
`opensportslib setup --vqa_qwen` for the Qwen backend. The Qwen backend
supports `Qwen/Qwen2.5-7B-Instruct` and `Qwen/Qwen3.5-9B-Base`.

For X-VARS, `feature_source: indexed_or_raw_clip` prefers indexed CLIP features
when available and falls back to raw-video CLIP extraction during `infer()`.
Pre-extracted features are still the preferred path for parity and throughput.
See [tools/vqa.md](../tools/vqa.md) for the full VQA setup workflow.

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


if __name__ == "__main__":
    main()
```
