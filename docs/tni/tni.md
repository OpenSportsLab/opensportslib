# Training & Inference

This section explains how to:

- Configure experiments
- Train models (single & multi-GPU)
- Run inference
- Use pretrained weights from HuggingFace

For full key-by-key config documentation and Python-only override workflow, see [Configuration Guide](config-guide.md).

---
## Configuration Sample (.yaml) file

The examples below are included directly from the latest canonical YAML files in
`opensportslib/configs/`, so the documentation stays aligned with the runnable
configs.

### 1. Classification

```yaml
--8<-- "opensportslib/configs/classification/video/classification.yaml"
```

### 2. Classification (Tracking)

```yaml
--8<-- "opensportslib/configs/classification/tracking/sngar-tracking.yaml"
```

### 3. Localization

```yaml
--8<-- "opensportslib/configs/localization/video/localization-ocv.yaml"
```

## Annotations (train/valid/test) (.json) Format

Download annotation files from the links below.

### 1. Classification

- **MVFouls**  
  https://huggingface.co/datasets/OpenSportsLab/opensportslib-classification-vars/tree/mvfouls  

- **SVFouls**  
  https://huggingface.co/datasets/OpenSportsLab/opensportslib-classification-vars/tree/svfouls  

### 2. Localization

- **Ball Action Spotting**  
  https://huggingface.co/datasets/OpenSportsLab/opensportslib-localization-snbas/tree/main  


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
```bash
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
```bash
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
```bash
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

metrics = myModel.evaluate(
    test_set="/path/to/test_annotations.json",
)

metrics_from_saved_predictions = myModel.evaluate(
    test_set="/path/to/test_annotations.json",
    predictions="/path/to/predictions.json",
)
```

## Test / Inference on Multiple GPU (DDP)
```bash
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
