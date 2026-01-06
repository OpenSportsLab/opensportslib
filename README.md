# SoccerNetPro
SoccerNetPro is the professional extension of the popular SoccerNet library, designed for advanced video understanding in soccer. It provides state-of-the-art tools for action recognition, spotting, retrieval, and captioning, making it ideal for researchers, analysts, and developers working with soccer video data.

## Installation

```bash
conda create -n SoccerNet python=3.12 pip
conda activate SoccerNet
pip install --pre soccernetpro
```

## Configuration Sample (.yaml) file
```bash
TASK: classification

DATA:
  dataset_name: mvfouls
  data_dir: /path/to/mvfouls
  view_type: single
  annotations:
    train: /path/to/train_annotations.json
    valid: /path/to/test_annotations.json
    test: /path/to/valid_annotations.json
  num_frames: 16               # 8 before + 8 after the foul
  input_fps: 25                # Original FPS of video
  target_fps: 16               # Temporal downsampling to 1s clip (approx)
  frame_size: [224, 224]       # Spatial resolution (HxW)
  augmentations:
    random_crop: true
    random_horizontal_flip: true
    flip_prob: 0.5
    color_jitter: false
    jitter_params: [0.4, 0.4, 0.4, 0.1]   # brightness, contrast, saturation, hue
    random_erasing: false
  num_workers: 1
  train_batch_size: 8
  valid_batch_size: 4

MODEL:
  type: huggingface
  backbone: video_mae
  pretrained_model: MCG-NJU/videomae-base
  num_classes: 7
  freeze_backbone: true
  

TRAIN:
  enabled: true
  use_weighted_sampler: true
  epochs: 10
  learning_rate: 0.005
  save_dir: ./checkpoints
  log_interval: 10
  save_every: 5

SYSTEM:
  log_dir: ./logs
  seed: 42
```

## Annotations (train/valid/test) (.json) format
```bash
{
  "version": "1.0",
  "date": "2025-11-11",
  "task": "action_classification",
  "dataset_name": "mvfouls",
  "metadata": {
    "source": "Professional Soccer Dataset",
    "license": "CC-BY-NC-4.0",
    "created_by": "AI Sports Lab",
    "notes": "Converted automatically from SoccerNet-like foul annotation structure."
  },
  "labels": {
    "foul_type": {
      "type": "single_label",
      "labels": [
        "Challenge",
        "Dive",
        "Elbowing",
        "High Leg",
        "Holding",
        "Pushing",
        "Standing Tackling",
        "Tackling"
      ]
    },
    "severity": {
      "type": "single_label",
      "labels": [
        "No Offence",
        "Offence + No Card",
        "Offence + Yellow Card",
        "Offence + Red Card"
      ]
    },
    "attributes": {
      "type": "multi_label",
      "labels": [
        "Intentional",
        "Reckless",
        "Dangerous Play",
        "VAR Checked",
        "InBox",
        "CounterAttack"
      ]
    }
  },
  "data": [
    {
      "id": "action_0",
      "inputs": [
        {
          "type": "video",
          "path": "Dataset/Train/action_0/clip_0",
          "metadata": {
            "camera_type": "Main camera center",
            "timestamp": 1730826,
            "replay_speed": 1.0
          }
        },
        {
          "type": "video",
          "path": "Dataset/Train/action_0/clip_1",
          "metadata": {
            "camera_type": "Close-up player or field referee",
            "timestamp": 1744173,
            "replay_speed": 1.8
          }
        }
      ],
      "labels": {
        "foul_type": {
          "label": "Challenge"
        },
        "severity": {
          "label": "Offence + No Card"
        },
        "attributes": {
          "labels": [
            "Reckless"
          ]
        }
      },
      "metadata": {
        "UrlLocal": "england_epl\\2014-2015\\2015-02-21 - 18-00 Chelsea 1 - 1 Burnley",
        "Contact": "With contact",
        "Bodypart": "Upper body",
        "Upper body part": "Use of shoulder",
        "Handball": "No handball"
      }
    }
    ]
}

```

## Train
```bash
from soccernetpro import model
import wandb

# Initialize model with config
myModel = model.classification(
    config="/path/to/classification.yaml"
)

# Optionally adjust config
myModel.config.TRAIN.learning_rate = 1e-4
myModel.config.MODEL.freeze_backbone = True

# Train on your dataset
myModel.train(
    train_set="/path/to/train_annotations.json",
    valid_set="/path/to/valid_annotations.json",
    pretrained=/path/to/  # or path to pretrained checkpoint
)
```

## Test / Inference
```bash
from soccernetpro import model

# Load trained model
myModel = model.classification(
    config="/path/to/classification.yaml"
)

# Run inference on test set
preds, metrics = myModel.infer(
    test_set="/path/to/test_annotations.json",
    pretrained="/path/to/checkpoints/final_model",
)
```

