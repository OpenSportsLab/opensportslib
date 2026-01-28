# SoccerNetPro
SoccerNetPro is the professional extension of the popular SoccerNet library, designed for advanced video understanding in soccer. It provides state-of-the-art tools for action recognition, spotting, retrieval, and captioning, making it ideal for researchers, analysts, and developers working with soccer video data.

## Development
```bash
### Clone the github repo
git clone https://github.com/OpenSportsLab/soccernetpro.git 

### Requirements and installation ###
conda create -n SoccerNet python=3.12 pip
conda activate SoccerNet
pip install -e .

or 

pip install -e .[localization]

### git branch and merge rules ###
1. Check and verify current branch is "dev" - git status

2. Create new branch from source "dev" - 
git pull
git checkout -b <new_feature/fix/bug>

3. Raise PR request to merge your branch <new_feature/fix/bug> to "dev" branch 
```

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
  data_dir: mvfouls
  view_type: multi  # multi or single
  annotations:
    train: /path/to/train_annotations.json
    valid: /path/to/test_annotations.json
    test: /path/to/valid_annotations.json
  num_frames: 16               # 8 before + 8 after the foul
  input_fps: 25                # Original FPS of video
  target_fps: 17               # Temporal downsampling to 1s clip (approx)
  start_frame: 63            # Start frame of clip relative to foul frame
  end_frame: 87              # End frame of clip relative to foul frame
  frame_size: [224, 224]       # Spatial resolution (HxW)
  augmentations:
    random_affine: true
    translate: [0.1, 0.1]        
    affine_scale: [0.9, 1.0]     
    random_perspective: true
    distortion_scale: 0.3        
    perspective_prob: 0.5
    random_rotation: true
    rotation_degrees: 5          
    color_jitter: true
    jitter_params: [0.2, 0.2, 0.2, 0.1]   # brightness, contrast, saturation, hue
    random_horizontal_flip: true
    flip_prob: 0.5
    random_crop: false
  num_workers: 1
  train_batch_size: 8
  valid_batch_size: 1

MODEL:
  type: custom # huggingface, custom 
  backbone: 
    type: mvit_v2_s # video_mae, r3d_18, mc3_18, r2plus1d_18, s3d, mvit_v2_s
  neck:
    type: MV_Aggregate
    agr_type: max   # max, mean, attention
  head: 
    type: MV_LinearLayer
  pretrained_model: mvit_v2_s # MCG-NJU/videomae-base, OpenGVLab/VideoMAEv2-Base, r3d_18, mc3_18, r2plus1d_18, s3d, mvit_v2_s
  num_classes: 8
  unfreeze_head: true  # for videomae backbone
  unfreeze_last_n_layers: 3 # for videomae backbone
    

TRAIN:
  enabled: true
  use_weighted_sampler: false
  use_weighted_loss: true
  epochs: 20 #20
  save_dir: ./checkpoints
  log_interval: 10
  save_every: 2 #5

  criterion:
    type: CrossEntropyLoss

  optimizer:
    type: AdamW
    lr: 0.0001  #0.001
    backbone_lr: 0.00005
    head_lr: 0.001
    betas: [0.9, 0.999]
    eps: 0.0000001
    weight_decay: 0.001 #0.01 - videomae, 0.001 - others
    amsgrad: false
  
  scheduler:
    type: StepLR
    step_size: 3
    gamma: 0.1

SYSTEM:
  log_dir: ./logs
  seed: 42
  device: cuda   # auto | cuda | cpu
  gpu_id: 0
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

