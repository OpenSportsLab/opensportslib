# Training & Inference

This section explains how to:

- Configure experiments
- Train models (single & multi-GPU)
- Run inference
- Use pretrained weights from HuggingFace

---
## Configuration Sample (.yaml) file

### 1. Classification 
```bash
TASK: classification

DATA:
  dataset_name: mvfouls
  data_dir: /home/vorajv/soccernetpro/SoccerNet/mvfouls
  data_modality: video
  view_type: multi  # multi or single
  num_classes: 8 # mvfoul
  train: 
    type: annotations_train.json
    video_path: ${DATA.data_dir}/train
    path: ${DATA.train.video_path}/annotations-train.json
    dataloader:
      batch_size: 8
      shuffle: true
      num_workers: 4
      pin_memory: true
  valid:
    type: annotations_valid.json
    video_path: ${DATA.data_dir}/valid
    path: ${DATA.valid.video_path}/annotations-valid.json
    dataloader:
      batch_size: 1
      num_workers: 1
      shuffle: false
  test:
    type: annotations_test.json
    video_path: ${DATA.data_dir}/test
    path: ${DATA.test.video_path}/annotations-test.json
    dataloader:
      batch_size: 1
      num_workers: 1
      shuffle: false
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
  unfreeze_head: true  # for videomae backbone
  unfreeze_last_n_layers: 3 # for videomae backbone
    

TRAIN:
  monitor: balanced_accuracy # balanced_accuracy, loss
  mode: max # max or min
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
  use_seed: false
  seed: 42
  GPU: 4
  device: cuda   # auto | cuda | cpu
  gpu_id: 0

```

### 2. Classification (Tracking)
```bash
TASK: classification

DATA:
  dataset_name: sngar
  data_modality: tracking_parquet
  data_dir: /home/karkid/soccernetpro/sngar-tracking
  preload_data: false
  train: 
    type: annotations_train.json
    video_path: ${DATA.data_dir}/train
    path: ${DATA.train.video_path}/train.json
    dataloader:
      batch_size: 32
      shuffle: true
      num_workers: 8
      pin_memory: true
  valid:
    type: annotations_valid.json
    video_path: ${DATA.data_dir}/valid
    path: ${DATA.valid.video_path}/valid.json
    dataloader:
      batch_size: 32
      num_workers: 8
      shuffle: false
  test:
    type: annotations_test.json
    video_path: ${DATA.data_dir}/test
    path: ${DATA.test.video_path}/test.json
    dataloader:
      batch_size: 32
      num_workers: 8
      shuffle: false
  num_frames: 16
  frame_interval: 9
  augmentations:
    vertical_flip: true
    horizontal_flip: true
    team_flip: true
  normalize: true
  num_objects: 23
  feature_dim: 8
  pitch_half_length: 85.0
  pitch_half_width: 50.0
  max_displacement: 110.0
  max_ball_height: 30.0

MODEL:
  type: custom
  backbone:
    type: graph_conv
    encoder: graphconv
    hidden_dim: 64
    num_layers: 20
    dropout: 0.1
  neck:
    type: TemporalAggregation
    agr_type: maxpool
    hidden_dim: 64
    dropout: 0.1
  head:
    type: TrackingClassifier
    hidden_dim: 64
    dropout: 0.1
    num_classes: 10
  edge: positional
  k: 8
  r: 15.0

TRAIN:
  monitor: loss # balanced_accuracy, loss
  mode: min # max or min
  enabled: true
  use_weighted_sampler: true
  use_weighted_loss: false
  samples_per_class: 4000
  epochs: 10
  patience: 10
  save_every: 20
  detailed_results: true

  optimizer:
    type: Adam
    lr: 0.001

  scheduler:
    type: ReduceLROnPlateau
    mode: ${TRAIN.mode}
    patience: 10
    factor: 0.1
    min_lr: 1e-8
  
  criterion:
    type: CrossEntropyLoss

  save_dir: ./checkpoints_tracking

SYSTEM:
 log_dir: ./logs
 use_seed: true
 seed: 42
 GPU: 4
 device: cuda   # auto | cuda | cpu
 gpu_id: 0
```

### 3. Localization
```bash
TASK: localization

dali: True

DATA:
  dataset_name: SoccerNet
  data_dir: /home/vorajv/soccernetpro/SoccerNet/annotations/
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
    
  epoch_num_frames: 500000
  mixup: true
  modality: rgb
  crop_dim: -1
  dilate_len: 0        # Dilate ground truth labels
  clip_len: 100
  input_fps: 25
  extract_fps: 2
  imagenet_mean: [0.485, 0.456, 0.406]
  imagenet_std: [0.229, 0.224, 0.225]
  target_height: 224
  target_width: 398

  train:
    type: VideoGameWithDali
    classes: ${DATA.classes}
    output_map: [data, label]
    video_path: ${DATA.data_dir}/train/
    path: ${DATA.train.video_path}/annotations-2024-224p-train.json
    dataloader:
      batch_size: 8
      shuffle: true
      num_workers: 4
      pin_memory: true

  valid:
    type: VideoGameWithDali
    classes: ${DATA.classes}
    output_map: [data, label]
    video_path: ${DATA.data_dir}/valid/
    path: ${DATA.valid.video_path}/annotations-2024-224p-valid.json
    dataloader:
      batch_size: 8
      shuffle: true

  valid_data_frames:
    type: VideoGameWithDaliVideo
    classes: ${DATA.classes}
    output_map: [data, label]
    video_path: ${DATA.valid.video_path}
    path: ${DATA.valid.path}
    overlap_len: 0
    dataloader:
      batch_size: 4
      shuffle: false

  test:
    type: VideoGameWithDaliVideo
    classes: ${DATA.classes}
    output_map: [data, label]
    video_path: ${DATA.data_dir}/test/
    path: ${DATA.test.video_path}/annotations-2024-224p-test.json
    results: results_spotting_test
    nms_window: 2 
    metric: tight
    overlap_len: 50
    dataloader:
      batch_size: 4
      shuffle: false

  challenge:
    type: VideoGameWithDaliVideo
    overlap_len: 50
    output_map: [data, label]
    path: ${DATA.data_dir}/challenge/annotations.json
    dataloader:
      batch_size: 4
      shuffle: false

MODEL:
    type: E2E
    runner:
      type: runner_e2e
    backbone:
      type: rny008_gsm
    head:
      type: gru
    multi_gpu: true
    load_weights: null
    save_dir: ./checkpoints
    work_dir: ${MODEL.save_dir}

TRAIN:
  type: trainer_e2e
  num_epochs: 10
  acc_grad_iter: 1
  base_num_valid_epochs: 30
  start_valid_epoch: 4
  valid_map_every: 1
  criterion_valid: map

  criterion:
    type: CrossEntropyLoss

  optimizer:
    type: AdamWithScaler
    lr: 0.01

  scheduler:
    type: ChainedSchedulerE2E
    acc_grad_iter: 1
    num_epochs: ${TRAIN.num_epochs}
    warm_up_epochs: 3

SYSTEM:
  log_dir: ./logs
  seed: 42
  GPU: 4         # number of gpus to use
  device: cuda   # auto | cuda | cpu
  gpu_id: 0      # device id for single gpu training
```

## Annotations (train/valid/test) (.json) Format

Download annotation files from the links below.

### 1. Classification

- **MVFouls**  
  https://huggingface.co/datasets/OpenSportsLab/soccernetpro-classification-vars/tree/mvfouls  

- **SVFouls**  
  https://huggingface.co/datasets/OpenSportsLab/soccernetpro-classification-vars/tree/svfouls  

### 2. Localization

- **Ball Action Spotting**  
  https://huggingface.co/datasets/OpenSportsLab/soccernetpro-localization-snbas/tree/main  


---

## Download Weights from HuggingFace

### 1. Classification (MViT)

**MVFoul Classification (MViT backbone)**  
https://huggingface.co/jeetv/snpro-classification-mvit/tree/main


### 2. Localization (E2E Spotting)

- **2023 Ball Action Spotting (2 classes)**  
  https://huggingface.co/jeetv/snpro-snbas-2023/tree/main  

- **2024 Ball Action Spotting (12 classes)**  
  https://huggingface.co/jeetv/snpro-snbas-2024/tree/main  

Usage:
```bash
### Load weights from HF ###

#### For Classification ####
myModel.infer(
    test_set="/path/to/annotations.json",
    pretrained="jeetv/snpro-classification-mvit", # classification (MViT)
)

#### For Localization ####
pretrained = "jeetv/snpro-snbas-2023" # SNBAS - 2 classes (E2E spot)
pretrained = "jeetv/snpro-snbas-2024" # SNBAS - 12 classes (E2E spot)
```

## Train on SINGLE GPU
```bash
from soccernetpro import model
import wandb

# Initialize model with config
myModel = model.classification(
    config="/path/to/classification.yaml"
)

## Localization ##
# myModel = model.localization(
#     config="/path/to/classification.yaml"
# )

# Train on your dataset
myModel.train(
    train_set="/path/to/train_annotations.json",
    valid_set="/path/to/valid_annotations.json",
    pretrained=/path/to/  # or path to pretrained checkpoint
)
```

## Train on Multiple GPU (DDP)
```bash
from soccernetpro import model

def main():
    myModel = model.classification(
        config="/path/to/classification.yaml",
        data_dir="/path/to/dataset_root"
    )

    ## Localization ##
    # myModel = model.localization(
    #     config="/path/to/classification.yaml"
    # )

    myModel.train(
        train_set="/path/to/train_annotations.json",
        valid_set="/path/to/valid_annotations.json",
        pretrained="/path/to/pretrained.pt",  # optional
        use_ddp=True,  # IMPORTANT
    )

if __name__ == "__main__":
    main()
```


## Test / Inference on SINGLE GPU
```bash
from soccernetpro import model

# Load trained model
myModel = model.classification(
    config="/path/to/classification.yaml"
)

## Localization ##
# myModel = model.localization(
#     config="/path/to/classification.yaml"
# )

# Run inference on test set
metrics = myModel.infer(
    test_set="/path/to/test_annotations.json",
    pretrained="/path/to/checkpoints/final_model",
    predictions="/path/to/predictions.json"
)
```

## Test / Inference on Multiple GPU (DDP)
```bash
from soccernetpro import model

def main():
    myModel = model.classification(
        config="/path/to/classification.yaml",
        data_dir="/path/to/dataset_root"
    )

    ## Localization ##
    # myModel = model.localization(
    #     config="/path/to/classification.yaml"
    # )

    metrics = myModel.infer(
        test_set="/path/to/test_annotations.json",
        pretrained="/path/to/checkpoints/best.pt",
        predictions="/path/to/predictions.json",
        use_ddp=True,   # optional (usually not needed)
    )

    print(metrics)

if __name__ == "__main__":
    main()
```