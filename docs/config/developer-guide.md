# Configuration Developer Guide

Developer-facing guidance for authoring, reviewing, and extending canonical config safely.

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
