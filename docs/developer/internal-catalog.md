# Internal Module Catalog

> **Internal.** This catalog is for maintainers who need to navigate the
> current codebase. Names here are not a compatibility promise. Private
> underscore-prefixed helpers are intentionally omitted.

## Configuration and runtime

| Module | Concrete classes/functions | Called from |
| --- | --- | --- |
| `core.config.schema` | `SystemConfig`, `DataConfig`, `ModelConfig`, `TrainConfig`, `IOConfig`, `ConfigDocument` | Schema/config tooling. |
| `core.config.runtime_adapter` | namespace conversion and runtime adaptation functions | Config loader and wrappers. |
| `core.trainer.classification_trainer` | `BaseTrainerClassification`, `MVTrainerClassification`, `TrackingTrainerClassification`, `FramesTrainerClassification`, `Trainer_Classification` | `ClassificationModel`. |
| `core.trainer.localization_trainer` | `Trainer_pl`, `Trainer_e2e`, `Inferer`, `Evaluator` | `LocalizationModel`. |
| `core.trainer.vqa_trainer` | backend trainers, SFT dataset/collator classes, `Trainer_VQA` | `VQAModel`. |

## Datasets

| Module | Concrete classes/functions | Caller/route |
| --- | --- | --- |
| `datasets.classification_dataset` | `ClassificationDataset`, `VideoDataset`, `TrackingDataset`, `HFTrackingDataset`, `H5TrackingDataset` | Classification dataset factory. |
| `datasets.localization_dataset` | `FrameReader`, `ActionSpotDataset`, `ActionSpotVideoDataset`, feature and SoccerNet dataset classes | `LocalizationDataset` selection. |
| `datasets.vqa_dataset` | `VQADataset` | VQA builder route. |
| `datasets.hf_json` / `hf_tracking` | prepared-split records and readers | Hub-backed manifest preparation. |

## Models

| Module family | Concrete implementation groups | Selected by |
| --- | --- | --- |
| `models.base` | E2E, ContextAware, LearnablePooling, tracking, VideoMAE/video, rule-based, X-VARS, and Qwen model classes | Top-level model builder. |
| `models.backbones.builder` | extract-feature classes, graph encoders, video backbones | `build_backbone`. |
| `models.neck.builder` | aggregation, temporal, pooling, VLAD/RVLAD classes | `build_neck`. |
| `models.heads.builder` | multi-view, tracking, temporal E2E, linear, and CALF heads | `build_head`. |
| `models.utils` | temporal-shift, prediction, prompting, feature-index, and base utilities | Model-family implementation. |

## Metrics and utilities

| Module | Concrete classes/functions | Notes |
| --- | --- | --- |
| `metrics.classification_metric` | classification metric functions and detailed artifact writer | Detailed report function writes files. |
| `metrics.localization_metric` | AP/mAP functions, `ErrorStat`, `ForegroundF1`, `AverageMeter`, prediction/result writers | Some functions write JSON/evaluation output. |
| `metrics.vqa_metric` | `compute_vqa_metrics` and internal normalization/scoring helpers | Returns serializable VQA scores. |
| `tools.*` | conversion and Hub-transfer implementation helpers | Public export boundary is documented in [Tool Reference](tools-reference.md). |

Use the [extension API pages](config-reference.md) for supported customization points and source links/signatures. Search the named module before changing an internal class, then add coverage in the closest existing test tier.
