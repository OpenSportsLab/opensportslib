# Model Construction Extension API

> **Extension point.** Builders define the model names and families available
> from supported configuration files. Concrete model modules are internal
> implementations.

## Dispatch rules

`models.builder.build_model()` accepts the supported runtime configuration and
selects a model from `TASK`. Classification selects an encoder route. Localization selects
`MODEL.metadata.family` (`RuleBased`, `E2E`, `ContextAware`, or
`LearnablePooling`). VQA selects its configured backend
(`xvars_videochatgpt`, `qwen_xvars_infer`, or `qwen_vl_native_infer`). An
unsupported name raises `ValueError`; OpenSportsLib does not dynamically import it.

| Builder | Responsibility |
| --- | --- |
| `models.builder` | Top-level canonical model route. |
| `models.backbones.builder` | Video, feature, graph, and image backbone construction. |
| `models.neck.builder` | Aggregation and temporal/pooling adapters. |
| `models.heads.builder` | Classification and spotting output heads. |

::: opensportslib.models.builder
    options:
      members:
        - build_model_canonical
        - build_model_from_config
        - build_model
      show_source: true

## Namespace-module builders

`models/backbones`, `models/neck`, and `models/heads` are namespace-style
source directories in this repository, so MkDocstrings cannot resolve them as
Python packages. Their concrete extension signatures are:

```python
build_backbone(cfg, default_args=None)
Add_Temporal_Shift_Modules(net, n_segment, is_gsm=False)
build_neck(cfg, default_args=None)
build_head(cfg, default_args=None)
```

The source modules are `models/backbones/builder.py`, `models/neck/builder.py`,
and `models/heads/builder.py`. They remain **extension points**, not public
imports.

## Concrete task models

> **Internal.** These classes are the concrete implementations selected by the
> top-level builder. They are cataloged for maintainers, not offered as stable
> application imports.

| Task / family | Module | Classes and functions | Builder selection |
| --- | --- | --- | --- |
| Classification multi-view | `models/base/vars.py` | `MVNetwork` | Video encoder route such as Torchvision video models. |
| Classification tracking | `models/base/tracking.py` | `TrackingModel` | `encoder` name `graph_conv`. |
| Classification video | `models/base/video.py` | `VideoModel`, `build_video_mae_backbone`, `load_video_mae_checkpoint` | Feature-extractor/video route; `video_mae` uses the dedicated builder. |
| Localization E2E | `models/base/e2e.py` | `E2EModel` | `MODEL.metadata.family: E2E`. |
| Localization context-aware | `models/base/contextaware.py` | `ContextAwareModel`, `LiteContextAwareModel` | `ContextAware`. |
| Localization pooling | `models/base/learnablepooling.py` | `LearnablePoolingModel`, `LiteLearnablePoolingModel` | `LearnablePooling`. |
| Rule-based HDF5 spotting | `models/base/rule_based.py` | H5 JSON dataset plus header/skeleton spotter variants; `build_rule_based_model` | `RuleBased`. |
| X-VARS VQA | `models/base/xvars_videochatgpt.py` | `XVarsVideoChatGPTCausalLM`, raw-video feature extractors, `XVarsVideoChatGPTModel` | `xvars_videochatgpt`. |
| Qwen-XVARS VQA | `models/base/qwen_xvars.py` | `QwenXVarsCausalLM`, `QwenXVarsModel` | `qwen_xvars_infer`. |
| Native Qwen VL | `models/base/qwen_vl_native.py` | `QwenVLNativeModel`, native collator/trainer/error classes | `qwen_vl_native_infer`. |

The rule-based module contains these named spotter implementations:
`H5HeaderSpotter`, distance/speed/angle variants,
`H5HeaderSkeletonSpotter`, recall variants, and their common base classes.
They are selected by canonical rule-variant configuration, not by direct public
construction.

## Backbone, neck, and head class catalog

| Component area | Internal classes |
| --- | --- |
| Backbones | `BaseExtractFeatures`, `ConvNextTinyExtractFeatures`, `RegnetyExtractFeatures`, `ResnetExtractFeatures`, `PreExtactedFeatures`, `TorchvisionVideoExtractFeatures`, `GraphEncoder`, `GraphSequenceEncoder`, `VideoBackbone` |
| Necks/adapters | `MVAggregate`, `WeightedAggregate`, `ViewMaxAggregate`, `ViewAvgAggregate`, `TemporalAggregation`, `CNN_temporally_aware`, max/average pool variants, `NetRVLAD`, `NetVLAD`, and their temporal/core variants |
| Heads | `MVHead`, `TrackingClassifierHead`, `TemporalE2EHead`, `LinearLayerHead`, `SpottingCALFHead` |

Backbone, neck, and head constructor details are intentionally read from the
builder configuration and component `params`; their exact source signatures
may change with the internal implementations.

## Model utilities

| Module | Non-private classes/functions | Purpose |
| --- | --- | --- |
| `models/utils/common.py` | `ABCModel`, `BaseRGBModel`, `step`, `SingleStageGRU`, `SingleStageTCN` | Common model and optimization-loop support. |
| `models/utils/litebase.py` | `LiteBaseModel` | Lightning base for lite localization families. |
| `models/utils/modules.py` | `FCPrediction`, `GRUPrediction`, `TCNPrediction`, `ASFormerPrediction` | Temporal prediction blocks. |
| `models/utils/shift.py` | `GatedShift`, `make_temporal_shift` | Temporal shift/GSM support. |
| `models/utils/xvars_clip_index.py` | feature/prediction index loaders and tensor validator | VQA feature/index lookup. |
| `models/utils/vqa_prompting.py`, `vqa_prediction_priors.py` | prompt/prior builders | VQA prompt construction. |
| `models/utils/utils.py` | NMS, prediction JSON, timestamp, and result helpers | Localization prediction formatting; some helpers write archives/JSON. |
| `models/utils/impl/` | CALF, ASFormer, GTAD, TSM/GSM implementation classes | Algorithm-specific internal layers. |

Utilities that write prediction JSON, zip results, or load feature files have
filesystem side effects. VQA feature/index utilities require the referenced
local artifacts to exist; model constructors may additionally download Hub
weights when their configured source requests them.

To add a component, first give it a canonical component name and parameters,
then route it through the matching builder and prove it with a config and test.
Do not expose a concrete `nn.Module` as a stable user API merely because a
builder uses it.
