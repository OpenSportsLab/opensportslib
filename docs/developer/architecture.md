# Architecture

OpenSportsLib executes canonical configuration through task-specific dispatch:

```text
YAML/JSON config
  -> Config loader (composition, legacy migration, validation, runtime adaptation)
  -> ClassificationModel | LocalizationModel | VQAModel
  -> datasets.builder.build_dataset + models.builder.build_model
  -> task trainer / inferer / evaluator
  -> metrics and OSL JSON prediction payload
```

`Config.from_file()` composes a canonical task config from root defaults, task defaults, and the selected YAML. Legacy input is migrated at ingestion; runtime model construction accepts canonical config only.

## Ownership boundaries

- `apis/` orchestrates public task operations and explicit prediction saving.
- `core/config/` owns config lifecycle; `core/trainer/` owns task execution routes.
- `datasets/`, `models/`, and `metrics/` provide task-specific implementations.
- `models/builder.py` and `datasets/builder.py` are the central dispatch points.

The package has no separate generic `dataloaders/`, `inference/`, or `evaluation/` package: these responsibilities live in dataset and trainer modules. The server is a separate project that wraps inference through FastAPI/RQ.

## Task dispatch in detail

| Layer | Classification | Localization | VQA |
| --- | --- | --- | --- |
| Public wrapper | `apis/classification.py` | `apis/localization.py` | `apis/vqa.py` |
| Dataset route | `classification_dataset.build(...)` | `LocalizationDataset(...)` | `VQADataset(...)` |
| Model route | Video, graph, VideoMAE, and feature-extractor branches | RuleBased, E2E, ContextAware, or LearnablePooling family | X-VARS VideoChatGPT, Qwen-XVARS, or native Qwen VL backend |
| Execution | `classification_trainer.py` | `localization_trainer.py` | `vqa_trainer.py` |
| Metrics | `classification_metric.py` | `localization_metric.py` | `vqa_metric.py` |

The dataset route is selected by `TASK`; modality-specific handling remains in the task dataset. Classification selects video, frame-array, or tracking behavior from configuration. Localization uses a single entry point whose configured split type and modality select video, feature, tracking, or HDF5 behavior.

The dataset builder and model builder route only `classification`, `localization`, and `vqa`. Another `TASK` token may pass generic config validation but is not a runnable OpenSportsLib task.

## Configuration lifecycle

1. `Config.from_file()` composes config layers for canonical task YAML files.
2. It resolves interpolation, migrates legacy-shaped input when needed, and validates the canonical document.
3. `Config.get_config()` returns a detached resolved dictionary; wrapper construction adapts it to runtime access.
4. The wrapper may apply call-specific split overrides, then routes to the task builder/trainer.
5. `Config.update(...)` and `model.update_config(...)` permit only registered editable options. Options requiring initialization must be set before creating the wrapper.

The loader selects a practical video backend at runtime: DALI is only used for compatible video inputs when available, while CPU and non-video paths use OpenCV where supported. Consume configuration through accessors/runtime adaptation rather than relying on raw YAML shape deep in a model or dataset.

## Model components and persistence

Canonical `MODEL.components` identifies a component by `kind` and `source.name`; `MODEL.topology` describes component connections. `models/builder.py` reads encoder, adapter, head, postprocessor, and VQA decoder settings through config accessors. It does not dynamically discover arbitrary Python classes: a new supported name needs an explicit builder route and a canonical config.

Task wrappers create run directories from `SYSTEM.paths.save_dir` and a run ID. Their public result contract is in-memory prediction data from `infer()` and explicit persistence through `save_predictions(...)`. Do not add hidden writes to `infer()`.
