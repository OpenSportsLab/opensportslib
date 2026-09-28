# Training, Inference, Evaluation, and Metrics API

> **Extension point.** Task trainers, inferers, evaluators, and metric
> functions are implementation-level contracts between a task wrapper and its
> data/model route.

## Runtime module map

| Area | Modules and non-private entry points | Role |
| --- | --- | --- |
| Classification | `Trainer_Classification` and trainer subclasses | Selects video, tracking, or frame training behavior. |
| Localization | `build_trainer`, `build_inferer`, `build_evaluator`, `Trainer_*`, `Inferer`, `Evaluator` | Routes model families and runners. |
| VQA | `Trainer_VQA`, LoRA trainer classes, SFT datasets/collators | Implements backend-specific fine-tuning and inference preparation. |
| Classification metrics | `process_preds_labels`, `compute_classification_metrics` | Computes classification scores from logits or labels. |
| Localization metrics | AP/mAP functions, prediction processors, result writers | Computes spotting metrics and may write evaluation artifacts. |
| VQA metrics | `compute_vqa_metrics` | Computes answer matching and configured referee-semantic metrics. |

`store_eval_files_json`, detailed classification metrics, and trainer/checkpoint
operations write files under their supplied/configured directories. Evaluation
functions may require ground truth in the task’s expected representation.

## Classification runtime

`Trainer_Classification` selects one concrete trainer based on the configured
input route. `BaseTrainerClassification` provides common behavior;
`MVTrainerClassification`, `TrackingTrainerClassification`, and
`FramesTrainerClassification` implement the multi-view, tracking, and frame
paths respectively. These classes are **internal**; `ClassificationModel` is
the stable operation entry point.

::: opensportslib.core.trainer.classification_trainer
    options:
      members:
        - BaseTrainerClassification
        - MVTrainerClassification
        - TrackingTrainerClassification
        - FramesTrainerClassification
        - Trainer_Classification
      show_source: true

::: opensportslib.core.trainer.localization_trainer
    options:
      members:
        - build_trainer
        - build_inferer
        - build_evaluator
        - Trainer
        - Trainer_pl
        - Trainer_e2e
        - Inferer
        - Evaluator
      show_source: true

::: opensportslib.core.trainer.vqa_trainer
    options:
      members:
        - build_vqa_sft_text
        - XVarsVideoChatGPTDataCollator
        - XVarsVideoChatGPTTrainer
        - VQAXVarsVideoChatGPTSFTDataset
        - VQANativeQwenVLSFTDataset
        - VQAXVarsVideoChatGPTLoraTrainer
        - VQAQwenXVarsLoraTrainer
        - VQAQwenVLNativeLoraTrainer
        - Trainer_VQA
      show_source: true

## VQA runtime

The VQA module keeps backend-specific collators, supervised fine-tuning
datasets, and LoRA trainer classes beside `Trainer_VQA`. They are **internal**
implementation classes selected by the configured backend and training mode;
they may save adapters, checkpoints, and metadata in the configured output
directory. Use `VQAModel.train()`, `infer()`, and `evaluate()` for the public
contract.

## Metric signatures

`opensportslib/metrics/` is a namespace-style source directory without an
`__init__.py`, so MkDocstrings cannot resolve it as a dotted package. Its
non-private metric entry points are maintained as source-linked signatures:

```python
# classification_metric.py
process_preds_labels(eval_pred, top_k=None)
compute_classification_metrics(eval_pred, top_k=None, mode="logits")
compute_detailed_classification_metrics(all_logits, all_labels, class_names, save_dir, set_name)

# localization_metric.py
compute_average_precision(pred, truth, tolerance=0, min_precision=0, plot_ax=None, plot_label=None, plot_raw_pr=True)
compute_mAPs_E2E(truth, pred, tolerances=[0, 1, 2, 3, 4], plot_pr=False)
build_snpro_prediction_json(pred_events, head_name="action", split=None, created_by="model")
infer_and_process_predictions_e2e(...)
compute_performances_mAP(...)

# vqa_metric.py
compute_vqa_metrics(predictions, dataset, eval_profile=None)
```

These functions are **extension points**. Their source files are
`metrics/classification_metric.py`, `metrics/localization_metric.py`, and
`metrics/vqa_metric.py`; detailed localization arguments should be read from
the source before extending an evaluation route.
