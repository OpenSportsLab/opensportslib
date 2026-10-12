# Training, Inference, and Evaluation

This page shows the shared workflow for the three supported task APIs. A task
wrapper is the Python object you use for one task. The examples use the public
Python API; `tools/training/` provides equivalent command-line scripts.

## Before running

- Install the dependency profile required by your configuration. DALI needs an NVIDIA GPU; graph/tracking workflows may need `--pyg`; X-VARS and Qwen VQA each need their own dependency profile.
- Use a supported YAML file under `opensportslib/configs/` or a compatible copy.
- Provide OSL JSON manifest files and the media files they reference. The package does not include a training dataset.
- Read the [configuration guide](../config/configuration-guide.md) and [OSL JSON format guide](../data/osl-json-format.md) before editing either file type.

## Common lifecycle

```python
from opensportslib.apis import ClassificationModel

model = ClassificationModel(config="/path/to/config.yaml", weights=None)
checkpoint = model.train(train_set="/path/to/train.json", valid_set="/path/to/valid.json")
predictions = model.infer(test_set="/path/to/test.json", weights=checkpoint)
metrics = model.evaluate(test_set="/path/to/test.json", predictions=predictions)
model.save_predictions("/path/to/predictions.json", predictions)
```

Split arguments replace the corresponding configured annotation path for that
one call. `infer()` returns predictions but does not save them automatically.
Pass those predictions, or their saved path, to `evaluate()` to avoid running
inference a second time.

## Task routes

### Classification

Use `ClassificationModel` with `opensportslib/configs/classification/video.yaml`, `sngar_frames.yaml`, or `sngar_tracking.yaml` as the starting point matching your input modality. The script route is:

```bash
python tools/training/classification.py --config /path/to/classification.yaml \
  --train-set /path/to/train.json --valid-set /path/to/valid.json --test-set /path/to/test.json
```

### Localization / action spotting

Use `LocalizationModel`. The supported configurations cover OpenCV and DALI
video inputs, CALF/NetVLAD features, tracking-based action spotting, E2E
SpoTTA, and selected HDF5 header spotters. The loader is chosen from the input
and available hardware: compatible non-video inputs and CPU runs use OpenCV
where appropriate.

```bash
python tools/training/localization.py --config /path/to/localization.yaml \
  --train-set /path/to/train.json --valid-set /path/to/valid.json --test-set /path/to/test.json
```

See [tracking action spotting](../spotting/tracking-action-spotting.md) and the [headers guide](../headers/README.md) for specialist routes.

### VQA

Use `VQAModel` and first install either `opensportslib setup --vqa_xvars` or `opensportslib setup --vqa_qwen`. VQA training can resume a Hugging Face Trainer checkpoint.

```bash
python tools/training/vqa.py --config /path/to/vqa.yaml \
  --train-set /path/to/train.json --valid-set /path/to/valid.json \
  --resume-from-checkpoint /path/to/checkpoint
```

`--skip-infer` makes the script train only; `--use-wandb` opts into Weights & Biases logging. See the [VQA setup guide](../tools/vqa.md) for backend-specific data and feature requirements.

## Checkpoints and pretrained models

Pass either a local checkpoint path or a Hugging Face model ID through
`weights`. If you do not supply a configuration, a Hub model repository must
include a compatible OpenSportsLib `config.yaml`; a Transformers-only
`config.json` is not enough. The [model zoo](../model-zoo.md) links each model
card and its recommended configuration.

## Multi-GPU and remote inference

Task behavior is controlled by the supported `SYSTEM` and `TRAIN.execution`
settings. Use the configuration reference for those settings rather than shell
launchers that this repository does not provide. SLURM wrappers are documented
in the [SLURM guide](../getting-started/slurm.md).

For remote prediction, construct a wrapper with `remote=...` and use its normal inference methods. The server owns model loading and accepts only inference; see [inference server](../server/inference-server.md).
