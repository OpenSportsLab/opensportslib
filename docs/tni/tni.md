# Training, Inference, and Evaluation

This page covers the common lifecycle for the three implemented task wrappers. It uses real Python APIs; the `tools/training/` scripts provide equivalent command-line entry points.

## Before running

- Install the dependency profile required by your config. DALI requires an NVIDIA GPU; graph/tracking paths may require `--pyg`; X-VARS and Qwen VQA require separate dependency profiles.
- Use a canonical YAML under `opensportslib/configs/` or a compatible copy.
- Supply OSL JSON manifests and the media they reference. The package does not include a training dataset.
- Review [configuration](../config/configuration-guide.md) and [OSL JSON](../data/osl-json-format.md).

## Common lifecycle

```python
from opensportslib.apis import ClassificationModel

model = ClassificationModel(config="/path/to/config.yaml", weights=None)
checkpoint = model.train(train_set="/path/to/train.json", valid_set="/path/to/valid.json")
predictions = model.infer(test_set="/path/to/test.json", weights=checkpoint)
metrics = model.evaluate(test_set="/path/to/test.json", predictions=predictions)
model.save_predictions("/path/to/predictions.json", predictions)
```

Split arguments override the corresponding configured annotation path for that call. `infer()` does not implicitly save its returned payload. Pass a payload or saved path to `evaluate()` to avoid a second inference run.

## Task routes

### Classification

Use `ClassificationModel` with `opensportslib/configs/classification/video.yaml`, `sngar_frames.yaml`, or `sngar_tracking.yaml` as the starting point matching your input modality. The script route is:

```bash
python tools/training/classification.py --config /path/to/classification.yaml \
  --train-set /path/to/train.json --valid-set /path/to/valid.json --test-set /path/to/test.json
```

### Localization / action spotting

Use `LocalizationModel`. The canonical configs include OpenCV and DALI video routes, feature-based CALF/NetVLAD routes, tracking action spotting, E2E SpoTTA, and selected HDF5 header spotters. Loader selection is not merely a preference: non-video modalities and CPU execution fall back to OpenCV where applicable.

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

Pass a local checkpoint path or a Hugging Face model ID through `weights`. If no explicit config is supplied, a Hub model ID must provide a compatible OpenSportsLib `config.yaml`; a Transformers-only `config.json` is insufficient. The [model zoo](../model-zoo.md) links model cards and their recommended configurations.

## Multi-GPU and remote inference

Task behavior is controlled by canonical `SYSTEM` and `TRAIN.execution` settings; use the configuration reference rather than shell launchers not provided by this repository. SLURM wrappers are documented in the [SLURM guide](../getting-started/slurm.md).

For remote prediction, construct a wrapper with `remote=...` and use its normal inference methods. The server owns model loading and accepts only inference; see [inference server](../server/inference-server.md).
