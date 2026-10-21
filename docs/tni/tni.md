# Training, Inference, and Evaluation

For the T-DEED localization family, see [development and reproduction](tdeed.md):
canonical YAML recipes, frame-indexed data, deterministic checkpoints, evaluation
protocols and the local experiment campaign.

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

For tennis stroke classification with online frozen skeleton extraction and the
existing graph classifier, see the
[dataset conversion and training launcher](../tools/tennis-classification.md).

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

For public tennis with model-owned pose/tracking, the localization examples use
[`OpenSportsLab/OSL-loc-tennis-public`](https://huggingface.co/datasets/OpenSportsLab/OSL-loc-tennis-public)
at revision `swing-data`, configured for the local `tennis_action` annotations:
six classes covering near/far court bounces, serves, and swings. Both examples
select checkpoints by validation loss and run full spotting evaluation after
training. Follow the
[download and training commands](../preprocessing/video-to-skeleton.md#public-tennis-localization-training)
for the OpenCV or DALI example.

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

For remote prediction, construct a wrapper with `remote=...` and use its normal inference methods. The server owns model loading and accepts only inference; see [inference server](../server/inference-server.md). Hugging Face registration supports canonical and legacy rule-runner configs without weights, as well as learned checkpoints including `.pth.tar` files.

## Frozen skeleton action inputs

The existing graph classification and E2E spotting models also accept frozen
model-owned video-to-skeleton preprocessing with ordinary video annotations/loaders.
Training extracts clips live with optional exact-clip caching; inference can extract directly from a video. Windowed streaming retains
bidirectional lookahead and returns finalized event batches. See the
[video-to-skeleton guide](../preprocessing/video-to-skeleton.md) for installation,
examples, limitations, and the public-tennis benchmark procedure.
After training, the [single-video inspection script](../../tools/inference/README.md)
exports actual model boxes/poses and overlays them with clip classifications or
timestamped localization events. It uses the checkpoint's saved configuration
and explicitly writes predictions, a pose trace and an annotated MP4.

For Dallas broadcasts, see the [binary classification pilot](../preprocessing/tennis-dallas.md):
three training games, one validation game, one test game, timestamp-preserving
swing/background clips, and paired RGB/graph training scripts.
The guide includes retraining until validation plateaus. Native classification
trainers honor `TRAIN.selection.patience` and `min_delta`, retaining the strict best
checkpoint while stopping on insufficient validation improvement. The epoch cap,
validation curve and stopping reason are recorded in `training_summary.json`.
For action spotting, use `tools/train/train_tennis_dallas_spotting_until_convergence.sh`.
It uses the original broadcast annotations, RegNetY-GSM/GRU versus pose/GIN/GRU,
fixed OpenCV validation windows (`TRAIN.execution.validation_seed`), and E2E
validation stopping (`TRAIN.selection.patience`/`min_delta`).
The spotting launcher repeats fixed training windows and reuses skeleton caches
by default. Set `TRAINING_SAMPLING=random` for fresh windows; Python exposes
`--training-sampling`, `--sampling-seed` and `--cache-mode`. The dataset-level
fixed-pool setting is `TRAIN.execution.training_seed`.
