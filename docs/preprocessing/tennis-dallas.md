# Dallas tennis classification and action spotting

Dallas contains one annotation class, `swing`. The conversion builds a binary
classification dataset with classes `[background, swing]`, and partitions the
original localization annotations by the same whole games:

| Split | Games | Localization swings | Classification clips |
|---|---|---:|---:|
| train | Cilic–Tien, Dimitrov–Michelsen, Fritz–Giron | 2,088 | 4,176 |
| valid | Shapovalov–Jodar | 540 | 1,080 |
| test | Shelton–Diallo | 509 | 1,018 |

Every classification split is balanced. These are Dallas 2026 matches, distinct
from the four-class public tennis conversion. No source match crosses splits.
The test game is used only after selecting a checkpoint on validation loss.

Positive clips contain up to 0.5 seconds before/after their annotation; neighboring
swings constrain the interval at their midpoint. MP4s keep the source 1024×576
resolution and frame presentation times. Background centers are sampled on a
1.25-second grid, at least 2.6 seconds from any annotation. Encoding checks that
the whole output window remains at least two seconds from annotated swings.
Each game supplies as many background clips as annotated swings, using seed 42.
Background is **inferred from annotation gaps, not manually verified negatives**.
This setup can reward camera-view detection and does not establish stroke-type
recognition or spotting quality. Localization manifests retain only original swings.

## Download and convert

Install OSL, PyAV and the optional GPU pose profile described in
[video-to-skeleton](video-to-skeleton.md). Use the `osl` Python environment.

```python
from opensportslib.tools import download_dataset_split_from_hf

download_dataset_split_from_hf(
    "OpenSportsLab/OSL-loc-tennis-dallas",
    "085a5e92597ebc0b943d043f17260879e7c1f3ae", "swing",
    "data/tennis-dallas-source", download_format="json")
```

The pilot pins source revision `085a5e92597ebc0b943d043f17260879e7c1f3ae`.
Check the returned `hf_commit` before rebuilding; the manifests record it.

```bash
python tools/convert/dallas_to_classification.py \
  --manifest data/tennis-dallas-source/085a5e92597ebc0b943d043f17260879e7c1f3ae/swing/swing.json
python tools/upload/upload_tennis_dallas.py --check-only
python tools/upload/upload_tennis_dallas.py
```

The converter resumes completed clips through SHA-256 receipts. Changing source
annotations or conversion settings requires another output root. Source videos
remain intact. Localization output uses symlinks locally, preserving the five
original broadcasts without another media copy.

The uploader uses OSL's Parquet/WebDataset converter for classification, publishes
the three JSON manifests and a dataset card to
[`OSL-cls-tennis-dallas`](https://huggingface.co/datasets/OpenSportsLab/OSL-cls-tennis-dallas),
and publishes localization splits to the existing
[`OSL-loc-tennis-dallas` `swing` branch](https://huggingface.co/datasets/OpenSportsLab/OSL-loc-tennis-dallas/tree/swing).
The source does not declare a dataset license; conversion does not create one.

## Train the two classifiers

The initial pilot runs for up to five epochs. Both use the same clips and 16 uniformly
sampled source frame indices. Repeated source indices are preserved for very
short clips. Both use the existing `VideoDataset`, classification task wrapper,
trainers, neck/head builders and classification metrics.

The skeleton route uses frozen YOLOX-m + RTMPose-s, conservative tennis selection,
two player nodes with 73 features each, four GIN layers of width 64 with LayerNorm,
and temporal max pooling. Tracking resets for every clip. Exact-clip skeleton
caches are explicitly enabled and reused across epochs; this does not change
which frames or frontend observations reach the graph.

Unchanged pose-weight checksums are reused within each process, keyed by path,
file size, modification time and change time. Cached training does not repeatedly read the
large ONNX files to establish each clip's cache identity; changed artifacts are
hashed again and checked against any configured pin.

```bash
python tools/train/train_tennis_dallas.py skeleton --epochs 5
```

With exact-clip caches already warmed on GPU, `--device cpu --num-workers 4`
trains/evaluates the small graph classifier without constructing a pose backend.
CPU mode uses read-only caches and fails explicitly on a missing request. This
can avoid GPU contention with the larger RGB model; measure on your hardware.

The RGB route uses OSL's VideoMAEv2-Base encoder with its final four transformer
blocks trainable, an averaging neck, and a two-class head. The other encoder
weights remain frozen. A deterministic uint8 frame cache applies the checkpoint's
224-pixel short-edge resize and center crop. OSL's frame loader then applies
ImageNet normalization. AMP is enabled through `TRAIN.execution.use_amp`.

Download the exact pretrained snapshot:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    "OpenGVLab/VideoMAEv2-Base",
    revision="78c337a418cc4adaf7f1ff6c7ba343418393966b",
    local_dir="data/dallas-models/videomaev2-base",
    allow_patterns=["*.json", "*.py", "*.safetensors", "README.md"])
```

[VideoMAEv2-Base](https://huggingface.co/OpenGVLab/VideoMAEv2-Base) declares
**CC-BY-NC-4.0** for its weights. This baseline is suitable for a research pilot;
commercial deployment requires a different permitted model or appropriate rights.
Loading the upstream custom architecture uses Transformers `trust_remote_code`;
review the pinned Python files when reproducing.

```bash
python tools/train/train_tennis_dallas.py prepare-video
python tools/train/train_tennis_dallas.py video --epochs 5
```

Outputs under `outputs/dallas-pilot/{skeleton,video}` include the resolved config,
validation-selected checkpoint, test predictions, per-class precision/recall/F1,
confusion matrix, accuracy, balanced accuracy and macro-F1. `results.json` records
the selected epoch and validation loss. Existing results are not silently replaced;
use `--output` for another experiment. Five epochs are a pilot, not evidence of
convergence. A single test match and weak background labels limit interpretation.

## Retrain both models until validation plateaus

From the repository root, using the prepared Dallas pilot data, frame cache,
pose caches and local pretrained snapshot:

```bash
conda activate osl
bash tools/train/train_tennis_dallas_until_convergence.sh
```

This starts **fresh training** with seed 42 and the same models/splits as the
pilot. It does not resume the five-epoch checkpoints. The graph model trains
first on CPU from read-only pose caches, then VideoMAEv2 trains on GPU. Running
sequentially avoids contention with pose extraction. Set `SKELETON_DEVICE=cuda`
if exact-clip caches still need filling; the video frame cache and pretrained
snapshot must already exist.

Defaults are a **100-epoch maximum**, **10 epochs of patience**, and an absolute
validation-loss improvement greater than **0.0001** to reset patience. Small gains
can accumulate against the last significant improvement. Every strict minimum
still qualifies for the best checkpoint, even below this stopping threshold.
The existing ReduceLROnPlateau scheduler halves the LR after two tolerated bad
epochs; its minimum-LR stopping rule also remains active. These are practical
stopping criteria, not a guarantee of mathematical convergence. If the run reaches
the epoch cap, inspect the validation curve before increasing it.

The native classification trainer now honors `TRAIN.selection.patience` (previously
unused) and `TRAIN.selection.min_delta` (default zero). Patience zero disables
validation-plateau stopping; it does not disable the minimum-LR stopping rule.

Each invocation writes a new directory under `outputs/dallas-convergence/`,
keeping pilot results intact. Both model directories contain `config.yaml`,
`training_summary.json` with the validation curve, completed epochs and stop
reason, and `results.json` with the selected checkpoint and test metrics. Each
model evaluates the test game once after checkpoint selection. Use validation,
rather than the already evaluated test game, to choose further training settings.

Override the defaults, devices, batch sizes, or worker counts with environment
variables, for example:

```bash
MAX_EPOCHS=200 PATIENCE=15 MIN_DELTA=0.0001 \
  OUTPUT_ROOT=outputs/dallas-convergence-long \
  bash tools/train/train_tennis_dallas_until_convergence.sh
```

`VIDEO_BATCH_SIZE`/`SKELETON_BATCH_SIZE` default to 4/8;
`VIDEO_WORKERS`/`SKELETON_WORKERS` to 2/4. To run either model separately:

```bash
python tools/train/train_tennis_dallas.py skeleton --device cpu --num-workers 4 \
  --epochs 100 --patience 10 --min-delta 0.0001 \
  --output outputs/dallas-convergence/skeleton-only
python tools/train/train_tennis_dallas.py video \
  --epochs 100 --patience 10 --min-delta 0.0001 \
  --output outputs/dallas-convergence/video-only
```

## Train video and skeleton action spotters

Use the full-broadcast manifests already prepared under
`data/tennis-dallas-localization/{train,valid,test}.json`. These keep only the
original `swing` events and the same 3/1/1 whole-match split. Background is the
implicit frame class zero; do not supply the classification clip manifests or
add a background event label.

```bash
conda activate osl
bash tools/train/train_tennis_dallas_spotting_until_convergence.sh
```

The script starts fresh runs sequentially on GPU:

- **Skeleton:** frozen YOLOX-m/RTMPose-s, two player nodes, four-layer GIN with
  LayerNorm, and the existing bidirectional GRU spotting head. Poses are extracted
  online from full-resolution, uncropped broadcast windows; tracking resets per
  sampled training window.
- **Video:** ImageNet-pretrained RegNetY-008-GSM and the same GRU head, trained
  end to end from RGB resized to 224×398. OSL/timm downloads pretrained weights
  on the first run unless already cached. `OSL_PRETRAINED_WEIGHTS=0` explicitly
  disables them. The classification VideoMAEv2 wrapper emits a clip embedding,
  so it cannot directly replace this frame-aligned spotting encoder.

Both sample 64-frame windows at 25 samples/s, with 32-frame overlap for full-match
evaluation. An epoch samples 64,000 training frames (1,000 windows) and 16,000
validation frames (250 windows). Validation uses seed 42 and the same windows
each epoch, independent of training's random sampling. The default batches are
four graph windows or two RGB windows for training; validation uses one window
per batch for both routes. Both use the existing OpenCV dataset
interfaces and E2E trainer; no DALI installation is needed for this script.

Defaults are a maximum of 100 epochs, patience of 10 validation checks, and
absolute validation-loss improvement greater than 0.0001 to reset patience.
Every strict validation-loss minimum remains eligible for the best checkpoint.
The existing batch-stepped warmup/cosine LR schedule remains in use. Full-match
validation runs once after stopping, followed by one test-match inference using
the selected checkpoint. This final broadcast evaluation can still take time,
particularly for online pose extraction.

Outputs go into a fresh timestamped directory under
`outputs/dallas-spotting-convergence/{run}/{skeleton,video}`. Each contains:

- `config.yaml`, `training_summary.json` and its validation-loss history;
- latest and best checkpoints inside the resolved checkpoint directory;
- explicitly saved `predictions.json`, `metrics.json` and `results.json`, including
  selected/completed epochs, stopping reason, manifest hashes and test metrics.

Test metrics use OSL's common spotting evaluator for both routes: mAP at
0/1/2/3/4-second tolerances and their mean, plus exact-frame event foreground F1.
The script normalizes video paths and applies two-frame local-max suppression
to RGB events; skeleton streaming already performs this suppression within shots.
The full-validation console summary uses the legacy evaluator and need not equal
the test metric definition. Temporal windows and cut handling differ between the
existing RGB and skeleton inference routes. Both have bidirectional lookahead;
these models do not provide fully causal spotting.

To inspect the generated YAMLs without loading weights or starting training:

```bash
CONFIG_ONLY=1 OUTPUT_ROOT=outputs/dallas-spotting-configs \
  bash tools/train/train_tennis_dallas_spotting_until_convergence.sh
```

For a longer run or smaller per-epoch sample budget:

```bash
MAX_EPOCHS=200 PATIENCE=15 EPOCH_NUM_FRAMES=32000 \
  OUTPUT_ROOT=outputs/dallas-spotting-long \
  bash tools/train/train_tennis_dallas_spotting_until_convergence.sh
```

`EPOCH_NUM_FRAMES` must be a multiple of `CLIP_LEN` and include at least four
windows. `VIDEO_BATCH_SIZE`, `SKELETON_BATCH_SIZE`, `VIDEO_WORKERS` and
`SKELETON_WORKERS` are configurable; workers default to zero. For a lighter RGB
encoder set `VIDEO_BACKBONE=rny002_gsm`. You can also run either route separately:

```bash
python tools/train/train_tennis_dallas_spotting.py skeleton \
  --epochs 100 --patience 10 --min-delta 0.0001 --output outputs/dallas-spotting/skeleton
python tools/train/train_tennis_dallas_spotting.py video \
  --epochs 100 --patience 10 --min-delta 0.0001 --output outputs/dallas-spotting/video
```

Skeleton caching defaults to `off`: random broadcast windows are distinct from
the 16-frame classification clips, and many will never repeat. Explicitly set
`SKELETON_CACHE_MODE=read_write` to cache exact spotting requests; monitor disk
usage as new random windows accumulate. `SKELETON_CACHE_DIR` chooses the location.
Continuous streaming inference remains separate from these independent-clip
caches. GPU pose dependencies are required; the classification caches cannot
be substituted to make the initial spotting run CPU-only.

E2E trainers accept `TRAIN.selection.patience` (default zero/disabled) and
`min_delta` (default zero). For mAP selection, patience counts only full-video
measurements scheduled by `valid_map_every`. OpenCV validation windows can be
fixed with `TRAIN.execution.validation_seed`; omitting it preserves random
sampling. The Dallas scripts enable this setting for both routes.

## Verification

Run `bash scripts/run_tests.sh`. Owning tests cover background exclusion distances,
match splits, preservation of every localization event, timestamp-preserving clips,
conversion resume, RGB frame-cache loading, two-class configurations, canonical AMP
resolution and actual gradients through VideoMAEv2's selected nested encoder blocks.
Validation-stopping tests cover minimizing/maximizing metrics, accumulated small
improvements, disabled patience, and checkpoint correctness at LR reductions/stops.
Spotting tests cover fixed validation requests, sparse mAP checks, zero-mAP best
checkpoints, stopping-epoch saves, fast NMS equivalence, and real CPU video/graph
training, checkpoint reload and inference with an external pose-backend fixture.
GPU model probes and the real-data pilot are separate from the offline suite.

## Measured five-epoch pilot (21 October 2026)

Both models trained for five epochs with seed 42. Checkpoints were selected by
minimum validation loss, then evaluated on all 1,018 Shelton–Diallo test clips
(509 background, 509 swing). The RGB encoder fine-tuned its final four blocks on
the NVIDIA GB10. The graph model trained on CPU using GPU-extracted frozen-pose
caches; a loader probe favored this route and avoided GPU contention.

| Model | Selected epoch | Test accuracy / balanced accuracy | Test macro-F1 |
|---|---:|---:|---:|
| VideoMAEv2-Base | 4 | 99.51% | 99.51% |
| Frozen YOLOX-m + RTMPose-s, four-layer GIN | 5 | 95.09% | 95.08% |

The RGB model made five errors (three background, two swing); the graph model
made 50 (44 background, six swing). A diagnostic that simply predicts swing
whenever a player observation is present scored **94.50%**, demonstrating strong
view/selection bias in this binary dataset. Of the test clips, 454 background
clips and one swing clip had no valid player observations at all. These results
do not establish fine-grained stroke recognition or performance on other matches.

All published Parquet sample payloads matched their JSON manifests. An OSL
download of the complete test split preserved all 1,018 MP4 checksums, labels and
metadata. The final graph checkpoint passed exact live/cache canonical-record and
graph-feature comparisons on a validation clip; live/cache and CPU/GPU logits
agreed within `atol=1e-5`, `rtol=1e-4`. The offline suite passed 708 tests, with five
optional tests skipped.

Local artifacts are under `outputs/dallas-pilot`: `comparison.json`,
`comparison.png`, each model's `results.json`, configs, checkpoint paths and test
predictions. Training wall times include GPU scheduling pauses and shared resource
use; they are not dedicated throughput benchmarks.
