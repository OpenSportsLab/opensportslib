# T-DEED development and reproduction

**Task:** action localization / spotting. **Paper / algorithm:** T-DEED.
This is one entry in the [action localization reproduction catalog](index.md),
which will grow to include other algorithms. The [paper reproduction overview](../index.md)
groups papers across all tasks. Current validation covers the released
FineDiving runs; full paper reproduction remains in progress.

See [results and model releases](tdeed-results.md) for measured benchmark
comparisons and the publication layout for configs, trained weights and provenance.
All T-DEED datasets, architectures, seeds and recipe/data versions share the
`OpenSportsLab/OSL-loc-TDEED` model repository, with separate named release
revisions and an index on `main`.

T-DEED is a localization family in the existing `LocalizationModel` wrapper.
The port uses [the authors' extended release](https://github.com/arturxe2/T-DEED)
at `d953b4de8a2ad941c565e1683c5b07255cbe644e`. The reference primitives retain
their GPLv3/E2E-Spot notices and licenses. Existing task APIs remain compatible.

```python
from opensportslib.apis import LocalizationModel

model = LocalizationModel(
    config="opensportslib/configs/localization/tdeed/FineDiving_small.yaml"
)
checkpoint = model.train(use_wandb=False)
predictions = model.infer(weights=checkpoint, use_wandb=False)
model.save_predictions("outputs/predictions.json", predictions)
metrics = model.evaluate(weights=checkpoint, use_wandb=False)
```

Use the campaign CLI for benchmark preflight checks. Direct API calls support
custom datasets and development fixtures; they do not certify paper reproduction.

## Implementation

- `models/base/tdeed.py`: RegNetY-200MF/800MF, GSF/GSM, learned temporal positions,
  SGP encoder-decoder, classification/displacement heads, weighted CE, background-
  inclusive displacement MSE and within-dataset mixup. Original checkpoint keys
  are retained. ImageNet checkpoint revisions are immutable.
- `datasets/tdeed.py`: native frame targets, released sliding-window sampling,
  black padding, foreground-only AS sampling and shared FigureSkating broadcasts.
  Joint BAS/AS samples datasets with probability one half and doubles epoch size.
- `core/trainer/tdeed_trainer.py`: deterministic training, validation selection,
  overlapping inference, test flips and epoch-boundary checkpoint resume.
- `metrics/tdeed.py`: PES event AP, greedy NMS, quadratic Soft-NMS and official
  SoccerNet scoring. FG-Start includes the 14 `_start` classes plus `VT_1`, `VT_2`.

Input is RGB in the 0–255 range, shaped `(B, clip_len, 3, H, W)`. The network owns
crop, augmentation and ImageNet normalization. Do not normalize in the loader.
Temporal length is fixed by position embeddings and shift modules. Missing frames
raise errors. Displacement relocation takes classwise maximum votes without
re-normalization; overlap support counts match the reference's nonzero votes.

`infer()` returns OSL JSON without saving predictions. Its event frame and
millisecond axes are sampled and performance-relative; source start/stride are
recorded in metadata. `evaluate(predictions=...)` scores a fixed prediction set.
`evaluate()` reports raw, NMS and Soft-NMS independently, without choosing a
postprocessor from test labels. PES tolerances are frame distances; SoccerNet's
official tolerances are full matching-window widths.

## Experiment recipes and differences

Canonical YAML recipes live in `opensportslib/configs/localization/tdeed/`.
`reproduction/tdeed/` contains original configurations, reference-file hashes,
published targets and the experiment matrix. Independent seeds `1`, `2` are
explicit choices; the authors' exact seed pair is unavailable.

Released settings and paper settings sometimes differ. Small FigureSkating
release configs use SGP ratio four; `_paper` configs use the paper's two.
FineGym and AS release configs use six warmup epochs. The released chained
scheduler advances warmup and cosine together; this behavior is retained.
AS stride 12 at 25 fps gives 2.0833 fps. Released AS clip-validation selection
uses a separate 6.25-fps scoring clock (`released_snas_clips`); full-game test
scoring uses the actual sampling clock.

Controlled ablations cover GRU/Transformer/SGP temporal modules, skip variants,
SGP-Mixer summation, displacement/dilation, shift scope, clip length and
postprocessing. The exact search-selected ablation recipes are not released,
so these configs are marked `controlled-ablation`, not exact best-search replicas.
The adapted TriDet search and private BAS challenge evaluations remain explicit
matrix entries requiring unavailable recipes/labels. Challenge scores also need
the exact all-split training/ensemble protocol. No leaderboard submission occurs.

## Data preflight

See [dataset preparation](../../tools/spotting-dataset-cleanup.md) for repositories,
permissions and source provenance. Configure local paths before launching.

```bash
python tools/reproduction/tdeed.py preflight \
  --config opensportslib/configs/localization/tdeed/FineDiving_small.yaml \
  --verify-hashes --output outputs/tdeed/preflight-finediving
```

Preflight checks pinned reference-file hashes, class order, split membership,
native events where available, declared frame counts and media presence.
`--verify-hashes` additionally verifies available media checksums.

- FineDiving resizes original JPEG-derived video frames to 224×224.
- FigureSkating resizes original broadcasts to 398×224, then crops to 224×224.
  Two source broadcasts have documented replacement encodings.
- FineGym requires all 5,374 padded clips. `--recover-originals` on the rebuild
  tool can recover original high-rate videos when archive versions fail timing
  checks; the same strict frame-geometry validation still applies.
- Tennis's current 144p media fails strict paper-resolution preflight.
- BAS's annotation-version discrepancy remains a strict reproduction blocker.
- AS/BAS require local frame-geometry adapters and their media. AS annotations
  outside the reference's 45-minute window remain ground truth for scoring;
  training windows retain the reference bounds.

Normalize existing OSL soccer manifests with `prepare_tdeed_soccer.py`:

```bash
python tools/reproduction/prepare_tdeed_soccer.py \
  --dataset soccernet --source-root /data/soccernet/224p-json \
  --fallback-source-root /data/soccernet/720p-json \
  --reference-root /data/reference/T-DEED/data/soccernet \
  --output data/spotting-cleanup/SoccerNet \
  --media-root data/spotting-cleanup/SoccerNet/official/videos-224p
```

Eighteen irregular-FPS 224p sources use corresponding 25-fps 720p files.
New preparations prefer `SoccerNet/SoccerNet_raw_HQ`, pinned to
`56a8be6be6788ed65cce25f68780d55ff4544aa2` (`videos-224p`) and
`280e9828acb8f99250eefdeb4c0eae9a1f784739` (`videos-720p`).
Per-video `media_repo_id` and `media_revision` identify the actual source;
old manifests retain their historical provenance rather than being relabelled.
Official media still requires sequential decode validation: container frame
counts and successful seeks near the end do not establish the number of decoded
frames or alignment of native frame indices.
The 2026-10-26 comparison found 63 short files among 65 changed official 224p
assets after full sequential decoding; the failing half's official 720p copy
also decoded short. Evidence and the repeatable audit command are in
`reproduction/training-runs/tdeed/diagnostics/soccernet-source-audit-20261026/`.
Existing normalized manifests are retained with their original media identity;
changing the preferred repository does not certify those experiments as ready.
`--download-media` acquires missing AS media subject to free-space reservations.
For existing normalized manifests, use `--existing-manifests` with `--output`,
`--media-root`, `--dataset soccernet` and `--download-media`; this downloads the
pinned media without rewriting annotations. `--reserve-gb` controls the free
space retained after acquisition (default 500 GB).
Joint BAS/AS uses a separate `SoccerNet448` adapter over high-resolution AS
videos, resized to 796×448; independent AS's low-resolution data cannot replace
it. Acquire BAS shards with the existing OSL split downloader, then normalize
the extracted manifests. Pixel identity to the authors' earlier resized JPEG
collections is not claimed for these explicit video preprocessing choices.

## Run, resume and compare

The publication campaign runs main results first, with SoccerNet taking
priority as soon as its data passes preflight:

```bash
python tools/reproduction/tdeed.py campaign --resume --verify-hashes \
  --stage main --skip-blocked \
  --priority SoccerNet_small --priority SoccerNet_big \
  --priority SoccerNetBall_challenge1 --priority SoccerNetBall_challenge2 \
  --output outputs/tdeed/campaign
```

`--stage main` includes `released-main` and `paper-main`. It excludes BAS-only
and controlled ablations. `--skip-blocked` rechecks pending blocked datasets
between runs, so completed downloads can promote SoccerNet ahead of other
pending work. It does not interrupt an active training run. If only blocked
jobs remain, the campaign exits and retains their errors in the ledger; resume
the same command after fixing prerequisites. Aggregates update after each run.

After publishing the main results, run ablations explicitly with
`--stage ablations --resume --skip-blocked` against the same output directory
to reuse the main checkpoints. The default `--stage all` retains the full-matrix
behavior. No CLI command publishes results or code automatically.

Individual experiments can also be selected:

```bash
python tools/reproduction/tdeed.py campaign --resume \
  --experiment FineDiving_small --experiment FineDiving_big \
  --output outputs/tdeed/campaign

python tools/reproduction/tdeed.py report --output outputs/tdeed/campaign
```

Omit experiment restrictions to visit the full matrix. `--group` selects
`released-main`, `paper-main`, `bas-ablation` or `controlled-ablation`.
Blocked data and failed runs remain in the ledger. Resume after prerequisites
are fixed. Postprocessing variants reuse their parent checkpoints.

Each run retains settings/environment, progress, history, `last.pt`, validation-
selected `best.pt`, explicit predictions and final metrics. Checkpoints preserve
optimizer, scaler, scheduler and Python/NumPy/CPU/CUDA RNG states. Resume requires
unchanged recipes, implementation hashes and dataset-manifest hashes. Seed
directories prevent overwriting independent runs.

Training console logs and small run artifacts are also retained in the
Git-visible `reproduction/training-runs/tdeed/` archive. Launch campaigns through
`tools/reproduction/record_training.py` to capture the command, source revision,
working-tree patch, determinism environment, histories and final metrics. See
`reproduction/training-runs/README.md` for the command and regression comparison
procedure. The recorder mirrors JSON
artifacts every 30 seconds; checkpoints and predictions remain in `outputs/`.

The ledger reports per-seed scores, means, standard deviations, published targets
and signed percentage-point gaps. Completing training never sets
`paper_reproduced` to true. Missing coverage, settings, data/environment differences
and rerun evidence must be resolved before a reproduction claim or PR approval.

## Verification

```bash
python tools/reproduction/check_tdeed_parity.py \
  --reference-root /data/reference/T-DEED --output outputs/tdeed/parity.json
python tools/reproduction/smoke_tdeed.py \
  --config opensportslib/configs/localization/tdeed/FineDiving_small.yaml \
  --output outputs/tdeed/gpu-repeat
bash scripts/run_tests.sh
```

Parity checks an independent pinned checkout's weights, forward values,
relocation, loss, gradients, PES AP and suppression. Real-data smoke runs check
repeated weights/histories bit for bit. Integration tests compare resumed and
continuous training and verify inference does not write prediction files.

Strict CUDA determinism stays enabled. Temporal resampling keeps native forward
operations and uses fixed-order backward reductions where PyTorch's atomic
backward is unsupported. These gradients are checked analytically; half-precision
accumulation can differ from the nondeterministic reference. The GB10 environment
uses newer PyTorch/timm than `requirements.reference.txt`; versions and this
implementation choice are recorded rather than presented as the original stack.
