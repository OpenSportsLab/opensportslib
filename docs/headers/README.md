# Header spotting from skeletal tracking

Detect headers directly from 3-D player joints and ball tracking, with no
trained model, and score the result against video annotations using the
library's action-spotting mAP.

## What this needs

| Input | Description |
|---|---|
| `live_joints.h5` | Player joints per frame, with `frame`, `player_id`, and `<joint>_x/y/z` columns |
| `live_ball.h5` | Ball track with `frame`, `x`, `y`, `z`, `half`, `timestamp_utc` |
| Annotation JSON | OSL v2 file holding `Header` and `Kickoff` events plus the video's `UTC_time_start` |

The annotation file carries both clocks, so no separate synchronisation file is
needed. Tracking data and annotations are not part of the repository; pass your
own with `--annotations` and `--data-root`.

Headers are annotated as positions inside the video file, while tracking rows
are stamped in UTC. The video's `UTC_time_start` bridges the two:

```
utc = UTC_time_start + position_ms          position_ms = utc - UTC_time_start
```

## Methods

Five rule variants are available. All require the ball to come close to a
player's head; they differ in what else they demand.

| Variant | Additional requirement |
|---|---|
| `skeleton` | Ball at head height, path bends at contact, hands farther from the ball than the head, player facing the ball, no acrobatic pose, contact is brief |
| `distance` | Nothing further |
| `distance_angle` | Ball direction changes by at least 25° |
| `distance_speed` | Ball speed changes by at least 25% |
| `distance_speed_angle` | Both the angle and the speed change |

`skeleton` is the strongest and the fastest. Thresholds live in
`DEFAULT_SKELETON_RULE_PARAMS` in `opensportslib/models/base/rule_based.py` and
can be overridden from a config.

## Running it

Everything runs on CPU. Use an environment with `h5py` installed.

Run every variant:

```bash
python scripts/run_header_spotting_eval.py
```

Run one variant:

```bash
python scripts/run_header_spotting_eval.py --variants skeleton
```

Point it at another game:

```bash
python scripts/run_header_spotting_eval.py \
    --annotations /path/to/annotations.json \
    --data-root /path/to/h5/root \
    --output-dir outputs/my_game
```

Re-score without re-detecting, once predictions are cached:

```bash
python scripts/run_header_spotting_eval.py --eval-only
```

Force fresh detection after changing a threshold:

```bash
python scripts/run_header_spotting_eval.py --variants skeleton --force
```

### Options

| Flag | Default | Meaning |
|---|---|---|
| `--annotations` | `WC22_multi.json` | Annotation JSON with headers, kickoffs and the video UTC start |
| `--data-root` | `/home/giancos/FIFA_data` | Directory the H5 paths in the annotation file resolve against |
| `--output-dir` | `outputs/header_spotting` | Where manifests, predictions and results are written |
| `--variants` | `all` | Comma-separated subset of the five variants |
| `--force` | off | Re-detect even when raw predictions are cached |
| `--eval-only` | off | Skip detection, only convert and score |

## What it produces

```
outputs/header_spotting/
├── manifest.json                     inputs and scan window given to the model
├── config_<variant>.yaml             generated config per variant
├── raw/predictions_<variant>.json    detections on the UTC clock, with diagnostics
├── video_clock/                      detections on the video clock, in play only
└── map_results.json                  the scores below, as JSON
```

Each raw detection carries diagnostics: contact distance, the player, the ball
height, the measured trajectory change and the dwell count. Useful for
inspecting why something was or was not detected.

## Reading the output

```
variant                   #pred    mAP@1s    mAP@2s   tight avg   rec@1s  prec@1s
skeleton                    111    63.03%    64.68%      64.85%    86.5%    81.1%
```

- `mAP@Ns` — mean average precision allowing a prediction to sit up to N
  seconds from the annotation.
- `tight avg` — the mAP values averaged over 1 to 5 seconds. The headline score.
- `rec@1s` — share of annotated headers found within one second.
- `prec@1s` — share of predictions that correspond to a real header.

Scoring is confined to the play windows, which run from each kickoff to the
last tracked ball sample of that half. Extra time is excluded from both the
annotations and the predictions, so the comparison stays fair.

## Three stages

The script runs these in order; each can be repeated on its own.

1. **Extract** — write a manifest naming the H5 pair and the scan window, then
   run the variant through `LocalizationModel.infer()`. Detections come out on
   the UTC clock.
2. **Convert** — subtract the video's UTC start to place each detection on the
   video clock, and drop anything outside the play windows.
3. **Evaluate** — bin annotations and predictions at 20 ms, then score with
   `delta_curve` from `opensportslib/metrics/localization_metric.py`.

## How the model is wired in

The variants are a `RuleBased` model family: no weights, no training, but the
same config schema, inference API and JSON output as a trained model.

- `family: RuleBased` in a config routes the builder to
  `build_rule_based_model` in `opensportslib/models/base/rule_based.py`.
- Dataset type `H5OSLJsonSpotting` reads the manifest and skips the dataloader.
- Runner `runner_h5_header_rule` makes `infer()` call `model.predict()` directly.
- `train()` raises `NotImplementedError`; these models are inference only.

A ready-made config for the strongest variant is at
`opensportslib/configs/localization/h5_header_skeleton.yaml`. To use a variant
directly, without the evaluation pipeline:

```python
import os
os.environ.setdefault("RUN_ID", "headers")
from opensportslib.apis import LocalizationModel

api = LocalizationModel(
    config="opensportslib/configs/localization/h5_header_skeleton.yaml")
predictions = api.infer(use_wandb=False)
api.save_predictions("predictions.json", predictions)
```

## Tests

```bash
python -m pytest tests/test_h5_header_skeleton_spotter.py \
                 tests/test_h5_header_rule_spotter.py
```

These build small synthetic H5 files, so they need no tracking data.

## Troubleshooting

**No events detected.** Check the scan window in `manifest.json` overlaps the
tracking data, and that ball `z` values are in metres. The `skeleton` variant
only considers the ball between 1.3 m and 3.0 m.

**Predictions land seconds away from annotations.** The clock bridge is wrong.
Confirm `UTC_time_start` by taking a kickoff's `position_ms`, adding it to
`UTC_time_start`, and checking it matches that kickoff's `timestamp_utc`.

**`ImportError` on `LocalizationModel`.** An installed copy of the library is
shadowing the checkout. The script prepends its own repository to `sys.path`;
if you import the modules yourself, do the same or install with `pip install -e .`.
