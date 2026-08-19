# Header spotting from skeletal tracking

Detect headers directly from 3-D player joints and ball tracking, with no
trained model, and score the result against video annotations using the
library's action-spotting mAP.

## Data

One directory per game, each holding two files:

| File | Required columns |
|---|---|
| `live_ball.h5` | `frame`, `x`, `y`, `z`, `timestamp_utc` |
| `live_joints.h5` | `frame`, `player_id`, and the joint coordinates below |

The joint coordinates the skeleton family reads are `l_wrist_x/y/z`,
`r_wrist_x/y/z`, `l_shoulder_x/y`, `r_shoulder_x/y`, `l_ankle_z`, `r_ankle_z`,
and `x/y/z` for every joint named in `head_joints`. Missing any of them raises
rather than silently degrading, so a bad file fails immediately.


Missing coordinates must be marked with the sentinel `-1.0`, which is what
`invalid_value` matches; those rows are skipped.

### Annotations, only for scoring

Detection needs no annotations at all. They are required only by
`--annotations`, which expects an OSL v2 file with `Header` and `Kickoff`
events and the video's `UTC_time_start`.

Tracking data and annotations are not part of the repository; pass your own with
`--data-root` and `--annotations`.

The annotations used to measure every number below are published in
[pixels vs positions](https://github.com/drishyakarki/pixels_vs_positions/blob/main/headers_data/FWC2022-final-224p-Milli-OpenSportsLib.json).
They cover the 2022 World Cup final and carry millisecond-precise header
timings, which matters: see [Reproducing these numbers](#reproducing-these-numbers).


Headers are annotated as positions inside the video file, while tracking rows
are stamped in UTC. The video's `UTC_time_start` bridges the two:

```
utc = UTC_time_start + position_ms          position_ms = utc - UTC_time_start
```

## Methods

Seven variants in two families. Every one starts from the same question, "did the
ball come close to a head", then differs in what else it demands before calling
that a header.

| Variant | Family | What it adds |
|---|---|---|
| `distance` | distance | Nothing. Ball within 25 cm of a head joint |
| `distance_angle` | distance | Ball path bends by 25° at the contact |
| `distance_speed` | distance | Ball speed changes by 25% at the contact |
| `distance_speed_angle` | distance | Both the bend and the speed change |
| `skeleton` | skeleton | Ball at head height, path bends, hands clear of the head, player facing the ball, feet on the ground, contact brief |
| `skeleton_recall` | skeleton | Hands and facing only, over a wide height band, matching any tracked head joint |
| `skeleton_max_recall` | skeleton | The same without the bend test, finding every annotated header |

The families differ in how they pair a player with the ball. The distance family
matches on **timestamp**, taking the ball sample nearest in time within 60 ms.
The skeleton family matches on **frame number and player id**, which is what
lets it follow one player across neighbouring frames and measure how long the
ball stayed near their head.

### The distance family

`distance` is the plain version. For every joint row it takes the nearest ball
sample in time, measures from six head joints (nose, neck, both eyes, both
ears), and keeps contacts under `distance_threshold_m`, which is 0.5 m. Two
filters then run: contacts within 1 m of the touchline are dropped as throw-ins,
and overlapping detections are reduced to one per second, keeping the most
confident.

Confidence is `1 - distance / 0.5`, and a detection also has to clear
`min_confidence`, which is 0.5. Those two settings interact, and the result is
not obvious from either alone: requiring confidence above 0.5 means the ball
must come within **25 cm**, half the nominal threshold.

The other three add a check on what the ball did. At each contact the spotter
takes a ball sample 200 ms before and 200 ms after, builds the vectors
`before to contact` and `contact to after`, then measures the angle between them
and the change in speed. `distance_angle` wants the angle to reach 25°,
`distance_speed` wants the speed to change by 25%, and `distance_speed_angle`
wants both. The measurement lives in `_trajectory_diagnostics`; each variant
only sets `trajectory_change_mode`.

Of the three, only the angle test earns its place. Speed is a poor signal for
heading, because a flick-on or a glancing contact barely disturbs the ball, so
both speed variants throw away more than half the real headers.

### The skeleton family

These judge the player, not only the ball, and that is where their advantage
comes from. A contact has to survive eight checks, applied in a fixed order:

| # | Check | Threshold | Rejects | Applied to |
|---|---|---|---|---|
| 1 | Ball at head height, coordinate valid, inside the scan window | `ball_height_min_m` 1.3, `ball_height_max_m` 3.0 | Ground passes and high clearances | Every ball sample at once |
| 2 | Ball speed changed | `velocity_change_min_mps` 2.0, `velocity_mag_min_mps` 1.0 | The ball drifting past untouched | One candidate ball sample |
| 3 | Ball path bent, and dropped or rose sharply, having arrived fast | `angle_change_min_deg` 10, `accel_z_change_min_mps2` 8.0, `incoming_speed_min_mps` 4.0 | A ball merely passing near a head | One candidate ball sample |
| 4 | A head joint close to the ball | `head_ball_distance_max_m` 0.4, over `head_joints` | Everyone not in contact | Every player on that frame at once |
| 5 | Player facing the ball | `facing_dot_min` -0.5 | The ball striking the back of the head | One player |
| 6 | Both hands farther from the ball than the head | `hand_check_enabled` | Arm contacts and keeper grabs | One player |
| 7 | Both ankles low | `ankle_height_max_m` 1.2 | Acrobatic poses where the geometry is accidental | One player |
| 8 | Ball near the head on few enough neighbouring frames | `dwell_max_frames` 5, over `dwell_window_frames` 3 | A ball carried or held, rather than headed | One player, across nearby frames |

Survivors are then reduced to one detection per `nms_window_frames`, which is 25
frames or half a second, keeping the closest contact.

The order does not change which contacts survive, since every check is an AND.
It changes only the cost. Step 1 is a single vectorised mask that discards about
96% of ball samples before any per-player work, step 4 screens every player on a
frame in one operation so steps 5 to 8 usually run on nobody, and the dwell check
is last because it is the most expensive: it has to look one player up across
neighbouring frames, so only contacts that already passed everything else pay
for it.

One consequence worth knowing: because a rejected contact never becomes an
event, the output cannot tell you which check turned it away. Diagnosing a
missed header means instrumenting the gates directly.

`skeleton_recall` is the same detector with checks 2, 7 and 8 switched off,
check 1 widened and check 3 reduced to the bend test at 10 degrees. It also
matches on any tracked head joint instead of the nose alone.

### How confidence is scored

Confidence is the contact distance and nothing else, in both families:

```
distance family    1 - distance / 0.5
skeleton family    1 - distance / 0.4
```

Every other check is a pass or fail gate, so a contact that scraped through the
trajectory test scores the same as one that sailed through it, if the distance
matches. The measured angle and speed are recorded in each event's `metadata`
for inspection, but nothing reads them back.

This matters when reading mAP. All variants rank their detections the same way,
by proximity, so a variant only improves its score by removing false positives,
not by ordering the survivors better. It also makes confidence the natural knob
for trimming `skeleton_recall`'s output after the fact.

### Where the settings are

All in `opensportslib/models/base/rule_based.py`:

| Name | Holds |
|---|---|
| `DEFAULT_HEADER_RULE_PARAMS` | Defaults for the distance family |
| `HEADER_RULE_VARIANTS` | Per-variant trajectory mode |
| `DEFAULT_SKELETON_RULE_PARAMS` | Defaults for the skeleton family |
| `SKELETON_RULE_VARIANTS` | The `skeleton_recall` overrides |

Any of it can be overridden from a config, under `MODEL.components.rule.params`.

### Configs

Two configs ship with the library, one per family:

| Config | Covers |
|---|---|
| `h5_header_distance.yaml` | All four distance variants |
| `h5_header_skeleton.yaml` | All three skeleton variants |

Each threshold in them is annotated with what it controls and what the parameter
sweep measured when it was changed.

#### Choosing the variant

One line, under `MODEL.components.rule.source`:

```yaml
      source:
        provider: opensportslib
        registry: rule_based
        name: h5_header_distance     # <- this line
```

| Config | Valid `name:` values |
|---|---|
| `h5_header_distance.yaml` | `h5_header_distance`, `h5_header_distance_angle`, `h5_header_distance_speed`, `h5_header_distance_speed_angle` |
| `h5_header_skeleton.yaml` | `h5_header_skeleton`, `h5_header_skeleton_recall`, `h5_header_skeleton_max_recall` |

Nothing else needs to change. The name selects the variant's overrides from
`HEADER_RULE_VARIANTS` or `SKELETON_RULE_VARIANTS`, so for the distance family it
alone decides which trajectory test runs, and for the skeleton family it decides
which of the two operating points you get.

#### Changing thresholds

Copy a config, edit the `params:` block, and run it with `--config`. Values you
set there win over the defaults in `rule_based.py`:

```bash
cp opensportslib/configs/localization/h5_header_distance.yaml my_run.yaml
# in my_run.yaml: set name: to the variant, then edit params:
python scripts/run_h5_header_rule_inference.py --config my_run.yaml --output test.json
```

Two things to know before you tune:

**Some settings cannot be overridden from a config.** Variant tables are
applied after your params. For the distance family that means
`trajectory_change_mode`: the `name:` line alone decides whether the angle test,
the speed test, both or neither runs. For `h5_header_skeleton_recall` it means
twelve keys, including the head joints, the trajectory gates, both height bounds
and the suppression window. Both configs list which ones.

**The sweep ignores your params.** `--games` and `--combined` generate their own
config carrying only `label` and `head_name`, so they always run the defaults.
Threshold experiments have to go through `--config`. Once you settle on values
worth keeping, add them to `HEADER_RULE_VARIANTS` or `SKELETON_RULE_VARIANTS` as
a named variant and they become available everywhere.

### How they score

Measured on the 2022 World Cup final against 105 annotated headers, at a
one-second tolerance. Tight average mAP averages the score over tolerances of
one to five seconds. Detection time is for that one game, on CPU, scanning the
whole 178-minute tracking file.

| Variant | Predictions | Recall | Precision | Tight avg mAP | Detection time |
|---|---|---|---|---|---|
| `skeleton` | 111 | 91.4% | **86.5%** | 69.2% | **12 s** |
| `skeleton_recall` | 147 | **97.1%** | 69.4% | **69.2%** | 38 s |
| `distance` | 142 | 95.2% | 70.4% | 59.4% | 368 s |
| `distance_angle` | 134 | 92.4% | 72.4% | 59.0% | 364 s |
| `distance_speed_angle` | 87 | 46.7% | 56.3% | 25.7% | 363 s |
| `distance_speed` | 88 | 46.7% | 55.7% | 25.1% | 367 s |

These numbers depend on the annotations being millisecond-precise. Scored
against a second-rounded copy of the same annotations every variant loses
three to seven points of mAP, because half a second of rounding noise sits
right on the one-second tolerance.

The two speed variants are not competitive: requiring the ball's speed to change
by a quarter throws away more than half the real headers, because flick-ons and
glancing contacts barely disturb it. Direction change is the gentler test, and
`distance_angle` trades 2.8 points of recall for 2 of precision against plain
`distance`, ending up marginally behind it on mAP.

The skeleton variants lead because they judge the player, not only the ball:
hands clear of the head, facing the ball, feet on the ground, contact brief.

They are also about thirty times faster, which comes from how each family pairs
a player with the ball. The distance family matches every joint row to a ball
sample by timestamp, a per-row search over millions of rows. The skeleton family
joins on frame number with a binary search and only reads joint rows near
candidate frames. The four distance variants all cost the same, around six
minutes a game, because the trajectory test runs after that matching and is
cheap by comparison.

### Reproducing these numbers

Everything in the table above comes from one command against one game. To check
it yourself:

**1. Get the annotations.**

```bash
curl -LO https://raw.githubusercontent.com/drishyakarki/pixels_vs_positions/main/headers_data/FWC2022-final-224p-Milli-OpenSportsLib.json
```

**2. Point at your tracking data.** You need the game's directory holding
`live_joints.h5` and `live_ball.h5`. The annotations describe game `128083`, the
2022 World Cup final.

**3. Run the strict variant and score it.**

```bash
python scripts/run_h5_header_rule_inference.py \
    --data-root /path/to/FIFA_data \
    --games 128083 \
    --variants skeleton \
    --annotations FWC2022-final-224p-Milli-OpenSportsLib.json
```

About 20 seconds, CPU only. Expect:

```
variant       #pred   mAP@1s   mAP@2s   ...   tight avg   rec@1s   prec@1s
skeleton        111   70.13%   70.13%         69.19%      91.4%    86.5%
```

**4. Compare variants** by naming more of them. Add `--force` to re-detect
rather than reuse the cache:

```bash
python scripts/run_h5_header_rule_inference.py \
    --data-root /path/to/FIFA_data --games 128083 \
    --variants skeleton,skeleton_recall \
    --annotations FWC2022-final-224p-Milli-OpenSportsLib.json
```

**5. Re-score without re-detecting.** Detection is cached per game and variant,
so this returns in seconds:

```bash
python scripts/run_h5_header_rule_inference.py \
    --data-root /path/to/FIFA_data --games 128083 \
    --variants skeleton --annotations FWC2022-final-224p-Milli-OpenSportsLib.json \
    --assemble-only
```

Results land in `outputs/header_spotting/`: `map_results.json` for the scores,
`raw/<variant>/<game>.json` for detections with their full diagnostics.


### Choosing between `skeleton` and `skeleton_recall`

They are the same detector at two operating points. `skeleton` finds 96 of the
105 headers with 15 false positives; `skeleton_recall` finds 102 with 45. Six
more real headers cost thirty more false ones, because the headers `skeleton`
misses are genuinely ambiguous contacts.

Their mAP is now level, 69.19 against 69.24, so neither ranks its detections
better than the other. The choice is purely recall against precision.

Use `skeleton` when a prediction should be trustworthy on its own. Use
`skeleton_recall` to build a candidate set that a human or a downstream
classifier will filter, where a missed header is worse than a false one.

#### How `skeleton_recall` was tuned

Each gate was switched off on its own and the result measured, rather than
guessed at. No single gate was to blame: each cost only one or two points of
recall alone, but the four trajectory gates and the narrow height band together
were rejecting about a tenth of the real headers, because a flick-on or a
glancing contact barely disturbs the ball.

Two gates earned their keep and stayed on. The hand check costs no recall at
all and returns three points of precision. The facing check rejects nothing on
this data, so it is free.

A later sweep of 82 configurations added a third: a 10 degree bend test costs
about three points of recall and returns eight of precision, so it is now on by
default. `header_skeleton_sweep.xlsx` records every configuration tried.

Widening the contact radius, the obvious next lever, backfires. At 0.6 m recall
does not move and precision halves, partly because the dwell radius is derived
from the contact threshold, so widening one tightens the other. `dwell_distance_m`
exists to decouple them.

#### Why any head joint, not just the nose

Across a 64-game run, only 23% of `skeleton_recall`'s detections match on the
nose:

| Joint | Detections |
|---|---|
| nose | 2,747 |
| l_eye | 2,293 |
| r_eye | 2,124 |
| r_ear | 1,971 |
| l_ear | 1,926 |
| neck | 1,008 |

The other 9,322 come from an eye, an ear or the neck. The nose-only rule skipped
a player outright whenever their nose was untracked, so every one of those
contacts was lost. Those counts come from a run made before the bend gate became
the default, so the totals have since changed; the proportion has not.


## Running it

One script covers everything. It runs on CPU; use an environment with `h5py`
installed. Roughly 45 seconds per game.

Spot headers across every game in a directory:

```bash
python scripts/run_h5_header_rule_inference.py --data-root /path/to/games
```

One game, one variant:

```bash
python scripts/run_h5_header_rule_inference.py --games 128083 --variants skeleton
```

Score against annotations, comparing variants:

```bash
python scripts/run_h5_header_rule_inference.py \
    --games 128083 \
    --variants skeleton,skeleton_recall,distance \
    --annotations WC22_multi.json
```

Rebuild the output files without re-detecting, once predictions are cached:

```bash
python scripts/run_h5_header_rule_inference.py --assemble-only
```

Run a hand-written config as-is, without the sweep:

```bash
python scripts/run_h5_header_rule_inference.py \
    --config opensportslib/configs/localization/h5_header_skeleton.yaml \
    --output predictions.json
```

Spot every game in a single pass and keep the spotter's own output, diagnostics
and all, instead of reassembling per-game runs:

```bash
python scripts/run_h5_header_rule_inference.py --combined \
    --variants skeleton_recall --output h5_header_predictions.json
```

### Three ways in

| Mode | What it does | Output |
|---|---|---|
| `--config` | Runs one YAML config exactly as written | Whatever that config covers |
| `--combined` | One manifest over every game, one `infer()` call | The spotter's own payload, full diagnostics |
| default sweep | One game at a time, cached, then reassembled | Trimmed events, and `--halves` selects periods |

`--combined` is closest to how the spotter is meant to be driven: the OSL JSON
manifest lists every game under `data`, and `predict()` walks them in one pass.
The sweep exists for the extras built on top: per-period filtering and
resuming an interrupted run.

`--annotations` works with either, so a run can be scored the same way whichever
mode produced it:

```bash
python scripts/run_h5_header_rule_inference.py --combined \
    --games 128083 --variants skeleton_recall --annotations WC22_multi.json
```

Scoring covers whichever game the annotations describe; the other games in the
run are still spotted and written, just not scored.

### Options

| Flag | Default | Meaning |
|---|---|---|
| `--config` | none | Run this YAML config as written, instead of sweeping a directory |
| `--combined` | off | Spot every game in one pass and save the spotter's own output |
| `--data-root` | `/home/giancos/FIFA_data` | Directory holding one sub-directory per game |
| `--games` | `all` | Comma-separated game directory names |
| `--variants` | `skeleton_recall` | Comma-separated subset of the seven variants |
| `--annotations` | none | Annotation JSON; when given, predictions are scored against it |
| `--output-dir` | `outputs/header_spotting` | Where manifests, predictions and results are written |
| `--output` | `<output-dir>/headers-<variant>.json` | Path of the combined OSL JSON |
| `--dataset-name` | `Header predictions` | Name recorded in the combined file |
| `--halves` | `all` | Comma-separated period tags to report |
| `--force` | off | Re-detect even when raw predictions are cached |
| `--assemble-only` | off | Skip detection, only rebuild the outputs |

### Periods

The ball track tags every sample with a `half`. Period 0 is everything
**outside** active play, meaning warm-up, stoppages and the interval, spread across the
whole broadcast; periods 1 and up are the played halves and any extra time.

Detection always scans the whole file, so `--halves` only filters what gets
reported and can be changed without re-detecting:

```bash
python scripts/run_h5_header_rule_inference.py --assemble-only --halves 0     # out of play only
python scripts/run_h5_header_rule_inference.py --assemble-only --halves 1,2   # regular time
```

## What it produces

```
outputs/header_spotting/
├── manifest_<game>.json              inputs and scan window given to the model
├── config_<game>_<variant>.yaml      generated config
├── raw/<variant>/<game>.json         detections on the UTC clock, with diagnostics
├── headers-<variant>.json            all games in one OSL JSON
└── map_results.json                  scores, when --annotations was given
```

The combined file follows the OSL v2 action-spotting schema, with one entry per
game under `data`:

```json
{
  "head": "Actions",
  "label": "Header",
  "position_ms": 53280,
  "timestamp_utc": "2022-12-03 15:00:23.285000",
  "confidence": 0.3993,
  "metadata": {"note": ""}
}
```

`position_ms` counts from the first sample of that game's tracking file. Each
game's `metadata` records `track_start_utc`, `track_end_utc` and the kickoff of
every period, so positions can be rebased onto any other clock.

Each raw detection also carries diagnostics: contact distance, which head joint
matched, the player, the ball height, the measured trajectory change and the
dwell count. Useful for inspecting why something was or was not detected.

## Reading the output

```
variant                   #pred    mAP@1s    mAP@2s   tight avg   rec@1s  prec@1s
skeleton                    111    63.03%    64.68%      64.85%    86.5%    81.1%
```

- `mAP@Ns`: mean average precision allowing a prediction to sit up to N
  seconds from the annotation.
- `tight avg`: the mAP values averaged over 1 to 5 seconds. The headline score.
- `rec@1s`: share of annotated headers found within one second.
- `prec@1s`: share of predictions that correspond to a real header.

Scoring is confined to the play windows, which run from each kickoff to the
last tracked ball sample of that half. Extra time is excluded from both the
annotations and the predictions, so the comparison stays fair.

## Three stages

The script runs these in order; each can be repeated on its own.

1. **Detect**: write a manifest naming the H5 pair and the scan window, then
   run the variant through `LocalizationModel.infer()`. Detections come out on
   the UTC clock and are cached per game and variant.
2. **Assemble**: collect every game's detections into one OSL JSON, converting
   each to a `position_ms` from that game's first tracking sample.
3. **Evaluate**: only with `--annotations`. Places detections on the video
   clock using the annotations' `UTC_time_start`, restricts both sides to the
   annotated play windows, bins them at 20 ms and scores with `delta_curve`
   from `opensportslib/metrics/localization_metric.py`.

## How the model is wired in

The variants are a `RuleBased` model family: no weights, no training, but the
same config schema, inference API and JSON output as a trained model.

- `family: RuleBased` in a config routes the builder to
  `build_rule_based_model` in `opensportslib/models/base/rule_based.py`.
- Dataset type `H5OSLJsonSpotting` reads the manifest and skips the dataloader.
- Runner `runner_h5_header_rule` makes `infer()` call `model.predict()` directly.
- `train()` raises `NotImplementedError`; these models are inference only.

Ready-made configs sit in `opensportslib/configs/localization/`, one per
family. To use a variant directly, without the evaluation pipeline:

```python
import os
os.environ.setdefault("RUN_ID", "headers")
from opensportslib.apis import LocalizationModel

api = LocalizationModel(config="opensportslib/configs/localization/h5_header_skeleton.yaml")

predictions = api.infer(use_wandb=False)

api.save_predictions("predictions.json", predictions)
```

## Tests

```bash
python -m pytest tests/test_h5_header_skeleton_spotter.py \
                 tests/test_h5_header_rule_spotter.py
```

These build small synthetic H5 files, so they need no tracking data.

