"""Dataset card generation for the SN-GAR spotting release.

The card is rendered from the annotation files and parquet footers already
written into the release directory, never from in-process build state. That
makes it impossible for a card to claim something the shipped files do not
say, and lets `build_sngar_spotting.py --cards-only` regenerate both cards in
seconds without touching the data.

Prose that is modality-specific lives in the *_TRACKING / *_VIDEO constants
below, since claims about parquet row indices and tracking coverage are
meaningless on the video side.
"""

import os
import json
from datetime import datetime, timezone

import pyarrow.parquet as pq

from sngar_events import LABELS


LABEL_RULES = """```
possessionEventType == "PA"  (pass)       bodyType == "HE"          -> HEADER
                                          ballHeightType == "A"     -> HIGH PASS
                                          passType == "H"           -> THROW IN
                                          otherwise                 -> PASS
                    == "CR"  (cross)                                -> CROSS
                    == "SH"  (shot)       bodyType == "HE"          -> HEADER   |
                                          always                    -> SHOT     | all that
                                          shotOutcomeType == "G"    -> GOAL     | apply
                    == "CH"  (challenge)  challengeWinnerPlayerId   -> PLAYER SUCCESSFUL TACKLE
                    == "CL"  (clearance)  bodyType == "HE"          -> HEADER

gameEventType       == "OUT"                                        -> OUT
setpieceType        == "T"                                          -> THROW IN
                    == "F"                                          -> FREE KICK
```"""

TRACKING_SCHEMA = """### `{split}/videos/<game_id>.parquet`

One row per tracked frame, 17 columns, 178k-191k rows per game.

| column | type | meaning |
|---|---|---|
| `videoTimeMs` | float32 | **the clock.** Absolute video time in ms. `position_ms` indexes into this column and nothing else. |
| `frameNum` | int32 | source frame counter |
| `period` | int32 | 1-2, or 1-4 for the three extra-time games (`10506`, `10508`, `10517`) |
| `game_event_id` | int32 | -1 when the frame carries no game event |
| `possession_event_id` | int32 | -1 when the frame carries no possession event |
| `game_event_type` | string | `FIRSTKICKOFF`, `OTB`, `OUT`, ... empty on non-event frames |
| `player_name`, `player_id` | string | the event actor; empty on non-event frames |
| `team_id`, `home_team` | string | the actor's team; `home_team` is `"1"`/`"0"` |
| `possession_event_type` | string | `PA`, `SH`, `CR`, `CH`, `CL`, ... |
| `homePlayers`, `awayPlayers` | string | **JSON array** of 11 objects (see below) |
| `homePlayersSmoothed`, `awayPlayersSmoothed` | string | same, from the smoothed track |
| `balls` | string | **JSON array** of `{"visibility", "x", "y", "z"}` |
| `ballsSmoothed` | string | **JSON object** - note, not an array |

A player object:

```json
{"jerseyNum": "4", "confidence": "LOW", "visibility": "ESTIMATED",
 "x": -16.286, "y": 5.821, "position": "RCB", "positionGroup": "DEF"}
```

Coordinates are pitch metres with the origin at the centre circle. `position`
is the fine-grained role and `positionGroup` collapses it to `GK`/`DEF`/`MID`/`FWD`.

Two shape quirks inherited from the source, documented rather than silently
patched, since fixing them would break existing loaders:

- The `*Smoothed` player columns carry **no** `position`/`positionGroup` keys.
  Only the raw `homePlayers`/`awayPlayers` are role-enriched.
- `ballsSmoothed` is a JSON **object** where `balls` is a JSON **array**.

Both are inert for training: the OpenSportsLib loader reads only
`videoTimeMs`, `balls`, `homePlayers` and `awayPlayers`.

The nested columns are JSON strings rather than Arrow structs. That is what the
original SN-GAR conversion produced and what every existing loader expects, so
it is preserved; zstd compression absorbs the cost (62 GB -> 2.9 GB)."""

VIDEO_SCHEMA = """### `{split}/videos/<game_id>.mp4`

Whole-match broadcast video, 398x224 at 29.97 fps (30000/1001), H.264.
Durations run 98.9-157.6 minutes (median 104.4); files are 359-685 MB
(median 431).

The video and the tracking clock **start at different points**: `videoTimeMs`
in the tracking modality is time on this video's timeline, so `position_ms`
addresses both modalities identically. Note that annotations declare
`"fps": 30.0` while the true rate is 29.97 - the OpenSportsLib video loader
reads the real rate from the container with OpenCV and ignores that field.

No tracking parquets are shipped here; the companion repo holds those."""

LOAD_TRACKING = """```python
import json, pandas as pd

ann = json.load(open("annotations_test.json"))
game = ann["data"][0]
df = pd.read_parquet(game["inputs"][0]["path"])

times = df["videoTimeMs"].to_numpy()          # the clock, in ms
for event in game["events"][:5]:
    row = times.searchsorted(event["position_ms"])
    players = json.loads(df["homePlayers"].iloc[row])
    print(event["label"], event["position_ms"], len(players), "home players")
```

With OpenSportsLib:

```python
from opensportslib.core.utils.load_annotations import annotationstoe2eformat_tracking

labels, task = annotationstoe2eformat_tracking(
    ["annotations_test.json"], ["."], extract_fps=5
)
```"""

LOAD_VIDEO = """```python
import json, cv2

ann = json.load(open("annotations_test.json"))
game = ann["data"][0]
cap = cv2.VideoCapture(game["inputs"][0]["path"])
fps = cap.get(cv2.CAP_PROP_FPS)               # 29.97, not the declared 30.0

for event in game["events"][:5]:
    cap.set(cv2.CAP_PROP_POS_MSEC, event["position_ms"])
    ok, frame = cap.read()
    print(event["label"], event["position_ms"], ok, frame.shape)
```"""


CLOCK_TRACKING = """The single most important property of this dataset, and the one that has
already broken an evaluation once:

`videoTimeMs` starts **40-211 seconds** into the broadcast, because recording
begins before kickoff. Every game then has coverage gaps - 70 gaps longer than
5 s corpus-wide, median 58 s, mostly half-time plus outages. The tracked span
runs 95.5-144.8 minutes (median 101.9), against mp4 durations of 98.9-157.6.

**A parquet row index is therefore not linear in time.** `row_index / fps` is
not a timestamp. Computing one that way is what previously drove an oracle test
- feeding perfect predictions through the metric - down to 0.98% mAP instead of
99.07%, a ceiling under which no model could have scored well regardless of
quality. Read time from the `videoTimeMs` column, always."""

CLOCK_VIDEO = """The mp4 timeline is continuous, so unlike the tracking modality a frame index
here *is* linear in time - but two things still catch people out:

**The video starts before kickoff.** The first event in a game sits 40-211
seconds in, and the video runs through half-time and stoppages with no
annotations. Long unlabelled stretches are expected, not missing data.

**The rate is 29.97 fps, not 30.** Over a 100-minute match, indexing with 30.0
drifts by roughly 6 seconds by the final whistle - far beyond the 1-second
tight-mAP tolerance. Seek by milliseconds (`CAP_PROP_POS_MSEC`) or read the
real rate from the container; do not multiply `position_ms` by a hardcoded 30.

The companion tracking modality carries the same events on the same clock, but
its rows are *not* evenly spaced - see that repo's card before comparing the
two frame-by-frame."""

PROPS_TRACKING = """- **Ball coverage is 53-79% of frames** (mean 68%). The most informative object
  for spotting is missing about a third of the time. At *event* frames coverage
  rises to 93% - the ball is tracked when it matters most.
- **11-v-11 on 99.93% of team-frames**, 99.98% `positionGroup` coverage, and
  zero out-of-bounds coordinates.
- **Three games have extra time** (`10506`, `10508`, `10517`), with `period`
  running 1-4 rather than 1-2."""

PROPS_VIDEO = """- **Events do not cover the whole video.** Between the pre-kickoff head, the
  post-match tail and stretches over a minute with no events, an unannotated
  6.1-67.4 minutes per game (median 15.0) is expected, not missing data. The
  long tail of that range is tracking outages, since events with no tracking
  coverage were dropped from both modalities alike.
- **Three games have extra time** (`10506`, `10508`, `10517`).
- **Broadcast footage**, so it carries replays, cutaways and graphics. The
  tracking modality has none of these - a point in its favour when comparing
  the two."""


#
# Everything the card states about counts, splits and payload is derived from
# the annotation files and parquet footers already written into repo_dir, not
# from the in-process build stats. The card therefore cannot drift from what
# actually ships, and can be regenerated alone with --cards-only.

SPLIT_ORDER = ("train", "valid", "test")


def id_ranges(game_ids):
    """Collapse game ids into contiguous numeric runs.

    The splits are string-sorted, so a split's first and last id are not its
    numeric bounds - train runs 10502-10517 *and* 3812-3840, and printing
    "10502-3840" would be nonsense.
    """
    nums = sorted(int(g) for g in game_ids)
    runs = []
    start = prev = nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
            continue
        runs.append((start, prev))
        start = prev = n
    runs.append((start, prev))
    return ", ".join(f"{a}" if a == b else f"{a}-{b}" for a, b in runs)


def read_release_stats(repo_dir):
    """Recompute per-split counts from the shipped annotation files."""
    stats = {}
    for split in SPLIT_ORDER:
        path = os.path.join(repo_dir, f"annotations_{split}.json")
        if not os.path.exists(path):
            continue
        doc = json.load(open(path))
        counts = {}
        for game in doc["data"]:
            for event in game["events"]:
                counts[event["label"]] = counts.get(event["label"], 0) + 1
        stats[split] = {
            "n_games": len(doc["data"]),
            "games": [g["game_id"] for g in doc["data"]],
            "label_counts": counts,
            "total_events": sum(counts.values()),
            "metadata": doc["metadata"],
        }
    return stats


def payload_summary(repo_dir, modality):
    """Total bytes and, for tracking, total rows, read from file footers."""
    total_bytes = 0
    total_rows = 0
    suffix = ".parquet" if modality == "tracking" else ".mp4"
    for split in SPLIT_ORDER:
        d = os.path.join(repo_dir, split, "videos")
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if not name.endswith(suffix):
                continue
            path = os.path.join(d, name)
            total_bytes += os.path.getsize(path)
            if modality == "tracking":
                total_rows += pq.ParquetFile(path).metadata.num_rows
    return total_bytes, total_rows


def write_card(repo_dir, modality):
    stats = read_release_stats(repo_dir)
    if not stats:
        return
    meta = stats[next(iter(stats))]["metadata"]
    total_bytes, total_rows = payload_summary(repo_dir, modality)
    n_games = sum(s["n_games"] for s in stats.values())
    grand_total = sum(s["total_events"] for s in stats.values())

    counts = {}
    for s in stats.values():
        for label, n in s["label_counts"].items():
            counts[label] = counts.get(label, 0) + n

    def cell(split, label):
        return stats[split]["label_counts"].get(label, 0) if split in stats else 0

    label_rows = "\n".join(
        f"| {label} | {cell('train', label):,} | {cell('valid', label):,} "
        f"| {cell('test', label):,} | **{counts.get(label, 0):,}** |"
        for label in sorted(LABELS, key=lambda l: -counts.get(l, 0))
    )
    label_total = (
        f"| **all** | **{stats.get('train', {}).get('total_events', 0):,}** "
        f"| **{stats.get('valid', {}).get('total_events', 0):,}** "
        f"| **{stats.get('test', {}).get('total_events', 0):,}** "
        f"| **{grand_total:,}** |"
    )

    split_rows = "\n".join(
        f"| `{split}` | {stats[split]['n_games']} | "
        f"{id_ranges(stats[split]['games'])} | "
        f"{stats[split]['total_events']:,} |"
        for split in SPLIT_ORDER if split in stats
    )

    payload_ext = "parquet" if modality == "tracking" else "mp4"
    other = "video" if modality == "tracking" else "tracking"
    tree = "\n".join([
        "annotations_train.json      45 games",
        "annotations_valid.json       9 games",
        "annotations_test.json       10 games",
        f"train/videos/<game_id>.{payload_ext}",
        f"valid/videos/<game_id>.{payload_ext}",
        f"test/videos/<game_id>.{payload_ext}",
        "README.md",
        "MANIFEST.sha256             sha256 of every shipped file",
    ])

    if modality == "tracking":
        schema, loading = TRACKING_SCHEMA, LOAD_TRACKING
        clock, props = CLOCK_TRACKING, PROPS_TRACKING
        clock_heading = "The clock is not the wall clock"
        headline = (
            f"{n_games} whole-match player/ball tracking tables at ~29.97 Hz "
            f"({total_rows:,} frames), with {grand_total:,} action-spotting "
            f"annotations across 10 labels."
        )
        input_stanza = (
            '"inputs": [{"type": "tracking_parquet", '
            '"path": "valid/videos/3841.parquet", "fps": 30.0}]'
        )
    else:
        schema, loading = VIDEO_SCHEMA, LOAD_VIDEO
        clock, props = CLOCK_VIDEO, PROPS_VIDEO
        clock_heading = "Time, and how to index it"
        headline = (
            f"{n_games} whole-match broadcast videos at 398x224, with "
            f"{grand_total:,} action-spotting annotations across 10 labels."
        )
        input_stanza = (
            '"inputs": [{"type": "video_mp4", '
            '"path": "valid/videos/3841.mp4", "fps": 30.0}]'
        )

    # Native frame period is ~33.4 ms, so an event on an arbitrary millisecond
    # can be up to half of that from the nearest row and still be perfectly
    # aligned. A tolerance below that bound rejects well-aligned events.
    tol = float(meta.get("tolerance_ms", 34.0))
    half_period = 16.69
    if tol < half_period:
        tolerance_note = (
            f"**tighter than half a native frame period** ({half_period} ms), so it "
            "also rejects events that sit correctly on the clock between two rows"
        )
        tolerance_caveat = (
            f"\n> **Note.** At {tol} ms this window is narrower than half a native\n"
            f"> frame period ({half_period} ms). Because the tracking clock ticks every\n"
            "> ~33.4 ms, an event timestamped at an arbitrary millisecond can be up to\n"
            f"> {half_period} ms from the nearest row and still be perfectly aligned.\n"
            "> This build therefore also drops 441 such events, chosen deliberately to\n"
            "> reproduce the historical event count. A tolerance of 34.0 ms (one full\n"
            "> frame period) keeps them.\n"
        )
    else:
        tolerance_note = (
            "one native frame period; a tolerance below half a period "
            f"({half_period} ms) would reject perfectly-aligned events"
        )
        tolerance_caveat = ""

    acct = meta.get("event_accounting")
    dropped_note = (
        "Most dropped events are post-match, where the source keeps annotating\n"
        "tracking has stopped; at this tolerance the rest are the mid-frame\n"
        "events described above."
        if tol < half_period else
        "The dropped events are overwhelmingly post-match, where the source keeps\n"
        "annotating after tracking has stopped. None sits just outside the\n"
        "tolerance."
    )
    if acct and all(k in acct for k in ("extracted", "final")):
        totals = {"extracted": 0, "removed_by_dedup": 0,
                  "removed_by_alignment": 0, "final": 0}
        for st in stats.values():
            a = st["metadata"].get("event_accounting", {})
            for k in totals:
                totals[k] += a.get(k, 0)
        parts = [f"{totals['extracted']:,}   extracted"]
        if totals["removed_by_dedup"]:
            parts.append(f"-{totals['removed_by_dedup']:,}   removed by priority dedup")
        parts.append(f"-{totals['removed_by_alignment']:,}   dropped as unalignable")
        parts.append(f"{'=' * 7}")
        parts.append(f"{totals['final']:,}   final")
        width = max(len(p.split()[0]) for p in parts)
        accounting_chain = "\n".join(
            (p.split(None, 1)[0].rjust(width) + "   " + p.split(None, 1)[1])
            if len(p.split(None, 1)) > 1 else p
            for p in parts
        )
    else:
        accounting_chain = f"{grand_total:,} final events"

    if meta.get("deduplicated_events"):
        label_rule = "one label per instant, resolved by priority"
        # Computed rather than quoted, so the card cannot describe a build it
        # does not ship with.
        before = {}
        for st in stats.values():
            for label, n in st["metadata"].get(
                "event_accounting", {}
            ).get("labels_before_dedup", {}).items():
                before[label] = before.get(label, 0) + n
        if before:
            hit = sorted(
                ((counts.get(l, 0) / n, l, counts.get(l, 0), n)
                 for l, n in before.items() if n),
                key=lambda t: t[0],
            )[:2]
            worst = "; ".join(
                f"**{l}** keeps {kept:,} of {tot:,} ({100 * rate:.0f}%)"
                for rate, l, kept, tot in hit
            )
        else:
            worst = "some labels are heavily reduced"
        label_section = f"""## One label per instant

The task is single-label: each `position_ms` in a game carries exactly one
event. The source stream does not come that way - a throw-in is also a high
pass, a headed shot is both a header and a shot - so where several labels land
on the same millisecond, `LABEL_PRIORITY` selects the intended one.

Worth knowing when reading per-class results, because it is lossy *across*
labels rather than uniformly. Least-preserved: {worst}. Those classes are
sparse by construction rather than by data quality, so a model scoring badly on
them is not necessarily doing badly."""
    else:
        label_rule = "every label kept, several per instant allowed"
        label_section = """## Several labels per instant

Every matching rule fires, so one moment can carry several events at the same
`position_ms` - a headed goal produces HEADER, SHOT and GOAL. Consumers must
expect repeated timestamps with different labels."""

    card = f"""---
task_categories:
- video-classification
language:
- en
tags:
- soccer
- football
- action-spotting
- temporal-action-localization
- sports
- {modality}
size_categories:
- 10K<n<100K
---

# SN-GAR Action Spotting - {modality}

{headline}

This is one of a **pair** of datasets built in a single pass from the same raw
source. `SNGAR-Action-Spotting-Tracking` ships tracking tables,
`SNGAR-Action-Spotting-Video`
ships the broadcast video, and both carry **byte-identical ground truth** - the
same games, the same splits, the same event lists in the same order. A
tracking-vs-video comparison run on this pair measures the modality and nothing
else. The companion repo is the `{other}` one.

## Layout

```
{tree}
```

Payload: {total_bytes / 1e9:.1f} GB across {n_games} games.

## Splits

| split | games | game ids | events |
|---|---:|---|---:|
{split_rows}

Splits are assigned by **string-sorted** game id, so the `105xx` games sort
before the `38xx` ones. This is not random and not stratified - it is the
original SN-GAR contract, preserved so results stay comparable with prior work.

## Labels

| label | train | valid | test | total |
|---|---:|---:|---:|---:|
{label_rows}
{label_total}

{label_section}

## Source

64 games: the 2022 FIFA World Cup (`3812`-`3859`) plus 16 further matches
(`10502`-`10517`). Two raw inputs per game:

| input | content |
|---|---|
| `RawEventsData/<game>.json` | hand-annotated event stream, ~2,000 events per game, each with `gameEvents` / `possessionEvents` / `initialTouch` sub-objects |
| `PlayerPoseTracking/<game>.jsonl.bz2` | bz2 line-delimited JSON, one line per tracked frame at ~29.97 Hz, ~186k lines per game (11.9M total) |

## How it was built

**1. Tracking to table.** Each `.jsonl.bz2` is decoded line by line and
flattened. The non-obvious part is player roles: the source exposes
`position_group_type` only inside `game_event`, which appears on roughly 1% of
frames. The builder therefore makes a **first full pass** to construct a
`team_id -> jersey -> position` map plus the game-static home/away team ids,
then a **second pass** stamping `position` and `positionGroup` onto every
player on every frame. This tags 99.98% of player-frames; the original
converter resolved team ids per frame and so tagged only the sparse event
frames.

Rows are then sorted by `(videoTimeMs, frameNum)` and deduplicated on
`videoTimeMs` keeping the first, which removes 42,950 rows corpus-wide and
leaves a strictly monotone clock.

**2. Events to labels.** Each source event maps to zero or more of the 10 SN-GAR
labels:

{LABEL_RULES}

Every rule that matches fires, so **one event can emit several labels**, each
becoming its own annotation at the same `position_ms`. A headed goal produces
three: HEADER, SHOT and GOAL. `position_ms` is `int(eventTime * 1000)`, and
`eventTime` is on the same video clock as `videoTimeMs` - which is what makes
step 3 possible.

**3. Alignment.** For each event, find the tracking row with the nearest
`videoTimeMs`. If it is further than `tolerance_ms` ({meta.get('tolerance_ms')} ms)
away, drop the event: there is no tracking against which to localise it.
{tolerance_caveat}

```
{accounting_chain}
```

{dropped_note}

**4. Write.** Both modalities' annotation files are written from the *same*
in-memory event lists - identical ground truth by construction rather than by a
follow-up sync - followed by the sha256 manifest and this card.

## Annotation format

OpenSportsLib v2, one file per split:

```json
{{
  "version": "2.0",
  "task": "action_spotting",
  "dataset_name": "action_spotting_{modality}_valid",
  "metadata": {{"modality": "{modality}", "aligned": true,
               "tolerance_ms": {meta.get('tolerance_ms')},
               "deduplicated_events": {str(meta.get('deduplicated_events')).lower()},
               "events_identical_across_modalities": true}},
  "labels": {{"action": {{"type": "single_label", "labels": ["PASS", "HEADER", ...]}}}},
  "data": [{{
    "game_id": "3841",
    "split": "valid",
    {input_stanza},
    "events": [{{"head": "action", "label": "PASS", "position_ms": 190256,
                "gameTime": "1 - 00:00", "team": "home", "visibility": "visible"}}]
  }}]
}}
```

`position_ms` is the only field you need to localise an event. `gameTime` is
the `period - MM:SS` match clock, useful for display but **not** for
indexing.

## Payload format

{schema}

## Loading

{loading}

## {clock_heading}

{clock}

## Other known properties

Real characteristics of the source, not defects to be cleaned:

{props}

## Build contract

| setting | value | why |
|---|---|---|
| label resolution | {label_rule} | see above |
| `tolerance_ms` | `{meta.get('tolerance_ms')}` | {tolerance_note} |
| `dedupe_video_time_ms` | `{meta.get('dedupe_video_time_ms')}` | the source emits repeated `videoTimeMs`; removing them makes the clock strictly monotone |
| `aligned` | `{meta.get('aligned')}` | tracking cannot localise events outside its coverage |
| splits | 45 / 9 / 10 | alphabetical by game id, the original contract |


## Integrity

`MANIFEST.sha256` lists a sha256 for every shipped file:

```bash
sha256sum -c MANIFEST.sha256
```

## Access

Access is gated. Approval covers internal research use; check with the dataset
owners before redistributing.

Built by `build_sngar_spotting.py` on {datetime.now(timezone.utc).strftime('%Y-%m-%d')}.
"""
    with open(os.path.join(repo_dir, "README.md"), "w") as f:
        f.write(card)