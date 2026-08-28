# SN-GAR Action Spotting Datasets

How the two SN-GAR action-spotting datasets are built from raw event and
tracking data, what guarantees the build carries, and how to rebuild or
publish them.

| script | purpose |
|---|---|
| `tools/convert/build_sngar_spotting.py` | the builder (raw source data → release-ready dataset pair) |
| `tools/convert/sngar_events.py` | event extraction and frame flattening primitives |
| `tools/convert/sngar_dataset_card.py` | dataset card rendering |
| `tools/convert/verify_sngar_spotting.py` | contract verification for a built release |
| `tools/download/push_sngar_spotting.py` | Hugging Face upload (dry run by default) |

Published datasets:

- [`OpenSportsLab/SNGAR-Action-Spotting-Tracking`](https://huggingface.co/datasets/OpenSportsLab/SNGAR-Action-Spotting-Tracking) — 3.0 GB
- [`OpenSportsLab/SNGAR-Action-Spotting-Video`](https://huggingface.co/datasets/OpenSportsLab/SNGAR-Action-Spotting-Video) — 28.2 GB

## Building

```bash
python tools/convert/build_sngar_spotting.py \
    --events-dir   <data-root>/RawEventsData \
    --tracking-dir <data-root>/PlayerPoseTracking \
    --video-dir    <data-root>/224p \
    --out-root     release \
    --num-workers  24
```

Around 11 minutes on 24 workers. Every default is contractual — running it with
no flags beyond the paths reproduces the published datasets byte for byte.

Regenerate only the cards and checksums, without rebuilding any data:

```bash
python tools/convert/build_sngar_spotting.py --out-root release --cards-only
```

### Inputs

Raw event and tracking data, laid out as:

```
<data-root>/
├── RawEventsData/       # one .json per game, source event stream
├── PlayerPoseTracking/  # one .jsonl.bz2 per game, ~29.97 Hz tracking
└── 224p/                # one .mp4 per game, 398x224 broadcast video
```

64 games: the 2022 World Cup (`3812`–`3859`) plus 16 further matches
(`10502`–`10517`).

### Outputs

```
release/
├── sngar-action-spotting-tracking/
│   ├── annotations_{train,valid,test}.json
│   ├── {train,valid,test}/videos/<game_id>.parquet
│   ├── README.md
│   └── MANIFEST.sha256
└── sngar-action-spotting-video/
    ├── annotations_{train,valid,test}.json   ← identical events
    ├── {train,valid,test}/videos/<game_id>.mp4
    ├── README.md
    └── MANIFEST.sha256
```

## The pipeline

**1. Tracking → table.** Each `.jsonl.bz2` is decoded line by line and
flattened into a 17-column frame table. Player roles need two passes: the
source exposes `position_group_type` only inside `game_event`, present on roughly 1% of
frames, so the builder first scans the whole file to build a
`team_id → jersey → position` map plus the game-static home/away team ids, then
stamps `position` and `positionGroup` onto every player on every frame. That
covers 99.98% of player-frames.

Rows are sorted by `(videoTimeMs, frameNum)` and deduplicated on `videoTimeMs`
keeping the first — 42,950 rows corpus-wide — leaving a strictly monotone clock.

**2. Events → labels.** Each source event maps to zero or more of the 10 SN-GAR
labels via `extract_expanded_annotations`. Every matching rule fires, so one
event can emit several labels at the same `position_ms`; a headed goal produces
HEADER, SHOT and GOAL.

**3. Alignment.** Events whose nearest tracking row is further than
`--tolerance-ms` (10.0) are dropped — they cannot be localised. Most are
post-match; see the tolerance caveat below for the rest.

**4. Write.** Both modalities' annotations are written from the *same* in-memory
event lists, then the sha256 manifest and the dataset cards.

```
94,285   extracted
-4,963   resolved to one label per instant
-1,383   dropped as unalignable
=======
87,939   final    train 62,159 · valid 12,091 · test 13,689
```

## Design decisions

Each of these is a deliberate choice with consequences, not a default that
happened to stick.

### Ground truth is shared by construction

Both modalities' annotation files are written from one event list in a single
pass. Earlier builds produced each modality separately and reconciled them with
a sync script afterwards; that reconciliation silently drifted, leaving the
video dataset with a different event set from tracking's. Any
tracking-vs-video comparison run on that pair was measuring a 4,880-event
ground-truth difference as if it were a modality effect.

If you change how events are produced, change it once — there is no second
place to keep in step.

### One label per instant

The task is single-label, so `LABEL_PRIORITY` resolves each `position_ms` to
one label where the source emits several (a throw-in is also a high pass; a headed
shot is both a header and a shot). This is **on** by default — it is the
published dataset, not a variant. `--no-event-dedup` keeps every label.

It is lossy *across* labels rather than uniformly, which matters when reading
per-class results: HIGH PASS keeps 89 of 2,697 candidates (3%) and SHOT 1,041
of 1,559 (67%). Those classes are sparse by construction, not by data quality.

### The tolerance is 10 ms, deliberately

10 ms is **narrower than half a native frame period**. The tracking clock ticks
every ~33.4 ms, so an event at an arbitrary millisecond can be up to 16.69 ms
from the nearest row and still be perfectly aligned.

This build therefore drops 441 correctly-aligned events — median gap 13.62 ms,
404 of them within half a period. `--tolerance-ms 34` keeps them, giving
88,380. 10 ms is kept because it reproduces the historical event count exactly;
it is recorded here so the 441 are not later mistaken for data loss.

### The defaults are the contract

A bare invocation reproduces the published dataset. The predecessor's defaults
disagreed with its own documentation, so its documented "canonical invocation"
built a different dataset than the one the docs described. Opting out of the
contract now requires saying so explicitly.

### Parquets are zstd-compressed

`--compression zstd --compression-level 9` with 50,000-row row groups. The
predecessor wrote `compression=None` in a single row group: 0.9–1.0 GB per game,
62 GB total, and unstreamable. The same bytes compress to 2.9 GB. Loaders are
unaffected — pandas and pyarrow decompress transparently.

### The video dataset ships no parquets

Event alignment needs the tracking clock even when building the video modality.
The predecessor wrote that scratch parquet into whichever output directory it
was filling and left it there, so the video dataset carried 53 GB of tracking
parquets it never reads. Here the clock stays in memory and only the mp4 ships.

### `video_url` is dropped

The column held internal film-room URLs that no loader reads.
`--keep-video-url` restores it.

### No training caches in the tree

OpenSportsLib's tracking loader writes `<game>.parquet.features.npy`,
`.positions.npy` and `.times.npy` next to each parquet on first access. Those
are build artifacts, regenerated on demand, and the upload excludes them —
about 9 GB that previously shipped as if it were data.

## Publishing

```bash
# inspect the plan; nothing is created or uploaded
python tools/download/push_sngar_spotting.py --out-root release

# actually push
python tools/download/push_sngar_spotting.py --out-root release --yes
```

The upload re-hashes every file against `MANIFEST.sha256` before sending
anything, which catches a parquet silently rewritten or truncated between build
and push — a size check would not.

Repos follow the OpenSportsLab house pattern: **public with `gated="manual"`**.
Gating is applied *before* the repo is made public, so files are never briefly
readable without an approved request. Pass `--private` to keep a repo private
instead (note the 100 GB free-tier cap applies to private storage only).

## Verifying a build

```bash
python tools/convert/verify_sngar_spotting.py --out-root release          # fast
python tools/convert/verify_sngar_spotting.py --out-root release --deep   # + parquets and hashes
```

The fast pass checks per-split and per-label event totals against the contract,
that both modalities carry identical events in identical game order, that
annotations are sorted and well formed, that every referenced payload exists
with nothing extra, that no loader caches leaked in, and that the manifest
covers the tree. `--deep` additionally reads every parquet to confirm each
clock is strictly monotone and every event lands within tolerance of a real
tracking row, then re-hashes everything against `MANIFEST.sha256`.

The contract constants are the dataset's identity. If a builder change moves
them, that is a new dataset version — not a test to update in place.

## Verifying a download

```bash
sha256sum -c MANIFEST.sha256
```

## Loading

```python
from opensportslib.core.utils.load_annotations import annotationstoe2eformat_tracking

labels, task = annotationstoe2eformat_tracking(
    ["annotations_test.json"], ["."], extract_fps=5
)
```

!!! warning "A parquet row index is not linear in time"

    `videoTimeMs` starts 40–211 s into the broadcast and every game has coverage
    gaps (70 gaps over 5 s, median 58 s). `row_index / fps` is **not** a
    timestamp. Computing one that way previously drove an oracle test — perfect
    predictions fed through the metric — down to 0.98% tight mAP instead of
    99.07%, a ceiling under which no model could score well regardless of
    quality. Read time from `videoTimeMs`, and report predictions on
    `frame_times`.

    The video modality does not have this problem: an mp4 frame index is linear.
    Its own trap is the frame rate — 29.97, not the 30.0 the annotations declare.
    Seek by milliseconds.

## Relationship to the other builders

`tools/convert/build_soccernet_gar_action_spotting.py` builds a spotting dataset
from an existing **GAR clip manifest**, inheriting splits and events from the
classification dataset. `build_sngar_spotting.py` goes direct from the raw source and
owns the whole contract. They are separate lineages; use this one for the
published SN-GAR spotting datasets.
