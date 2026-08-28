# Action spotting from tracking data

Spot the ten SN-GAR actions across a whole match from player and ball
coordinates instead of pixels, scored with the same tight mAP as the video
baseline.

The design rule was: **change the input, change nothing else.** The tracking
path reuses the library's E2E-Spot training loop, GRU head, sliding-window
inference and mAP evaluator unmodified. Only the data layer and the encoder are
new. That is what makes the two modalities comparable — a difference in the
final number is a difference in the input, not in the recipe.

| | Video baseline | Tracking |
|---|---|---|
| Config | `configs/localization/video_action_spotting.yaml` | `configs/localization/tracking_action_spotting.yaml` |
| Input | 224×398 mp4, 5 fps | parquet rows, 5 fps |
| Per frame | RGB image | 23-node graph |
| Encoder | `rny008_gsm` | `graph_conv_seq` |
| Head | GRU | GRU |
| Clip | 300 frames (60 s) | 300 frames (60 s) |
| lr | 0.0008 | 0.001 |

## 1. The dataset

The dataset is **not** built from the GAR manifests. It is built from raw event and tracking
data by `tools/convert/build_sngar_spotting.py`, documented in full in
[docs/tools/sngar-spotting.md](../tools/sngar-spotting.md). Both modalities are
produced in one pass so their ground truth is identical by construction.

```bash
python tools/convert/build_sngar_spotting.py \
    --events-dir <data-root>/RawEventsData --tracking-dir <data-root>/PlayerPoseTracking \
    --video-dir <data-root>/224p --out-root release --num-workers 24
```

Published as
[`OpenSportsLab/SNGAR-Spotting-Tracking`](https://huggingface.co/datasets/OpenSportsLab/SNGAR-Spotting-Tracking)
and
[`OpenSportsLab/SNGAR-Spotting-Video`](https://huggingface.co/datasets/OpenSportsLab/SNGAR-Spotting-Video).

> Superseded: `create_action_spotting_dataset.py` and
> `sync_video_events_from_tracking.py`, which lived beside the data rather than
> in this repo. The event-extraction rules they defined are preserved verbatim
> in `tools/convert/sngar_events.py`; the surrounding build is rewritten.

| Input | Content |
|---|---|
| `RawEventsData/<game>.json` | source event stream, one JSON list per game |
| `PlayerPoseTracking/<game>.jsonl.bz2` | ~29.97 Hz tracking frames, bz2 line-delimited JSON |

64 games, split alphabetically by game id into 45 / 9 / 10. Three stages:

1. **Tracking → parquet.** Each `jsonl.bz2` is decoded and flattened into one
   whole-match table. Team IDs are resolved once per game, then every frame's
   players are tagged `GK/DEF/MID/FWD`. Rows are sorted by
   `(videoTimeMs, frameNum)` and deduplicated on `videoTimeMs`, keeping the
   first, so the clock is strictly monotone.
2. **Events → annotations.** Possession and game events map to the 10 SN-GAR
   labels, reproducing the original SN-GAR converter rule for rule. One instant
   may legitimately emit several labels (a headed shot is `HEADER` *and* `SHOT`).
3. **Alignment.** Events whose nearest tracking row is more than `--tolerance-ms`
   (default 34 ms, one native frame period) away are dropped — they cannot be
   localised in tracking. 94,285 extracted → 87,939 kept.

Output is OSL v2:

```
sngar-action-spotting-tracking/
├── annotations_{train,valid,test}.json    task=action_spotting, 10 labels
└── {train,valid,test}/videos/<game>.parquet
```

### The event set

The published dataset is **87,939 events** — one label per instant, alignment
tolerance 10 ms. That is the historical SN-GAR action-spotting count, and the
build defaults reproduce it with no flags.

!!! note "Results in §5 used 88,380 events"

    The runs below were measured against a build with tolerance 34 ms rather
    than 10 ms, which keeps 441 additional events. Those 441 sit within half a
    native frame period of a tracking row — correctly aligned, and excluded
    only because 10 ms is narrower than the clock's own half-period. Totals
    therefore differ by 0.5%; per-class ordering does not change.

    Rebuild with `--tolerance-ms 34` to reproduce §5 exactly. The parquets are
    identical either way — only the annotation JSONs differ, so
    `--annotations-only` regenerates them in about two minutes.

Historically the video modality was given its event set by a separate sync
script, which drifted: the two modalities ended up thousands of events apart, a
difference any modality comparison would have read as a modality effect. The
current builder writes both from one event list, so this cannot recur.

> `tools/convert/build_soccernet_gar_action_spotting.py` in this repo is a
> *separate* path that regroups the GAR clip manifests into spotting
> annotations. It did not produce the dataset the results were measured on.

### Verification

`verify_sngar_spotting_dataset.py` runs 20 checks in four levels: builder logic
against the original converter, byte-identical rebuild from raw data, data
quality, and trainer consumability. The one to know is **D3, the oracle test**:
feed perfect predictions through the real evaluator and it scores **99.07%**
tight mAP. Before the evaluation fixes in §4 the same test scored **0.98%** — a
1% ceiling no model could ever beat. Run it before trusting any spotting number.

## 2. Parquet to graph

Everything below lives in `opensportslib/datasets/utils/tracking.py`.

**Feature layout.** Each frame becomes 23 objects × 8 features. Slot 0 is always
the ball, 1–11 home, 12–22 away, players ordered by jersey number so a slot
means something consistent over time.

| Index | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| Field | x | y | is_ball | is_home | is_away | dx | dy | z |

Objects that aren't tracked in a frame get the sentinel `-200.0`, which becomes
`-2.0` after normalization — deliberately outside the valid range, so the model
can tell "absent" from "at the centre spot".

**The whole-match cache.** A spotting parquet is a full match: ~175k rows,
0.8–1.2 GB, one row group with no predicate pushdown. Re-parsing it for every
randomly sampled window is not viable. So each game is parsed exactly once into
three memory-mapped sidecars next to the parquet:

```
<game>.parquet.features.npy    (rows, 23, 8)  float32
<game>.parquet.times.npy       (rows,)        float64   videoTimeMs
<game>.parquet.positions.npy   (rows, 23)     int8      GK/DEF/MID/FWD codes
```

The cache is raw on purpose — no velocities, no normalization, no augmentation.
Those are window- and config-dependent, and are applied after slicing.

**Per window.** `slice_window` takes every `stride`-th row from the cache
(`stride` = native rate ÷ `extract_fps`), pads out-of-range frames with the
missing sentinel, then computes `dx, dy` **on the decimated sequence** — velocity
must reflect the gap between sampled frames, not raw rows. Augmentations
(horizontal flip, vertical flip, team swap — all label-preserving) run next, then
normalization by the pitch constants, then one PyG `Data` graph per frame.

**Edges.** `edge_type` on the encoder component picks the connectivity:
`none`, `full`, `knn`, `distance`, `ball_knn`, `ball_distance`, or `positional`
(the default — formation lines GK↔DEF↔MID↔FWD within each team, plus the ball
connected to every player).

## 3. Model

`graph_conv_seq` (`GraphSequenceEncoder`, in `models/backbones/builder.py`) is a
thin wrapper on the existing GAR `GraphEncoder`. The classification encoder pools
to one vector per graph; spotting needs one embedding *per frame*, so the wrapper
reshapes `(B*T, H)` back to `(B, T, H)` using the `seq_len` the collate attaches
to the PyG `Batch`. The stock GRU head then emits per-frame logits, exactly as it
does for RGB features.

The collate flattens a batch's B×T per-frame graphs into a single PyG `Batch`
(`core/utils/data.py`). Eval collate requires `batch_size: 1`, mirroring how the
E2E inferer indexes `clip["frame"][0]`.

Mixup is off: blending node features across two differently-structured graphs is
not defined. DALI is off too — it decodes video, and is now pinned away from
non-video modalities rather than failing with `KeyError: 'frame'`.

## 4. The part that was actually hard: clock vs row

For decoded video a frame index and a clock position are interchangeable —
`t = frame / fps`. **For a tracking parquet they are not.** Recording starts
mid-broadcast (row 0 can be ~80 s into the match clock) and the clock jumps
60–90 s across half-time. Every path that turned an index into a timestamp by
dividing by fps was wrong, and wrong *silently* — none of them raised, each
returned a plausible number.

The fix is two indices per event, kept apart:

| Field | Meaning | Used by |
|---|---|---|
| `frame` | parquet row ÷ stride | window slicer, `get_labels` |
| `eval_frame` | `fps × position_ms / 1000` | mAP evaluation |

Datasets additionally expose `frame_times`, the absolute match clock of every
decimated frame. `process_frame_predictions._locate()` uses it to map a clip
index to `(timestamp_ms, eval index)`, re-deriving the eval index the same way
the ground truth does, so both sides land in one coordinate system. RGB has no
`frame_times`, keeps the old arithmetic, and is untouched.

Three related traps closed at the same time (commit `5e22f2f`):

- **mAP tolerances were in frames**, so strictness scaled with `extract_fps` —
  4 frames is 1.9 s at 2 fps but 0.8 s at 5 fps. Now given in seconds and
  converted with the run's own rate.
- **The evaluation vector was clamped at 90 minutes**, folding stoppage-time
  events onto the final bin and merging distinct events. Now sized from the
  actual labels and predictions.
- **Class order was permuted on checkpoint reload** — a `{name: index}` mapping
  round-tripped through YAML comes back alphabetical, so `list(mapping)` remapped
  labels. `classes_to_ordered_list()` now orders by index.

Any result measured before that commit should be treated as suspect.

## 5. Running it

```bash
RUN_ID=my_run CUDA_VISIBLE_DEVICES=0 \
    python3 -u run_spotting.py opensportslib/configs/localization/tracking_action_spotting.yaml
```

`run_spotting.py` trains, infers and evaluates in one go, reading annotation
paths from the config's own `DATA.common.data_root` so config and data cannot
disagree. Swap in `video_action_spotting.yaml` for the baseline — same script.

First run over a new dataset builds the per-game caches (slow, once); afterwards
they are memory-mapped. The cache rebuilds automatically whenever the parquet is
newer than its sidecars.

## Where things live

| Path | What |
|---|---|
| `tools/convert/build_sngar_spotting.py` | Raw source data → the published spotting dataset pair; see [docs/tools/sngar-spotting.md](../tools/sngar-spotting.md) |
| `tools/convert/sngar_events.py` | The event-extraction rules that define the SN-GAR label set |
| `tools/convert/verify_sngar_spotting.py` | Contract verification for a built release |
| `tools/download/push_sngar_spotting.py` | Publishes the pair to the Hugging Face Hub |
| `tools/convert/build_soccernet_gar_action_spotting.py` | Separate path: GAR manifests → spotting annotations |
| `opensportslib/datasets/utils/tracking.py` | Feature layout, cache, edges, augmentations |
| `opensportslib/datasets/localization_dataset.py` | `TrackingActionSpot{,Video}Dataset` |
| `opensportslib/core/utils/load_annotations.py` | `annotationstoe2eformat_tracking`, `frame`/`eval_frame` |
| `opensportslib/core/utils/data.py` | Graph collate functions |
| `opensportslib/models/backbones/builder.py` | `GraphSequenceEncoder` |
| `opensportslib/metrics/localization_metric.py` | `_locate`, second-based tolerances |
