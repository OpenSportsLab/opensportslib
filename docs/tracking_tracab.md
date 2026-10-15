# TRACAB tracking

Use `type: "tracking_tracab"` for an original classic ASCII DAT. The original
JSON and the distinct TF05/TF08 XML feeds are companion files. Keep source values
in those files; `metadata` holds only the additional UTC calibration and the
explicit mapping between DAT team codes and home/away.

```json
{
  "type": "tracking_tracab",
  "path": "tracking/match.dat",
  "metadata_path": "tracking/match_metadata.json",
  "tf05_path": "tracking/match_TF05.xml",
  "tf08_path": "tracking/match_TF08.xml",
  "UTC_time_start": "2020-01-01 12:00:00.000000",
  "duration_ms": 6000,
  "metadata": {
    "team_codes": {"home": 1, "away": 0},
    "periods": [
      {"period": 1, "start_utc": "2020-01-01 12:00:01.000000"},
      {"period": 2, "start_utc": "2020-01-01 12:00:04.000000"}
    ]
  }
}
```

All paths resolve from the same manifest/media root. XML paths are optional;
when supplied, the actual root feed ID must match the reference. TF05 and TF08
are not interchangeable. Transfers, TAR packaging, inference bundles and VAT
Save As include/rebase every referenced file. Source files are never rewritten.

The JSON reader supports the observed flat metadata schema, for example:

```json
{
  "GameID": "001",
  "FrameRate": 25,
  "PitchLongSide": 10500,
  "PitchShortSide": 6800,
  "Phase1StartFrame": 100,
  "Phase1EndFrame": 102,
  "Phase2StartFrame": 200,
  "Phase2EndFrame": 202,
  "HomeTeam": {"TeamID": "H", "Players": [
    {"PlayerID": "0009", "JerseyNo": 9, "StartFrameCount": 100, "EndFrameCount": 202}
  ]},
  "AwayTeam": {"TeamID": "A", "Players": [
    {"PlayerID": "0003", "JerseyNo": 3, "StartFrameCount": 100, "EndFrameCount": 202}
  ]}
}
```

FPS, actual pitch dimensions, rosters, active intervals and inclusive period
bounds come from this JSON. They are not repeated under input `metadata`.
`period: N` refers to `PhaseNStartFrame`/`PhaseNEndFrame`. Only populated phases
may be calibrated. Source frame UTC is `start_utc + (frame - PhaseNStartFrame) /
FrameRate`; playback position subtracts `UTC_time_start`. No UTC is inferred
from frame numbering or timezone-free XML timestamps. Naive OSL timestamp
strings mean UTC. Video alignment remains a constant offset within each period.

The native API preserves input coordinates, field tokens, frame IDs, original
bytes, all targets and unknown fields. VAT and the tracking graph loaders
explicitly request the normalized view: classic centimeter coordinates and
height become centered meters, preserving original axes and supplied speed.
Pitch dimensions are required for this view; no standard pitch is substituted.
There is no automatic mirroring, team reversal, smoothing or speed recomputation.

Native timing duplicates, gaps and out-of-order records are preserved and
reported. The synchronized playback view rejects duplicate/out-of-order frames,
invalid calibration, overlapping periods, missing boundary frames and timing
outside the declared duration. It displays an empty pitch before play, during
halftime/missing frames and after play. Its virtual empty rows and graph sampling
do not alter native records. Player rendering excludes non-player team codes;
native access retains them. Ambiguous roster identities fail explicitly.

This supports the inspected classic ASCII layout, not all TRACAB products or
versions. XML attributes and unknown sections remain accessible without guessed
semantics. Heatmap strings retain leading zeros; decoding requires independently
established grid dimensions, cell order, orientation and value semantics.

**BREAKING CHANGE from the unreleased prototype:** replace input type
`tracab_dat` with `tracking_tracab`, rename `tracab` to `metadata`, and replace
embedded period frame bounds with `period` references when using `metadata_path`.
The old modality/key are not aliases. Regenerate converter-produced manifests.
Standalone descriptors without source metadata remain available for synthetic
or custom inputs, with explicit FPS, pitch dimensions and period frame bounds.

## Native API and field mapping

```python
from opensportslib.tracking import TracabDatReader, read_document

for record in TracabDatReader("match.dat", mode="permissive"):
    # Default native access: every source line in file order, no resampling.
    print(record.frame_id, record.targets, record.balls, record.diagnostics)
    original_bytes = record.raw

metadata = read_document("match_metadata.json", format="json")
tf05 = read_document("match_TF05.xml", format="xml")
tf08 = read_document("match_TF08.xml", format="xml")
for heatmap in tf05.heatmaps():
    print(heatmap["attribute"], heatmap["encoded"])
```

`TracabDatReader` streams binary sources. It closes files it opens and leaves
caller-owned streams open. Strict mode rejects malformed required records with
file/line/frame diagnostics. Permissive mode yields a rejected `NativeFrame`
with its complete bytes and diagnostics; parsing may be partial after the first
error, but no source bytes are lost. Both modes retain duplicate/out-of-order
records and report gaps. A set of observed frame IDs detects repeated IDs;
records themselves are not accumulated in memory. Unsupported block layouts
produce an error, not a guessed alternative interpretation.

DAT columns have no native names. These positional names describe the supported
classic layout; `fields` retains every original string, including additional
columns and leading zeros. Empty blocks and trailing delimiters stay in `raw`.
Absent balls produce an empty list. Empty required numeric tokens are rejected,
not confused with valid zeros. No undocumented sentinel is interpreted.

| Source | Native representation | Explicit playback/graph view |
| --- | --- | --- |
| Frame prefix | `frame_id` string, original line bytes | Integer source frame; calibrated UTC/video time |
| Target fields 0–2 | `team_code`, `tracking_id`, `shirt_number` strings | `teamCode`, `trackId`, `shirtNumber`; numeric `jerseyNum` for graph slots |
| Target fields 3–5 | `x`, `y`, `speed` strings | X/Y divided by 100; supplied speed parsed without conversion |
| Extra target fields | `additional_fields`, exact strings | Retained as `additional_fields` |
| Ball fields 0–3 | `x`, `y`, `z`, `speed` strings | X/Y/Z divided by 100; speed unchanged |
| Ball fields 4–5 | `possession`, `status`, original codes | `possession`, `state`, unchanged codes |
| Extra ball fields | `additional_fields`, including empty tokens | `flags`, unchanged strings |
| JSON | All original keys/values plus raw bytes | FrameRate, pitch, periods and roster identity join |
| XML | Raw bytes, namespace declarations, ElementTree preserving repeated/unknown elements, attributes, text, comments and processing instructions | Known identity fields checked; no invented TF08 statistical mappings |
| TF05 heatmaps | Exact strings, source element path and attribute name | `decode_heatmap` requires explicit dimensions/order/orientation/semantics; retains digit strings |

Native JSON retains source types, and XML/token identifiers remain strings.
The normalized join uses explicit home/away code mapping, team identity and
shirt number within the roster's active frame intervals. A shared shirt number
alone never merges players. Reused jerseys require disjoint active intervals.
System tracking IDs remain separate and may change. An observation outside a
known roster interval is marked `identityStatus: unresolved_outside_roster_interval`
without inventing a player ID; its native target remains accessible.

`metadata_conflicts` checks only observed match/team/player identity attributes
across JSON and XML. Conflicts identify both sources and values. Unknown XML
fields are preserved, not assigned meanings. The normalized view refuses
known identity conflicts, duplicate FPS/geometry/rosters that disagree, and
calibrated frame bounds that disagree with their source phase.

## Explicit normalized API

```python
from opensportslib.tracking import TracabTrackingReader

with TracabTrackingReader("tracking/match.dat", input_descriptor,
                          metadata_base_dir=".") as tracking:
    row = tracking.row_at(1000)  # video-relative milliseconds
    original = tracking.native_frame(tracking.time_axis_ms.index(1000))
    print(tracking.transformations)
    print(tracking.documents["metadata_path"].value)
```

The normalized reader owns its DAT handle, indexes byte offsets for random
access, and exposes source documents plus `native_frame(index)`. Local handles
reopen after pickling/fork; caller-supplied seekable streams serialize a snapshot.
Thread access uses a lock. For TAR assets, pass `NativeDocument` objects as
`metadata` and in `xml_metadata={"tf05_path": ..., "tf08_path": ...}`.
`resolve_tracab_descriptor` creates an internal view without modifying the
manifest. Its derived fields are not written back by readers or VAT.

`reader[index]` returns `videoTimeMs`, `timestamp_utc`, `sourceFrame`, `homePlayers`,
`awayPlayers`, and `balls`. `iter_uniform_rows()` is an explicitly sampled video
timeline used only by the graph adapters. Missing nodes retain the existing
sentinel; displacement resets across missing native frames. Native iteration
never resamples, interpolates, removes stoppages, or concatenates periods.

The supported classic coordinate convention is corroborated by
[Kloppy's TRACAB coordinate system](https://kloppy.pysport.org/reference/domain/models/common/coordinate-system/).
This is external format evidence, not a unit declaration in DAT. Source speed
units and unknown codes remain unresolved; no speed conversions are performed.

## Dataset loading and persistence

Set `DATA.inputs.<name>.modality: tracking_tracab`. Local `source.format: dat`
uses the split `annotation_path` and `source_path` (the common asset root).
HF JSON uses `source.format: hf_json` and `source.input_type: tracking_tracab`.
Indexed TAR uses `source.format: hf_webdataset` with the same modality. Every
companion file is staged/packaged with the DAT; tracking bytes never become
Parquet. Classification reads TAR members; localization stages raw files for
its internal feature cache. Train/valid/test and dataloader workers are supported.

`params.annotation_head` defaults to `action` if present, otherwise the sole
head; other ambiguity requires an explicit choice. Select `headers` for that
vocabulary. Localization prefers event `timestamp_utc` and falls back to
`position_ms`. Select inputs by configured modality, not their array position.

The graph layout remains 23 objects × 8 features. Internal `.npy` caches use
parser version, source size/mtime, calibration and resolved source metadata
fingerprints. JSON/XML edits invalidate caches; per-worker memoization checks
companion file size/mtime. Cache writes use locks and atomic replacement.

VAT uses `TracabTrackingMediaBackend`, shared `PitchMediaBackend` rendering and
the asynchronous raster runtime. Explorer resolves all companion paths; source
identity includes calibration and companion file signatures. Inference resolves
paths before staging temporary manifests. Save/reopen preserves source references
and calibration through existing owners and HistoryManager boundaries.

Requires the updated editable OpenSportsLib checkout until a containing release
is published. Tests use synthetic fixtures and offline Hub stubs via the required
`bash scripts/run_tests.sh`. Real delivery audits and comparison scripts belong
outside tracked generic repository tests.
