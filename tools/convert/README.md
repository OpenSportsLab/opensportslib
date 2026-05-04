# Convert Tools

Scripts for building OpenSportsLib (OSL) datasets from raw sources, and for converting OSL JSON annotations to and from a Parquet + WebDataset
representation suited for large-scale training.

## Scripts

Build (raw source -> OSL JSON):

- `build_soccernet_gar.py`: PFF FC raw data -> SoccerNet-GAR classification dataset (OSL JSON manifest of windowed action clips).
- `build_soccernet_gar_action_spotting.py`: SoccerNet-GAR classification manifest -> SoccerNet-GAR action-spotting dataset (per-game manifest with
  event timestamps).

Convert (OSL JSON <-> Parquet + WebDataset):

- `osl_json_to_parquet_webdataset.py`: OSL JSON -> Parquet + WebDataset.
- `parquet_webdataset_to_osl_json.py`: Parquet + WebDataset -> OSL JSON.

## Pipeline overview

Stage 1 and stage 2 are SoccerNet-GAR-specific (they know about PFF schemas, event labels, and clip windowing). The conversion scripts at the right are generic OSL tooling: they accept any OSL JSON manifest and don't assume a particular sport or task.

## Build scripts

### `build_soccernet_gar.py`

Builds the SoccerNet-GAR classification dataset from raw PFF FC data. The script has two CLI subcommands:

- `convert`: Organize raw PFF data into per-split (train/valid/test) folders and produce a per-split manifest JSON file. Tracking files (`.jsonl.bz2`) are converted to Parquet; video files are copied as-is.
- `extract`: Read a `convert` output and emit fixed-length action clips centered on each annotated event. Each clip carries a tracking-window
  (Parquet) and/or a frames-window (NumPy) modality.

Per-sample metadata is written into each entry's `metadata` block in the OSL JSON, not at the top level. Fields: `game_id`, `game_time`, `position_ms`, `team`, `source_fps` (rate of the underlying video), `effective_fps` (rate the clip was sampled at), `window_size`, `frame_interval`.

The `convert` subcommand reads from raw PFF FC data, available on Hugging Face: https://huggingface.co/datasets/OpenSportsLab/PFF. Download it so the local layout matches the structure below. If you only want to run `extract` against a previously-built dataset, you can skip this download.

Expected input layout:

```
PFF-FC/
├── RawEventsData/      # one .json per game (PFF event format)
├── PlayerPoseTracking/ # one .jsonl.bz2 per game (PFF tracking)
└── 224p/               # one .mp4 per game (broadcast video)
```

CLI usage:

```bash
# Stage 1a: convert tracking files (.jsonl.bz2 -> .parquet) per split
python tools/convert/build_soccernet_gar.py convert \
    --modality tracking \
    --events-dir PFF-FC/RawEventsData \
    --tracking-dir PFF-FC/PlayerPoseTracking \
    --output-dir data/tracking_dataset \
    --num-workers 24 \
    --fps 30

# Stage 1b: copy videos per split
python tools/convert/build_soccernet_gar.py convert \
    --modality video \
    --events-dir PFF-FC/RawEventsData \
    --video-dir PFF-FC/224p \
    --output-dir data/video_dataset \
    --fps 30

# Stage 2: extract windowed clips (modality: frames, tracking, or both)
python tools/convert/build_soccernet_gar.py extract \
    --video-dir data/video_dataset \
    --tracking-dir data/tracking_dataset \
    --output-dir data/soccernet_gar \
    --modality both \
    --window-size 16 \
    --frame-interval 9 \
    --num-workers 24
```

# Stage 2 alternative: express the sampling rate directly. --target-fps replaces --frame-interval. The two are mutually exclusive. Stride is derived per game as round(source_fps / target_fps); a 16-frame window at 2 Hz covers 8 seconds.
python tools/convert/build_soccernet_gar.py extract \
    --video-dir data/video_dataset \
    --tracking-dir data/tracking_dataset \
    --output-dir data/soccernet_gar \
    --modality both \
    --window-size 16 \
    --target-fps 2 \
    --num-workers 24
```

Stage 1 has a skip-if-output-exists guard. If you change upstream data or the conversion logic, delete the output directory before rerunning.

### `build_soccernet_gar_action_spotting.py`

Reads a SoccerNet-GAR classification manifest (clip-level) and emits a SoccerNet-GAR action-spotting dataset (per-game manifest with all events
sorted by `position_ms`). Splits are inherited from the input manifest, so each game stays in the split it had for classification.

The script does not re-derive events; it groups the same clips by `game_id` and reformats. Two modalities supported, run independently:

- `video`: copy `{game_id}.mp4` from `--source-dir` to
  `{output-dir}/{split}/{game_id}.mp4`.
- `tracking`: read `{game_id}.parquet` from
  `{source-dir}/{split}/videos/`, sort by `(videoTimeMs, frameNum)`, drop
  duplicate rows, and write to `{output-dir}/{split}/{game_id}.parquet`.

CLI usage:

```bash
# video spotting dataset
python tools/convert/build_soccernet_gar_action_spotting.py \
    --modality video \
    --manifest-dir sngar-frames \
    --source-dir /path/to/PFF-FC/720p \
    --output-dir data/spotting_video

# tracking spotting dataset
python tools/convert/build_soccernet_gar_action_spotting.py \
    --modality tracking \
    --manifest-dir sngar-frames \
    --source-dir data/tracking_dataset \
    --output-dir data/spotting_tracking
```

## Convert scripts

### JSON -> Parquet + WebDataset

```bash
python tools/convert/osl_json_to_parquet_webdataset.py <json_path> <media_root> <output_dir> [options]
```

### Parquet + WebDataset -> JSON

```bash
python tools/convert/parquet_webdataset_to_osl_json.py <dataset_dir> <output_json_path> [options]
```

## Python API

```python
from opensportslib.tools import convert_json_to_parquet, convert_parquet_to_json
```

## Round-trip examples

```bash
# Localization
python tools/convert/osl_json_to_parquet_webdataset.py \
    /path/to/Localization/gymnastics/annotations.json \
    /path/to/Localization/gymnastics \
    /tmp/gymnastics_wds \
    --overwrite

python tools/convert/parquet_webdataset_to_osl_json.py \
    /tmp/gymnastics_wds \
    /tmp/gymnastics_reconstructed.json

# Classification
python tools/convert/osl_json_to_parquet_webdataset.py \
    /path/to/Classification/svfouls/annotations_test.json \
    /path/to/Classification/svfouls \
    /path/to/svfouls_parquet_webdataset \
    --shard-size 500MB \
    --missing-policy skip \
    --overwrite

python tools/convert/parquet_webdataset_to_osl_json.py \
    /path/to/svfouls_parquet_webdataset \
    /path/to/svfouls_back_to_json/reconstructed_annotations.json \
    --extract-media \
    --output-media-root /path/to/svfouls_back_to_json \
    --indent 2

# SN-GAR-tracking
python tools/convert/osl_json_to_parquet_webdataset.py \
    /path/to/sngar-tracking/annotations_test.json \
    /path/to/sngar-tracking \
    /path/to/sngar-tracking_parquet_webdataset \
    --shard-size 500MB \
    --missing-policy skip \
    --overwrite

python tools/convert/parquet_webdataset_to_osl_json.py \
    /path/to/sngar-tracking_parquet_webdataset \
    /path/to/sngar-tracking_back_to_json/reconstructed_annotations.json \
    --extract-media \
    --output-media-root /path/to/sngar-tracking_back_to_json \
    --indent 2
```

## End-to-end SoccerNet-GAR example

Build the classification dataset from raw PFF, derive the spotting variant, then convert both to Parquet + WebDataset:

```bash
# 1. classification: raw PFF -> OSL JSON (clip-level)
python tools/convert/build_soccernet_gar.py convert --modality tracking
python tools/convert/build_soccernet_gar.py convert --modality video
python tools/convert/build_soccernet_gar.py extract --modality both \
    --output-dir data/sngar_frames

# 2. spotting: classification manifest -> per-game OSL JSON
python tools/convert/build_soccernet_gar_action_spotting.py \
    --modality tracking \
    --manifest-dir data/sngar_frames \
    --source-dir data/tracking_dataset \
    --output-dir data/spotting_tracking

# 3. either dataset -> Parquet + WebDataset for training
python tools/convert/osl_json_to_parquet_webdataset.py \
    data/sngar_frames/annotations_train.json \
    data/sngar_frames \
    data/sngar_frames_wds \
    --shard-size 500MB \
    --missing-policy skip \
    --overwrite
```