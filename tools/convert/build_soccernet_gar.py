"""
PFF FC -> SoccerNet-GAR clip dataset pipeline.

This script combines the two original conversion stages into a single CLI:

  Stage 1 (convert): Convert raw PFF FC data into a SoccerNet-GAR-style
    structured dataset. Tracking data (.jsonl.bz2) is converted to Parquet,
    or video files (.mp4) are organized into train/valid/test splits.
    Per-split manifest JSON files are produced from the PFF event files.

  Stage 2 (extract): Take the structured dataset from Stage 1 and produce
    windowed action clips (numpy frames and/or tracking parquet slices)
    centered on each annotated event, ready for training.

Usage:
    # Stage 1a - tracking conversion (.jsonl.bz2 -> .parquet)
    python tools/convert/build_soccernet_gar.py convert --modality tracking \
        --events-dir PFF-FC/RawEventsData \
        --tracking-dir PFF-FC/PlayerPoseTracking \
        --output-dir data/tracking_dataset

    # Stage 1b - video copy
    python tools/convert/build_soccernet_gar.py convert --modality video \
        --events-dir PFF-FC/RawEventsData \
        --video-dir PFF-FC/224p \
        --output-dir data/video_dataset

    # Stage 2 - clip extraction (frames, tracking, or both)
    python tools/convert/build_soccernet_gar.py extract --modality both \
        --video-dir data/video_dataset \
        --tracking-dir data/tracking_dataset \
        --output-dir data/soccernet_gar
"""

import argparse
import bz2
import concurrent.futures
import json
import os
import shutil
from collections import defaultdict
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------------
# shared constants
# ---------------------------------------------------------------------------

# dataset splits expressed as (start_idx, end_idx) ranges over games sorted
# alphabetically by filename. these ranges define which games go into each
# split and are used by both stages of the pipeline.
SPLITS = {
    "train": (0, 45),
    "valid": (45, 54),
    "test": (54, 64),
}

# activity classes (group activity recognition labels) used for both
# manifest generation in stage 1 and clip labeling in stage 2.
LABELS = [
    "PASS", "HEADER", "HIGH PASS", "OUT", "CROSS",
    "THROW IN", "SHOT", "PLAYER SUCCESSFUL TACKLE", "FREE KICK", "GOAL",
]

# mapping from detailed pff position codes to coarse positional groups,
# used in stage 1 when annotating tracking frames with player roles.
POSITION_GROUPS = {
    "GK": "GK",
    "LCB": "DEF", "RCB": "DEF", "MCB": "DEF",
    "LB":  "DEF", "RB":  "DEF", "LWB": "DEF", "RWB": "DEF",
    "CM":  "MID", "AM":  "MID", "DM":  "MID", "LM":  "MID", "RM": "MID",
    "CF":  "FWD", "LW":  "FWD", "RW":  "FWD",
}

# label priority used in stage 2 when multiple annotations share the exact
# same timestamp. higher value wins.
LABEL_PRIORITY = {
    "PASS": 0,
    "HIGH PASS": 1,
    "OUT": 2,
    "CROSS": 3,
    "SHOT": 4,
    "HEADER": 5,
    "PLAYER SUCCESSFUL TACKLE": 6,
    "THROW IN": 7,
    "FREE KICK": 8,
    "GOAL": 9,
}

# tolerance (in milliseconds) for matching an annotation timestamp to the
# nearest tracking frame. annotations farther than this are skipped.
TIME_TOLERANCE_MS = 10


# ---------------------------------------------------------------------------
# stage 1: pff -> soccernet-gar structured dataset
# ---------------------------------------------------------------------------

def build_position_mapping(jsonl_path):
    """
    Scan a tracking jsonl.bz2 file and build a team -> jersey -> position
    lookup table from any embedded ``game_event`` records.

    Returns a plain dict-of-dicts keyed by string team_id and string
    jersey number.
    """
    position_map = defaultdict(lambda: defaultdict(lambda: None))

    with bz2.open(jsonl_path, "rt") as f:
        for line in f:
            try:
                frame = json.loads(line)
                game_event = frame.get("game_event")

                if game_event and isinstance(game_event, dict):
                    team_id = game_event.get("team_id")
                    shirt_num = game_event.get("shirt_number")
                    position = game_event.get("position_group_type")

                    if team_id and shirt_num and position:
                        position_map[str(team_id)][str(shirt_num)] = position
            except json.JSONDecodeError:
                # skip malformed lines and keep scanning.
                continue

    return dict(position_map)


def add_positions_to_players(players_list, team_id, position_map):
    """
    Annotate each player dict in ``players_list`` with ``position`` and
    ``positionGroup`` fields, looking values up in ``position_map``.

    Mutates and returns the list. Returns the input unchanged for
    empty or non-list inputs.
    """
    if not players_list or not isinstance(players_list, list):
        return players_list

    team_positions = position_map.get(str(team_id), {})

    for player in players_list:
        if isinstance(player, dict):
            jersey = str(player.get("jerseyNum", ""))
            position = team_positions.get(jersey)
            player["position"] = position if position else None
            player["positionGroup"] = POSITION_GROUPS.get(position) if position else None

    return players_list


# fixed set of game_event fields copied verbatim into each flattened frame.
_GAME_EVENT_KEYS = [
    "game_event_type", "player_name", "player_id",
    "team_id", "home_team", "video_url",
]


def flatten_frame(frame, position_map):
    """
    Flatten one nested JSON tracking frame into a single-level dict suitable
    for column-oriented Parquet storage.

    Player and ball collections are serialized as JSON strings to keep the
    schema flat while preserving the full per-frame payload.
    """
    flat = {
        "videoTimeMs": frame.get("videoTimeMs"),
        "frameNum": frame.get("frameNum"),
        "period": frame.get("period"),
        "game_event_id": frame.get("game_event_id"),
        "possession_event_id": frame.get("possession_event_id"),
    }

    # game_event block: copy a fixed set of keys, and figure out which team
    # is home vs away so positions can be attached to the correct list.
    game_event = frame.get("game_event", {})
    home_team_id = None
    away_team_id = None

    if isinstance(game_event, dict):
        for key in _GAME_EVENT_KEYS:
            flat[key] = game_event.get(key, "")

        if game_event.get("home_team"):
            home_team_id = game_event.get("team_id")
        else:
            away_team_id = game_event.get("team_id")
    else:
        for key in _GAME_EVENT_KEYS:
            flat[key] = ""

    # possession_event block: only one field is needed downstream.
    possession_event = frame.get("possession_event", {})
    if isinstance(possession_event, dict):
        flat["possession_event_type"] = possession_event.get("possession_event_type", "")
    else:
        flat["possession_event_type"] = ""

    # tag players with position information before serialization.
    home_players = frame.get("homePlayers", [])
    away_players = frame.get("awayPlayers", [])

    if home_team_id:
        home_players = add_positions_to_players(home_players, home_team_id, position_map)
    if away_team_id:
        away_players = add_positions_to_players(away_players, away_team_id, position_map)

    # serialize all per-frame collections as json strings.
    flat["homePlayers"] = json.dumps(home_players if home_players else [])
    flat["homePlayersSmoothed"] = json.dumps(frame.get("homePlayersSmoothed", []))
    flat["awayPlayers"] = json.dumps(away_players if away_players else [])
    flat["awayPlayersSmoothed"] = json.dumps(frame.get("awayPlayersSmoothed", []))
    flat["balls"] = json.dumps(frame.get("balls", []))
    flat["ballsSmoothed"] = json.dumps(frame.get("ballsSmoothed", []))

    return flat


def convert_jsonl_to_parquet(jsonl_path, parquet_path):
    """
    Read a compressed jsonl tracking file, flatten every frame, and write
    the result as a Parquet file. Returns the number of frames written.
    """
    position_map = build_position_mapping(jsonl_path)
    frames = []

    with bz2.open(jsonl_path, "rt") as f:
        for line in f:
            try:
                frame = json.loads(line)
                frames.append(flatten_frame(frame, position_map))
            except json.JSONDecodeError:
                continue

    df = pd.DataFrame(frames)

    # cast columns to their expected dtypes for compact, predictable storage.
    int_columns = ["frameNum", "period", "game_event_id", "possession_event_id"]
    for col in int_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype("int32")

    float_columns = ["videoTimeMs"]
    for col in float_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    string_columns = [c for c in df.columns if c not in int_columns + float_columns]
    for col in string_columns:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    df.to_parquet(parquet_path, engine="pyarrow", compression=None, index=False)
    return len(frames)


def process_tracking_file(args):
    """
    Worker function (used with ProcessPoolExecutor) that converts a single
    tracking file and returns the (game_id, split_name) it was placed in,
    or None if the index falls outside any defined split range.
    """
    idx, tracking_file, tracking_dir, output_dir = args
    game_id = tracking_file.replace(".jsonl.bz2", "")

    split_name = None
    for split, (start, end) in SPLITS.items():
        if start <= idx < end:
            split_name = split
            break

    if split_name:
        src = os.path.join(tracking_dir, tracking_file)
        dst = os.path.join(output_dir, split_name, "videos", f"{game_id}.parquet")

        if not os.path.exists(dst):
            convert_jsonl_to_parquet(src, dst)

        return (game_id, split_name)

    return None


def extract_annotations(events_path):
    """
    Parse a PFF event JSON file and emit a list of SoccerNet-GAR-style
    annotations sorted by their position (in milliseconds).

    Each PFF event can yield zero, one, or several output annotations,
    depending on which combination of possession/game-event fields maps
    onto the configured LABELS.
    """
    with open(events_path, "r") as f:
        data = json.load(f)

    annotations = []

    for event in data:
        labels = []

        possession = event.get("possessionEvents", {})
        event_type = possession.get("possessionEventType", "")
        body_type = possession.get("bodyType", "")

        # possession-event-driven labels.
        if event_type == "PA":
            if body_type == "HE":
                labels.append("HEADER")
            elif possession.get("ballHeightType") == "A":
                labels.append("HIGH PASS")
            elif possession.get("passType") == "H":
                labels.append("THROW IN")
            else:
                labels.append("PASS")

        elif event_type == "CR":
            labels.append("CROSS")

        elif event_type == "SH":
            if body_type == "HE":
                labels.append("HEADER")
            labels.append("SHOT")
            if possession.get("shotOutcomeType") == "G":
                labels.append("GOAL")

        elif event_type == "CH":
            if possession.get("challengeWinnerPlayerId"):
                labels.append("PLAYER SUCCESSFUL TACKLE")

        elif event_type == "CL" and body_type == "HE":
            labels.append("HEADER")

        # game-event-driven labels (independent of the possession event type).
        game_events = event.get("gameEvents", {})
        if game_events.get("gameEventType") == "OUT":
            labels.append("OUT")

        setpiece = game_events.get("setpieceType", "")
        if setpiece == "T":
            labels.append("THROW IN")
        elif setpiece == "F":
            labels.append("FREE KICK")

        # emit one annotation per label that survives the labels filter.
        for label in labels:
            if label in LABELS:
                annotation = {
                    "gameTime": f"{game_events.get('period', 1)} - {game_events.get('startFormattedGameClock', '00:00')}",
                    "label": label,
                    "position": int(event.get("eventTime", 0) * 1000),
                    "team": "home" if game_events.get("homeTeam", False) else "away",
                    "visibility": "visible",
                }
                annotations.append(annotation)

    return sorted(annotations, key=lambda x: x["position"])


def create_split_manifests(events_dir, file_mapping, output_dir, file_format, fps):
    """
    For every game in ``file_mapping`` (game_id -> split), build a per-video
    manifest entry (with annotations if the corresponding events file
    exists) and write a {split}.json file under output_dir/{split}/.

    Returns the per-split list of video entries actually written.
    """
    split_data = {split: [] for split in SPLITS}

    print("Creating split manifests...")
    for game_id, split in tqdm(file_mapping.items()):
        file_ext = ".parquet" if file_format == "parquet" else ".mp4"

        # source_fps is the rate of the underlying video/tracking, recorded
        # per video so that mixed-rate datasets stay well-defined. stage 2
        # uses this as the reference rate when deriving effective fps from
        # the target sampling rate.
        video_entry = {
            "path": f"videos/{game_id}{file_ext}",
            "input_type": file_format,
            "source_fps": fps,
            "gameId": game_id,
        }

        events_path = os.path.join(events_dir, f"{game_id}.json")
        if os.path.exists(events_path):
            video_entry["annotations"] = extract_annotations(events_path)

        split_data[split].append(video_entry)

    # write one manifest per non-empty split.
    for split, videos in split_data.items():
        if videos:
            output_file = os.path.join(output_dir, split, f"{split}.json")
            split_json = {
                "version": 1,
                "format": file_format,
                "videos": videos,
                "labels": LABELS,
            }
            with open(output_file, "w") as f:
                json.dump(split_json, f, indent=2)

    return split_data


def process_tracking_modality(events_dir, tracking_dir, output_dir, num_workers, fps):
    """
    Stage 1 entry point for the tracking modality: parallel-convert every
    .jsonl.bz2 tracking file into Parquet, then build the manifests.
    Skips work if a manifest already exists at the destination.
    """
    if os.path.exists(os.path.join(output_dir, "train", "train.json")):
        print("Tracking dataset already exists. Skipping conversion.")
        return

    print("Converting tracking data to parquet format...")

    for split in SPLITS:
        os.makedirs(os.path.join(output_dir, split, "videos"), exist_ok=True)

    # alphabetical order is what defines the train/valid/test partition.
    tracking_files = sorted([f for f in os.listdir(tracking_dir) if f.endswith(".jsonl.bz2")])
    args_list = [(idx, f, tracking_dir, output_dir) for idx, f in enumerate(tracking_files)]

    # executor.map preserves input order, so file_mapping insertion order
    # follows the alphabetically-sorted input list. this makes manifest
    # contents deterministic across runs.
    file_mapping = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for result in tqdm(executor.map(process_tracking_file, args_list),
                           total=len(args_list), desc="Converting"):
            if result:
                game_id, split_name = result
                file_mapping[game_id] = split_name

    split_data = create_split_manifests(events_dir, file_mapping, output_dir, "parquet", fps)

    print(f"\nTracking dataset created at: {output_dir}")
    for split in SPLITS:
        print(f"  {split}: {len(split_data[split])} videos")


def process_video_modality(events_dir, video_dir, output_dir, fps):
    """
    Stage 1 entry point for the video modality: copy every .mp4 into the
    correct split folder and build the manifests. Skips work if a manifest
    already exists at the destination.
    """
    if os.path.exists(os.path.join(output_dir, "train", "train.json")):
        print("Video dataset already exists. Skipping copy.")
        return

    print("Copying video files...")

    for split in SPLITS:
        os.makedirs(os.path.join(output_dir, split, "videos"), exist_ok=True)

    video_files = sorted([f for f in os.listdir(video_dir) if f.endswith(".mp4")])
    file_mapping = {}

    for idx, video_file in enumerate(tqdm(video_files, desc="Copying")):
        game_id = video_file.replace(".mp4", "")

        split_name = None
        for split, (start, end) in SPLITS.items():
            if start <= idx < end:
                split_name = split
                break

        if split_name:
            src = os.path.join(video_dir, video_file)
            dst = os.path.join(output_dir, split_name, "videos", video_file)

            if not os.path.exists(dst):
                shutil.copy2(src, dst)

            file_mapping[game_id] = split_name

    split_data = create_split_manifests(events_dir, file_mapping, output_dir, "mp4", fps)

    print(f"\nVideo dataset created at: {output_dir}")
    for split in SPLITS:
        print(f"  {split}: {len(split_data[split])} videos")


# ---------------------------------------------------------------------------
# stage 2: structured dataset -> action clips
# ---------------------------------------------------------------------------

def deduplicate_annotations(annotations):
    """
    Safety net for cases where multiple annotations land on the exact same
    position_ms: keep only the highest-priority label per timestamp.
    """
    by_position = defaultdict(list)
    for ann in annotations:
        by_position[ann["position"]].append(ann)

    deduped = []
    for position_ms in sorted(by_position.keys()):
        candidates = by_position[position_ms]
        if len(candidates) == 1:
            deduped.append(candidates[0])
        else:
            # pick the highest priority label.
            best = max(candidates, key=lambda a: LABEL_PRIORITY.get(a["label"], -1))
            deduped.append(best)

    return deduped


def _read_frames_sequential(cap, wanted_frames, total_frames, seek_threshold=64):
    """
    read a set of frames from a videocapture without random seeking per frame.

    walks the video sequentially, decoding only the frames present in
    ``wanted_frames`` (a sorted list of unique non-negative ints). for gaps
    larger than ``seek_threshold`` we issue a single cap.set() to skip past
    the gap; for smaller gaps we just decode through them, since one seek
    typically costs more than decoding ~60 frames on h.264.

    returns dict: frame_idx -> np.uint8 array (224, 224, 3) in rgb.
    frames outside [0, total_frames) are returned as zero arrays.
    """
    out = {}
    if not wanted_frames:
        return out

    # filter to in-range targets; out-of-range get zero-filled later.
    in_range = [f for f in wanted_frames if 0 <= f < total_frames]
    if not in_range:
        return out

    current = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    for target in in_range:
        gap = target - current
        if gap < 0:
            # rare: targets should be sorted, but be defensive.
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            current = target
        elif gap > seek_threshold:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            current = target
        else:
            # decode through the gap. cap.grab() decodes without copying
            # the pixel data out, which is faster than read() when we
            # don't need the intermediate frames.
            for _ in range(gap):
                if not cap.grab():
                    break
                current += 1

        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            out[target] = frame
        current += 1

    return out


def process_game(args):
    """
    Worker function that, for a single game, extracts a windowed clip
    (frames and/or tracking) per annotation, aligned to the tracking
    timeline by ``videoTimeMs``.

    Returns a list of result dicts, one per successfully extracted clip.
    """
    (game_id, video_path, parquet_path, annotations,
     window_size, frame_interval, modality, source_fps) = args

    # only open the video reader if frames are actually needed. when frames
    # are needed we read the video file's fps directly for window indexing
    # (it's the ground truth for the underlying mp4); when only tracking is
    # requested we fall back to the manifest's source_fps so the value can
    # still flow into per-sample metadata.
    cap = None
    total_video_frames = 0
    if modality in ("frames", "both"):
        cap = cv2.VideoCapture(video_path)
        # cv2 reports the source video's frame rate; this is the same
        # quantity the manifest carries as source_fps. we trust cv2 here
        # because it sees the actual file.
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # tracking data is always required: it provides the videotimems <-> framenum
    # mapping used to align annotations and to populate the tracking clips.
    tracking_df = pd.read_parquet(parquet_path).sort_values(
        ["videoTimeMs", "frameNum"], ascending=[True, True]
    ).reset_index(drop=True)

    # precompute lookup structures once per game to avoid o(n) pandas scans
    # inside the per-annotation loop. with ~150k tracking rows and ~1500
    # annotations this turns the inner loop from minutes to seconds.

    # contiguous numpy view of timestamps for searchsorted (o(log n) lookup).
    times_ms = tracking_df["videoTimeMs"].to_numpy()

    # framenum -> first row position. dict construction iterates once; lookups
    # are o(1). taking the first occurrence preserves the original code's
    # `frame_data.iloc[0]` semantics for any duplicated framenums.
    frame_nums_arr = tracking_df["frameNum"].to_numpy()
    frame_to_row = {}
    for i, fnum in enumerate(frame_nums_arr):
        # `int(fnum)` keeps keys as plain python ints so lookups don't depend
        # on numpy's int subclass equality semantics.
        key = int(fnum)
        if key not in frame_to_row:
            frame_to_row[key] = i

    # ----- pass 1: build per-annotation specs and collect all wanted frames.
    # this lets us fetch video frames in sorted order in pass 2, which is
    # vastly cheaper than per-frame random seeks.

    specs = []           # list of dicts describing each kept annotation
    wanted_video = set() # global set of video frame indices to decode

    for ann in annotations:
        position_ms = ann["position"]
        label = ann["label"]

        if label not in LABELS:
            continue

        # locate the closest tracking row by videotimems using searchsorted.
        ins = np.searchsorted(times_ms, position_ms)
        if ins == 0:
            closest_idx = 0
        elif ins == len(times_ms):
            closest_idx = len(times_ms) - 1
        else:
            left_diff = position_ms - times_ms[ins - 1]
            right_diff = times_ms[ins] - position_ms
            # tie-break to the left to match the original idxmin() behavior,
            # which returns the lowest index when multiple rows tie.
            closest_idx = ins - 1 if left_diff <= right_diff else ins

        closest_diff = abs(times_ms[closest_idx] - position_ms)

        # skip if no tracking data lies within tolerance.
        if closest_diff > TIME_TOLERANCE_MS:
            specs.append(None)  # placeholder so we can count skips in pass 3
            continue

        center_frame_num = int(tracking_df.iloc[closest_idx]["frameNum"])

        # window of tracking frame numbers centered on the event.
        half_window = window_size // 2
        tracking_window = [
            center_frame_num + (i - half_window) * frame_interval
            for i in range(window_size)
        ]

        # window of video frame indices, computed from position_ms (not from
        # tracking-derived center_frame_num) so frames-only and frames+tracking
        # modes use the same indexing strategy.
        video_window = None
        if modality in ("frames", "both"):
            center_video_frame = int((position_ms / 1000.0) * source_fps)
            video_start = center_video_frame - half_window * frame_interval
            video_window = [video_start + i * frame_interval for i in range(window_size)]
            for f in video_window:
                if 0 <= f < total_video_frames:
                    wanted_video.add(f)

        specs.append({
            "ann": ann,
            "tracking_window": tracking_window,
            "video_window": video_window,
        })

    # ----- pass 2: decode all wanted video frames in one sequential walk.
    frame_cache = {}
    if modality in ("frames", "both") and wanted_video:
        frame_cache = _read_frames_sequential(
            cap, sorted(wanted_video), total_video_frames
        )

    # ----- pass 3: assemble per-annotation results from caches.
    results = []
    skipped = 0
    zero_frame = np.zeros((224, 224, 3), dtype=np.uint8)

    for spec in specs:
        if spec is None:
            skipped += 1
            continue

        ann = spec["ann"]

        # tracking clip from the o(1) framenum cache.
        clip_rows = []
        for fnum in spec["tracking_window"]:
            row_idx = frame_to_row.get(int(fnum))
            if row_idx is not None:
                clip_rows.append(tracking_df.iloc[row_idx])
            else:
                clip_rows.append(pd.Series({
                    "videoTimeMs": np.nan,
                    "frameNum": fnum,
                    "period": -1,
                    "balls": "[]",
                    "homePlayers": "[]",
                    "awayPlayers": "[]",
                }))
        clip_df = pd.DataFrame(clip_rows).reset_index(drop=True)

        # video clip from the sequential-read cache.
        frames = None
        if spec["video_window"] is not None:
            frames = np.array([
                frame_cache.get(f, zero_frame) for f in spec["video_window"]
            ])

        results.append({
            "frames": frames,
            "tracking": clip_df,
            "label": ann["label"],
            "game_id": game_id,
            "game_time": ann["gameTime"],
            "position_ms": ann["position"],
            "team": ann["team"],
            "source_fps": source_fps,
            "frame_interval": frame_interval,
        })

    if cap is not None:
        cap.release()

    if skipped > 0:
        print(f"  {game_id}: skipped {skipped} events (time tolerance > {TIME_TOLERANCE_MS}ms)")

    return results


def print_split_stats(split, data_entries):
    """Print per-class clip counts for a single split."""
    counts = defaultdict(int)
    for entry in data_entries:
        counts[entry["labels"]["action"]["label"]] += 1

    total = sum(counts.values())
    print(f"\n  {split} stats: {total} clips")
    for label in LABELS:
        c = counts[label]
        pct = (c / total * 100) if total > 0 else 0
        print(f"    {label}: {c} ({pct:.1f}%)")


def print_overall_stats(all_stats):
    """Print combined per-class counts aggregated across all splits."""
    print(f"\n{'='*50}")
    print("overall dataset stats")
    print("=" * 50)

    total_counts = defaultdict(int)
    grand_total = 0

    for split in ["train", "valid", "test"]:
        if split in all_stats:
            for label, count in all_stats[split].items():
                total_counts[label] += count
                grand_total += count

    print(f"  total clips: {grand_total}")
    for label in LABELS:
        c = total_counts[label]
        pct = (c / grand_total * 100) if grand_total > 0 else 0
        print(f"    {label}: {c} ({pct:.1f}%)")


def run_clip_extraction(video_dir, tracking_dir, output_dir, modality,
                        window_size, frame_interval, target_fps, num_workers):
    """
    Stage 2 entry point: iterate over train/valid/test, deduplicate
    annotations from each game's manifest, extract windowed clips in
    parallel, save them to disk, and write per-split annotation files.

    Sampling stride can be specified in one of two ways:
      - frame_interval: explicit stride in source-frame units.
      - target_fps: desired effective sampling rate; the stride is derived
        per game as round(source_fps / target_fps). The achieved effective
        rate is reported when it differs from the requested rate.

    If neither is given, frame_interval defaults to 9.
    """
    # exactly one of (frame_interval, target_fps) is honored. mutual
    # exclusion is enforced upstream by argparse; we just resolve defaults.
    fixed_default_used = frame_interval is None and target_fps is None
    if fixed_default_used:
        frame_interval = 9

    print("Clip extraction config:")
    print(f"  modality:       {modality}")
    print(f"  window_size:    {window_size} frames")
    if target_fps is not None:
        print(f"  target_fps:     {target_fps} Hz (frame_interval derived per game)")
    else:
        print(f"  frame_interval: {frame_interval}")
    print(f"  num_workers:    {num_workers}")

    all_stats = {}

    for split in ["train", "valid", "test"]:
        print(f"\n{'='*50}")
        print(f"processing {split}")
        print("=" * 50)

        # output layout: single-modality runs flatten the directory tree;
        # 'both' runs separate frames and tracking into sibling subtrees.
        if modality == "frames":
            frames_out = os.path.join(output_dir, split)
            os.makedirs(frames_out, exist_ok=True)
        elif modality == "tracking":
            tracking_out = os.path.join(output_dir, split)
            os.makedirs(tracking_out, exist_ok=True)
        else:
            frames_out = os.path.join(output_dir, "frames_npy", split)
            tracking_out = os.path.join(output_dir, "tracking_parquet", split)
            os.makedirs(frames_out, exist_ok=True)
            os.makedirs(tracking_out, exist_ok=True)

        # load the per-split manifest produced by stage 1.
        manifest_path = os.path.join(video_dir, split, f"{split}.json")
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        # track which (source_fps, achieved_effective_fps) pairs we've
        # already reported, so the printout doesn't repeat once per game.
        reported_rates = set()

        # build per-game arg tuples for the worker pool.
        game_args = []
        for video_data in manifest["videos"]:
            game_id = video_data["gameId"]
            video_path = os.path.join(video_dir, split, "videos", f"{game_id}.mp4")
            parquet_path = os.path.join(tracking_dir, split, "videos", f"{game_id}.parquet")

            # source_fps is the underlying video/tracking rate. fall back
            # to None so missing entries surface as null in the output
            # rather than silently defaulting to a wrong value.
            source_fps = video_data.get("source_fps")

            # derive this game's stride from target_fps if requested. each
            # game uses its own source_fps so mixed-rate datasets resolve
            # cleanly to the same effective rate across games.
            if target_fps is not None:
                if source_fps is None or source_fps <= 0:
                    print(f"skipping {game_id}: cannot derive stride, source_fps missing or invalid")
                    continue
                game_frame_interval = max(1, round(source_fps / target_fps))
                achieved_fps = source_fps / game_frame_interval
                key = (source_fps, game_frame_interval)
                if key not in reported_rates:
                    reported_rates.add(key)
                    if abs(achieved_fps - target_fps) > 1e-6:
                        print(
                            f"  source_fps={source_fps} Hz, target={target_fps} Hz: "
                            f"using stride={game_frame_interval} -> effective {achieved_fps:.3f} Hz"
                        )
                    else:
                        print(
                            f"  source_fps={source_fps} Hz, target={target_fps} Hz: "
                            f"using stride={game_frame_interval}"
                        )
            else:
                game_frame_interval = frame_interval

            if not os.path.exists(parquet_path):
                print(f"skipping {game_id}: missing parquet (required for alignment)")
                continue

            if modality in ("frames", "both") and not os.path.exists(video_path):
                print(f"skipping {game_id}: missing video")
                continue

            # collapse same-timestamp duplicates before dispatch.
            raw_annotations = video_data.get("annotations", [])
            annotations = deduplicate_annotations(raw_annotations)

            if len(annotations) != len(raw_annotations):
                removed = len(raw_annotations) - len(annotations)
                print(f"  {game_id}: removed {removed} duplicate annotations via priority resolution")

            game_args.append((
                game_id, video_path, parquet_path, annotations,
                window_size, game_frame_interval, modality, source_fps,
            ))

        # run extraction across the worker pool. executor.map preserves
        # input order, so clip indices are deterministic across runs.
        all_results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            for results in tqdm(executor.map(process_game, game_args),
                                total=len(game_args), desc=f"{split}"):
                all_results.extend(results)

        # persist clips and assemble manifest entries.
        data_entries = []
        for idx, result in enumerate(tqdm(all_results, desc="saving")):
            frames_filename = f"clip_{idx:06d}.npy"
            tracking_filename = f"clip_{idx:06d}.parquet"

            if modality == "frames":
                np.save(os.path.join(frames_out, frames_filename), result["frames"])
            elif modality == "tracking":
                result["tracking"].to_parquet(
                    os.path.join(tracking_out, tracking_filename), index=False
                )
            else:
                np.save(os.path.join(frames_out, frames_filename), result["frames"])
                result["tracking"].to_parquet(
                    os.path.join(tracking_out, tracking_filename), index=False
                )

            inputs = []
            if modality == "frames":
                inputs.append({"type": "frames_npy", "path": f"{split}/{frames_filename}"})
            elif modality == "tracking":
                inputs.append({"type": "tracking_parquet", "path": f"{split}/{tracking_filename}"})
            else:
                inputs.append({"type": "frames_npy", "path": f"frames_npy/{split}/{frames_filename}"})
                inputs.append({"type": "tracking_parquet", "path": f"tracking_parquet/{split}/{tracking_filename}"})

            # effective_fps is the rate the clip was actually sampled at,
            # independent of the source video's frame rate. consumers
            # should prefer this over source_fps when interpreting the
            # clip's temporal axis.
            sample_source_fps = result["source_fps"]
            sample_frame_interval = result["frame_interval"]
            sample_effective_fps = (
                sample_source_fps / sample_frame_interval
                if sample_source_fps and sample_frame_interval
                else None
            )

            data_entries.append({
                "id": f"{split}_{idx:06d}",
                "inputs": inputs,
                "labels": {
                    "action": {"label": result["label"]},
                },
                "metadata": {
                    "game_id": result["game_id"],
                    "game_time": result["game_time"],
                    "position_ms": result["position_ms"],
                    "team": result["team"],
                    "source_fps": sample_source_fps,
                    "effective_fps": sample_effective_fps,
                    "window_size": window_size,
                    "frame_interval": sample_frame_interval,
                },
            })

        # write the per-split annotation file. source_fps, effective_fps,
        # window_size, and frame_interval live in each sample's metadata
        # block (one source of truth per sample) rather than at the top
        # level.
        annotation = {
            "version": "2.0",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "task": "action_classification",
            "modalities": (
                ["frames_npy"] if modality == "frames" else
                ["tracking_parquet"] if modality == "tracking" else
                ["frames_npy", "tracking_parquet"]
            ),
            "dataset_name": f"soccernet_gar_{'multimodal' if modality == 'both' else modality}_{split}",
            "labels": {
                "action": {
                    "type": "single_label",
                    "labels": LABELS,
                },
            },
            "data": data_entries,
        }

        ann_path = os.path.join(output_dir, f"annotations_{split}.json")
        with open(ann_path, "w") as f:
            json.dump(annotation, f, indent=2)

        # collect per-split counts for the final overall stats summary.
        split_counts = defaultdict(int)
        for entry in data_entries:
            split_counts[entry["labels"]["action"]["label"]] += 1
        all_stats[split] = dict(split_counts)

        print_split_stats(split, data_entries)

    print_overall_stats(all_stats)


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="PFF FC -> SoccerNet-GAR clip dataset pipeline (combined script).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # stage 1: convert raw pff data into a structured per-split dataset.
    convert_p = subparsers.add_parser(
        "convert",
        help="Stage 1: convert PFF tracking/video into a SoccerNet-GAR-style structured dataset.",
    )
    convert_p.add_argument("--modality", choices=["tracking", "video"], required=True)
    convert_p.add_argument("--events-dir", default="PFF-FC/RawEventsData")
    convert_p.add_argument("--tracking-dir", default="PFF-FC/PlayerPoseTracking")
    convert_p.add_argument("--video-dir", default="PFF-FC/224p")
    convert_p.add_argument("--output-dir", default=None)
    convert_p.add_argument("--num-workers", type=int, default=24,
                           help="Parallel worker processes for tracking conversion. Ignored for video modality.")
    convert_p.add_argument("--fps", type=int, default=30,
                           help="Source video frame rate written into each split manifest.")

    # stage 2: extract windowed action clips from a stage-1 structured dataset.
    extract_p = subparsers.add_parser(
        "extract",
        help="Stage 2: extract windowed action clips from a stage-1 structured dataset.",
    )
    extract_p.add_argument("--video-dir", default="data/video_dataset")
    extract_p.add_argument("--tracking-dir", default="data/tracking_dataset")
    extract_p.add_argument("--output-dir", default="data/soccernet_gar")
    extract_p.add_argument("--modality", choices=["frames", "tracking", "both"], default="both")
    extract_p.add_argument("--window-size", type=int, default=16)
    extract_p.add_argument(
        "--frame-interval", type=int, default=None,
        help="Stride between sampled frames in source-frame units. Mutually "
             "exclusive with --target-fps. Defaults to 9 if neither is given.",
    )
    extract_p.add_argument(
        "--target-fps", type=float, default=None,
        help="Desired effective sampling rate (Hz). Derives --frame-interval "
             "as round(source_fps / target_fps). Mutually exclusive with "
             "--frame-interval.",
    )
    extract_p.add_argument("--num-workers", type=int, default=24)

    return parser


def main():
    args = build_parser().parse_args()

    if args.command == "convert":
        if args.modality == "tracking":
            output_dir = args.output_dir if args.output_dir else "data/tracking_dataset"
            process_tracking_modality(args.events_dir, args.tracking_dir, output_dir,
                                      args.num_workers, args.fps)
        elif args.modality == "video":
            output_dir = args.output_dir if args.output_dir else "data/video_dataset"
            process_video_modality(args.events_dir, args.video_dir, output_dir, args.fps)

    elif args.command == "extract":
        # validate --target-fps / --frame-interval combination. they're
        # different ways of saying the same thing, so allowing both opens
        # the door to silent contradictions.
        if args.target_fps is not None and args.frame_interval is not None:
            build_parser().error(
                "--target-fps and --frame-interval are mutually exclusive; pass one."
            )

        run_clip_extraction(
            video_dir=args.video_dir,
            tracking_dir=args.tracking_dir,
            output_dir=args.output_dir,
            modality=args.modality,
            window_size=args.window_size,
            frame_interval=args.frame_interval,
            target_fps=args.target_fps,
            num_workers=args.num_workers,
        )


if __name__ == "__main__":
    main()
