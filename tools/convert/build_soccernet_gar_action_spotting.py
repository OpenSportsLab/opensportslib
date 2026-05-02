"""
Build OpenSportsLib action spotting datasets from GAR (classification) manifests.

Reads sngar-frames/annotations_{split}.json (clip-level), groups clips by game_id,
and emits one entry per game with all its events sorted by position_ms.

Splits are inherited from the GAR manifests, so each game stays in the split it
already had for classification.

Video mode:
  Copies {game_id}.mp4 from --source-dir into {output-dir}/{split}/{game_id}.mp4

Tracking mode:
  Reads {game_id}.parquet from --source-dir/{split}/videos/, sorts by
  (videoTimeMs, frameNum), drops duplicate rows, writes to
  {output-dir}/{split}/{game_id}.parquet

Usage:
    # video dataset
    python build_soccernet_gar_action_spotting.py --modality video \
        --manifest-dir sngar-frames \
        --source-dir /home/karkid/PFF/720p \
        --output-dir data/spotting_video

    # tracking dataset
    python build_soccernet_gar_action_spotting.py --modality tracking \
        --manifest-dir sngar-frames \
        --source-dir /home/karkid/temporal-localization/data/tracking_dataset \
        --output-dir data/spotting_tracking
"""

import argparse
import json
import os
import shutil
from collections import OrderedDict, defaultdict
from datetime import datetime

import pandas as pd
from tqdm import tqdm


LABELS = [
    "PASS", "HEADER", "HIGH PASS", "OUT", "CROSS",
    "THROW IN", "SHOT", "PLAYER SUCCESSFUL TACKLE", "FREE KICK", "GOAL",
]


def load_manifest(path):
    """Load a JSON manifest from disk."""
    with open(path, "r") as f:
        return json.load(f)


def group_clips_by_game(manifest):
    """Group GAR clip-level entries by game_id and emit per-game event lists.

    Returns an OrderedDict mapping game_id (str) to a list of event dicts
    sorted by position_ms.
    """
    by_game = OrderedDict()

    for entry in manifest.get("data", []):
        meta = entry.get("metadata", {})
        game_id = meta.get("game_id")
        position_ms = meta.get("position_ms")
        game_time = meta.get("game_time", "")
        label = entry.get("labels", {}).get("action", {}).get("label")

        if game_id is None or position_ms is None or label is None:
            continue
        if label not in LABELS:
            continue

        game_id = str(game_id)
        by_game.setdefault(game_id, []).append({
            "head": "action",
            "label": label,
            "position_ms": int(position_ms),
            "gameTime": game_time,
        })

    for events in by_game.values():
        events.sort(key=lambda e: e["position_ms"])

    return by_game


def copy_video_if_missing(src, dst):
    """Copy an mp4 from src to dst if dst does not exist."""
    if os.path.exists(dst):
        return False, "already_present"
    shutil.copy2(src, dst)
    return True, "copied"


def write_sorted_tracking_if_missing(src, dst):
    """Read a per-game tracking parquet, sort and dedup it, and write to dst.

    Returns a tuple (did_write, status, dups_removed). If dst already exists,
    no work is done and dups_removed is None.
    """
    if os.path.exists(dst):
        return False, "already_present", None

    df = pd.read_parquet(src)
    rows_before = len(df)

    df = (
        df.sort_values(["videoTimeMs", "frameNum"])
          .drop_duplicates(subset=["videoTimeMs", "frameNum"], keep="first")
          .reset_index(drop=True)
    )

    df.to_parquet(dst, engine="pyarrow", compression=None, index=False)
    return True, "written", rows_before - len(df)


def build_split(split, manifest_dir, source_dir, output_dir, modality, fps):
    """Build the spotting dataset for a single split.

    Reads the GAR clip-level manifest for the split, copies or writes the
    underlying media files, and emits annotations_{split}.json in the
    SoccerNet action spotting format.
    """
    manifest_path = os.path.join(manifest_dir, f"annotations_{split}.json")
    if not os.path.exists(manifest_path):
        print(f"  no manifest for {split} at {manifest_path}, skipping")
        return None

    manifest = load_manifest(manifest_path)
    by_game = group_clips_by_game(manifest)

    split_out = os.path.join(output_dir, split)
    os.makedirs(split_out, exist_ok=True)

    data_entries = []
    label_counts = defaultdict(int)
    skipped_missing = []
    counts = {"copied": 0, "written": 0, "already_present": 0}
    total_dups_removed = 0

    print(f"\n{'=' * 60}")
    print(
        f"processing {split}: {len(by_game)} games, "
        f"{sum(len(v) for v in by_game.values())} events"
    )
    print("=" * 60)

    for game_id, events in tqdm(by_game.items(), desc=f"{split}"):
        if modality == "video":
            src = os.path.join(source_dir, f"{game_id}.mp4")
            ext = ".mp4"
            input_type = "video"
        else:
            src = os.path.join(source_dir, split, "videos", f"{game_id}.parquet")
            ext = ".parquet"
            input_type = "tracking"

        if not os.path.exists(src):
            skipped_missing.append((game_id, src))
            continue

        dst = os.path.join(split_out, f"{game_id}{ext}")

        if modality == "video":
            _, status = copy_video_if_missing(src, dst)
            counts[status] += 1
        else:
            _, status, dups = write_sorted_tracking_if_missing(src, dst)
            counts[status] += 1
            if dups:
                total_dups_removed += dups
                tqdm.write(f"  {game_id}: removed {dups} duplicate tracking rows")

        for ev in events:
            label_counts[ev["label"]] += 1

        data_entries.append({
            "inputs": [{
                "type": input_type,
                "path": f"{split}/{game_id}{ext}",
                "fps": fps,
            }],
            "events": events,
        })

    annotation = {
        "version": "2.0",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "task": "action_spotting",
        "dataset_name": f"SoccerNet GAR Spotting ({split})",
        "metadata": {
            "source": "PFF FC -> SoccerNet GAR",
            "created_by": "Drishya Karki",
            "notes": f"modality={modality}",
            "split": split,
        },
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

    total = sum(label_counts.values())
    print(f"\n  {split} summary")
    print(f"    games: {len(data_entries)}")
    if modality == "video":
        print(
            f"    files copied: {counts['copied']} "
            f"(already present: {counts['already_present']})"
        )
    else:
        print(
            f"    files written: {counts['written']} "
            f"(already present: {counts['already_present']})"
        )
        print(f"    total duplicate rows removed: {total_dups_removed}")
    print(f"    events: {total}")
    for label in LABELS:
        c = label_counts[label]
        pct = (c / total * 100) if total else 0
        print(f"      {label}: {c} ({pct:.1f}%)")

    if skipped_missing:
        print(f"\n  skipped {len(skipped_missing)} games due to missing source files:")
        for game_id, src in skipped_missing[:5]:
            print(f"    {game_id}: {src}")
        if len(skipped_missing) > 5:
            print(f"    ... and {len(skipped_missing) - 5} more")

    return label_counts


def main():
    parser = argparse.ArgumentParser(
        description="Build SoccerNet-style action spotting dataset from GAR clip manifests."
    )
    parser.add_argument("--modality", choices=["video", "tracking"], required=True)
    parser.add_argument(
        "--manifest-dir", required=True,
        help="directory containing annotations_{split}.json from GAR stage 2",
    )
    parser.add_argument(
        "--source-dir", required=True,
        help=(
            "video: directory containing {game_id}.mp4. "
            "tracking: directory containing {split}/videos/{game_id}.parquet"
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--fps", type=float, default=29.97,
        help="fps to record in the inputs entry (default 29.97 for 720p mp4s)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    overall = defaultdict(int)
    for split in ("train", "valid", "test"):
        counts = build_split(
            split, args.manifest_dir, args.source_dir,
            args.output_dir, args.modality, args.fps,
        )
        if counts:
            for k, v in counts.items():
                overall[k] += v

    grand_total = sum(overall.values())
    print(f"\n{'=' * 60}")
    print("overall dataset stats")
    print("=" * 60)
    print(f"  total events: {grand_total}")
    for label in LABELS:
        c = overall[label]
        pct = (c / grand_total * 100) if grand_total else 0
        print(f"    {label}: {c} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
