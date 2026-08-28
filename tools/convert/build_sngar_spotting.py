"""Build the SN-GAR action-spotting datasets from raw event and tracking data.

Produces the tracking and video modalities in a single pass, so both carry the
same ground truth by construction rather than by a follow-up sync step.

Output layout:

    <out-root>/sngar-action-spotting-tracking/
        annotations_{train,valid,test}.json
        {train,valid,test}/videos/<game_id>.parquet   zstd, chunked row groups
        README.md
        MANIFEST.sha256
    <out-root>/sngar-action-spotting-video/
        annotations_{train,valid,test}.json           same events, same order
        {train,valid,test}/videos/<game_id>.mp4
        README.md
        MANIFEST.sha256

Every default is contractual: a bare invocation reproduces the published
dataset of 87,939 events, one label per instant, alignment tolerance 10 ms.
--no-event-dedup and --tolerance-ms opt out of that deliberately.

Label extraction, priority resolution and the alignment filter come from
sngar_events.py, which this script calls without modifying.

Behaviour worth knowing:

  * Parquets are zstd-compressed with bounded row groups, which takes the
    corpus from roughly 62 GB uncompressed to under 3 GB. Compression is
    transparent to pandas and pyarrow readers.
  * The video modality ships mp4 only. Aligning events needs the tracking
    clock, but that clock is held in memory rather than written beside the
    videos.
  * An internal URL column carried by the source tracking stream is dropped,
    since no loader reads it. Pass --keep-video-url to retain it.
  * Loader feature caches (*.npy) are never written into the dataset tree.
  * Each modality gets a sha256 manifest and a generated dataset card.

Usage instructions are at the bottom of this file.
"""

import os
import bz2
import json
import shutil
import hashlib
import argparse
import concurrent.futures
import multiprocessing as mp
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from sngar_dataset_card import write_card
from sngar_events import (
    LABELS,
    SPLITS,
    assign_splits,
    build_position_mapping,
    deduplicate_annotations,
    extract_expanded_annotations,
    filter_aligned_annotations,
    flatten_frame,
)

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    def tqdm(iterable=None, total=None, desc=None, **kwargs):
        return iterable if iterable is not None else range(total or 0)


TRACKING_REPO = "sngar-action-spotting-tracking"
VIDEO_REPO = "sngar-action-spotting-video"

INT_COLUMNS = ["frameNum", "period", "game_event_id", "possession_event_id"]
FLOAT_COLUMNS = ["videoTimeMs"]


# --------------------------------------------------------------------------
# stage 1: tracking jsonl.bz2 -> dataframe
# --------------------------------------------------------------------------

def tracking_frame_table(jsonl_path, dedupe_video_time_ms, keep_video_url):
    """Decode one game's tracking stream into the release dataframe."""
    position_map, home_team_id, away_team_id = build_position_mapping(jsonl_path)

    frames = []
    with bz2.open(jsonl_path, "rt") as f:
        for line in f:
            try:
                frame = json.loads(line)
            except json.JSONDecodeError:
                continue
            frames.append(flatten_frame(frame, position_map, home_team_id, away_team_id))

    df = pd.DataFrame(frames)
    original_rows = len(df)

    for col in INT_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype("int32")

    for col in FLOAT_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    for col in df.columns:
        if col not in INT_COLUMNS + FLOAT_COLUMNS:
            df[col] = df[col].fillna("").astype(str)

    removed = 0
    if dedupe_video_time_ms and "videoTimeMs" in df.columns:
        before = len(df)
        df = (
            df.sort_values(["videoTimeMs", "frameNum"], ascending=[True, True])
            .drop_duplicates(subset=["videoTimeMs"], keep="first")
            .reset_index(drop=True)
        )
        removed = before - len(df)

    if not keep_video_url and "video_url" in df.columns:
        df = df.drop(columns=["video_url"])

    return df, original_rows, removed


def write_parquet(df, path, compression, compression_level, row_group_size):
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(
        table,
        path,
        compression=compression,
        compression_level=compression_level,
        row_group_size=row_group_size,
        version="2.6",
    )


# --------------------------------------------------------------------------
# stage 2: one game, both modalities
# --------------------------------------------------------------------------

def process_game(task):
    game_id = task["game_id"]
    split = task["split"]

    df, original_rows, removed_dupe_rows = tracking_frame_table(
        task["jsonl_path"], task["dedupe_video_time_ms"], task["keep_video_url"]
    )

    if task["build_tracking"]:
        parquet_dir = os.path.join(task["out_root"], TRACKING_REPO, split, "videos")
        os.makedirs(parquet_dir, exist_ok=True)
        write_parquet(
            df,
            os.path.join(parquet_dir, f"{game_id}.parquet"),
            task["compression"],
            task["compression_level"],
            task["row_group_size"],
        )

    if task["build_video"]:
        video_dir = os.path.join(task["out_root"], VIDEO_REPO, split, "videos")
        os.makedirs(video_dir, exist_ok=True)
        link_media(task["video_path"], os.path.join(video_dir, f"{game_id}.mp4"), task["link_mode"])

    if os.path.exists(task["events_path"]):
        events = extract_expanded_annotations(task["events_path"])
    else:
        events = []
    n_extracted = len(events)
    labels_extracted = {}
    for e in events:
        labels_extracted[e["label"]] = labels_extracted.get(e["label"], 0) + 1

    if task["deduplicate_events"]:
        events = deduplicate_annotations(events)
    n_after_dedup = len(events)

    # Filter against the in-memory clock so the video modality needs no
    # scratch parquet on disk and both modalities use the same rows.
    clock = df[["videoTimeMs", "frameNum"]]
    if task["align"]:
        events, n_unaligned = filter_aligned_annotations_df(events, clock, task["tolerance_ms"])
    else:
        n_unaligned = 0

    return {
        "game_id": game_id,
        "split": split,
        "order": task["order"],
        "events": events,
        "n_extracted": n_extracted,
        "labels_extracted": labels_extracted,
        "n_deduped": n_extracted - n_after_dedup,
        "n_unaligned": n_unaligned,
        "original_rows": original_rows,
        "final_rows": len(df),
        "removed_dupe_rows": removed_dupe_rows,
    }


def filter_aligned_annotations_df(annotations, clock_df, tolerance_ms):
    """filter_aligned_annotations against an in-memory clock.

    Nearest-row semantics are identical to the parquet-backed version; only
    the source of the dataframe differs.
    """
    if not annotations:
        return annotations, 0

    tracking_df = clock_df.sort_values(
        ["videoTimeMs", "frameNum"], ascending=[True, True]
    ).reset_index(drop=True)

    kept = []
    skipped = 0
    for ann in annotations:
        time_diff = (tracking_df["videoTimeMs"] - ann["position_ms"]).abs()
        closest_idx = time_diff.idxmin()
        if time_diff.loc[closest_idx] > tolerance_ms:
            skipped += 1
            continue
        kept.append(ann)

    return kept, skipped


def link_media(src, dst, mode):
    if os.path.lexists(dst):
        os.remove(dst)
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    if mode in ("symlink", "hardlink"):
        try:
            os.symlink(os.path.abspath(src), dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


# --------------------------------------------------------------------------
# stage 3: annotations
# --------------------------------------------------------------------------

def annotation_document(split, results, args, modality):
    results = sorted(results, key=lambda r: r["order"])

    # The card renders the extracted -> resolved -> aligned chain from these,
    # so it always describes the build it ships with.
    labels_before_dedup = {}
    for r in results:
        for label, n in r.get("labels_extracted", {}).items():
            labels_before_dedup[label] = labels_before_dedup.get(label, 0) + n

    accounting = {
        "labels_before_dedup": labels_before_dedup,
        "extracted": sum(r["n_extracted"] for r in results),
        "removed_by_dedup": sum(r["n_deduped"] for r in results),
        "removed_by_alignment": sum(r["n_unaligned"] for r in results),
        "final": sum(len(r["events"]) for r in results),
    }

    if modality == "tracking":
        input_for = lambda gid: {
            "type": "tracking_parquet",
            "path": f"{split}/videos/{gid}.parquet",
            "fps": args.fps,
        }
    else:
        input_for = lambda gid: {
            "type": "video_mp4",
            "path": f"{split}/videos/{gid}.mp4",
            "fps": args.fps,
        }

    return {
        "version": "2.0",
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "task": "action_spotting",
        "dataset_name": f"{args.dataset_name}_{modality}_{split}",
        "metadata": {
            "source": args.source,
            "created_by": args.created_by,
            "split": split,
            "modality": modality,
            "aligned": args.align,
            "tolerance_ms": args.tolerance_ms,
            "deduplicated_events": args.deduplicate_events,
            "dedupe_video_time_ms": args.dedupe_video_time_ms,
            "events_identical_across_modalities": True,
            "event_accounting": accounting,
        },
        "labels": {"action": {"type": "single_label", "labels": LABELS}},
        "data": [
            {
                "game_id": r["game_id"],
                "split": split,
                "inputs": [input_for(r["game_id"])],
                "events": r["events"],
            }
            for r in results
        ],
    }


# --------------------------------------------------------------------------
# stage 4: manifest + card
# --------------------------------------------------------------------------

def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# huggingface_hub writes .cache/ into the folder it uploads to track progress.
# It is not dataset content and must stay out of the manifest.
MANIFEST_IGNORE_DIRS = {".cache", "__pycache__", ".ipynb_checkpoints"}


def write_manifest(repo_dir, num_workers):
    paths = []
    for root, dirs, files in os.walk(repo_dir):
        dirs[:] = [d for d in dirs if d not in MANIFEST_IGNORE_DIRS]
        for name in sorted(files):
            if name == "MANIFEST.sha256":
                continue
            paths.append(os.path.join(root, name))
    paths.sort()

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(num_workers, 16)) as pool:
        digests = list(tqdm(pool.map(sha256_file, paths), total=len(paths), desc="sha256"))

    lines = [
        f"{digest}  {os.path.relpath(path, repo_dir)}\n"
        for digest, path in zip(digests, paths)
    ]
    with open(os.path.join(repo_dir, "MANIFEST.sha256"), "w") as f:
        f.writelines(lines)

    total_bytes = sum(os.path.getsize(p) for p in paths)
    return len(paths), total_bytes




def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--events-dir", default="RawEventsData")
    parser.add_argument("--tracking-dir", default="PlayerPoseTracking")
    parser.add_argument("--video-dir", default="224p")
    parser.add_argument("--out-root", default="release")
    parser.add_argument("--modality", choices=["tracking", "video", "both"], default="both")
    parser.add_argument("--splits", nargs="+", default=list(SPLITS.keys()))
    parser.add_argument("--games", nargs="+", default=None,
                        help="restrict to these game ids (smoke tests only; the\n"
                             "annotation files it writes are partial)")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--tolerance-ms", type=float, default=10.0,
                        help="max gap between an event and the nearest tracking\n"
                             "row (default 10.0, the published contract)")
    parser.add_argument("--no-align", dest="align", action="store_false")
    parser.add_argument("--event-dedup", dest="deduplicate_events", action="store_true",
                        help="resolve each instant to one label (default)")
    parser.add_argument("--no-event-dedup", dest="deduplicate_events",
                        action="store_false",
                        help="keep every label, allowing several per instant")
    parser.add_argument("--no-dedupe-video-time-ms", dest="dedupe_video_time_ms",
                        action="store_false")
    parser.add_argument("--keep-video-url", action="store_true",
                        help="retain the source video_url column")
    parser.add_argument("--compression", default="zstd", choices=["zstd", "snappy", "gzip", "none"])
    parser.add_argument("--compression-level", type=int, default=9)
    parser.add_argument("--row-group-size", type=int, default=50000)
    parser.add_argument("--link-mode", choices=["hardlink", "symlink", "copy"], default="hardlink")
    parser.add_argument("--annotations-only", action="store_true",
                        help="re-derive annotations from the parquets already in\n"
                             "--out-root, then rewrite cards and manifests. The\n"
                             "payload is unaffected by --event-dedup and\n"
                             "--tolerance-ms, so changing either needs no rebuild.")
    parser.add_argument("--cards-only", action="store_true",
                        help="regenerate README.md and MANIFEST.sha256 from the\n"
                             "annotations already in --out-root; no rebuild")
    parser.add_argument("--num-workers", type=int, default=max(1, mp.cpu_count() // 2))
    parser.add_argument("--source", default="",
                        help="value recorded in each annotation file's metadata.source")
    parser.add_argument("--created-by", default="",
                        help="value recorded in metadata.created_by")
    parser.add_argument("--dataset-name", default="action_spotting")
    # These defaults are the published dataset; a bare invocation reproduces it.
    parser.set_defaults(align=True, deduplicate_events=True, dedupe_video_time_ms=True)

    args = parser.parse_args()
    if args.compression == "none":
        args.compression = None
        args.compression_level = None
    return args


def rebuild_annotations(args, build_tracking, build_video):
    """Re-derive the event lists against the parquets already on disk.

    Only the annotations depend on --event-dedup and --tolerance-ms, so this
    reads each parquet's clock column and leaves the payload untouched.
    """
    tracking_dir = os.path.join(args.out_root, TRACKING_REPO)
    if not os.path.isdir(tracking_dir):
        raise SystemExit(f"{tracking_dir} does not exist; run a full build first")

    split_map = assign_splits(args.tracking_dir, ".jsonl.bz2")
    by_split = {}
    for game_id, (split, order) in sorted(split_map.items(), key=lambda kv: kv[1][1]):
        if split not in args.splits:
            continue
        parquet = os.path.join(tracking_dir, split, "videos", f"{game_id}.parquet")
        if not os.path.isfile(parquet):
            raise SystemExit(f"missing {parquet}")

        events_path = os.path.join(args.events_dir, f"{game_id}.json")
        events = (
            extract_expanded_annotations(events_path)
            if os.path.exists(events_path) else []
        )
        n_extracted = len(events)
        labels_extracted = {}
        for e in events:
            labels_extracted[e["label"]] = labels_extracted.get(e["label"], 0) + 1
        if args.deduplicate_events:
            events = deduplicate_annotations(events)
        n_after_dedup = len(events)
        if args.align:
            events, n_unaligned = filter_aligned_annotations(
                events, parquet, args.tolerance_ms
            )
        else:
            n_unaligned = 0

        by_split.setdefault(split, []).append({
            "game_id": game_id,
            "order": order,
            "events": events,
            "n_extracted": n_extracted,
            "labels_extracted": labels_extracted,
            "n_deduped": n_extracted - n_after_dedup,
            "n_unaligned": n_unaligned,
        })

    repos = []
    if build_tracking:
        repos.append((TRACKING_REPO, "tracking"))
    if build_video:
        repos.append((VIDEO_REPO, "video"))

    for repo_name, modality in repos:
        repo_dir = os.path.join(args.out_root, repo_name)
        if not os.path.isdir(repo_dir):
            continue
        for split, results in by_split.items():
            doc = annotation_document(split, results, args, modality)
            with open(os.path.join(repo_dir, f"annotations_{split}.json"), "w") as f:
                json.dump(doc, f, indent=2)
        write_card(repo_dir, modality)
        n_files, total_bytes = write_manifest(repo_dir, args.num_workers)
        print(f"{repo_name}: annotations + card + manifest rewritten "
              f"({n_files} files, {total_bytes / 1e9:.2f} GB)")

    print()
    grand = 0
    for split in ("train", "valid", "test"):
        if split not in by_split:
            continue
        rs = by_split[split]
        final = sum(len(r["events"]) for r in rs)
        grand += final
        print(f"  {split:5s} extracted={sum(r['n_extracted'] for r in rs):6d} "
              f"dedup=-{sum(r['n_deduped'] for r in rs):5d} "
              f"unaligned=-{sum(r['n_unaligned'] for r in rs):5d} final={final:6d}")
    print(f"  TOTAL final={grand}")


def main():
    args = parse_args()

    build_tracking = args.modality in ("tracking", "both")
    build_video = args.modality in ("video", "both")

    if args.annotations_only:
        rebuild_annotations(args, build_tracking, build_video)
        return

    if args.cards_only:
        for repo_name, modality, wanted in (
            (TRACKING_REPO, "tracking", build_tracking),
            (VIDEO_REPO, "video", build_video),
        ):
            repo_dir = os.path.join(args.out_root, repo_name)
            if not wanted or not os.path.isdir(repo_dir):
                continue
            write_card(repo_dir, modality)
            n_files, total_bytes = write_manifest(repo_dir, args.num_workers)
            print(f"{repo_name}: card + manifest rewritten "
                  f"({n_files} files, {total_bytes / 1e9:.2f} GB)")
        return

    split_map = assign_splits(args.tracking_dir, ".jsonl.bz2")

    tasks = []
    for game_id, (split, order) in sorted(split_map.items(), key=lambda kv: kv[1][1]):
        if split not in args.splits:
            continue
        if args.games and game_id not in args.games:
            continue
        jsonl_path = os.path.join(args.tracking_dir, f"{game_id}.jsonl.bz2")
        video_path = os.path.join(args.video_dir, f"{game_id}.mp4")
        if build_video and not os.path.exists(video_path):
            raise SystemExit(f"{split}/{game_id}: missing mp4 at {video_path}")
        tasks.append({
            "game_id": game_id,
            "split": split,
            "order": order,
            "jsonl_path": jsonl_path,
            "events_path": os.path.join(args.events_dir, f"{game_id}.json"),
            "video_path": video_path,
            "out_root": args.out_root,
            "build_tracking": build_tracking,
            "build_video": build_video,
            "align": args.align,
            "tolerance_ms": args.tolerance_ms,
            "deduplicate_events": args.deduplicate_events,
            "dedupe_video_time_ms": args.dedupe_video_time_ms,
            "keep_video_url": args.keep_video_url,
            "compression": args.compression,
            "compression_level": args.compression_level,
            "row_group_size": args.row_group_size,
            "link_mode": args.link_mode,
        })

    print(f"{len(tasks)} games -> {args.out_root} "
          f"(tracking={build_tracking}, video={build_video}, workers={args.num_workers})")

    results = []
    if args.num_workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            futures = [pool.submit(process_game, t) for t in tasks]
            for fut in tqdm(concurrent.futures.as_completed(futures),
                            total=len(futures), desc="games"):
                results.append(fut.result())
    else:
        results = [process_game(t) for t in tqdm(tasks, desc="games")]

    by_split = {}
    for r in results:
        by_split.setdefault(r["split"], []).append(r)

    stats_by_split = {}
    for split, split_results in by_split.items():
        label_counts = {}
        for r in split_results:
            for e in r["events"]:
                label_counts[e["label"]] = label_counts.get(e["label"], 0) + 1
        stats_by_split[split] = {
            "n_games": len(split_results),
            "label_counts": label_counts,
            "total_events": sum(label_counts.values()),
            "n_extracted": sum(r["n_extracted"] for r in split_results),
            "n_deduped": sum(r["n_deduped"] for r in split_results),
            "n_unaligned": sum(r["n_unaligned"] for r in split_results),
            "original_rows": sum(r["original_rows"] for r in split_results),
            "final_rows": sum(r["final_rows"] for r in split_results),
            "removed_dupe_rows": sum(r["removed_dupe_rows"] for r in split_results),
        }

    repos = []
    if build_tracking:
        repos.append((TRACKING_REPO, "tracking"))
    if build_video:
        repos.append((VIDEO_REPO, "video"))

    for repo_name, modality in repos:
        repo_dir = os.path.join(args.out_root, repo_name)
        os.makedirs(repo_dir, exist_ok=True)
        for split, split_results in by_split.items():
            doc = annotation_document(split, split_results, args, modality)
            with open(os.path.join(repo_dir, f"annotations_{split}.json"), "w") as f:
                json.dump(doc, f, indent=2)
        write_card(repo_dir, modality)
        n_files, total_bytes = write_manifest(repo_dir, args.num_workers)
        print(f"{repo_name}: {n_files} files, {total_bytes / 1e9:.2f} GB")

    print("\nevent accounting")
    for split in ("train", "valid", "test"):
        if split not in stats_by_split:
            continue
        s = stats_by_split[split]
        print(f"  {split:5s} games={s['n_games']:2d} extracted={s['n_extracted']:6d} "
              f"dedup=-{s['n_deduped']:5d} unaligned=-{s['n_unaligned']:5d} "
              f"final={s['total_events']:6d}")
    print(f"  TOTAL final={sum(s['total_events'] for s in stats_by_split.values())}")

    print("\nframe rows")
    for split in ("train", "valid", "test"):
        if split not in stats_by_split:
            continue
        s = stats_by_split[split]
        print(f"  {split:5s} raw={s['original_rows']:9d} "
              f"dupes=-{s['removed_dupe_rows']:6d} kept={s['final_rows']:9d}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
#
# Build both modalities. Every default is contractual, so this reproduces the
# published dataset exactly:
#
#     python build_sngar_spotting.py \
#         --events-dir   <data-root>/RawEventsData \
#         --tracking-dir <data-root>/PlayerPoseTracking \
#         --video-dir    <data-root>/224p \
#         --out-root     release \
#         --num-workers  24
#
# Roughly 11 minutes on 24 workers.
#
# Re-derive the annotations against parquets already on disk, without
# rebuilding the payload. --event-dedup and --tolerance-ms affect only the
# annotation JSONs, so changing either needs no full rebuild:
#
#     python build_sngar_spotting.py --out-root release --annotations-only \
#         --events-dir <data-root>/RawEventsData \
#         --tracking-dir <data-root>/PlayerPoseTracking
#
# Re-render the dataset cards and checksums only:
#
#     python build_sngar_spotting.py --out-root release --cards-only
#
# Build one modality, or a subset of games for a smoke test:
#
#     python build_sngar_spotting.py --modality tracking ...
#     python build_sngar_spotting.py --games 3841 3850 ...
#
# Then verify and publish:
#
#     python verify_sngar_spotting.py --out-root release
#     python push_sngar_spotting.py --out-root release          # dry run
#     python push_sngar_spotting.py --out-root release --yes
