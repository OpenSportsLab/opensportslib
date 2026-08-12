"""Run rule-based H5 header spotting and save OSL JSON predictions.

Each game directory must hold `live_joints.h5` and `live_ball.h5`. The tracking
file is scanned end to end and `position_ms` is measured from its first sample,
with every event also carrying an absolute `timestamp_utc`.

There are three ways in:

    --config      run a single YAML config exactly as written
    --combined    one manifest over every game, one pass, the spotter's own
                  output saved as-is
    (default)     sweep games one at a time, cache each, and reassemble them
                  into a combined file; `--halves` then selects which periods
                  to report and `--annotations` scores the result

The sweep runs in three stages, each reusable on its own:

    detect     per game and variant -> raw predictions on the UTC clock, cached
    assemble   raw predictions -> one OSL JSON covering every game
    evaluate   optional, with --annotations: mAP against video annotations

Re-running skips games whose raw predictions exist, so an interrupted sweep
resumes where it stopped.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# Prefer the checkout this script lives in over any installed copy.
sys.path.insert(0, str(REPO_ROOT))

VARIANTS = [
    "skeleton",
    "skeleton_recall",
    "distance",
    "distance_speed",
    "distance_angle",
    "distance_speed_angle",
]
FPS = 50
DELTAS_S = [1, 2, 3, 4, 5]
PERIOD_END_BUFFER_S = 60

CONFIG_TEMPLATE = """\
TASK: localization
VERSION: 2

SYSTEM:
  paths:
    work_dir: {work_dir}
  device: cpu
  gpu:
    count: 0

DATA:
  common:
    dataset_name: headers_{game_id}
    data_root: {game_dir}
    classes:
      - header
    splits:
      test:
        type: H5OSLJsonSpotting
        annotation_path: {manifest}
        source_path: {game_dir}
        dataloader:
          batch_size: 1
          shuffle: false
          num_workers: 0
          pin_memory: false
  inputs:
    tracking:
      modality: player_joints_h5
      representation: raw
      source:
        format: h5
      sampling: {{}}
      transform: {{}}
      augmentations: {{}}
      params: {{}}

MODEL:
  metadata:
    family: RuleBased
    runner:
      type: runner_h5_header_rule
  components:
    rule:
      kind: algorithm
      source:
        provider: opensportslib
        registry: rule_based
        name: h5_header_{variant}
      params:
        label: header
        head_name: action
  topology: []

TRAIN:
  trainer:
    type: trainer_rule_based
  execution:
    enabled: false
"""


def parse_utc_string(value):
    """Parse a timestamp from the tracking or annotation files.

    Args:
        value (str): Timestamp such as "2022-12-03 15:00:03.356000".

    Returns:
        moment (datetime): Parsed instant, or None when the field is absent.
    """
    if value in (None, "", "None"):
        return None
    return datetime.fromisoformat(value)


def read_game_metadata(ball_path):
    """Describe a game from its ball track.

    Args:
        ball_path (Path): Path to the game's `live_ball.h5`.

    Returns:
        metadata (dict): `game_id`, `home`, `away`, the `track_start` and
            `track_end` of the whole file, and `periods` as
            (tag, kickoff, end) for each played period.

    Raises:
        ValueError: If the ball track has no timestamps.
    """
    import h5py
    import numpy as np

    def text(dataset, index=0):
        value = dataset[index]
        return value.decode("utf-8") if isinstance(value, bytes) else str(value)

    with h5py.File(ball_path, "r") as f:
        stamps = f["timestamp_utc"][:]
        if stamps.size == 0:
            raise ValueError(f"No timestamps in {ball_path}")
        # Fixed-width ISO strings sort chronologically.
        track_start = parse_utc_string(min(stamps).decode("utf-8"))
        track_end = parse_utc_string(max(stamps).decode("utf-8"))

        half = f["half"][:]
        starts = f["half_start_utc"] if "half_start_utc" in f else None
        periods = []
        for period in sorted(int(v) for v in np.unique(half) if int(v) >= 1):
            rows = np.flatnonzero(half == period)
            if rows.size == 0:
                continue
            period_stamps = stamps[rows]
            kickoff = None
            if starts is not None:
                kickoff = parse_utc_string(text(starts, int(rows.min())))
            if kickoff is None:
                kickoff = parse_utc_string(min(period_stamps).decode("utf-8"))
            end = parse_utc_string(max(period_stamps).decode("utf-8"))
            periods.append((period, kickoff, end))

        return {
            "game_id": text(f["game_id"]) if "game_id" in f else ball_path.parent.name,
            "home": text(f["home_name"]) if "home_name" in f else None,
            "away": text(f["away_name"]) if "away_name" in f else None,
            "track_start": track_start,
            "track_end": track_end,
            "periods": periods,
        }


def read_half_by_timestamp(ball_path):
    """Map each ball sample's instant to the period it is tagged with.

    Period 0 marks samples outside active play; 1 and up are the played
    periods. A sample's tag is how an event is attributed to a period, since
    every event is spotted at a ball sample.

    Args:
        ball_path (Path): Path to the game's `live_ball.h5`.

    Returns:
        halves (dict): UTC instant to period tag.
    """
    import h5py

    with h5py.File(ball_path, "r") as f:
        halves = f["half"][:]
        stamps = f["timestamp_utc"][:]
    return {parse_utc_string(stamp.decode("utf-8")): int(half)
            for stamp, half in zip(stamps, halves)}


def find_games(data_root, wanted="all"):
    """List game directories holding both required tracking files.

    Args:
        data_root (Path): Directory containing one sub-directory per game.
        wanted (str): "all", or a comma-separated list of game directory names.
            Default: "all".

    Returns:
        games (List[Path]): Game directories, sorted by name.
    """
    games = sorted(
        d for d in data_root.iterdir()
        if d.is_dir()
        and (d / "live_joints.h5").exists()
        and (d / "live_ball.h5").exists()
    )
    if wanted != "all":
        names = {name.strip() for name in wanted.split(",")}
        games = [g for g in games if g.name in names]
    return games


# ---------------------------------------------------------------- detect
def write_manifest(path, entries):
    """Write the OSL JSON manifest the spotter reads its inputs from.

    Args:
        path (Path): Where to write the manifest.
        entries (List[dict]): One entry per game, each with an `id`, the H5
            paths under `inputs`, and optionally a `metadata` scan window.

    Returns:
        path (Path): The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "version": "2.0",
        "task": "action_spotting",
        "labels": {"action": {"type": "single_label", "labels": ["header"]}},
        "data": entries,
    }, indent=2))
    return path


def manifest_entry(game_id, joints_path, ball_path, window=None):
    """Describe one game for the manifest.

    Args:
        game_id (str): Identifier carried through to the predictions.
        joints_path (str): Player joints H5, relative to the config's
            `source_path`.
        ball_path (str): Ball H5, relative to the same.
        window (tuple): (start, end) UTC datetimes to restrict the scan to.
            Scans the whole file when None. Default: None.

    Returns:
        entry (dict): A manifest `data` entry.
    """
    entry = {
        "id": game_id,
        "inputs": [{
            "type": "player_joints_h5",
            "path": joints_path,
            "ball_path": ball_path,
        }],
    }
    if window is not None:
        entry["metadata"] = {
            "start_utc": window[0].strftime("%Y-%m-%d %H:%M:%S.%f"),
            "end_utc": window[1].strftime("%Y-%m-%d %H:%M:%S.%f"),
        }
    return entry


def write_config(path, work_dir, source_root, manifest_path, variant, name):
    """Write the localization config that points the spotter at a manifest.

    Args:
        path (Path): Where to write the config.
        work_dir (Path): Working directory recorded in the config.
        source_root (Path): Directory the manifest's relative paths resolve
            against.
        manifest_path (Path): Manifest the config should read.
        variant (str): Rule variant, without the `h5_header_` prefix.
        name (str): Suffix for the config's dataset name.

    Returns:
        path (Path): The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(CONFIG_TEMPLATE.format(
        work_dir=work_dir, game_dir=source_root, manifest=manifest_path,
        variant=variant, game_id=name))
    return path


def detect(games, variants, work_dir, force=False):
    """Spot headers in each game, caching raw predictions per game and variant.

    Args:
        games (List[Path]): Game directories to process.
        variants (List[str]): Rule variants to run.
        work_dir (Path): Directory for manifests, configs and raw predictions.
        force (bool): Re-run combinations whose raw predictions already exist.
            Default: False.
    """
    os.environ.setdefault("RUN_ID", "header-spotting")
    from opensportslib.apis import LocalizationModel

    total = len(games) * len(variants)
    step = 0
    for game_dir in games:
        game_id = game_dir.name
        meta = None
        for variant in variants:
            step += 1
            raw_path = work_dir / "raw" / variant / f"{game_id}.json"
            if raw_path.exists() and not force:
                print(f"[{step}/{total}] {game_id} {variant}: cached", flush=True)
                continue
            raw_path.parent.mkdir(parents=True, exist_ok=True)

            started = time.time()
            try:
                if meta is None:
                    meta = read_game_metadata(game_dir / "live_ball.h5")

                manifest_path = write_manifest(
                    work_dir / f"manifest_{game_id}.json",
                    [manifest_entry(game_id, "live_joints.h5", "live_ball.h5",
                                    (meta["track_start"], meta["track_end"]))])
                config_path = write_config(
                    work_dir / f"config_{game_id}_{variant}.yaml",
                    work_dir, game_dir, manifest_path, variant, game_id)

                api = LocalizationModel(config=str(config_path))
                events = api.infer(use_wandb=False)["data"][0]["events"]

                raw_path.write_text(json.dumps({
                    "game_id": meta["game_id"],
                    "home": meta["home"],
                    "away": meta["away"],
                    "variant": variant,
                    "track_start": meta["track_start"].strftime("%Y-%m-%d %H:%M:%S.%f"),
                    "track_end": meta["track_end"].strftime("%Y-%m-%d %H:%M:%S.%f"),
                    "kickoffs": [[period, kickoff.strftime("%Y-%m-%d %H:%M:%S.%f")]
                                 for period, kickoff, _ in meta["periods"]],
                    "events": events,
                }, indent=2))
                print(f"[{step}/{total}] {game_id} {variant}: {len(events)} headers "
                      f"in {time.time() - started:.0f}s "
                      f"({meta['home']} v {meta['away']})", flush=True)
            except Exception as exc:  # keep the sweep going; report and move on
                print(f"[{step}/{total}] {game_id} {variant}: FAILED - "
                      f"{type(exc).__name__}: {exc}", flush=True)


# ---------------------------------------------------------------- assemble
def assemble(games, variant, work_dir, output_path, dataset_name, halves=None):
    """Collect every game's raw predictions into one OSL JSON file.

    Args:
        games (List[Path]): Game directories that were processed.
        variant (str): Which variant's predictions to collect.
        work_dir (Path): Directory holding the cached raw predictions.
        output_path (Path): Where to write the combined file.
        dataset_name (str): Name recorded in the file header.
        halves (set): Period tags to keep, or None for every period.
            Default: None.

    Returns:
        payload (dict): The written OSL JSON payload.
    """
    data, missing, total, dropped = [], [], 0, 0
    for game_dir in games:
        raw_path = work_dir / "raw" / variant / f"{game_dir.name}.json"
        if not raw_path.exists():
            missing.append(game_dir.name)
            continue
        raw = json.loads(raw_path.read_text())
        track_start = parse_utc_string(raw["track_start"])

        kept = raw["events"]
        if halves is not None:
            half_by_instant = read_half_by_timestamp(game_dir / "live_ball.h5")
            kept = [e for e in kept
                    if half_by_instant.get(parse_utc_string(e["timestamp_utc"]))
                    in halves]
            dropped += len(raw["events"]) - len(kept)

        events = [{
            "head": "Actions",
            "label": "Header",
            "position_ms": int(round(
                (parse_utc_string(e["timestamp_utc"]) - track_start).total_seconds()
                * 1000)),
            "timestamp_utc": e["timestamp_utc"],
            "confidence": round(float(e["confidence_score"]), 4),
            "metadata": {"note": ""},
        } for e in kept]
        events.sort(key=lambda e: e["position_ms"])
        total += len(events)

        data.append({
            "id": raw["game_id"],
            "inputs": [{
                "type": "player_joints_h5",
                "path": f"{game_dir.name}/live_joints.h5",
                "ball_path": f"{game_dir.name}/live_ball.h5",
            }],
            "metadata": {
                "home": raw["home"],
                "away": raw["away"],
                "track_start_utc": raw["track_start"],
                "track_end_utc": raw["track_end"],
                "kickoffs_utc": {str(period): kickoff
                                 for period, kickoff in raw["kickoffs"]},
            },
            "events": events,
        })

    payload = {
        "version": "2.0",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "task": "action_spotting",
        "dataset_name": dataset_name,
        "description": (
            f"Header predictions from skeletal tracking, produced by the "
            f"h5_header_{variant} rule-based spotter"
            + (f", restricted to period {sorted(halves)}" if halves is not None else "")
            + ". position_ms is measured from the first sample of each game's "
            "tracking file; timestamp_utc gives the absolute instant."
        ),
        "modalities": ["player_joints_h5"],
        "metadata": {
            "type": "predictions",
            "created_by": f"h5_header_{variant}_rule",
            "num_games": len(data),
            "num_events": total,
            "periods": "all" if halves is None else sorted(halves),
        },
        "labels": {"Actions": {"type": "single_label", "labels": ["Header"]}},
        "data": data,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"[assemble] {variant}: {len(data)} games, {total} headers"
          + (f" ({dropped} dropped outside period {sorted(halves)})"
             if halves is not None else "")
          + f" -> {output_path}", flush=True)
    if missing:
        print(f"[assemble] {variant}: no predictions for {len(missing)} games: "
              f"{', '.join(missing[:10])}{' ...' if len(missing) > 10 else ''}")
    return payload


# ---------------------------------------------------------------- evaluate
def load_game_events(work_dir, variant, game_id, predictions_path=None):
    """Find one game's spotted events, from either output layout.

    Args:
        work_dir (Path): Directory holding the per-game cache.
        variant (str): Rule variant whose predictions to load.
        game_id (str): Game to look for.
        predictions_path (Path): A combined predictions file to read instead of
            the cache. Default: None.

    Returns:
        events (List[dict]): The game's events, or None when absent.
    """
    if predictions_path and Path(predictions_path).exists():
        payload = json.loads(Path(predictions_path).read_text())
        for entry in payload.get("data", []):
            if str(entry.get("id")) == str(game_id):
                return entry.get("events", [])
        return None

    raw_path = work_dir / "raw" / variant / f"{game_id}.json"
    if raw_path.exists():
        return json.loads(raw_path.read_text()).get("events", [])
    return None



class GroundTruth:
    """Video annotations plus the clock bridge to the tracking data.

    Headers are annotated as positions inside a video file while tracking rows
    are stamped in UTC, so scoring needs the instant the video starts. Play
    windows keep both sides to the periods the annotations cover.
    """

    def __init__(self, annotation_path, data_root):
        """Read annotations and derive the play windows.

        Args:
            annotation_path (str): OSL JSON with Header and Kickoff events.
            data_root (Path): Directory the H5 input paths resolve against.

        Raises:
            ValueError: If no item carries tracking inputs.
        """
        payload = json.loads(Path(annotation_path).read_text())
        game = next(
            (item for item in payload["data"]
             if any(i.get("type") == "player_joints_h5"
                    for i in item.get("inputs", []))),
            None,
        )
        if game is None:
            raise ValueError(
                f"No item with a player_joints_h5 input in {annotation_path}")

        video = next(i for i in game["inputs"] if i["type"] == "video")
        joints = next(i for i in game["inputs"] if i["type"] == "player_joints_h5")
        self.game_id = Path(joints["path"]).parent.name or game["id"]
        self.video_start = parse_utc_string(video["UTC_time_start"])
        self.header_ms = sorted(e["position_ms"] for e in game["events"]
                                if e["label"] == "Header")

        # A window runs from an annotated kickoff to the last tracked sample of
        # that period, so half-time and the gaps between periods stay out.
        kickoffs = sorted((e for e in game["events"] if e["label"] == "Kickoff"),
                          key=lambda e: e["position_ms"])
        meta = read_game_metadata(Path(data_root) / joints["ball_path"])
        ends = [end for _, _, end in meta["periods"]]
        self.windows = [
            (parse_utc_string(kickoff["timestamp_utc"]),
             ends[index] + timedelta(seconds=PERIOD_END_BUFFER_S))
            for index, kickoff in enumerate(kickoffs)
            if index < len(ends)
        ]

    def in_play_ms(self, position_ms):
        """Whether a video position falls inside a play window.

        Args:
            position_ms (int): Position in the video file, milliseconds.

        Returns:
            in_play (bool): True when inside any window.
        """
        moment = self.video_start + timedelta(milliseconds=position_ms)
        return any(start <= moment <= end for start, end in self.windows)

    def to_position_ms(self, moment):
        """Convert a UTC instant to a position in the video file.

        Args:
            moment (datetime): UTC instant, as carried by a prediction.

        Returns:
            position_ms (int): Position in the video file, milliseconds.
        """
        return int(round((moment - self.video_start).total_seconds() * 1000))


def evaluate(gt, variants, work_dir, predictions_paths=None):
    """Score predictions for one game against video annotations.

    Args:
        gt (GroundTruth): Annotations and the clock bridge.
        variants (List[str]): Rule variants to score.
        work_dir (Path): Directory holding the cached raw predictions.
        predictions_paths (dict): Variant to a combined predictions file, for
            scoring a `--combined` run rather than the per-game cache.
            Default: None.

    Returns:
        results (dict): Per-variant prediction count, mAP per tolerance, tight
            average mAP, and recall and precision at one second.
    """
    import numpy as np
    from opensportslib.metrics.localization_metric import (
        delta_curve,
        get_closest_action_index,
        label2vector,
        predictions2vector,
    )

    gt_ms = [ms for ms in gt.header_ms if gt.in_play_ms(ms)]
    print(f"[eval] ground truth headers: {len(gt.header_ms)} total | "
          f"in play: {len(gt_ms)} | "
          f"excluded: {len(gt.header_ms) - len(gt_ms)}", flush=True)

    event_dict = {"Header": 0}
    vector_size = int(FPS * (max(gt_ms) / 1000.0)) + FPS * 60
    # label2vector reads "gameTime" before falling back to "position".
    dense_labels = label2vector(
        [{"label": "Header", "position": ms, "gameTime": None} for ms in gt_ms],
        num_classes=1, framerate=FPS, EVENT_DICTIONARY=event_dict,
        vector_size=vector_size)
    closest = get_closest_action_index(dense_labels, np.zeros(dense_labels.shape) - 1)

    results = {}
    for variant in variants:
        events = load_game_events(work_dir, variant, gt.game_id,
                                  (predictions_paths or {}).get(variant))
        if events is None:
            print(f"[eval] {variant}: no predictions for game {gt.game_id}")
            continue

        positions, confidences = [], {}
        for event in events:
            moment = parse_utc_string(event["timestamp_utc"])
            position = gt.to_position_ms(moment)
            if not gt.in_play_ms(position):
                continue
            positions.append(position)
            confidences[position] = float(event["confidence_score"])
        positions.sort()

        dense_predictions = predictions2vector(
            [{"label": "Header", "position": p, "confidence": confidences[p]}
             for p in positions],
            num_classes=1, framerate=FPS, EVENT_DICTIONARY=event_dict,
            vector_size=vector_size)
        # Signature order: (targets, closests, detections, framerate, deltas).
        mAP_per_delta = delta_curve([dense_labels], [closest], [dense_predictions],
                                    FPS, np.array(DELTAS_S))[0]

        matched = sum(1 for ms in gt_ms
                      if any(abs(p - ms) <= 1000 for p in positions))
        results[variant] = {
            "num_predictions": len(positions),
            "mAP_per_delta": {f"{d}s": float(m)
                              for d, m in zip(DELTAS_S, mAP_per_delta)},
            "tight_avg_mAP": float(np.mean(mAP_per_delta)),
            "recall_at_1s": matched / len(gt_ms) if gt_ms else 0.0,
            "precision_at_1s": matched / len(positions) if positions else 0.0,
        }

    _print_table(results, len(gt_ms), len(gt.header_ms))
    (work_dir / "map_results.json").write_text(json.dumps(results, indent=2))
    print(f"[eval] results saved -> {work_dir / 'map_results.json'}", flush=True)
    return results


def _print_table(results, gt_in_play, gt_total):
    """Print the per-variant mAP table.

    Args:
        results (dict): Output of `evaluate`.
        gt_in_play (int): Ground truth headers inside the play windows.
        gt_total (int): Ground truth headers in the annotation file.
    """
    header = (f"{'variant':<24}{'#pred':>7}"
              + "".join(f"{f'mAP@{d}s':>10}" for d in DELTAS_S)
              + f"{'tight avg':>12}{'rec@1s':>9}{'prec@1s':>9}")
    print("\n" + header)
    print("-" * len(header))
    for variant, row in results.items():
        print(f"{variant:<24}{row['num_predictions']:>7}"
              + "".join(f"{row['mAP_per_delta'][f'{d}s'] * 100:>9.2f}%"
                        for d in DELTAS_S)
              + f"{row['tight_avg_mAP'] * 100:>11.2f}%"
              + f"{row['recall_at_1s'] * 100:>8.1f}%"
              + f"{row['precision_at_1s'] * 100:>8.1f}%")
    print(f"\n({gt_in_play} in-play headers of {gt_total} annotated; "
          f"{FPS} fps bins)")


def parse_args():
    """Parse command line arguments.

    Returns:
        args (argparse.Namespace): Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Rule-based header spotting on H5 tracking data.")
    parser.add_argument("--config", default=None,
                        help="Run this YAML config as written and save its "
                             "predictions, instead of sweeping a directory.")
    parser.add_argument("--data-root", default="/home/giancos/FIFA_data",
                        help="Directory holding one sub-directory per game.")
    parser.add_argument("--games", default="all",
                        help="Comma-separated game directory names, or 'all'.")
    parser.add_argument("--variants", default="skeleton_recall",
                        help=f"Comma-separated subset of: {', '.join(VARIANTS)}.")
    parser.add_argument("--annotations", default=None,
                        help="OSL JSON with Header and Kickoff events. When "
                             "given, predictions are scored against it.")
    parser.add_argument("--output-dir",
                        default=str(REPO_ROOT / "outputs" / "header_spotting"),
                        help="Directory for manifests, configs and raw predictions.")
    parser.add_argument("--output", default=None,
                        help="Path of the combined OSL JSON. Defaults to "
                             "headers-<variant>.json beside the output directory.")
    parser.add_argument("--dataset-name", default="Header predictions",
                        help="Dataset name recorded in the combined file.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run detection even when raw predictions exist.")
    parser.add_argument("--halves", default="all",
                        help="Comma-separated period tags to report, or 'all'. "
                             "Period 0 is the samples outside active play; 1 "
                             "and up are the played periods.")
    parser.add_argument("--combined", action="store_true",
                        help="Spot every game in one pass and save the "
                             "spotter's own output, instead of running games "
                             "separately and reassembling them.")
    parser.add_argument("--assemble-only", action="store_true",
                        help="Skip detection and only rebuild the outputs.")
    return parser.parse_args()


def run_combined(games, variant, data_root, work_dir, output_path):
    """Spot every game in a single pass and save one predictions file.

    Writes one manifest listing all the games and runs it through a single
    `infer()` call, so the output is what the spotter itself produced rather
    than something reassembled afterwards. Events keep their diagnostics and
    `position_ms` stays relative to each game's own tracking file.

    Args:
        games (List[Path]): Game directories to include.
        variant (str): Rule variant to run.
        data_root (Path): Directory the manifest paths resolve against.
        work_dir (Path): Directory for the manifest and config.
        output_path (Path): Where to save the OSL JSON predictions.

    Returns:
        saved (Path): The path written.
    """
    manifest_path = write_manifest(
        work_dir / "manifest_all.json",
        [manifest_entry(game.name,
                        f"{game.name}/live_joints.h5",
                        f"{game.name}/live_ball.h5")
         for game in games])
    config_path = write_config(work_dir / f"config_all_{variant}.yaml",
                               work_dir, data_root, manifest_path, variant, "all")

    print(f"[combined] {len(games)} games, variant h5_header_{variant} "
          f"-> {manifest_path}", flush=True)
    saved = run_config(config_path, output_path)
    payload = json.loads(Path(saved).read_text())
    print(f"[combined] {sum(len(g['events']) for g in payload['data'])} headers "
          f"across {len(payload['data'])} games", flush=True)
    return saved


def run_config(config_path, output_path):
    """Run one YAML config as written and save its predictions.

    Args:
        config_path (str): Path to a localization config naming a rule variant
            and the H5 inputs to run it over.
        output_path (Path): Where to save the OSL JSON predictions.

    Returns:
        saved (Path): The path written.
    """
    os.environ.setdefault("RUN_ID", "h5-header-rule")
    from opensportslib.apis import LocalizationModel

    api = LocalizationModel(config=str(config_path))
    predictions = api.infer(use_wandb=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    saved = Path(api.save_predictions(str(output_path), predictions))
    print(f"[config] {len(predictions['data'][0]['events'])} headers -> {saved}",
          flush=True)
    return saved


def main():
    """Detect headers, write the combined file, and score if asked."""
    args = parse_args()
    data_root = Path(args.data_root)
    work_dir = Path(args.output_dir)

    if args.config:
        run_config(args.config,
                   Path(args.output) if args.output
                   else work_dir / "h5_header_predictions.json")
        return

    variants = [name.strip() for name in args.variants.split(",")]
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown:
        raise SystemExit(f"Unknown variants: {unknown}. Expected any of {VARIANTS}.")

    halves = (None if args.halves == "all"
              else {int(h.strip()) for h in args.halves.split(",")})

    games = find_games(data_root, args.games)
    if not games:
        raise SystemExit(f"No games with tracking files found under {data_root}")

    if args.combined:
        written = {}
        for variant in variants:
            output = (Path(args.output) if args.output and len(variants) == 1
                      else work_dir / f"h5_header_predictions_{variant}.json")
            written[variant] = run_combined(games, variant, data_root,
                                            work_dir, output)
        if args.annotations:
            evaluate(GroundTruth(args.annotations, data_root), variants,
                     work_dir, predictions_paths=written)
        return

    print(f"[batch] {len(games)} games x {len(variants)} variants", flush=True)
    if not args.assemble_only:
        detect(games, variants, work_dir, force=args.force)

    for variant in variants:
        output = (Path(args.output) if args.output and len(variants) == 1
                  else work_dir / f"headers-{variant}.json")
        assemble(games, variant, work_dir, output, args.dataset_name, halves)

    if args.annotations:
        evaluate(GroundTruth(args.annotations, data_root), variants, work_dir)


if __name__ == "__main__":
    main()
