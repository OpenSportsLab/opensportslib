"""Spot headers in H5 tracking data and score them against video annotations.

Ground truth, the kickoffs and the video UTC start all come from one OSL JSON
file, so no separate synchronisation file is needed. Predictions are spotted on
the tracking clock, converted onto the video clock, and scored with the
library's action-spotting mAP at several tolerances.

The pipeline runs in three stages, each reusable on its own:

    extract   H5 tracking -> raw predictions on the UTC clock, per variant
    convert   raw predictions -> OSL JSON on the video clock, in-play only
    evaluate  predictions vs ground truth -> mAP table and map_results.json
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
    "distance",
    "distance_speed",
    "distance_angle",
    "distance_speed_angle",
]
FPS = 50
DELTAS_S = [1, 2, 3, 4, 5]
HALF_END_BUFFER_S = 60

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
    dataset_name: {dataset_name}
    data_root: {data_root}
    classes:
      - header
    splits:
      test:
        type: H5OSLJsonSpotting
        annotation_path: {manifest}
        source_path: {data_root}
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
    """Parse an ISO timestamp from the annotation file.

    Args:
        value (str): Timestamp such as "2022-12-18 15:00:04.857000".

    Returns:
        moment (datetime): Parsed instant.
    """
    return datetime.fromisoformat(value)


class GroundTruth:
    """Annotations plus the clock bridge between video and tracking.

    Header annotations are positions inside the video file while tracking rows
    are stamped in UTC, so scoring needs the instant the video starts. Play
    windows keep both sides to the periods the annotations cover.
    """

    def __init__(self, annotation_path, data_root):
        """Read annotations and derive the play windows.

        Args:
            annotation_path (str): OSL JSON with header and kickoff events.
            data_root (str): Directory the H5 input paths resolve against.

        Raises:
            ValueError: If no item carries tracking inputs, or the item does
                not hold exactly two kickoffs.
        """
        payload = json.loads(Path(annotation_path).read_text())
        game = next(
            (item for item in payload["data"]
             if any(i.get("type") == "player_joints_h5" for i in item.get("inputs", []))),
            None,
        )
        if game is None:
            raise ValueError(
                f"No item with a player_joints_h5 input in {annotation_path}")

        video = next(i for i in game["inputs"] if i["type"] == "video")
        joints = next(i for i in game["inputs"] if i["type"] == "player_joints_h5")

        self.sample_id = game["id"]
        self.video_path = video["path"]
        self.joints_path = joints["path"]
        self.ball_path = joints["ball_path"]
        self.video_start = parse_utc_string(video["UTC_time_start"])
        self.header_ms = sorted(
            event["position_ms"] for event in game["events"]
            if event["label"] == "Header"
        )

        kickoffs = sorted(
            (event for event in game["events"] if event["label"] == "Kickoff"),
            key=lambda event: event["position_ms"],
        )
        if len(kickoffs) != 2:
            raise ValueError(
                f"Expected 2 kickoffs in {annotation_path}, found {len(kickoffs)}")
        self.windows = self._play_windows(kickoffs, Path(data_root) / self.ball_path)

    @staticmethod
    def _play_windows(kickoffs, ball_path):
        """Bound each half by its kickoff and its last tracked ball sample.

        Args:
            kickoffs (List[dict]): The two kickoff events, in order.
            ball_path (Path): Path to the ball H5 file, whose `half` column
                marks which period each sample belongs to.

        Returns:
            windows (List[tuple]): One (start, end) UTC pair per half.

        Raises:
            ValueError: If the ball file has no samples for a half.
        """
        import h5py
        import numpy as np

        with h5py.File(ball_path, "r") as f:
            half = f["half"][:]
            timestamps = f["timestamp_utc"]
            ends = {}
            for number in (1, 2):
                rows = np.flatnonzero(half == number)
                if rows.size == 0:
                    raise ValueError(f"No ball samples tagged half {number}")
                # Fixed-width ISO strings sort chronologically.
                ends[number] = parse_utc_string(
                    max(timestamps[np.sort(rows)]).decode("utf-8"))

        return [
            (parse_utc_string(kickoff["timestamp_utc"]),
             ends[number] + timedelta(seconds=HALF_END_BUFFER_S))
            for number, kickoff in zip((1, 2), kickoffs)
        ]

    def in_play(self, moment):
        """Whether an instant falls inside a play window.

        Args:
            moment (datetime): UTC instant to test.

        Returns:
            in_play (bool): True when inside any window.
        """
        return any(start <= moment <= end for start, end in self.windows)

    def in_play_ms(self, position_ms):
        """Whether a video position falls inside a play window.

        Args:
            position_ms (int): Position in the video file, milliseconds.

        Returns:
            in_play (bool): True when inside any window.
        """
        return self.in_play(self.video_start + timedelta(milliseconds=position_ms))

    def to_position_ms(self, moment):
        """Convert a UTC instant to a position in the video file.

        Args:
            moment (datetime): UTC instant, as carried by a prediction.

        Returns:
            position_ms (int): Position in the video file, milliseconds.
        """
        return int(round((moment - self.video_start).total_seconds() * 1000))


def extract(gt, work_dir, data_root, variants, force=False):
    """Run the rule-based spotters over the tracking data.

    Args:
        gt (GroundTruth): Supplies the input paths and the scan window.
        work_dir (Path): Directory for the manifest, configs and raw output.
        data_root (Path): Directory the H5 input paths resolve against.
        variants (List[str]): Rule variants to run.
        force (bool): Re-run variants whose raw output already exists.
            Default: False.
    """
    (work_dir / "raw").mkdir(parents=True, exist_ok=True)

    scan_start, scan_end = gt.windows[0][0], gt.windows[-1][1]
    manifest_path = work_dir / "manifest.json"
    manifest_path.write_text(json.dumps({
        "version": "2.0",
        "task": "action_spotting",
        "labels": {"action": {"type": "single_label", "labels": ["header"]}},
        "data": [{
            "id": gt.sample_id,
            "inputs": [{
                "type": "player_joints_h5",
                "path": gt.joints_path,
                "ball_path": gt.ball_path,
            }],
            "metadata": {
                "start_utc": scan_start.strftime("%Y-%m-%d %H:%M:%S.%f"),
                "end_utc": scan_end.strftime("%Y-%m-%d %H:%M:%S.%f"),
            },
        }],
    }, indent=2))

    os.environ.setdefault("RUN_ID", "header-spotting")
    from opensportslib.apis import LocalizationModel

    for variant in variants:
        raw_path = work_dir / "raw" / f"predictions_{variant}.json"
        if raw_path.exists() and not force:
            print(f"[extract] {variant}: cached -> {raw_path}", flush=True)
            continue

        config_path = work_dir / f"config_{variant}.yaml"
        config_path.write_text(CONFIG_TEMPLATE.format(
            work_dir=work_dir,
            data_root=data_root,
            manifest=manifest_path,
            variant=variant,
            dataset_name=f"headers_{gt.sample_id}",
        ))
        started = time.time()
        print(f"[extract] {variant}: running ...", flush=True)
        api = LocalizationModel(config=str(config_path))
        predictions = api.infer(use_wandb=False)
        api.save_predictions(str(raw_path), predictions)
        print(f"[extract] {variant}: {len(predictions['data'][0]['events'])} events "
              f"in {time.time() - started:.0f}s -> {raw_path}", flush=True)


def convert(gt, work_dir, variants):
    """Move raw predictions onto the video clock and drop out-of-play events.

    Args:
        gt (GroundTruth): Supplies the clock bridge and the play windows.
        work_dir (Path): Directory holding `raw`; receives `video_clock`.
        variants (List[str]): Rule variants to convert.
    """
    (work_dir / "video_clock").mkdir(parents=True, exist_ok=True)

    for variant in variants:
        raw = json.loads((work_dir / "raw" / f"predictions_{variant}.json").read_text())
        events, dropped = [], 0
        for event in raw["data"][0]["events"]:
            moment = parse_utc_string(event["timestamp_utc"])
            if not gt.in_play(moment):
                dropped += 1
                continue
            events.append({
                "head": "Actions",
                "label": "Header",
                "position_ms": gt.to_position_ms(moment),
                "confidence": float(event["confidence_score"]),
            })
        events.sort(key=lambda event: event["position_ms"])

        out_path = work_dir / "video_clock" / f"predictions_{variant}.json"
        out_path.write_text(json.dumps({
            "version": "2.0",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "task": "action_spotting",
            "metadata": {
                "type": "predictions",
                "created_by": f"h5_header_{variant}_rule",
            },
            "labels": {"Actions": {"type": "single_label", "labels": ["Header"]}},
            "data": [{
                "id": gt.sample_id,
                "inputs": [{"type": "video", "path": gt.video_path, "fps": float(FPS)}],
                "events": events,
            }],
        }, indent=2))
        print(f"[convert] {variant}: kept {len(events)} in-play events "
              f"(dropped {dropped}) -> {out_path}", flush=True)


def evaluate(gt, work_dir, variants):
    """Score converted predictions against the ground truth.

    Args:
        gt (GroundTruth): Supplies the annotations and the play windows.
        work_dir (Path): Directory holding `video_clock`; receives the results.
        variants (List[str]): Rule variants to score.

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
    per_window = [
        sum(1 for ms in gt_ms
            if start <= gt.video_start + timedelta(milliseconds=ms) <= end)
        for start, end in gt.windows
    ]
    print(f"[eval] ground truth headers: {len(gt.header_ms)} total | "
          f"in play: {len(gt_ms)} {per_window} | "
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
        payload = json.loads(
            (work_dir / "video_clock" / f"predictions_{variant}.json").read_text())
        events = payload["data"][0]["events"]
        dense_predictions = predictions2vector(
            [{"label": e["label"], "position": e["position_ms"],
              "confidence": e["confidence"]} for e in events],
            num_classes=1, framerate=FPS, EVENT_DICTIONARY=event_dict,
            vector_size=vector_size)

        mAP_per_delta = delta_curve(
            [dense_labels], [closest], [dense_predictions],
            FPS, np.array(DELTAS_S))[0]

        positions = [event["position_ms"] for event in events]
        matched = sum(1 for ms in gt_ms
                      if any(abs(p - ms) <= 1000 for p in positions))
        results[variant] = {
            "num_predictions": len(events),
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
        description="Rule-based header spotting on H5 tracking, scored against "
                    "video annotations.")
    parser.add_argument(
        "--annotations",
        default=str(REPO_ROOT / "WC22_multi.json"),
        help="OSL JSON with Header and Kickoff events plus the video UTC start.")
    parser.add_argument(
        "--data-root",
        default="/home/giancos/FIFA_data",
        help="Directory the H5 input paths in the annotation file resolve against.")
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "outputs" / "header_spotting"),
        help="Directory for manifests, configs, predictions and results.")
    parser.add_argument(
        "--variants",
        default="all",
        help=f"Comma-separated subset of: {', '.join(VARIANTS)}.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run extraction even when raw predictions are cached.")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip extraction and only convert and evaluate cached predictions.")
    return parser.parse_args()


def main():
    """Extract, convert and evaluate header predictions for one game."""
    args = parse_args()

    variants = VARIANTS if args.variants == "all" else [
        name.strip() for name in args.variants.split(",")]
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown:
        raise SystemExit(f"Unknown variants: {unknown}. Expected any of {VARIANTS}.")

    work_dir = Path(args.output_dir)
    data_root = Path(args.data_root)
    gt = GroundTruth(args.annotations, data_root)

    if not args.eval_only:
        extract(gt, work_dir, data_root, variants, force=args.force)
    convert(gt, work_dir, variants)
    evaluate(gt, work_dir, variants)


if __name__ == "__main__":
    main()
