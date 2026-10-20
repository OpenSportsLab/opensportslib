"""Build timestamp-aligned tennis stroke clips from local spotting annotations.

Only decoding/encoding runs here. Frozen pose extraction remains in the model.
Completed clips have checksum receipts so an interrupted build can be resumed.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import time

import av
import numpy as np

CLASSES = ("far_court_serve", "far_court_swing", "near_court_serve", "near_court_swing")
SOURCE_SPLITS = {"train": "train", "valid": "validation", "test": "test"}


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def load_sources(root):
    root = Path(root).resolve()
    jobs, hashes, seen = [], {}, set()
    for split, folder in SOURCE_SPLITS.items():
        manifest = root / folder / f"{folder}.json"
        document = json.loads(manifest.read_text())
        hashes[split] = digest(manifest)
        heads = document["labels"]
        head = "tennis_action" if "tennis_action" in heads else next(iter(heads))
        if not set(CLASSES).issubset(heads[head]["labels"]):
            raise ValueError(f"{manifest}: missing requested stroke classes")
        for record in document["data"]:
            key = (split, record["id"])
            if key in seen:
                raise ValueError(f"Duplicate source record: {key}")
            seen.add(key)
            events = []
            for index, event in enumerate(record.get("events", [])):
                if event.get("label") not in CLASSES or event.get("head", head) != head:
                    continue
                seconds = float(event["position_ms"]) / 1000
                if not math.isfinite(seconds) or seconds < 0:
                    raise ValueError(f"{key}: invalid event timestamp")
                events.append({"index": index, "time": seconds, "event": event})
            if not events:
                continue
            events.sort(key=lambda event: event["time"])
            if any(a["time"] == b["time"] for a, b in zip(events, events[1:])):
                raise ValueError(f"{key}: simultaneous target events cannot form unambiguous clips")
            inputs = [item for item in record["inputs"] if item.get("type") == "video"]
            if len(inputs) != 1:
                raise ValueError(f"{key}: expected exactly one video")
            relative = Path(inputs[0]["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"{key}: unsafe video path")
            source = (manifest.parent / relative).resolve()
            if not source.is_relative_to(root) or not source.is_file():
                raise ValueError(f"{key}: missing video under source root: {relative}")
            match = record.get("metadata", {}).get("base_video")
            if not match:
                raise ValueError(f"{key}: metadata.base_video is required for match-level splits")
            jobs.append({"source_split": split, "record_id": record["id"],
                         "match": match, "source": str(source), "events": events,
                         "source_relative_path": source.relative_to(root).as_posix(),
                         "source_manifest_sha256": hashes[split]})
    return jobs, hashes


def assign_splits(jobs, policy, validation_fraction, seed):
    if policy not in {"match", "preserve"} or not 0 < validation_fraction < 1:
        raise ValueError("Choose split policy match/preserve and a validation fraction between 0 and 1")
    pool = {job["match"] for job in jobs if job["source_split"] != "test"}
    test = {job["match"] for job in jobs if job["source_split"] == "test"}
    if pool & test:
        raise ValueError(f"Test source matches overlap train/validation: {sorted(pool & test)}")
    if policy == "match":
        if len(pool) < 2:
            raise ValueError("At least two source matches are needed to split train/validation")
        count = max(1, min(len(pool) - 1, round(len(pool) * validation_fraction)))
        held_out = set(random.Random(seed).sample(sorted(pool), count))
    else:
        held_out = {job["match"] for job in jobs if job["source_split"] == "valid"}
    for job in jobs:
        job["split"] = (job["source_split"] if policy == "preserve" or job["source_split"] == "test"
                        else "valid" if job["match"] in held_out else "train")
    matches = {split: sorted({job["match"] for job in jobs if job["split"] == split})
               for split in SOURCE_SPLITS}
    for split in SOURCE_SPLITS:
        labels = {event["event"]["label"] for job in jobs if job["split"] == split for event in job["events"]}
        if labels != set(CLASSES):
            raise ValueError(f"{split}: selected matches do not cover all four classes; choose another seed")
    return matches


def probe(source):
    """Keep source clocks, never decoded image buffers, for window planning."""
    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        stream.thread_count = 1
        tb = stream.time_base
        pts, durations = [], []
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base != tb:
                raise ValueError(f"Missing or changing source time base: {source}")
            if pts and frame.pts <= pts[-1]:
                raise ValueError(f"Non-monotonic source timestamps: {source}")
            pts.append(frame.pts)
            durations.append(frame.duration)
        if not pts:
            raise ValueError(f"Empty video: {source}")
        rate = stream.average_rate or stream.guessed_rate
        if not rate or rate <= 0:
            raise ValueError(f"Invalid source frame rate: {source}")
        durations[-1] = durations[-1] or (pts[-1] - pts[-2] if len(pts) > 1 else round(1 / rate / tb))
        times = [(value - pts[0]) * float(tb) for value in pts]
        return {"pts": pts, "times": times, "durations": durations,
                "time_base": tb, "rate": rate, "width": stream.width, "height": stream.height,
                "duration": times[-1] + durations[-1] * float(tb)}


def plan_windows(job, info, before, after, trim):
    windows = []
    events = job["events"]
    for index, event in enumerate(events):
        center = event["time"]
        if center >= info["duration"]:
            raise ValueError(f"{job['record_id']}: event outside video: {center}")
        start, end = max(0., center - before), min(info["duration"], center + after)
        if trim:
            if index:
                start = max(start, (events[index - 1]["time"] + center) / 2)
            if index + 1 < len(events):
                end = min(end, (center + events[index + 1]["time"]) / 2)
        first = bisect_left(info["times"], start)
        stop = bisect_left(info["times"], end)
        if first >= stop or center < info["times"][first]:
            raise ValueError(f"{job['record_id']}: no source frames covering target event")
        actual_end = (info["times"][stop] if stop < len(info["times"]) else info["duration"])
        if not info["times"][first] <= center < actual_end:
            raise ValueError(f"{job['record_id']}: target not covered by decoded window")
        if trim and any(info["times"][first] <= other["time"] < actual_end
                        for other in events if other is not event):
            raise ValueError(f"{job['record_id']}: frame rounding includes another target event")
        key = f"{job['source_split']}:{job['record_id']}:{event['index']}"
        safe = re.sub(r"[^a-zA-Z0-9_.-]", "_", job["record_id"])[:100]
        sample_id = f"{safe}__e{event['index']:04d}_{fingerprint(key)[:10]}"
        windows.append({"id": sample_id, "event": event, "first": first, "stop": stop,
                        "requested_start": start, "requested_end": end, "actual_end": actual_end})
    return windows


class ClipWriter:
    def __init__(self, destination, window, info, crf, preset):
        self.destination, self.window, self.info = destination, window, info
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = destination.with_suffix(".part.mp4")
        self.container = av.open(str(self.temporary), "w", format="mp4")
        self.stream = self.container.add_stream("libx264", rate=info["rate"])
        self.stream.width, self.stream.height = info["width"], info["height"]
        self.stream.pix_fmt = "yuv420p"
        self.stream.time_base = info["time_base"]
        self.stream.codec_context.time_base = info["time_base"]
        self.stream.codec_context.thread_count = 1
        self.stream.options = {"crf": str(crf), "preset": preset}

    def push(self, image, index):
        frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        frame.pts = self.info["pts"][index] - self.info["pts"][self.window["first"]]
        frame.time_base = self.info["time_base"]
        frame.duration = self.info["durations"][index]
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def finish(self):
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()
        self.container = None
        first, stop = self.window["first"], self.window["stop"]
        expected = np.asarray(self.info["times"][first:stop]) - self.info["times"][first]
        with av.open(str(self.temporary)) as container:
            stream = container.streams.video[0]
            stream.thread_count = 1
            actual = [float(frame.pts * frame.time_base) for frame in container.decode(stream)]
            tolerance = 2 * max(float(stream.time_base), float(self.info["time_base"]))
            if len(actual) != len(expected) or not np.allclose(actual, expected, atol=tolerance, rtol=0):
                raise ValueError(f"Encoded timestamps/frame count changed: {self.destination}")
            if (stream.width, stream.height) != (self.info["width"], self.info["height"]):
                raise ValueError(f"Encoded dimensions changed: {self.destination}")
            duration = float(stream.duration * stream.time_base)
            event_offset = self.window["event"]["time"] - self.info["times"][first]
            if not 0 <= event_offset < duration:
                raise ValueError(f"Target event outside encoded clip: {self.destination}")
        os.replace(self.temporary, self.destination)
        return {"frame_count": len(actual), "duration_seconds": duration,
                "timestamps_sha256": fingerprint(actual)}

    def close(self):
        if self.container is not None:
            self.container.close()
        self.temporary.unlink(missing_ok=True)


def convert_source(job, output, identity, settings):
    info = probe(job["source"])
    source_sha = digest(job["source"])
    excluded = []
    if settings["invalid_events"] == "skip":
        excluded = [{"source_split": job["source_split"], "source_record_id": job["record_id"],
                     "source_video": job["source_relative_path"], "source_video_sha256": source_sha,
                     "source_event_index": event["index"], "event": event["event"],
                     "source_duration_seconds": info["duration"], "reason": "event_outside_decoded_video"}
                    for event in job["events"] if event["time"] >= info["duration"]]
        job = {**job, "events": [event for event in job["events"] if event["time"] < info["duration"]]}
    windows = plan_windows(job, info, settings["before_seconds"], settings["after_seconds"], settings["trim_neighbors"])
    entries, pending = [], []
    for window in windows:
        relative = Path("clips") / job["split"] / f"{window['id']}.mp4"
        destination = output / relative
        receipt = output / "receipts" / f"{window['id']}.json"
        window.update(relative=relative, destination=destination, receipt=receipt)
        if receipt.is_file() and destination.is_file():
            saved = json.loads(receipt.read_text())
            if (saved.get("identity") == identity and saved.get("source_sha256") == source_sha
                    and saved.get("output_sha256") == digest(destination)):
                entries.append(saved["sample"])
                continue
        pending.append(window)
    active, next_window = [], 0
    writers = []
    try:
        if pending:
            with av.open(job["source"]) as container:
                stream = container.streams.video[0]
                stream.thread_count = 1
                for index, frame in enumerate(container.decode(stream)):
                    if frame.pts != info["pts"][index]:
                        raise ValueError(f"Source changed between decode passes: {job['source']}")
                    while next_window < len(pending) and pending[next_window]["first"] == index:
                        window = pending[next_window]
                        writer = ClipWriter(window["destination"], window, info, settings["crf"], settings["preset"])
                        writers.append(writer)
                        active.append(writer)
                        next_window += 1
                    if active:
                        image = frame.to_ndarray(format="rgb24")
                        for writer in active:
                            writer.push(image, index)
                    remaining = []
                    for writer in active:
                        if index + 1 != writer.window["stop"]:
                            remaining.append(writer)
                            continue
                        clip = writer.finish()
                        window = writer.window
                        event = window["event"]
                        sample = {"id": window["id"], "inputs": [{"type": "video", "path": window["relative"].as_posix(),
                                   "fps": float(info["rate"])}],
                                  "labels": {"action": {"label": event["event"]["label"]}},
                                  "metadata": {"game_id": job["match"], "base_video": job["match"],
                                    "source_split": job["source_split"], "source_record_id": job["record_id"],
                                    "source_event_index": event["index"], "source_event": event["event"],
                                    "conversion": {"identity": identity, "source_video": job["source_relative_path"],
                                      "source_video_sha256": source_sha,
                                      "source_manifest_sha256": job["source_manifest_sha256"],
                                      "source_start_frame": window["first"], "source_end_frame_exclusive": window["stop"],
                                      "source_start_pts": info["pts"][window["first"]],
                                      "source_time_base": [info["time_base"].numerator, info["time_base"].denominator],
                                      "source_fps": str(info["rate"]), "width": info["width"], "height": info["height"],
                                      "requested_start_seconds": window["requested_start"],
                                      "requested_end_seconds": window["requested_end"],
                                      "source_start_seconds": info["times"][window["first"]],
                                      "source_end_seconds": window["actual_end"],
                                      "event_position_ms": float(event["event"]["position_ms"]),
                                      "event_offset_ms": (event["time"] - info["times"][window["first"]]) * 1000,
                                      **clip}}}
                        write_json(window["receipt"], {"identity": identity, "source_sha256": source_sha,
                                   "output_sha256": digest(window["destination"]), "sample": sample})
                        entries.append(sample)
                    active = remaining
                    if not active and next_window == len(pending):
                        break
            if active or next_window != len(pending):
                raise ValueError(f"Source ended before all clips were encoded: {job['source']}")
    finally:
        for writer in writers:
            writer.close()
    if len(entries) != len(windows):
        raise ValueError(f"Clip count mismatch for {job['record_id']}")
    return job["split"], entries, len(pending), excluded


def build(source_root, output_root, *, before_seconds=.5, after_seconds=.5,
          trim_neighbors=True, split_policy="match", validation_fraction=.2,
          seed=42, workers=2, crf=18, preset="veryfast", invalid_events="error"):
    source_root, output = Path(source_root).resolve(), Path(output_root).resolve()
    if source_root == output or output.is_relative_to(source_root) or source_root.is_relative_to(output):
        raise ValueError("Source and output trees must be separate")
    if workers < 1 or before_seconds <= 0 or after_seconds <= 0 or not 0 <= crf <= 51:
        raise ValueError("Invalid worker count, clip window, or CRF")
    if invalid_events not in {"error", "skip"}:
        raise ValueError("Invalid-event policy must be error or skip")
    jobs, manifests = load_sources(source_root)
    matches = assign_splits(jobs, split_policy, validation_fraction, seed)
    settings = dict(before_seconds=before_seconds, after_seconds=after_seconds,
                    trim_neighbors=trim_neighbors, split_policy=split_policy,
                    validation_fraction=validation_fraction, seed=seed, crf=crf, preset=preset,
                    invalid_events=invalid_events)
    identity = fingerprint({"settings": settings, "manifests": manifests, "matches": matches,
                            "converter_sha256": digest(__file__), "av": av.__version__})
    report_path = output / "reports" / "build.json"
    if report_path.is_file():
        previous = json.loads(report_path.read_text())
        if previous.get("identity") != identity:
            raise ValueError("Output belongs to another conversion; choose a new output root")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("Output root is not empty and has no conversion report")
    report = {"status": "running", "identity": identity, "settings": settings,
              "source_root": str(source_root), "source_manifests_sha256": manifests,
              "matches": matches, "source_videos": len(jobs),
              "requested_events": sum(len(job["events"]) for job in jobs),
              "expected_clips": sum(len(job["events"]) for job in jobs), "completed_videos": 0}
    write_json(report_path, report)
    started = time.monotonic()
    entries = {split: [] for split in SOURCE_SPLITS}
    excluded = []
    encoded = 0
    executor = ThreadPoolExecutor(max_workers=workers)
    futures = []
    try:
        futures = [executor.submit(convert_source, job, output, identity, settings) for job in jobs]
        for future in as_completed(futures):
            split, samples, count, skipped = future.result()
            entries[split].extend(samples)
            excluded.extend(skipped)
            report["expected_clips"] = report["requested_events"] - len(excluded)
            encoded += count
            report["completed_videos"] += 1
            if report["completed_videos"] % 20 == 0 or report["completed_videos"] == len(jobs):
                report.update(completed_clips=sum(map(len, entries.values())), elapsed_seconds=time.monotonic()-started)
                write_json(report_path, report)
                print(f"{report['completed_videos']}/{len(jobs)} videos; {report['completed_clips']}/{report['expected_clips']} clips", flush=True)
        if sum(map(len, entries.values())) + len(excluded) != report["requested_events"]:
            raise ValueError("Output/exclusion counts do not cover every requested event")
        counts = {}
        for split, samples in entries.items():
            samples.sort(key=lambda sample: sample["id"])
            counts[split] = dict(Counter(sample["labels"]["action"]["label"] for sample in samples))
            if len({sample["id"] for sample in samples}) != len(samples):
                raise ValueError(f"Duplicate output IDs in {split}")
            document = {"version": "2.0", "task": "action_classification", "dataset_name": "tennis-strokes",
                        "labels": {"action": {"type": "single_label", "labels": list(CLASSES)}},
                        "modalities": ["video"], "conversion_identity": identity, "data": samples}
            write_json(output / f"{split}.json", document)
        excluded.sort(key=lambda record: (record["source_split"], record["source_record_id"], record["source_event_index"]))
        write_json(output / "reports" / "excluded_events.json", excluded)
        report.update(status="complete", class_counts=counts, excluded_events=len(excluded),
                      split_counts={split: len(samples) for split, samples in entries.items()},
                      encoded_clips=encoded, reused_clips=report["expected_clips"]-encoded,
                      elapsed_seconds=time.monotonic()-started)
        write_json(report_path, report)
        return report
    except Exception as error:
        for future in futures:
            future.cancel()
        report.update(status="failed", error=str(error))
        write_json(report_path, report)
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("data/tennis-localization/swing-data"))
    parser.add_argument("--output-root", type=Path, default=Path("data/tennis-classification"))
    parser.add_argument("--before-seconds", type=float, default=.5)
    parser.add_argument("--after-seconds", type=float, default=.5)
    parser.add_argument("--keep-neighbors", action="store_true", help="Keep fixed context even if another selected stroke is present")
    parser.add_argument("--split-policy", choices=("match", "preserve"), default="match")
    parser.add_argument("--validation-fraction", type=float, default=.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--invalid-events", choices=("error", "skip"), default="error",
                        help="Fail on out-of-video events, or explicitly exclude and report them")
    parser.add_argument("--preset", default="veryfast", choices=("ultrafast", "veryfast", "medium", "slow"))
    args = parser.parse_args()
    report = build(args.source_root, args.output_root, before_seconds=args.before_seconds,
                   after_seconds=args.after_seconds, trim_neighbors=not args.keep_neighbors,
                   split_policy=args.split_policy, validation_fraction=args.validation_fraction,
                   seed=args.seed, workers=args.workers, crf=args.crf, preset=args.preset,
                   invalid_events=args.invalid_events)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
