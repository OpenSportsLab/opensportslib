"""Build balanced Dallas swing/background clips and disjoint localization splits.

Background is inferred from annotation gaps, not verified absence of strokes.
Reuse the timestamp-preserving tennis converter; never modify source annotations.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import json
from pathlib import Path
import random
import sys
import time

import av
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.convert.tennis_to_classification import convert_source, digest, fingerprint, write_json

SOURCE_REPO = "OpenSportsLab/OSL-loc-tennis-dallas"
SOURCE_REVISION = "085a5e92597ebc0b943d043f17260879e7c1f3ae"
CLASSES = ["background", "swing"]
MATCHES = {
    "train": ["Cilic_vs_Tien_2_10_26", "Dimitrov_vs_Michelsen_2_10_26", "Fritz_vs_Giron_2_10_26"],
    "valid": ["Shapovalov_vs_Jodar_2_10_26"],
    "test": ["Shelton_vs_Diallo_2_10_26"],
}


def background_centers(duration, positives, *, count, seed=42, guard_seconds=2.):
    """Sample disjoint one-second windows, with a frame-rounding safety margin."""
    if guard_seconds < 0 or not np.isfinite(duration) or duration <= 1:
        raise ValueError("Invalid duration or background guard")
    positive = np.asarray(sorted(positives), dtype=float)
    if not len(positive) or not np.all(np.isfinite(positive)):
        raise ValueError("Finite positive events are required")
    grid = np.arange(.5, duration - .5, 1.25)
    distance = np.full(len(grid), np.inf)
    for center in positive:
        distance = np.minimum(distance, np.abs(grid - center))
    candidates = grid[distance >= guard_seconds + .5 + .1].tolist()
    if len(candidates) < count:
        raise ValueError(f"Only {len(candidates)} safe background windows for {count} positives")
    return sorted(random.Random(seed).sample(candidates, count))


def prepare_jobs(manifest, *, matches=None, seed=42, guard_seconds=2.):
    manifest = Path(manifest).resolve()
    document = json.loads(manifest.read_text())
    matches = MATCHES if matches is None else matches
    names = [name for split in ("train", "valid", "test") for name in matches[split]]
    if len(names) != 5 or len(set(names)) != 5 or [len(matches[s]) for s in matches] != [3, 1, 1]:
        raise ValueError("Require five disjoint games: three train, one valid, one test")
    if document["labels"]["swing"]["labels"] != ["swing"]:
        raise ValueError("Expected the Dallas swing-only annotations")
    records = {record["id"]: record for record in document["data"]}
    if set(records) != set(names) or len(records) != len(document["data"]):
        raise ValueError("Source games do not match the requested disjoint splits")
    jobs, original = [], {}
    annotation_sha = digest(manifest)
    for split, games in matches.items():
        original[split] = []
        for name in games:
            record = copy.deepcopy(records[name])
            inputs = record["inputs"]
            if len(inputs) != 1 or inputs[0]["type"] != "video":
                raise ValueError(f"{name}: expected one video")
            relative = Path(inputs[0]["path"])
            source = (manifest.parent / relative).resolve()
            if relative.is_absolute() or not source.is_relative_to(manifest.parent) or not source.is_file():
                raise ValueError(f"{name}: missing or unsafe video path")
            with av.open(str(source)) as container:
                stream = container.streams.video[0]
                duration = float(stream.duration * stream.time_base)
            events = []
            for index, event in enumerate(record["events"]):
                seconds = float(event["position_ms"]) / 1000
                if (event["head"] != "swing" or event["label"] != "swing"
                        or not np.isfinite(seconds) or not 0 <= seconds < duration):
                    raise ValueError(f"{name}: invalid or out-of-video swing annotation")
                events.append({"index": index, "time": seconds, "event": copy.deepcopy(event)})
            if len({item["time"] for item in events}) != len(events):
                raise ValueError(f"{name}: duplicate swing timestamps")
            centers = background_centers(duration, [item["time"] for item in events],
                                         count=len(events), seed=seed, guard_seconds=guard_seconds)
            positives = len(events)
            for index, center in enumerate(centers, positives):
                events.append({"index": index, "time": center, "event": {
                    "head": "action", "label": "background", "position_ms": center * 1000,
                    "label_source": "inferred_from_unannotated_gap", "guard_seconds": guard_seconds}})
            events.sort(key=lambda item: item["time"])
            jobs.append({"source_split": "swing", "record_id": name, "match": name,
                         "source": str(source), "source_relative_path": relative.as_posix(),
                         "source_manifest_sha256": annotation_sha, "events": events, "split": split})
            record.setdefault("metadata", {}).update(base_video=name, game_id=name)
            original[split].append(record)
    return jobs, original, document


def build(manifest, output_root, localization_root, *, workers=2, seed=42, guard_seconds=2., matches=None):
    output, localization = Path(output_root).resolve(), Path(localization_root).resolve()
    manifest = Path(manifest).resolve()
    for destination in (output, localization):
        if destination == manifest.parent or destination.is_relative_to(manifest.parent):
            raise ValueError("Source and output trees must be separate")
    jobs, original, source = prepare_jobs(manifest, matches=matches, seed=seed, guard_seconds=guard_seconds)
    settings = dict(before_seconds=.5, after_seconds=.5, trim_neighbors=True, crf=18,
                    preset="veryfast", invalid_events="error", seed=seed, guard_seconds=guard_seconds,
                    background_grid_seconds=1.25, rounding_margin_seconds=.1, split_policy="match")
    split_matches = {split: [record["id"] for record in records] for split, records in original.items()}
    provenance = {"repo_id": SOURCE_REPO, "revision": source.get("hf_commit", SOURCE_REVISION),
                  "branch": "swing", "annotation_sha256": digest(manifest)}
    identity = fingerprint({"settings": settings, "source": provenance, "matches": split_matches,
                            "converter_sha256": digest(__file__),
                            "clip_converter_sha256": digest(Path(__file__).with_name("tennis_to_classification.py")),
                            "av_version": av.__version__})
    report_path = output / "reports/build.json"
    if report_path.is_file():
        if json.loads(report_path.read_text()).get("identity") != identity:
            raise ValueError("Output belongs to another conversion; choose another output root")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("Output is nonempty without a conversion report")
    report = dict(status="running", identity=identity, settings=settings, source=provenance,
                  matches=split_matches, requested_events=sum(len(job["events"]) for job in jobs),
                  source_videos=len(jobs), completed_videos=0)
    write_json(report_path, report)
    entries = {split: [] for split in original}
    encoded, started = 0, time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(convert_source, job, output, identity, settings) for job in jobs]
            for future in as_completed(futures):
                split, samples, count, excluded = future.result()
                if excluded:
                    raise ValueError("Dallas conversion must not silently exclude events")
                for sample in samples:
                    if sample["labels"]["action"]["label"] == "background":
                        timing = sample["metadata"]["conversion"]
                        positives = [float(event["position_ms"]) / 1000 for record in original[split]
                                     if record["id"] == sample["metadata"]["base_video"] for event in record["events"]]
                        if any(timing["source_start_seconds"] - guard_seconds < point
                               < timing["source_end_seconds"] + guard_seconds for point in positives):
                            raise ValueError("Encoded background violates the annotation guard")
                entries[split].extend(samples)
                encoded += count
                report["completed_videos"] += 1
                report.update(completed_clips=sum(map(len, entries.values())), elapsed_seconds=time.monotonic()-started)
                write_json(report_path, report)
                print(f"Completed {report['completed_videos']}/5 games; {report['completed_clips']} clips", flush=True)
        for split, samples in entries.items():
            samples.sort(key=lambda sample: sample["id"])
            write_json(output / f"{split}.json", {
                "version": "2.0", "task": "action_classification", "dataset_name": "tennis-dallas-binary",
                "modalities": ["video"], "labels": {"action": {"type": "single_label", "labels": CLASSES}},
                "conversion_identity": identity, "metadata": {"source": provenance, "split": split,
                  "background_labels": "weak labels inferred from annotation gaps, not verified negatives"}, "data": samples})
            for record in original[split]:
                for item in record["inputs"]:
                    target = localization / item["path"]
                    target.parent.mkdir(parents=True, exist_ok=True)
                    source_video = (manifest.parent / item["path"]).resolve()
                    if target.is_symlink() and target.resolve() != source_video:
                        raise ValueError(f"Conflicting localization link: {target}")
                    if not target.exists():
                        target.symlink_to(source_video)
            write_json(localization / f"{split}.json", {
                "version": "2.0", "task": "action_spotting", "dataset_name": "tennis-dallas",
                "modalities": ["video"], "labels": source["labels"],
                "metadata": {"source": provenance, "split": split, "match_disjoint": True}, "data": original[split]})
        counts = {split: {label: sum(sample["labels"]["action"]["label"] == label for sample in samples)
                          for label in CLASSES} for split, samples in entries.items()}
        total = sum(map(len, entries.values()))
        if total != report["requested_events"] or any(count["swing"] != count["background"] for count in counts.values()):
            raise ValueError("Incomplete or unbalanced conversion")
        report.update(status="complete", expected_clips=total, encoded_clips=encoded, reused_clips=total-encoded,
                      split_counts={s: len(v) for s, v in entries.items()}, class_counts=counts,
                      elapsed_seconds=time.monotonic()-started)
        write_json(report_path, report)
        return report
    except Exception as error:
        report.update(status="failed", error=str(error))
        write_json(report_path, report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/tennis-dallas-source/swing/swing/swing.json"))
    parser.add_argument("--output-root", type=Path, default=Path("data/tennis-dallas-classification"))
    parser.add_argument("--localization-root", type=Path, default=Path("data/tennis-dallas-localization"))
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(build(args.manifest, args.output_root, args.localization_root, workers=args.workers), indent=2))


if __name__ == "__main__":
    main()
