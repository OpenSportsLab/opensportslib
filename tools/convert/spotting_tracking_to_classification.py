"""Expand sampled GAR tracking clips to their full native-frame windows.

Uses reference frameNum values, never tracking row_index / fps. Missing source
frames are explicitly marked, not interpolated. Commands: audit, pilot, build.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import os
import shutil
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

# Reuse the existing classification identity, path, and report contracts without
# changing the MP4 converter (which may be running independently).
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.convert.spotting_to_classification import (
    SPLITS, ValidationError, digest, fingerprint, manifest_path, match_samples,
    read_json, resolve_media, select_pilot, write_json,
)

OBSERVATIONS = ("balls", "ballsSmoothed", "homePlayers", "homePlayersSmoothed",
                "awayPlayers", "awayPlayersSmoothed")


def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    for key in ("spotting_root", "reference_root", "output_root"):
        cfg[key] = str(Path(cfg[key]).expanduser().resolve())
    out = Path(cfg["output_root"])
    for key in ("spotting_root", "reference_root"):
        source = Path(cfg[key])
        if out == source or out in source.parents or source in out.parents:
            raise ValidationError("Input and output directories must be separate")
    cfg.setdefault("gap_policy", "pad")
    cfg.setdefault("pilot_count", 18)
    cfg.setdefault("reference_frames", 16)
    cfg.setdefault("frame_interval", 9)
    cfg.setdefault("alignment_tolerance_ms", 10.0)
    cfg.setdefault("timestamp_tolerance_ms", 1.0)
    if cfg["gap_policy"] not in ("pad", "observed"):
        raise ValidationError("gap_policy must be pad or observed")
    for key in ("pilot_count", "reference_frames", "frame_interval"):
        if not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValidationError(f"{key} must be a positive integer")
    if cfg["alignment_tolerance_ms"] < 0 or cfg["timestamp_tolerance_ms"] < 0:
        raise ValidationError("Timestamp tolerances cannot be negative")
    return cfg


class Dataset:
    def __init__(self, cfg):
        self.cfg, self.out = cfg, Path(cfg["output_root"])
        self.samples, self.headers, self.manifests = [], {}, {}
        self.groups = defaultdict(list)
        self.hashes = {}
        for split in SPLITS:
            sp = manifest_path(cfg["spotting_root"], split)
            rp = manifest_path(cfg["reference_root"], split)
            spot, ref = read_json(sp), read_json(rp)
            items = match_samples(spot, ref, split)
            self.samples.extend(items)
            self.manifests[split] = (sp, rp)
            self.headers[split] = {k: v for k, v in ref.items() if k != "data"}
            for item in items:
                self.groups[(split, str(item["game"]["game_id"]))].append(item)
        self.identity = fingerprint({"config": cfg, "converter": digest(__file__),
            "helpers": digest(Path(__file__).with_name("spotting_to_classification.py")),
            "pyarrow": pa.__version__, "numpy": np.__version__,
            "manifests": {s: [digest(p) for p in paths] for s, paths in self.manifests.items()}})

    def path(self, item, reference=False):
        payload = item["sample"] if reference else item["game"]
        inputs = [x for x in payload["inputs"] if x["type"] == "tracking_parquet"]
        if len(inputs) != 1:
            raise ValidationError("Expected exactly one tracking_parquet input")
        root = self.cfg["reference_root" if reference else "spotting_root"]
        return resolve_media(root, self.manifests[item["split"]][int(reference)], inputs[0]["path"])

    def file_hash(self, path):
        st = path.stat()
        key = (str(path), st.st_size, st.st_mtime_ns)
        if key not in self.hashes:
            self.hashes[key] = digest(path)
        return self.hashes[key]

    def receipt(self, item):
        # Filenames derive from canonical IDs that have been checked for uniqueness.
        sample_id = item["sample"]["id"]
        if Path(sample_id).name != sample_id or sample_id in (".", ".."):
            raise ValidationError("Unsafe sample ID")
        return self.out / "reports/completed" / item["split"] / f"{sample_id}.json"


def integer_frames(column):
    values = column.to_numpy(zero_copy_only=False)
    if not np.all(np.isfinite(values)) or not np.all(values == np.floor(values)):
        raise ValidationError("frameNum must contain finite integer values")
    return values.astype(np.int64)


def source_clock(table):
    frames = integer_frames(table["frameNum"])
    times = table["videoTimeMs"].to_numpy(zero_copy_only=False).astype(np.float64)
    if len(frames) < 2 or not np.all(np.isfinite(times)):
        raise ValidationError("Source clock must contain at least two finite timestamps")
    if np.any(np.diff(frames) <= 0) or np.any(np.diff(times) <= 0):
        raise ValidationError("Source frame numbers and timestamps must be unique and increasing")
    measured_fps = (int(frames[-1]) - int(frames[0])) * 1000 / (times[-1] - times[0])
    return frames, times, float(measured_fps)


def reference_window(reference, position_ms, cfg):
    frames = integer_frames(reference["frameNum"])
    if len(frames) != cfg["reference_frames"] or not np.all(np.diff(frames) == cfg["frame_interval"]):
        raise ValidationError("Reference does not match the configured sampled-frame schedule")
    times = reference["videoTimeMs"].to_numpy(zero_copy_only=False).astype(np.float64)
    center = len(frames) // 2
    if not np.isfinite(times[center]) or abs(times[center] - position_ms) > cfg["alignment_tolerance_ms"]:
        raise ValidationError("Reference center does not align with its annotated action")
    return {"start_frame": int(frames[0]), "end_frame_exclusive": int(frames[-1]) + cfg["frame_interval"],
            "center_frame": int(frames[center]), "sampled_frame_numbers": frames.tolist(),
            "center_timestamp_ms": float(times[center]),
            "alignment_error_ms": float(times[center] - position_ms)}


def positions_for(frames, wanted):
    positions = np.searchsorted(frames, wanted)
    present = positions < len(frames)
    present[present] &= frames[positions[present]] == wanted[present]
    return positions, present


def validate_reference(reference, frames, times, cfg):
    """Require every real sampled timestamp to exist unchanged in spotting."""
    wanted = integer_frames(reference["frameNum"])
    expected = reference["videoTimeMs"].to_numpy(zero_copy_only=False).astype(np.float64)
    positions, present = positions_for(frames, wanted)
    observed = np.isfinite(expected)
    if np.any(observed & ~present):
        raise ValidationError("An observed reference frame is absent from the spotting source")
    if np.any(np.abs(times[positions[observed]] - expected[observed]) > cfg["timestamp_tolerance_ms"]):
        raise ValidationError("Reference timestamps differ from the spotting source")
    return {"observed_reference_frames": int(observed.sum()),
            "reference_missing_frames": int((~observed).sum()),
            "recovered_reference_frames": int((~observed & present).sum())}


def slice_window(source, frames, window, gap_policy):
    """Keep all native slots or all observed rows, preserving the source schema."""
    if "tracking_valid" in source.column_names:
        raise ValidationError("Source already contains reserved tracking_valid column")
    wanted = np.arange(window["start_frame"], window["end_frame_exclusive"], dtype=np.int64)
    positions, present = positions_for(frames, wanted)
    missing = wanted[~present].tolist()
    if gap_policy == "observed":
        result = source.take(pa.array(positions[present], type=pa.int64()))
        flags = np.ones(result.num_rows, dtype=bool)
    else:
        indices = pa.array([int(p) if ok else None for p, ok in zip(positions, present)], type=pa.int64())
        result = source.take(indices)
        field = source.schema.field("frameNum")
        result = result.set_column(result.schema.get_field_index("frameNum"), field,
                                   pa.array(wanted, type=field.type))
        for name in ("period", *OBSERVATIONS):
            if name in result.column_names:
                index = result.schema.get_field_index(name)
                field = result.schema.field(name)
                replacement = -1 if name == "period" else "[]"
                result = result.set_column(index, field, pc.if_else(
                    pa.array(present), result[name], pa.scalar(replacement, type=field.type)))
        flags = present
    if not result.num_rows:
        raise ValidationError("Tracking window contains no observed rows")
    result = result.append_column("tracking_valid", pa.array(flags))
    # pandas metadata describes the old column list; data column types stay intact.
    metadata = {k: v for k, v in (source.schema.metadata or {}).items() if k != b"pandas"}
    result = result.replace_schema_metadata(metadata or None)
    return result, {"window_slots": len(wanted), "output_rows": result.num_rows,
                    "observed_rows": int(present.sum()), "missing_frame_numbers": missing}


def audit(dataset):
    path = dataset.out / "reports/audit.json"
    previous = read_json(path) if path.exists() else {}
    cached = {x["path"]: x for x in previous.get("files", [])}
    report = {"identity": dataset.identity, "status": "running", "files": [], "games": [],
              "splits": dict(Counter(x["split"] for x in dataset.samples)),
              "verification": "local manifests, source/reference timestamps and SHA-256; no Hub download"}
    def record(file):
        st = file.stat()
        old = cached.get(str(file), {})
        sha = old.get("sha256") if old.get("stat") == [st.st_size, st.st_mtime_ns] else None
        sha = sha or dataset.file_hash(file)
        dataset.hashes[(str(file), st.st_size, st.st_mtime_ns)] = sha
        report["files"].append({"path": str(file), "stat": [st.st_size, st.st_mtime_ns], "sha256": sha})
    try:
        if dataset.cfg.get("expected_counts") and report["splits"] != dataset.cfg["expected_counts"]:
            raise ValidationError("Unexpected classification split counts")
        for (split, game), items in dataset.groups.items():
            source_path = dataset.path(items[0])
            clock = pq.read_table(source_path, columns=["frameNum", "videoTimeMs"])
            frames, times, fps = source_clock(clock)
            record(source_path)
            windows_with_gaps, missing_slots = 0, 0
            for item in items:
                reference_path = dataset.path(item, reference=True)
                reference = pq.read_table(reference_path, columns=["frameNum", "videoTimeMs"])
                window = reference_window(reference, item["sample"]["metadata"]["position_ms"], dataset.cfg)
                validate_reference(reference, frames, times, dataset.cfg)
                lo, hi = np.searchsorted(frames, [window["start_frame"], window["end_frame_exclusive"]])
                missing = window["end_frame_exclusive"] - window["start_frame"] - (hi - lo)
                windows_with_gaps += int(missing > 0)
                missing_slots += int(missing)
                record(reference_path)
            report["games"].append({"split": split, "game_id": game, "samples": len(items),
                                    "measured_fps": fps, "windows_with_gaps": windows_with_gaps,
                                    "missing_slots": missing_slots})
            print(f"audit {split}/{game}: {len(items)} samples, {windows_with_gaps} windows with gaps", flush=True)
            write_json(path, report)
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        write_json(path, report)
    return report


def require_audit(dataset):
    path = dataset.out / "reports/audit.json"
    report = read_json(path) if path.exists() else {}
    if report.get("status") != "passed" or report.get("identity") != dataset.identity:
        raise ValidationError("Run audit with these inputs/settings before pilot or build")
    for entry in report["files"]:
        path = Path(entry["path"])
        st = path.stat()
        if [st.st_size, st.st_mtime_ns] != entry["stat"]:
            raise ValidationError(f"Input changed since audit: {path}")
        dataset.hashes[(str(path), st.st_size, st.st_mtime_ns)] = entry["sha256"]


def materialize(dataset, item, source, frames, times, fps):
    reference_path, source_path = dataset.path(item, True), dataset.path(item)
    sample = item["sample"]
    reference = pq.read_table(reference_path, columns=["frameNum", "videoTimeMs"])
    window = reference_window(reference, sample["metadata"]["position_ms"], dataset.cfg)
    checks = validate_reference(reference, frames, times, dataset.cfg)
    provenance = {"identity": dataset.identity, "reference_sample_id": sample["id"],
                  "source_path": str(source_path), "source_sha256": dataset.file_hash(source_path),
                  "reference_sha256": dataset.file_hash(reference_path),
                  "reference_hf_commit": dataset.headers[item["split"]].get("hf_commit"),
                  "measured_source_fps": fps, "gap_policy": dataset.cfg["gap_policy"],
                  "frame_interval": 1, "reference_frame_interval": dataset.cfg["frame_interval"],
                  **window, **checks}
    token = fingerprint(provenance)
    filename = Path(next(x for x in sample["inputs"] if x["type"] == "tracking_parquet")["path"]).name
    output = dataset.out / "tracking" / item["split"] / filename
    receipt = dataset.receipt(item)
    if receipt.exists():
        old = read_json(receipt)
        if old.get("token") == token and output.is_file() and digest(output) == old.get("sha256"):
            return old
    table, coverage = slice_window(source, frames, window, dataset.cfg["gap_policy"])
    provenance.update(coverage)
    provenance["nominal_duration_seconds"] = coverage["window_slots"] / fps
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".parquet.part")
    try:
        pq.write_table(table, temporary, compression="zstd")
        reloaded = pq.read_table(temporary)
        if not reloaded.equals(table):
            raise ValidationError(f"Tracking Parquet round-trip changed data: {sample['id']}")
        observed = reloaded.filter(reloaded["tracking_valid"]).drop(["tracking_valid"])
        source_positions, _ = positions_for(frames, integer_frames(observed["frameNum"]))
        if not observed.equals(source.take(pa.array(source_positions)), check_metadata=False):
            raise ValidationError(f"Observed tracking rows changed: {sample['id']}")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    entry = copy.deepcopy(sample)
    entry["inputs"] = [{"type": "tracking_parquet", "path": f"{item['split']}/{filename}",
                        "fps": round(fps, 3)}]
    entry["metadata"]["conversion"] = provenance
    record = {"token": token, "entry": entry, "path": str(output), "sha256": digest(output),
              "bytes": output.stat().st_size}
    write_json(receipt, record)
    return record


def convert_items(dataset, selected):
    selected_ids = {(x["split"], x["sample"]["id"]) for x in selected}
    for (split, game), items in dataset.groups.items():
        wanted = [x for x in items if (split, x["sample"]["id"]) in selected_ids]
        if not wanted:
            continue
        # At most one full game is retained; no corpus of clip tables is cached.
        source = pq.read_table(dataset.path(items[0]))
        frames, times, fps = source_clock(source)
        for item in wanted:
            yield item, materialize(dataset, item, source, frames, times, fps)
        del source


def write_manifests(dataset, selected, prefix):
    from opensportslib.core.utils.load_annotations import load_annotations
    for split in SPLITS:
        items = [x for x in selected if x["split"] == split]
        header = {k: v for k, v in dataset.headers[split].items() if not k.startswith("hf_")}
        header.update(modalities=["tracking_parquet"], dataset_name=f"soccernet_gar_tracking_full_fps_{split}")
        path = dataset.out / "tracking" / f"{prefix}_{split}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.part")
        with temporary.open("w") as handle:
            handle.write(json.dumps(header)[:-1] + ', "data": [')
            for index, item in enumerate(items):
                if index:
                    handle.write(",")
                json.dump(read_json(dataset.receipt(item))["entry"], handle)
            handle.write("]}\n")
        loaded, _ = load_annotations(str(temporary), input_type="tracking_parquet")
        if len(loaded) != len(items):
            raise ValidationError("Classification annotation loader lost tracking samples")
        del loaded
        os.replace(temporary, path)


def pilot(dataset):
    require_audit(dataset)
    selected = select_pilot(dataset.samples, dataset.cfg["pilot_count"])
    report = {"identity": dataset.identity, "status": "running", "samples": [],
              "gap_policy": dataset.cfg["gap_policy"]}
    started = time.monotonic()
    try:
        total_bytes = 0
        for item, record in convert_items(dataset, selected):
            conversion = record["entry"]["metadata"]["conversion"]
            report["samples"].append({"id": item["sample"]["id"], "split": item["split"], **conversion})
            total_bytes += record["bytes"]
            print(f"pilot {item['sample']['id']}: {conversion['observed_rows']}/{conversion['window_slots']} observed frames", flush=True)
        write_manifests(dataset, selected, "pilot")
        report.update(status="passed", estimated_total_bytes=int(total_bytes / len(selected) * len(dataset.samples)),
                      estimated_seconds=(time.monotonic() - started) / len(selected) * len(dataset.samples))
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(dataset.out / "reports/pilot.json", report)
    return report


def build(dataset):
    require_audit(dataset)
    gate_path = dataset.out / "reports/pilot.json"
    gate = read_json(gate_path) if gate_path.exists() else {}
    if gate.get("identity") != dataset.identity or gate.get("status") != "passed":
        raise ValidationError("Full build requires a passing pilot for these inputs/settings")
    existing = sum(p.stat().st_size for p in (dataset.out / "tracking").rglob("*.parquet"))
    if shutil.disk_usage(dataset.out).free < max(0, gate["estimated_total_bytes"] - existing) * 1.1:
        raise ValidationError("Insufficient free space for estimated output plus 10%")
    report = {"identity": dataset.identity, "status": "running", "completed": 0}
    started = time.monotonic()
    try:
        for item, _ in convert_items(dataset, dataset.samples):
            report["completed"] += 1
            if report["completed"] % 100 == 0:
                report["elapsed_seconds"] = time.monotonic() - started
                print(f"build: {report['completed']}/{len(dataset.samples)}", flush=True)
                write_json(dataset.out / "reports/build.json", report)
        write_manifests(dataset, dataset.samples, "annotations")
        report.update(status="passed", splits=dict(Counter(x["split"] for x in dataset.samples)))
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(dataset.out / "reports/build.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "pilot", "build"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--then-build", action="store_true")
    args = parser.parse_args(argv)
    if args.then_build and args.command != "pilot":
        parser.error("--then-build requires pilot")
    try:
        dataset = Dataset(load_config(args.config))
        result = {"audit": audit, "pilot": pilot, "build": build}[args.command](dataset)
        if args.then_build:
            result = build(dataset)
        print(json.dumps({k: v for k, v in result.items() if k not in ("files", "samples", "games")}, indent=2))
        return 0
    except (ValidationError, FileNotFoundError) as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
