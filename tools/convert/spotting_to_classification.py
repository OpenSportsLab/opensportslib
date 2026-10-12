"""Resumable GAR classification videos and optional exact frame reproduction.

Temporal validation preserves action identities and native source timing.
Exact validation additionally requires identical reference frame arrays.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from fractions import Fraction
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import av
import cv2
import numpy as np
import yaml

SPLITS = ("train", "valid", "test")
SHAPE = (16, 224, 224, 3)
SPOTTING = "OpenSportsLab/SNGAR-Action-Spotting"
REFERENCE = "OpenSportsLab/SoccerNet-GAR"
SPOTTING_REV = "a4a69ae4d40e706080183bf39fe5bf6f1d3a47ba"
REFERENCE_REV = "629043a5f3fe74e53095c7b8d3c0b489ff0d0269"
# This implementation used the same sampling/encoding contract, but rejected
# small accumulated differences between a stream's declared and measured rates.
# Only this known predecessor may reuse audit/pilot gates and validated receipts.
RESUME_COMPATIBLE_CONVERTERS = ("fd435d1fb6a3ebe88b3fd0c95943fa009ef3cd7a7bdfc23862427287a638517e",)


class ValidationError(RuntimeError):
    """A required dataset invariant failed; scaling must stop."""


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def manifest_path(root, split):
    root = Path(root)
    for path in (root / split / f"{split}.json", root / f"{split}.json",
                 root / f"annotations_{split}.json"):
        if path.is_file():
            return path
    raise FileNotFoundError(f"No {split} manifest under {root}")


def resolve_media(root, manifest, relative):
    """Resolve downloaded split-relative and flat Hub layouts without guessing names."""
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValidationError(f"Unsafe dataset path: {relative}")
    candidates = [Path(manifest).parent / relative, Path(root) / relative,
                  Path(root) / Path(relative).parts[0] / relative]
    existing = list(dict.fromkeys(p.resolve() for p in candidates if p.is_file()))
    if len(existing) > 1 and digest(existing[0]) != digest(existing[1]):
        raise ValidationError(f"Ambiguous media path: {relative}")
    if not existing:
        raise FileNotFoundError(f"Missing {relative}; searched {candidates}")
    return existing[0]


def action_key(game, event):
    return (str(game), event["position_ms"], event["label"])


def match_samples(spotting, reference, split):
    events = {}
    for game in spotting["data"]:
        for event in game["events"]:
            key = action_key(game["game_id"], event)
            if key in events:
                raise ValidationError(f"Duplicate spotting action: {key}")
            events[key] = (game, event)
    seen, ids, result = set(), set(), []
    if spotting["labels"] != reference["labels"]:
        raise ValidationError(f"Label vocabulary differs in {split}")
    for sample in reference["data"]:
        meta = sample["metadata"]
        key = (str(meta["game_id"]), meta["position_ms"], sample["labels"]["action"]["label"])
        if key in seen or sample["id"] in ids or key not in events:
            raise ValidationError(f"Missing or duplicate classification action: {sample['id']}")
        game, event = events[key]
        if meta.get("team") != event.get("team") or meta.get("game_time") != event.get("gameTime"):
            raise ValidationError(f"Action metadata differs: {sample['id']}")
        seen.add(key)
        ids.add(sample["id"])
        result.append({"split": split, "sample": sample, "game": game})
    if seen != set(events):
        raise ValidationError(f"Unrepresented spotting actions in {split}")
    return result


def select_pilot(samples, count=18):
    chosen = {}
    def add(item):
        chosen[(item["split"], item["sample"]["id"])] = item
    for split in SPLITS:
        subset = [x for x in samples if x["split"] == split]
        if subset:
            add(subset[0])
            add(subset[-1])
    labels = sorted({x["sample"]["labels"]["action"]["label"] for x in samples})
    for label in labels:
        if not any(x["sample"]["labels"]["action"]["label"] == label for x in chosen.values()):
            add(next(x for x in samples if x["sample"]["labels"]["action"]["label"] == label))
    if len(chosen) > count:
        raise ValidationError("Pilot size cannot cover all labels and split boundaries")
    for i in np.random.default_rng(42).permutation(len(samples)):
        if len(chosen) >= count:
            break
        add(samples[i])
    return [x for x in samples if (x["split"], x["sample"]["id"]) in chosen]


def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    for key in ("spotting_root", "reference_root", "output_root"):
        cfg[key] = str(Path(cfg[key]).expanduser().resolve())
    out = Path(cfg["output_root"])
    for key in ("spotting_root", "reference_root"):
        source = Path(cfg[key])
        if out == source or out in source.parents or source in out.parents:
            raise ValidationError("Output and input roots must be separate directory trees")
    cfg.setdefault("verify_hub", True)
    cfg.setdefault("pilot_count", 18)
    cfg.setdefault("frame_interval", 9)
    cfg.setdefault("crf", 18)
    cfg.setdefault("validation", "exact")
    cfg.setdefault("preset", "medium")
    cfg.setdefault("workers", 1)
    if cfg["validation"] not in ("exact", "temporal"):
        raise ValidationError("validation must be exact or temporal")
    if cfg["preset"] not in ("ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"):
        raise ValidationError("Unsupported x264 preset")
    if not isinstance(cfg["workers"], int) or not 1 <= cfg["workers"] <= 8:
        raise ValidationError("workers must be an integer between 1 and 8")
    if cfg["frame_interval"] <= 0 or not 0 <= cfg["crf"] <= 51:
        raise ValidationError("Invalid frame interval or CRF")
    return cfg


class Dataset:
    def __init__(self, cfg):
        self.cfg = cfg
        self.out = Path(cfg["output_root"])
        self.headers, self.samples, self.manifests = {}, [], {}
        self.hashes = {}
        self.video_infos = {}
        for split in SPLITS:
            sp = self.out / "verified_inputs/spotting" / split / f"{split}.json"
            rp = self.out / "verified_inputs/reference" / split / f"{split}.json"
            if not sp.is_file():
                sp = manifest_path(cfg["spotting_root"], split)
            if not rp.is_file():
                rp = manifest_path(cfg["reference_root"], split)
            spot, ref = read_json(sp), read_json(rp)
            self.headers[split] = {k: v for k, v in ref.items() if k != "data"}
            self.samples.extend(match_samples(spot, ref, split))
            self.manifests[split] = (sp, rp)
        identity_payload = {"config": cfg, "manifests": {
            split: [digest(p) for p in paths] for split, paths in self.manifests.items()},
            "converter": digest(__file__), "opencv": cv2.__version__, "av": av.__version__,
            "numpy": np.__version__, "spotting_revision": SPOTTING_REV,
            "reference_revision": REFERENCE_REV}
        self.identity = fingerprint(identity_payload)
        self.compatible_identities = {self.identity} | {
            fingerprint({**identity_payload, "converter": version})
            for version in RESUME_COMPATIBLE_CONVERTERS
        }

    def media(self, item, kind):
        split = item["split"]
        source = item["sample"] if kind == "frames_npy" else item["game"]
        root = self.cfg["reference_root" if kind == "frames_npy" else "spotting_root"]
        manifest = self.manifests[split][kind == "frames_npy"]
        inp = next(x for x in source["inputs"] if x["type"] == kind)
        replacement = self.out / "verified_inputs" / ("reference" if kind == "frames_npy" else "spotting") / split / inp["path"]
        if replacement.is_file():
            return replacement
        return resolve_media(root, manifest, inp["path"])

    def video_info(self, path):
        path = Path(path)
        st = path.stat()
        key = (str(path), st.st_size, st.st_mtime_ns)
        if key not in self.video_infos:
            self.video_infos[key] = video_info(path)
        return self.video_infos[key]

    def file_hash(self, path):
        path = Path(path)
        st = path.stat()
        key = (str(path), st.st_size, st.st_mtime_ns)
        if key not in self.hashes:
            self.hashes[key] = digest(path)
        return self.hashes[key]


def remote_file(repo, revision, filename, cache):
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo, filename, repo_type="dataset", revision=revision,
                               cache_dir=str(cache)))


def prepare_manifests(cfg):
    """Repair missing/stale manifests into a separate tree; never edit inputs."""
    if not cfg["verify_hub"]:
        return
    import pyarrow.parquet as pq
    out = Path(cfg["output_root"])
    for split in SPLITS:
        spot = read_json(remote_file(SPOTTING, SPOTTING_REV, f"{split}.json", out / "hub_cache"))
        rows = pq.read_table(remote_file(REFERENCE, REFERENCE_REV,
                            f"{split}/metadata.parquet", out / "hub_cache")).to_pylist()
        ref = json.loads(rows[0]["header"])
        ref["data"] = [json.loads(row["sample_payload"]) for row in rows]
        ref.update(hf_repo_id=REFERENCE, hf_branch="frames", hf_split=split,
                   hf_format="parquet", hf_commit=REFERENCE_REV)
        for kind, key, remote in (("spotting", "spotting_root", spot),
                                  ("reference", "reference_root", ref)):
            override = out / "verified_inputs" / kind / split / f"{split}.json"
            try:
                local = read_json(override if override.is_file() else manifest_path(cfg[key], split))
            except (FileNotFoundError, json.JSONDecodeError):
                local = {}
            if local.get("data") != remote["data"] or local.get("labels") != remote["labels"]:
                write_json(override, remote)


def audit(dataset):
    """Hub metadata has no per-NPY checksum; state that limitation explicitly."""
    cfg, out = dataset.cfg, dataset.out
    report = {"identity": dataset.identity, "status": "running", "splits": {},
              "spotting_revision": SPOTTING_REV, "reference_revision": REFERENCE_REV,
              "hub_verified": cfg["verify_hub"], "media": [],
              "reference_verification": "local SHA-256, array header, pinned Hub sample/member metadata; "
              "Hub shard manifests do not provide per-array checksums"}
    report_path = out / "reports/audit.json"
    previous = read_json(report_path) if report_path.exists() else {}
    cached = {x["path"]: x for x in previous.get("media", [])}
    hub_files = {}
    if cfg["verify_hub"]:
        from huggingface_hub import HfApi
        hub_files = {x.rfilename: x for x in HfApi().dataset_info(
            SPOTTING, revision=SPOTTING_REV, files_metadata=True).siblings}
    try:
        for split in SPLITS:
            sp, rp = dataset.manifests[split]
            spot, ref = read_json(sp), read_json(rp)
            if cfg["verify_hub"]:
                import pyarrow.parquet as pq
                remote = read_json(remote_file(SPOTTING, SPOTTING_REV, f"{split}.json", out / "hub_cache"))
                if spot["data"] != remote["data"] or spot["labels"] != remote["labels"]:
                    raise ValidationError(f"Local spotting manifest differs from pinned Hub: {split}")
                rows = pq.read_table(remote_file(REFERENCE, REFERENCE_REV,
                    f"{split}/metadata.parquet", out / "hub_cache")).to_pylist()
                if [json.loads(x["sample_payload"]) for x in rows] != ref["data"]:
                    raise ValidationError(f"Local reference manifest differs from pinned Hub: {split}")
                members = pq.read_table(remote_file(REFERENCE, REFERENCE_REV,
                    f"{split}/shard_manifest.parquet", out / "hub_cache")).to_pylist()
                member_ids = {x["sample_id"] for x in members if x["status"] == "ok" and x["input_type"] == "frames_npy"}
                if member_ids != {x["id"] for x in ref["data"]}:
                    raise ValidationError(f"Reference shard membership differs: {split}")
            samples = [x for x in dataset.samples if x["split"] == split]
            for i, item in enumerate(samples):
                try:
                    path = dataset.media(item, "frames_npy")
                    arr = np.load(path, mmap_mode="r", allow_pickle=False)
                    if arr.shape != SHAPE or arr.dtype != np.uint8:
                        raise ValidationError(f"Invalid reference array: {path}")
                except (FileNotFoundError, ValueError, ValidationError):
                    if not cfg["verify_hub"]:
                        raise
                    # Use the existing selective downloader, isolated from the supplied copy.
                    from opensportslib.tools import download_dataset_sample_inputs_from_hf
                    repair = out / "verified_inputs/reference" / split / "download.json"
                    payload = copy.deepcopy(ref)
                    payload.update(hf_repo_id=REFERENCE, hf_branch="frames", hf_split=split,
                                   hf_format="parquet", hf_commit=REFERENCE_REV)
                    write_json(repair, payload)
                    download_dataset_sample_inputs_from_hf(str(repair), item["sample"]["id"])
                    path = dataset.media(item, "frames_npy")
                    arr = np.load(path, mmap_mode="r", allow_pickle=False)
                    if arr.shape != SHAPE or arr.dtype != np.uint8:
                        raise ValidationError(f"Downloaded reference array is invalid: {path}")
                st = path.stat()
                old = cached.get(str(path), {})
                sha = old.get("sha256") if old.get("stat") == [st.st_size, st.st_mtime_ns] else None
                sha = sha or dataset.file_hash(path)
                report["media"].append({"path": str(path), "stat": [st.st_size, st.st_mtime_ns], "sha256": sha})
                if i % 5000 == 0:
                    print(f"audit {split}: {i}/{len(samples)} reference arrays", flush=True)
                    write_json(report_path, report)
            games = {x["game"]["game_id"]: x for x in samples}
            # Hash large source files with four bounded readers; decoding remains serial.
            def prehash(pair):
                item, kind = pair
                try:
                    path = dataset.media(item, kind)
                except FileNotFoundError:
                    return
                st = path.stat()
                old = cached.get(str(path), {})
                if old.get("stat") == [st.st_size, st.st_mtime_ns] and old.get("sha256"):
                    dataset.hashes[(str(path), st.st_size, st.st_mtime_ns)] = old["sha256"]
                else:
                    dataset.file_hash(path)
            with ThreadPoolExecutor(max_workers=4) as pool:
                for _ in pool.map(prehash, [(item, inp["type"]) for item in games.values()
                                            for inp in item["game"]["inputs"]]):
                    pass
            for game, item in games.items():
                for inp in item["game"]["inputs"]:
                    kind = inp["type"]
                    info = hub_files.get(inp["path"])
                    expected = info.lfs.sha256 if info and info.lfs else None
                    try:
                        path = dataset.media(item, kind)
                    except FileNotFoundError:
                        path = None
                    sha = None
                    if path:
                        st = path.stat()
                        old = cached.get(str(path), {})
                        sha = old.get("sha256") if old.get("stat") == [st.st_size, st.st_mtime_ns] else None
                        sha = sha or dataset.file_hash(path)
                    if path is None or (expected and sha != expected):
                        if not cfg["verify_hub"]:
                            raise FileNotFoundError(inp["path"])
                        downloaded = remote_file(SPOTTING, SPOTTING_REV, inp["path"], out / "hub_cache")
                        sha = digest(downloaded)
                        if not expected or sha != expected:
                            raise ValidationError(f"Hub checksum failure: {inp['path']}")
                        path = out / "verified_inputs/spotting" / split / inp["path"]
                        path.parent.mkdir(parents=True, exist_ok=True)
                        temporary = path.with_suffix(path.suffix + ".part")
                        shutil.copyfile(downloaded, temporary)
                        os.replace(temporary, path)
                    if cfg["verify_hub"] and not expected:
                        raise ValidationError(f"Missing Hub checksum for {inp['path']}")
                    st = path.stat()
                    report["media"].append({"path": str(path), "stat": [st.st_size, st.st_mtime_ns],
                                            "sha256": sha, "hub_sha256": expected})
                print(f"audit {split}: verified game {game}", flush=True)
                write_json(report_path, report)
            report["splits"][split] = len(samples)
        expected_counts = cfg.get("expected_counts")
        if expected_counts and report["splits"] != expected_counts:
            raise ValidationError("Dataset counts differ from configured release")
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        write_json(report_path, report)
    return report


def require_audit(dataset):
    path = dataset.out / "reports/audit.json"
    report = read_json(path) if path.exists() else {}
    if report.get("status") != "passed" or report.get("identity") not in dataset.compatible_identities:
        raise ValidationError("Run audit with this configuration before pilot/build")
    for record in report["media"]:
        path = Path(record["path"])
        st = path.stat()
        if [st.st_size, st.st_mtime_ns] != record["stat"]:
            raise ValidationError(f"Input changed since audit: {path}")
        dataset.hashes[(str(path), st.st_size, st.st_mtime_ns)] = record["sha256"]
    return report


def video_info(path):
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValidationError(f"Cannot open video {path}")
        result = {"opencv_fps": cap.get(cv2.CAP_PROP_FPS),
                  "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}
    finally:
        cap.release()
    with av.open(str(path)) as source:
        stream = source.streams.video[0]
        result.update(rate=str(stream.base_rate or stream.average_rate), time_base=str(stream.time_base),
                      start_pts=stream.start_time, width=stream.width, height=stream.height)
    if result["opencv_fps"] <= 0:
        raise ValidationError(f"Invalid video frame rate: {path}")
    return result


def frame_indices(position_ms, info, profile, stride):
    rate = info["opencv_fps"] if profile["clock"] == "source" else 30.0
    center = int(position_ms / 1000 * rate) + profile["offset"]
    indices = [center + (i - 8) * stride for i in range(16)]
    if indices[0] < 0 or indices[-1] + stride > info["frames"]:
        raise ValidationError("Full sampling window exceeds source video; padding is not permitted")
    return indices


def decode_samples(path, indices, profile):
    cap = cv2.VideoCapture(str(path))
    arrays = []
    methods = {"linear": cv2.INTER_LINEAR, "area": cv2.INTER_AREA,
               "cubic": cv2.INTER_CUBIC, "lanczos": cv2.INTER_LANCZOS4}
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, indices[0])
        wanted = set(indices)
        for index in range(indices[0], indices[-1] + 1):
            ok, frame = cap.read() if index in wanted else (cap.grab(), None)
            if not ok:
                raise ValidationError(f"Video decode failed at frame {index}: {path}")
            if index in wanted:
                if profile["color"] == "rgb":
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if profile["resize"] == "224p_linear":
                    frame = cv2.resize(frame, (398, 224))
                    interpolation = cv2.INTER_LINEAR
                else:
                    interpolation = methods[profile["resize"]]
                arrays.append(cv2.resize(frame, (224, 224), interpolation=interpolation))
    finally:
        cap.release()
    return np.stack(arrays)


def compare_arrays(actual, expected):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return {"exact": False, "error": "shape or dtype differs"}
    delta = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
    return {"exact": bool(np.array_equal(actual, expected)), "mae": float(delta.mean()),
            "max_error": int(delta.max()), "different_values": int(np.count_nonzero(delta)),
            "exact_frames": int(np.all(delta == 0, axis=(1, 2, 3)).sum())}


def profiles():
    # Keep the historical rule first, then bounded diagnostic alternatives.
    return [{"clock": clock, "offset": offset, "resize": resize, "color": color}
            for clock in ("source", "30") for offset in (0, -1, 1, -2, 2)
            for resize in ("linear", "area", "cubic", "lanczos", "224p_linear")
            for color in ("rgb", "bgr")]


def investigate(path, position_ms, info, candidates, stride, reference):
    """Decode each candidate source frame once; retain only resized pixels."""
    windows = [frame_indices(position_ms, info, p, stride) for p in candidates]
    wanted = sorted({index for window in windows for index in window})
    methods = {"linear": cv2.INTER_LINEAR, "area": cv2.INTER_AREA,
               "cubic": cv2.INTER_CUBIC, "lanczos": cv2.INTER_LANCZOS4}
    resized = {name: {} for name in (*methods, "224p_linear")}
    cap = cv2.VideoCapture(str(path))
    current = 0
    try:
        for index in wanted:
            if index - current > 64 or index < current:
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            else:
                for _ in range(index - current):
                    if not cap.grab():
                        raise ValidationError(f"Video decode failed before frame {index}")
            ok, frame = cap.read()
            if not ok:
                raise ValidationError(f"Video decode failed at frame {index}")
            current = index + 1
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            for name, method in methods.items():
                resized[name][index] = cv2.resize(frame, (224, 224), interpolation=method)
            resized["224p_linear"][index] = cv2.resize(cv2.resize(frame, (398, 224)), (224, 224))
    finally:
        cap.release()
    comparisons = []
    for profile, indices in zip(candidates, windows):
        actual = np.stack([resized[profile["resize"]][index] for index in indices])
        if profile["color"] == "bgr":
            actual = actual[..., ::-1]
        comparisons.append({"profile": profile, "indices": indices, **compare_arrays(actual, reference)})
    return comparisons


def encode_clip(source_path, destination, indices, stride, info, crf, preset="medium"):
    """Decode exact native indices and retain their PTS deltas in the new MP4.

    Recover indices using the measured clock, matching OpenCV's sampling rate.
    A rounded codec rate can drift by a fraction of a frame over a long game.
    Preserve the original PTS deltas; non-contiguous/ambiguous timing still fails.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".part.mp4")
    start, stop = indices[0], indices[-1] + stride
    rate = Fraction(info["rate"])
    expected_times = []
    try:
        with av.open(str(source_path)) as source, av.open(str(temporary), "w") as output:
            stream = source.streams.video[0]
            stream.codec_context.thread_count = 1
            tb = stream.time_base
            origin = stream.start_time or 0
            clock_rate = stream.average_rate or Fraction(str(info["opencv_fps"]))
            if clock_rate <= 0:
                raise ValidationError(f"Invalid measured frame rate: {source_path}")
            source.seek(origin + int(Fraction(start, 1) / clock_rate / tb), stream=stream, backward=True)
            encoder = output.add_stream("libx264", rate=rate)
            encoder.width, encoder.height = stream.width, stream.height
            encoder.pix_fmt = "yuv420p"
            encoder.time_base = tb
            encoder.codec_context.time_base = tb
            encoder.codec_context.thread_count = 2
            encoder.options = {"crf": str(crf), "preset": preset}
            first_pts = None
            next_index = start
            for frame in source.decode(stream):
                if frame.pts is None:
                    raise ValidationError("Video frame lacks PTS")
                index_float = (frame.pts - origin) * tb * clock_rate
                index = round(index_float)
                if abs(index_float - index) > Fraction(1, 4):
                    raise ValidationError(
                        f"Irregular source timing in {source_path}: pts={frame.pts}, "
                        f"measured_rate={clock_rate}, index={float(index_float):.6f}, "
                        f"window=[{start}, {stop})")
                if index < start:
                    continue
                if index >= stop:
                    break
                if index != next_index:
                    raise ValidationError(f"Non-contiguous source frames: expected {next_index}, got {index}")
                next_index += 1
                if first_pts is None:
                    first_pts = frame.pts
                relative_pts = frame.pts - first_pts
                expected_times.append(float(relative_pts * tb))
                frame.pts = relative_pts
                frame.time_base = tb
                for packet in encoder.encode(frame):
                    output.mux(packet)
            if next_index != stop:
                raise ValidationError("Source ended before full sampling window")
            for packet in encoder.encode():
                output.mux(packet)
        with av.open(str(temporary)) as output:
            stream = output.streams.video[0]
            stream.codec_context.thread_count = 1
            times = [float(f.pts * f.time_base) for f in output.decode(stream)]
            tolerance = max(float(stream.time_base), float(Fraction(info["time_base"]))) * 2
            if len(times) != stop - start or not np.allclose(times, expected_times, atol=tolerance, rtol=0):
                raise ValidationError("Encoded clip frame count or timestamps differ")
            if (stream.width, stream.height) != (info["width"], info["height"]):
                raise ValidationError("Encoded clip dimensions differ")
            duration = float(stream.duration * stream.time_base)
            encoded_average_rate = str(stream.average_rate)
            expected_duration = expected_times[-1] + float(1 / rate)
            if abs(duration - expected_duration) > max(tolerance, 0.001):
                raise ValidationError("Encoded clip duration differs")
        os.replace(temporary, destination)
        return {"start_frame": start, "end_frame_exclusive": stop, "frame_count": stop - start,
                "sampled_timestamps": expected_times[::stride],
                "timestamps_sha256": fingerprint(expected_times), "duration_seconds": duration,
                "source_start_pts": first_pts, "source_time_base": info["time_base"],
                "native_rate": info["rate"], "index_clock_rate": str(clock_rate),
                "encoded_average_rate": encoded_average_rate,
                "width": info["width"], "height": info["height"]}
    finally:
        temporary.unlink(missing_ok=True)


def temporal_profile():
    # Use the historical timestamp rule, never a per-clip best-pixel offset.
    return {"clock": "source", "offset": 0, "resize": "area", "color": "rgb"}


def output_kinds(cfg):
    if cfg.get("validation", "exact") == "temporal":
        return (("video", "video"),)
    return (("frames_npy", "frames"), ("video", "video"))


def validate_temporal_alignment(position_ms, indices, info, clip, stride):
    expected = frame_indices(position_ms, info, temporal_profile(), stride)
    if indices != expected or clip["frame_count"] != len(indices) * stride:
        raise ValidationError("Clip does not use the action's full native sampling window")
    start_seconds = (clip["source_start_pts"] - (info["start_pts"] or 0)) * Fraction(info["time_base"])
    action_offset = position_ms / 1000 - float(start_seconds)
    center_error = clip["sampled_timestamps"][8] - action_offset
    tolerance = 1 / info["opencv_fps"] + 2 * float(Fraction(info["time_base"]))
    if not 0 <= action_offset < clip["duration_seconds"] or abs(center_error) > tolerance:
        raise ValidationError(f"Action timestamp is not aligned to clip center: error={center_error:.6f}s")
    return {"passed": True, "action_offset_seconds": action_offset,
            "center_error_seconds": center_error, "tolerance_seconds": tolerance,
            "basis": "published action timestamp and historical 16-frame, stride-based sampling"}


def materialize(dataset, item, profile):
    sample, split = item["sample"], item["split"]
    video, ref_path = dataset.media(item, "video"), dataset.media(item, "frames_npy")
    info = dataset.video_info(video)
    temporal = dataset.cfg.get("validation", "exact") == "temporal"
    if temporal and profile != temporal_profile():
        raise ValidationError("Temporal mode requires the canonical source-timestamp profile")
    indices = frame_indices(sample["metadata"]["position_ms"], info, profile, dataset.cfg["frame_interval"])
    provenance = {"identity": dataset.identity, "profile": profile,
                  "validation": dataset.cfg.get("validation", "exact"),
                  "source_sha256": dataset.file_hash(video), "reference_sha256": dataset.file_hash(ref_path),
                  "spotting_revision": SPOTTING_REV, "reference_revision": REFERENCE_REV,
                  "source_path": str(video), "reference_sample_id": sample["id"],
                  "sampled_frame_indices": indices, "source_video": info}
    token = fingerprint(provenance)
    stem = Path(sample["inputs"][0]["path"]).stem
    frame_path = dataset.out / "frames" / split / f"{stem}.npy"
    video_path = dataset.out / "video" / split / f"{stem}.mp4"
    paths = [("video", video_path)] if temporal else [("frames_npy", frame_path), ("video", video_path)]
    receipt = dataset.out / "reports/completed" / split / f"{sample['id']}.json"
    if receipt.exists():
        try:
            old = read_json(receipt)
        except (ValueError, OSError):
            old = {}
        old_identity = old.get("entries", {}).get("video", {}).get("metadata", {}).get("conversion", {}).get("identity")
        compatible_token = fingerprint({**provenance, "identity": old_identity})
        if (old_identity in dataset.compatible_identities and old.get("token") == compatible_token
                and all(p.is_file() and digest(p) == old.get("hashes", {}).get(str(p)) for _, p in paths)):
            if old_identity != dataset.identity:
                for entry in old["entries"].values():
                    entry["metadata"]["conversion"].update(
                        identity=dataset.identity, reused_from_identity=old_identity)
                old["token"] = token
                write_json(receipt, old)
            return old
        # Do not leave a corrupt or stale completed artifact behind if retry fails.
        receipt.unlink()
        for _, path in paths:
            path.unlink(missing_ok=True)
    comparison = None
    if not temporal:
        expected = np.load(ref_path, allow_pickle=False)
        actual = decode_samples(video, indices, profile)
        comparison = compare_arrays(actual, expected)
        if not comparison["exact"]:
            raise ValidationError(f"Exact pixel reproduction failed for {sample['id']}: {comparison}")
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = frame_path.with_suffix(".npy.part")
        with temporary.open("wb") as handle:
            np.save(handle, actual, allow_pickle=False)
        os.replace(temporary, frame_path)
    clip = encode_clip(video, video_path, indices, dataset.cfg["frame_interval"], info,
                       dataset.cfg["crf"], dataset.cfg.get("preset", "medium"))
    provenance["clip"] = clip
    if temporal:
        try:
            provenance["alignment"] = validate_temporal_alignment(
                sample["metadata"]["position_ms"], indices, info, clip, dataset.cfg["frame_interval"])
        except Exception:
            video_path.unlink(missing_ok=True)
            raise
    entries = {}
    for kind, path in paths:
        entry = copy.deepcopy(sample)
        entry["inputs"] = [{"type": kind, "path": f"{split}/{path.name}"}]
        if kind == "video":
            entry["inputs"][0]["fps"] = float(Fraction(info["rate"]))
        entry["metadata"]["conversion"] = provenance
        entries[kind] = entry
    record = {"token": token, "entries": entries, "comparison": comparison,
              "hashes": {str(p): digest(p) for _, p in paths},
              "bytes": sum(p.stat().st_size for _, p in paths)}
    write_json(receipt, record)
    return record


def write_manifests(dataset, records, name="annotations"):
    """Stream receipt-backed entries instead of retaining the corpus in RAM."""
    from opensportslib.core.utils.load_annotations import load_annotations
    for split in SPLITS:
        selected = [(item, record) for item, record in records if item["split"] == split]
        for kind, directory in output_kinds(dataset.cfg):
            header = {k: v for k, v in dataset.headers[split].items() if not k.startswith("hf_")}
            header.update(modalities=[kind], dataset_name=f"soccernet_gar_{directory}_{split}")
            path = dataset.out / directory / f"{name}_{split}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".json.part")
            with temporary.open("w") as handle:
                handle.write(json.dumps(header)[:-1] + ', "data": [')
                for index, (_, record) in enumerate(selected):
                    if isinstance(record, Path):
                        record = read_json(record)
                    if index:
                        handle.write(",")
                    json.dump(record["entries"][kind], handle)
                handle.write("]}\n")
            loaded = load_annotations(str(temporary), input_type=kind)
            if len(loaded[0]) != len(selected):
                raise ValidationError(f"Classification loader lost samples: {path}")
            del loaded
            os.replace(temporary, path)


def pilot(dataset):
    require_audit(dataset)
    selected = select_pilot(dataset.samples, dataset.cfg["pilot_count"])
    temporal = dataset.cfg.get("validation", "exact") == "temporal"
    candidates = [temporal_profile()] if temporal else profiles()
    common = set(range(len(candidates)))
    report = {"identity": dataset.identity, "status": "running", "samples": [],
              "validation": dataset.cfg.get("validation", "exact"),
              "sample_ids": [x["sample"]["id"] for x in selected]}
    report_path = dataset.out / "reports/pilot.json"
    started = time.monotonic()
    try:
        for item in selected:
            sample = item["sample"]
            video = dataset.media(item, "video")
            ref = np.load(dataset.media(item, "frames_npy"), allow_pickle=False)
            info = dataset.video_info(video)
            comparisons = investigate(video, sample["metadata"]["position_ms"], info,
                                      candidates, dataset.cfg["frame_interval"], ref)
            matched = {i for i, result in enumerate(comparisons) if result["exact"]}
            common &= matched
            best = min(comparisons, key=lambda x: x.get("mae", float("inf")))
            report["samples"].append({"id": sample["id"], "split": item["split"],
                                      "game_id": sample["metadata"]["game_id"],
                                      "label": sample["labels"]["action"]["label"],
                                      "exact_profiles": sorted(matched), "best": best,
                                      "comparisons": comparisons})
            # Contact sheet: reference / closest regenerated frame / amplified absolute error.
            actual = decode_samples(video, best["indices"], best["profile"])
            rows = []
            for i in (0, 8, 15):
                error = np.minimum(np.abs(actual[i].astype(np.int16) - ref[i]) * 4, 255).astype(np.uint8)
                rows.append(np.concatenate([ref[i], actual[i], error], axis=1))
            image_path = dataset.out / "reports/comparisons" / f"{sample['id']}.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(image_path), cv2.cvtColor(np.concatenate(rows), cv2.COLOR_RGB2BGR))
            print(f"pilot {sample['id']}: {len(matched)} exact profiles; best MAE={best['mae']:.4f}", flush=True)
            write_json(report_path, report)
        if not temporal and not common:
            raise ValidationError("No single extraction profile exactly reproduces all pilot arrays. "
                                  "Original 224p source/preprocessing may be required. Full conversion is blocked.")
        profile = temporal_profile() if temporal else candidates[min(common)]
        # Preflight every action, including games not selected for the pilot.
        for item in dataset.samples:
            frame_indices(item["sample"]["metadata"]["position_ms"],
                          dataset.video_info(dataset.media(item, "video")), profile,
                          dataset.cfg["frame_interval"])
        report["preflight_samples"] = len(dataset.samples)
        conversion_started = time.monotonic()
        records = [(item, materialize(dataset, item, profile)) for item in selected]
        elapsed = time.monotonic() - conversion_started
        write_manifests(dataset, records, name="pilot")
        total_bytes = sum(r["bytes"] for _, r in records)
        report.update(status="passed", profile=profile,
                      estimated_total_bytes=int(total_bytes / len(records) * len(dataset.samples)),
                      estimated_seconds=elapsed / len(records) * len(dataset.samples),
                      timing_validated_clips=len(records),
                      pixel_equality_required=not temporal)
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(report_path, report)
    return report


def build(dataset):
    require_audit(dataset)
    path = dataset.out / "reports/pilot.json"
    gate = read_json(path) if path.exists() else {}
    if gate.get("status") != "passed" or gate.get("identity") not in dataset.compatible_identities:
        raise ValidationError("Full conversion requires a passing pilot for these exact inputs/settings")
    free = shutil.disk_usage(dataset.out).free
    existing_bytes = sum(p.stat().st_size for directory in ("frames", "video")
                         for p in (dataset.out / directory).rglob("*") if p.is_file())
    remaining = max(0, gate["estimated_total_bytes"] - existing_bytes)
    if free < remaining * 1.1:
        raise ValidationError("Insufficient free space for estimated output plus 10% margin")
    records = []
    report = {"identity": dataset.identity, "status": "running", "completed": 0}
    started = time.monotonic()
    try:
        workers = dataset.cfg.get("workers", 1)
        def convert(item):
            try:
                materialize(dataset, item, gate["profile"])
            except Exception as exc:
                raise ValidationError(f"{item['sample']['id']}: {exc}") from exc
            return item, dataset.out / "reports/completed" / item["split"] / f"{item['sample']['id']}.json"
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for start in range(0, len(dataset.samples), workers):
                # A bounded batch avoids queuing 87k futures and keeps manifest order stable.
                for item, receipt in pool.map(convert, dataset.samples[start:start + workers]):
                    records.append((item, receipt))
                    report["completed"] += 1
                if report["completed"] % 10 == 0:
                    report["elapsed_seconds"] = time.monotonic() - started
                    write_json(dataset.out / "reports/build.json", report)
                    print(f"build: {report['completed']}/{len(dataset.samples)}", flush=True)
        write_manifests(dataset, records)
        report.update(status="passed", splits=dict(Counter(x["split"] for x, _ in records)))
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(dataset.out / "reports/build.json", report)
    return report


def repair(dataset):
    """Remove incomplete/checksum-invalid generated outputs; retain verified clips.

    Run with no conversion process active. Source media is never modified.
    A completed receipt proves decode validation; its checksum proves the file
    has not changed since that validation, avoiding a second corpus-wide decode.
    """
    require_audit(dataset)
    report = {"identity": dataset.identity, "status": "running", "verified": 0,
              "removed": [], "invalid_samples": []}
    for index, item in enumerate(dataset.samples):
        sample, split = item["sample"], item["split"]
        stem = Path(sample["inputs"][0]["path"]).stem
        receipt = dataset.out / "reports/completed" / split / f"{sample['id']}.json"
        paths = [dataset.out / directory / split / (stem + (".mp4" if kind == "video" else ".npy"))
                 for kind, directory in output_kinds(dataset.cfg)]
        def remove(path):
            if path.is_file():
                path.unlink()
                report["removed"].append(str(path))
        for path in paths:
            remove(path.with_suffix(".part.mp4") if path.suffix == ".mp4" else path.with_suffix(".npy.part"))
        remove(receipt.with_suffix(".json.part"))
        record = {}
        if receipt.exists():
            try:
                record = read_json(receipt)
            except (ValueError, OSError):
                pass
        identity = record.get("entries", {}).get("video", {}).get("metadata", {}).get("conversion", {}).get("identity")
        valid = (identity in dataset.compatible_identities
                 and all(p.is_file() and digest(p) == record.get("hashes", {}).get(str(p)) for p in paths))
        if valid:
            report["verified"] += 1
        elif receipt.exists() or any(p.exists() for p in paths):
            report["invalid_samples"].append(sample["id"])
            for path in paths:
                remove(path)
            remove(receipt)
        if (index + 1) % 500 == 0:
            print(f"repair: checked {index + 1}/{len(dataset.samples)}, verified {report['verified']}, "
                  f"removed {len(report['removed'])} files", flush=True)
    report["status"] = "passed"
    write_json(dataset.out / "reports/repair.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "pilot", "build", "repair"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--then-build", action="store_true", help="After a passing pilot, run full conversion")
    args = parser.parse_args(argv)
    if args.then_build and args.command != "pilot":
        parser.error("--then-build is only valid with pilot")
    try:
        cfg = load_config(args.config)
        if args.command == "audit":
            prepare_manifests(cfg)
        dataset = Dataset(cfg)
        result = {"audit": audit, "pilot": pilot, "build": build, "repair": repair}[args.command](dataset)
        if args.then_build:
            result = build(dataset)
        print(json.dumps({k: v for k, v in result.items() if k not in ("media", "samples")}, indent=2))
        return 0
    except (ValidationError, FileNotFoundError) as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
