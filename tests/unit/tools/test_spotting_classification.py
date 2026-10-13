"""Offline contract checks for strict spotting-to-classification conversion."""
import copy
import importlib.util
import json
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "spotting_conversion", Path(__file__).parents[3] / "tools/convert/spotting_to_classification.py"
)
conversion = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(conversion)


@pytest.fixture
def source_dataset(tmp_path):
    movie = tmp_path / "source.mp4"
    rate = Fraction(30000, 1001)
    with av.open(str(movie), "w") as container:
        stream = container.add_stream("libx264", rate=rate)
        stream.width, stream.height = 64, 48
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, 90000)
        stream.codec_context.time_base = stream.time_base
        yy, xx = np.indices((48, 64))
        for index in range(350):
            pixels = np.stack(((xx * 3 + index) % 256, (yy * 4 + index * 2) % 256,
                               (xx + yy + index * 3) % 256), axis=-1).astype(np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = 1890 + index * 3003
            frame.time_base = stream.time_base
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    cap = cv2.VideoCapture(str(movie))
    fps = cap.get(cv2.CAP_PROP_FPS)
    decoded = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        decoded.append(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (224, 224)))
    cap.release()
    labels = {"action": {"type": "single_label", "labels": ["PASS", "HEADER"]}}
    for split in conversion.SPLITS:
        spotdir = tmp_path / "spotting" / split
        refdir = tmp_path / "reference" / split
        (spotdir / split / "video").mkdir(parents=True)
        (spotdir / split / "video/game.mp4").symlink_to(movie)
        (refdir / split).mkdir(parents=True)
        entries, events = [], []
        for index, (position, label) in enumerate(((5000, "PASS"), (7000, "HEADER"))):
            center = int(position / 1000 * fps)
            frames = np.stack([decoded[center + (i - 8) * 9] for i in range(16)])
            np.save(refdir / split / f"clip_{index:06d}.npy", frames)
            entries.append({"id": f"{split}_{index:06d}", "inputs": [{"type": "frames_npy",
                            "path": f"{split}/clip_{index:06d}.npy"}],
                            "labels": {"action": {"label": label}},
                            "metadata": {"game_id": "game", "position_ms": position,
                                         "team": "home", "game_time": "1 - 00:05"}})
            events.append({"position_ms": position, "label": label, "team": "home", "gameTime": "1 - 00:05"})
        conversion.write_json(refdir / f"{split}.json", {"version": "2.0", "task": "action_classification",
                               "labels": labels, "modalities": ["frames_npy"], "data": entries})
        conversion.write_json(spotdir / f"{split}.json", {"labels": labels, "data": [{"game_id": "game",
                               "inputs": [{"type": "video", "path": f"{split}/video/game.mp4"}], "events": events}]})
    return {"spotting_root": str(tmp_path / "spotting"), "reference_root": str(tmp_path / "reference"),
            "output_root": str(tmp_path / "output"), "verify_hub": False, "pilot_count": 6,
            "frame_interval": 9, "crf": 18}, movie


def test_matching_requires_unique_complete_events_and_metadata(source_dataset):
    cfg, _ = source_dataset
    sp = conversion.read_json(conversion.manifest_path(cfg["spotting_root"], "train"))
    ref = conversion.read_json(conversion.manifest_path(cfg["reference_root"], "train"))
    assert len(conversion.match_samples(sp, ref, "train")) == 2
    wrong = copy.deepcopy(ref)
    wrong["data"][0]["metadata"]["team"] = "away"
    with pytest.raises(conversion.ValidationError, match="metadata differs"):
        conversion.match_samples(sp, wrong, "train")
    sp["data"][0]["events"].append(sp["data"][0]["events"][0])
    with pytest.raises(conversion.ValidationError, match="Duplicate"):
        conversion.match_samples(sp, ref, "train")


def test_path_layouts_missing_and_unsafe(tmp_path):
    file = tmp_path / "train/video.mp4"
    file.parent.mkdir()
    file.write_bytes(b"a")
    assert conversion.resolve_media(tmp_path, tmp_path / "train.json", "train/video.mp4") == file
    assert conversion.resolve_media(tmp_path.parent, tmp_path / "train.json", "train/video.mp4") == file
    with pytest.raises(FileNotFoundError, match="Missing"):
        conversion.resolve_media(tmp_path, tmp_path / "train.json", "absent.mp4")
    with pytest.raises(conversion.ValidationError, match="Unsafe"):
        conversion.resolve_media(tmp_path, tmp_path / "train.json", "../outside.mp4")


def test_fractional_sampling_and_full_window_boundaries():
    info = {"opencv_fps": 30000 / 1001, "frames": 1000}
    profile = {"clock": "source", "offset": 0}
    indices = conversion.frame_indices(5000, info, profile, 9)
    assert indices == list(range(77, 213, 9))
    assert indices[-1] + 9 - indices[0] == 144
    with pytest.raises(conversion.ValidationError, match="exceeds"):
        conversion.frame_indices(0, info, profile, 9)
    with pytest.raises(conversion.ValidationError, match="exceeds"):
        conversion.frame_indices(33000, info, profile, 9)


def test_exactness_is_not_perceptual():
    ref = np.zeros(conversion.SHAPE, dtype=np.uint8)
    candidate = ref.copy()
    candidate[0, 0, 0, 0] = 1
    result = conversion.compare_arrays(candidate, ref)
    assert not result["exact"]
    assert result["different_values"] == 1
    assert result["exact_frames"] == 15


def test_native_mp4_retains_source_offset_and_fractional_timing(source_dataset, tmp_path):
    _, movie = source_dataset
    info = conversion.video_info(movie)
    assert info["start_pts"] > 0
    assert Fraction(info["rate"]) == Fraction(30000, 1001)
    destination = tmp_path / "clip.mp4"
    indices = list(range(77, 213, 9))
    result = conversion.encode_clip(movie, destination, indices, 9, info, 18)
    assert result["frame_count"] == 144
    assert result["duration_seconds"] == pytest.approx(144 * 1001 / 30000, abs=0.001)
    assert result["sampled_timestamps"][1] == pytest.approx(9 * 1001 / 30000)
    with av.open(str(destination)) as container:
        assert not container.streams.audio
        assert len(list(container.decode(video=0))) == 144
    assert not destination.with_suffix(".part.mp4").exists()


def test_audit_pilot_build_and_resume_use_real_media(source_dataset):
    cfg, _ = source_dataset
    dataset = conversion.Dataset(cfg)
    assert conversion.audit(dataset)["status"] == "passed"
    pilot = conversion.pilot(dataset)
    assert pilot["status"] == "passed"
    assert conversion.build(dataset)["completed"] == 6
    item = dataset.samples[0]
    record = conversion.materialize(dataset, item, pilot["profile"])
    output = next(Path(p) for p in record["hashes"] if p.endswith(".mp4"))
    timestamp = output.stat().st_mtime_ns
    assert conversion.materialize(dataset, item, pilot["profile"]) == record
    assert output.stat().st_mtime_ns == timestamp
    output.write_bytes(b"interrupted output")
    repaired = conversion.materialize(dataset, item, pilot["profile"])
    assert conversion.digest(output) == repaired["hashes"][str(output)]
    npy = next(Path(p) for p in record["hashes"] if p.endswith(".npy"))
    assert np.array_equal(np.load(npy), np.load(dataset.media(item, "frames_npy")))


def test_mismatch_blocks_pilot_and_full_build(source_dataset):
    cfg, _ = source_dataset
    dataset = conversion.Dataset(cfg)
    path = dataset.media(dataset.samples[0], "frames_npy")
    reference = np.load(path)
    reference[0, 0, 0, 0] ^= 127
    np.save(path, reference)
    conversion.audit(dataset)
    with pytest.raises(conversion.ValidationError, match="No single extraction profile"):
        conversion.pilot(dataset)
    assert conversion.read_json(dataset.out / "reports/pilot.json")["status"] == "failed"
    with pytest.raises(conversion.ValidationError, match="passing pilot"):
        conversion.build(dataset)
    assert not list((dataset.out / "video").rglob("*.mp4"))


def test_modified_input_invalidates_audit(source_dataset):
    cfg, _ = source_dataset
    dataset = conversion.Dataset(cfg)
    conversion.audit(dataset)
    path = dataset.media(dataset.samples[0], "frames_npy")
    path.write_bytes(b"changed")
    with pytest.raises(conversion.ValidationError, match="changed since audit"):
        conversion.require_audit(dataset)


def test_manifest_repair_is_isolated_and_media_can_still_resolve(source_dataset, monkeypatch, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    cfg, _ = source_dataset
    cfg["verify_hub"] = True
    remote = {}
    for split in conversion.SPLITS:
        sp = conversion.manifest_path(cfg["spotting_root"], split)
        rp = conversion.manifest_path(cfg["reference_root"], split)
        reference = conversion.read_json(rp)
        header = {k: v for k, v in reference.items() if k != "data"}
        metadata = tmp_path / f"{split}.parquet"
        pq.write_table(pa.Table.from_pylist([{"header": json.dumps(header), "sample_payload": json.dumps(x)}
                                          for x in reference["data"]]), metadata)
        remote[(conversion.SPOTTING, f"{split}.json")] = sp
        remote[(conversion.REFERENCE, f"{split}/metadata.parquet")] = metadata
    stale = conversion.manifest_path(cfg["reference_root"], "train")
    old = conversion.read_json(stale)
    old["data"][0]["labels"]["action"]["label"] = "WRONG"
    conversion.write_json(stale, old)
    before = stale.read_bytes()
    monkeypatch.setattr(conversion, "remote_file", lambda repo, revision, filename, cache: remote[(repo, filename)])
    conversion.prepare_manifests(cfg)
    assert stale.read_bytes() == before
    dataset = conversion.Dataset(cfg)
    assert dataset.samples[0]["sample"]["labels"]["action"]["label"] == "PASS"
    assert dataset.media(dataset.samples[0], "frames_npy").is_file()


def test_gate_rejects_changed_configuration(source_dataset):
    cfg, _ = source_dataset
    original = conversion.Dataset(cfg)
    conversion.audit(original)
    changed = conversion.Dataset({**cfg, "crf": 20})
    with pytest.raises(conversion.ValidationError, match="Run audit"):
        conversion.require_audit(changed)


def test_pilot_selection_is_repeatable_and_covers_boundaries(source_dataset):
    cfg, _ = source_dataset
    dataset = conversion.Dataset(cfg)
    selection = conversion.select_pilot(dataset.samples, 6)
    assert selection == conversion.select_pilot(dataset.samples, 6)
    assert {x["split"] for x in selection} == set(conversion.SPLITS)
    assert {x["sample"]["labels"]["action"]["label"] for x in selection} == {"PASS", "HEADER"}
    with pytest.raises(conversion.ValidationError, match="cannot cover"):
        conversion.select_pilot(dataset.samples, 2)


def test_temporal_mode_accepts_pixel_differences_and_creates_video_only(source_dataset):
    cfg, _ = source_dataset
    cfg.update(validation="temporal", workers=2, preset="veryfast")
    dataset = conversion.Dataset(cfg)
    reference_path = dataset.media(dataset.samples[0], "frames_npy")
    reference = np.load(reference_path)
    reference[0, 0, 0, 0] ^= 127
    np.save(reference_path, reference)
    conversion.audit(dataset)
    pilot = conversion.pilot(dataset)
    assert pilot["status"] == "passed"
    assert not pilot["pixel_equality_required"]
    assert pilot["preflight_samples"] == 6
    assert not pilot["samples"][0]["best"]["exact"]
    assert conversion.build(dataset)["completed"] == 6
    assert not (dataset.out / "frames").exists()
    for split in conversion.SPLITS:
        manifest = conversion.read_json(dataset.out / "video" / f"annotations_{split}.json")
        expected = [x["sample"] for x in dataset.samples if x["split"] == split]
        assert [x["id"] for x in manifest["data"]] == [x["id"] for x in expected]
        for entry, original in zip(manifest["data"], expected):
            assert entry["labels"] == original["labels"]
            assert entry["metadata"]["position_ms"] == original["metadata"]["position_ms"]
            meta = entry["metadata"]["conversion"]
            assert meta["reference_sample_id"] == original["id"]
            assert meta["alignment"]["passed"]
            assert meta["clip"]["frame_count"] == 144
            assert meta["clip"]["width"] == 64
            assert (dataset.out / "video" / entry["inputs"][0]["path"]).is_file()


def test_temporal_mode_rejects_time_shift_even_when_pixel_checks_are_disabled(source_dataset, tmp_path):
    cfg, movie = source_dataset
    cfg["validation"] = "temporal"
    dataset = conversion.Dataset(cfg)
    shifted = {**conversion.temporal_profile(), "offset": -1}
    with pytest.raises(conversion.ValidationError, match="canonical source-timestamp profile"):
        conversion.materialize(dataset, dataset.samples[0], shifted)
    info = conversion.video_info(movie)
    indices = conversion.frame_indices(5000, info, conversion.temporal_profile(), 9)
    clip = conversion.encode_clip(movie, tmp_path / "timed.mp4", indices, 9, info, 18)
    assert conversion.validate_temporal_alignment(5000, indices, info, clip, 9)["passed"]
    clip["source_start_pts"] += int(1 / Fraction(info["time_base"]))
    with pytest.raises(conversion.ValidationError, match="not aligned"):
        conversion.validate_temporal_alignment(5000, indices, info, clip, 9)


_TRACK_SPEC = importlib.util.spec_from_file_location(
    "tracking_conversion", Path(__file__).parents[3] / "tools/convert/spotting_tracking_to_classification.py"
)
tracking_conversion = importlib.util.module_from_spec(_TRACK_SPEC)
_TRACK_SPEC.loader.exec_module(tracking_conversion)


@pytest.fixture
def tracking_source_dataset(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    frame_numbers = [i for i in range(1000, 1300) if i not in (1005, 1006, 1120)]
    source = pa.table({
        "frameNum": pa.array(frame_numbers, type=pa.int32()),
        "videoTimeMs": pa.array([i * 1000 / 29.97 for i in frame_numbers], type=pa.float64()),
        "period": pa.array([1] * len(frame_numbers), type=pa.int32()),
        "balls": ['[{"x": 1, "y": 2, "z": 0.5}]'] * len(frame_numbers),
        "homePlayers": ['[{"jerseyNum": 1, "x": 5, "y": 2}]'] * len(frame_numbers),
        "awayPlayers": ["[]"] * len(frame_numbers),
        "extra_column": pa.array(frame_numbers, type=pa.int16()),
    })
    labels = {"action": {"type": "single_label", "labels": ["PASS", "HEADER"]}}
    for split in conversion.SPLITS:
        sp = tmp_path / "spotting" / split
        rp = tmp_path / "reference" / split
        (sp / split).mkdir(parents=True)
        (rp / split).mkdir(parents=True)
        pq.write_table(source, sp / split / "game.parquet")
        entries, events = [], []
        for index, center in enumerate((1000, 1150)):
            wanted = [center + (i - 8) * 9 for i in range(16)]
            times = [i * 1000 / 29.97 if i in frame_numbers else None for i in wanted]
            # Older reference-only columns are not injected into the new source data.
            ref = pa.table({"frameNum": wanted, "videoTimeMs": pa.array(times, type=pa.float64()),
                            "video_url": ["old-reference-field"] * 16})
            relative = f"{split}/clip_{index:06d}.parquet"
            pq.write_table(ref, rp / relative)
            position = round(center * 1000 / 29.97)
            label = ("PASS", "HEADER")[index]
            entries.append({"id": f"{split}_{index:06d}", "inputs": [{"type": "tracking_parquet", "path": relative}],
                            "labels": {"action": {"label": label}}, "metadata": {"game_id": "game",
                            "position_ms": position, "team": "home", "game_time": "1 - 00:01"}})
            events.append({"position_ms": position, "label": label, "team": "home", "gameTime": "1 - 00:01"})
        conversion.write_json(rp / f"{split}.json", {"version": "2.0", "task": "action_classification",
                               "labels": labels, "modalities": ["tracking_parquet"], "data": entries})
        conversion.write_json(sp / f"{split}.json", {"labels": labels, "data": [{"game_id": "game",
                               "inputs": [{"type": "tracking_parquet", "path": f"{split}/game.parquet"}], "events": events}]})
    cfg = {"spotting_root": str(tmp_path / "spotting"), "reference_root": str(tmp_path / "reference"),
           "output_root": str(tmp_path / "output"), "gap_policy": "pad", "reference_frames": 16,
           "frame_interval": 9, "pilot_count": 6, "alignment_tolerance_ms": 10.0, "timestamp_tolerance_ms": 1.0}
    return cfg, source


def test_tracking_expands_native_frames_and_marks_gaps(tracking_source_dataset):
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    from opensportslib.datasets.utils.tracking import parse_frame, MISSING_VALUE
    cfg, source = tracking_source_dataset
    t = tracking_conversion
    dataset = t.Dataset(cfg)
    assert t.audit(dataset)["status"] == "passed"
    assert t.pilot(dataset)["status"] == "passed"
    assert t.build(dataset)["completed"] == 6
    output = pq.read_table(dataset.out / "tracking/train/clip_000000.parquet")
    assert output.num_rows == 144
    assert output["frameNum"].to_pylist() == list(range(928, 1072))
    assert output["tracking_valid"].to_pylist().count(False) == 74
    padded = output.slice(0, 1).to_pandas().iloc[0]
    assert np.isnan(padded.videoTimeMs)
    assert padded.period == -1 and padded.balls == "[]"
    features, _ = parse_frame(padded)
    assert np.all(features == MISSING_VALUE)
    real = output.filter(output["tracking_valid"]).drop(["tracking_valid"])
    expected = source.filter(pc.less(source["frameNum"], 1072))
    assert real.equals(expected, check_metadata=False)
    assert output.schema.field("extra_column").type == source.schema.field("extra_column").type
    assert "video_url" not in output.column_names
    manifest = t.read_json(dataset.out / "tracking/annotations_train.json")
    entry = manifest["data"][0]
    assert entry["id"] == "train_000000"
    assert entry["inputs"][0]["fps"] == 29.97
    assert entry["metadata"]["conversion"]["center_frame"] == 1000


def test_tracking_observed_mode_does_not_compact_time_or_fabricate_rows(tracking_source_dataset):
    import pyarrow.parquet as pq
    cfg, source = tracking_source_dataset
    cfg["gap_policy"] = "observed"
    t = tracking_conversion
    dataset = t.Dataset(cfg)
    t.audit(dataset)
    t.pilot(dataset)
    output = pq.read_table(dataset.out / "tracking/train/clip_000000.parquet")
    assert output.num_rows == 70
    assert all(output["tracking_valid"].to_pylist())
    assert output["frameNum"].to_pylist()[:7] == [1000, 1001, 1002, 1003, 1004, 1007, 1008]
    assert output["videoTimeMs"][0].as_py() == source["videoTimeMs"][0].as_py()


def test_tracking_resume_rebuilds_corrupt_outputs(tracking_source_dataset):
    cfg, _ = tracking_source_dataset
    t = tracking_conversion
    dataset = t.Dataset(cfg)
    t.audit(dataset)
    t.pilot(dataset)
    item = dataset.samples[0]
    record = t.read_json(dataset.receipt(item))
    output = Path(record["path"])
    before = output.stat().st_mtime_ns
    list(t.convert_items(dataset, [item]))
    assert output.stat().st_mtime_ns == before
    output.write_bytes(b"partial output")
    records = list(t.convert_items(dataset, [item]))
    assert t.digest(output) == records[0][1]["sha256"]


def test_tracking_rejects_clock_mismatch_and_missing_observed_reference(tracking_source_dataset):
    import pyarrow as pa
    cfg, source = tracking_source_dataset
    t = tracking_conversion
    frames, times, _ = t.source_clock(source)
    wrong_time = pa.table({"frameNum": [1000], "videoTimeMs": [1.0]})
    with pytest.raises(t.ValidationError, match="timestamps differ"):
        t.validate_reference(wrong_time, frames, times, cfg)
    missing = pa.table({"frameNum": [1005], "videoTimeMs": [1005 * 1000 / 29.97]})
    with pytest.raises(t.ValidationError, match="absent"):
        t.validate_reference(missing, frames, times, cfg)
    duplicate = pa.table({"frameNum": [1, 1], "videoTimeMs": [0.0, 33.3]})
    with pytest.raises(t.ValidationError, match="unique and increasing"):
        t.source_clock(duplicate)
    malformed = pa.table({"frameNum": list(range(16)), "videoTimeMs": list(range(16))})
    with pytest.raises(t.ValidationError, match="sampled-frame schedule"):
        t.reference_window(malformed, 8, cfg)


def test_tracking_stale_configuration_and_failed_pilot_block_build(tracking_source_dataset):
    cfg, _ = tracking_source_dataset
    t = tracking_conversion
    dataset = t.Dataset(cfg)
    t.audit(dataset)
    with pytest.raises(t.ValidationError, match="passing pilot"):
        t.build(dataset)
    changed = t.Dataset({**cfg, "gap_policy": "observed"})
    with pytest.raises(t.ValidationError, match="Run audit"):
        t.require_audit(changed)


def test_declared_rate_drift_uses_measured_clock_and_retains_pts(source_dataset, tmp_path):
    _, movie = source_dataset
    info = conversion.video_info(movie)
    # Simulate a rounded declared codec rate. At source frame 300 its clock
    # differs by 0.3 frames, beyond the previous quarter-frame rejection.
    info["rate"] = "30"
    indices = list(range(180, 316, 9))
    result = conversion.encode_clip(movie, tmp_path / "drift.mp4", indices, 9, info, 18)
    assert result["frame_count"] == 144
    assert result["sampled_timestamps"][8] == pytest.approx(72 * 1001 / 30000, abs=0.00003)
    assert result["source_start_pts"] == info["start_pts"] + 180 * 3003
    with av.open(str(tmp_path / "drift.mp4")) as output:
        assert Fraction(result["encoded_average_rate"]) == output.streams.video[0].average_rate


def test_legacy_resume_and_repair_preserve_valid_outputs(source_dataset, monkeypatch):
    cfg, _ = source_dataset
    cfg["validation"] = "temporal"
    legacy = conversion.Dataset(cfg)
    legacy.identity = next(x for x in legacy.compatible_identities if x != legacy.identity)
    conversion.audit(legacy)
    # Build a legacy receipt using the identical sampling/encoding contract.
    good = conversion.materialize(legacy, legacy.samples[0], conversion.temporal_profile())
    bad = conversion.materialize(legacy, legacy.samples[1], conversion.temporal_profile())
    good_path = next(Path(p) for p in good["hashes"])
    bad_path = next(Path(p) for p in bad["hashes"])
    stamp = good_path.stat().st_mtime_ns
    bad_path.write_bytes(b"corrupt")
    orphan = legacy.out / "video/valid/clip_000000.mp4"
    orphan.parent.mkdir(parents=True, exist_ok=True)
    orphan.write_bytes(b"interrupted without receipt")
    partial = good_path.with_suffix(".part.mp4")
    partial.write_bytes(b"interrupted encoding")
    current = conversion.Dataset(cfg)
    result = conversion.repair(current)
    assert result["verified"] == 1
    assert not bad_path.exists()
    assert not partial.exists()
    # Use the actual input stem for the unreceipted-output invariant below.
    item = current.samples[2]
    orphan = current.out / "video" / item["split"] / (Path(item["sample"]["inputs"][0]["path"]).stem + ".mp4")
    orphan.parent.mkdir(parents=True, exist_ok=True)
    orphan.write_bytes(b"interrupted without receipt")
    conversion.repair(current)
    assert not orphan.exists()
    def no_encode(*args, **kwargs):
        pytest.fail("Valid legacy clip should be reused without encoding")
    monkeypatch.setattr(conversion, "encode_clip", no_encode)
    reused = conversion.materialize(current, current.samples[0], conversion.temporal_profile())
    assert good_path.stat().st_mtime_ns == stamp
    provenance = reused["entries"]["video"]["metadata"]["conversion"]
    assert provenance["identity"] == current.identity
    assert provenance["reused_from_identity"] == legacy.identity
    # An unrelated converter identity never satisfies the audit gate.
    audit = conversion.read_json(current.out / "reports/audit.json")
    audit["identity"] = "unknown-converter"
    conversion.write_json(current.out / "reports/audit.json", audit)
    with pytest.raises(conversion.ValidationError, match="Run audit"):
        conversion.require_audit(current)
