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

_TENNIS_SPEC = importlib.util.spec_from_file_location(
    "tennis_conversion", Path(__file__).parents[3] / "tools/convert/tennis_to_classification.py"
)
tennis = importlib.util.module_from_spec(_TENNIS_SPEC)
_TENNIS_SPEC.loader.exec_module(tennis)


@pytest.fixture
def dallas_dataset(tmp_path):
    import os
    from tests.helpers.video_pose import write_video
    from tools.convert import dallas_to_classification as dallas

    source = tmp_path / "dallas-source"
    source.mkdir()
    video = write_video(source / "source.mp4", count=400, width=64, height=48)
    records = []
    for names in dallas.MATCHES.values():
        for name in names:
            filename = f"{name}.mp4"
            os.link(video, source / filename)
            records.append({"id": name, "inputs": [{"type": "video", "path": filename}],
                            "events": [{"head": "swing", "label": "swing", "position_ms": ms}
                                       for ms in (3000, 10000)]})
    manifest = source / "swing.json"
    tennis.write_json(manifest, {"labels": {"swing": {"type": "single_label", "labels": ["swing"]}}, "data": records})
    return manifest


def test_dallas_binary_conversion_and_localization_keep_disjoint_original_events(dallas_dataset, tmp_path):
    from tools.convert import dallas_to_classification as dallas
    from tools.train.train_tennis_dallas import training_config

    root, localization = tmp_path / "classification", tmp_path / "localization"
    report = dallas.build(dallas_dataset, root, localization)
    assert report["split_counts"] == {"train": 12, "valid": 4, "test": 4}
    assert report["class_counts"]["train"] == {"background": 6, "swing": 6}
    original = json.loads(dallas_dataset.read_text())
    original_events = {record["id"]: record["events"] for record in original["data"]}
    for split, matches in dallas.MATCHES.items():
        local = json.loads((localization / f"{split}.json").read_text())
        assert [record["id"] for record in local["data"]] == matches
        for record in local["data"]:
            assert record["events"] == original_events[record["id"]]
            assert (localization / record["inputs"][0]["path"]).is_file()
        clips = json.loads((root / f"{split}.json").read_text())
        assert clips["labels"]["action"]["labels"] == ["background", "swing"]
        for clip in clips["data"]:
            if clip["labels"]["action"]["label"] == "background":
                conversion = clip["metadata"]["conversion"]
                assert clip["metadata"]["source_event"]["label_source"] == "inferred_from_unannotated_gap"
                assert all(not conversion["source_start_seconds"] - 2 < point < conversion["source_end_seconds"] + 2
                           for point in (3, 10))
    config = training_config(root, "skeleton", tmp_path / "pilot", epochs=5).get_config()
    assert config["TRAIN"]["epochs"] == 5
    assert config["MODEL"]["components"]["task_head"]["params"]["num_classes"] == 2
    assert config["MODEL"]["components"]["pose_tracking"]["params"]["cache"]["mode"] == "read_write"
    assert config["DATA"]["common"]["splits"]["valid"]["source_path"] == str(root)
    cpu = training_config(root, "skeleton", tmp_path / "cpu-pilot", device="cpu").get_config()
    assert cpu["SYSTEM"]["device"] == "cpu"
    assert cpu["MODEL"]["components"]["pose_tracking"]["params"]["cache"]["mode"] == "read"
    convergence = training_config(root, "skeleton", tmp_path / "convergence", epochs=100,
                                  patience=12, min_delta=.002).get_config()
    assert convergence["TRAIN"]["epochs"] == 100
    assert convergence["TRAIN"]["selection"] == {"monitor": "loss", "mode": "min", "patience": 12, "min_delta": .002}
    resumed = dallas.build(dallas_dataset, root, localization)
    assert resumed["encoded_clips"] == 0 and resumed["reused_clips"] == 20
    with pytest.raises(ValueError, match="epochs must be positive"):
        training_config(root, "skeleton", tmp_path / "pilot", epochs=0)
    with pytest.raises(ValueError, match="patience must be nonnegative"):
        training_config(root, "skeleton", tmp_path / "pilot", patience=-1)
    with pytest.raises(ValueError, match="min_delta must be finite"):
        training_config(root, "skeleton", tmp_path / "pilot", min_delta=float("nan"))
    with pytest.raises(ValueError, match="another conversion"):
        dallas.build(dallas_dataset, root, localization, guard_seconds=1.)


def test_dallas_background_sampler_rejects_insufficient_gaps_and_is_reproducible():
    from tools.convert.dallas_to_classification import background_centers

    first = background_centers(20., [3., 10.], count=3)
    assert first == background_centers(20., [3., 10.], count=3)
    assert min(abs(center-point) for center in first for point in (3., 10.)) >= 2.6
    with pytest.raises(ValueError, match="safe background"):
        background_centers(4., [1., 3.], count=2)


def test_dallas_spotters_keep_original_events_and_fixed_validation_requests(dallas_dataset, tmp_path):
    import random
    import torch
    from tools.convert.dallas_to_classification import build
    from tools.train.train_tennis_dallas_spotting import training_config, run
    from opensportslib.core.config.runtime_adapter import dict_to_namespace
    from opensportslib.datasets.builder import build_dataset

    root = tmp_path / "localization"
    build(dallas_dataset, tmp_path / "classification", root)
    requested = []
    training_requests = []
    for model in ("skeleton", "video"):
        payload = training_config(root, model, tmp_path / model, epochs=100, patience=12,
                                  min_delta=.002, epoch_num_frames=16, clip_len=4).get_config()
        assert payload["MODEL"]["task"] == "localization" and payload["DATA"]["common"]["classes"] == ["swing"]
        assert payload["TRAIN"]["selection"]["patience"] == 12
        assert payload["TRAIN"]["selection"]["min_delta"] == .002
        assert payload["TRAIN"]["execution"]["criterion_valid"] == "loss"
        assert payload["TRAIN"]["execution"]["training_seed"] == 42
        assert payload["TRAIN"]["scheduler"]["warm_up_epochs"] < payload["TRAIN"]["epochs"]
        for split in payload["DATA"]["common"]["splits"].values():
            assert split["results"] is None and split["overlap_len"] == 2
        # RGB validation transforms stay deterministic and small for this test.
        if model == "video":
            payload["DATA"]["inputs"]["video"]["transform"]["resize"] = {"height": 48, "width": 64}
            assert "pose_tracking" not in payload["MODEL"]["components"]
        else:
            cache = payload["MODEL"]["components"]["pose_tracking"]["params"]["cache"]
            assert cache == {"mode": "read_write", "path": str(root / "pose-cache-clips")}
            another = training_config(root, model, tmp_path / "another-run").get_config()
            assert another["MODEL"]["components"]["pose_tracking"]["params"]["cache"] == cache
            disabled = training_config(root, model, tmp_path / "disabled", cache_mode="off").get_config()
            assert disabled["MODEL"]["components"]["pose_tracking"]["params"]["cache"]["mode"] == "off"
        wrapper = build_dataset(dict_to_namespace(payload), split="valid")
        dataset = wrapper.building_dataset(wrapper.cfg, default_args=wrapper.default_args)
        def observe_request(original=dataset._sample_uniform):
            record, start = original()
            requested.append((record["video"], start))
            return record, start
        dataset._sample_uniform = observe_request
        state = random.getstate()
        first = dataset[0]
        assert random.getstate() == state
        random.random()
        second = dataset[0]
        torch.testing.assert_close(torch.as_tensor(first["frame"]), torch.as_tensor(second["frame"]))
        np.testing.assert_array_equal(first["label"], second["label"])
        wrapper = build_dataset(dict_to_namespace(payload), split="train")
        pool = wrapper.building_dataset(wrapper.cfg, default_args=wrapper.default_args)
        def observe_training(original=pool._sample_uniform):
            record, start = original()
            training_requests.append((record["video"], start))
            return record, start
        pool._sample_uniform = observe_training
        for index in (0, 1, 0):
            state = random.getstate()
            pool[index]
            assert random.getstate() == state
    assert len(requested) == 4 and all(request == requested[0] for request in requested)
    assert training_requests[:3] == training_requests[3:]
    assert training_requests[0] == training_requests[2]
    random_payload = training_config(root, "skeleton", tmp_path / "random",
                                     training_sampling="random").get_config()
    assert random_payload["TRAIN"]["execution"]["training_seed"] is None
    wrapper = build_dataset(dict_to_namespace(random_payload), split="train")
    random_pool = wrapper.building_dataset(wrapper.cfg, default_args=wrapper.default_args)
    state = random.getstate()
    random_pool[0]
    assert random.getstate() != state
    seeded = training_config(root, "skeleton", tmp_path / "seeded", sampling_seed=19).get_config()
    assert seeded["TRAIN"]["execution"]["training_seed"] == 19
    # Configuration-only mode validates local full videos without building a
    # GPU model or using the classification frame/pose caches.
    path = run(root, "skeleton", tmp_path / "config-only", config_only=True, epochs=1)
    assert path.is_file() and not (path.parent / "results.json").exists()
    with pytest.raises(ValueError, match="at least four clips"):
        training_config(root, "video", tmp_path / "bad", epoch_num_frames=64)
    with pytest.raises(ValueError, match="fixed or random"):
        training_config(root, "video", tmp_path / "bad", training_sampling="unsupported")
    document = json.loads((root / "valid.json").read_text())
    document["data"][0]["id"] = "Shelton_vs_Diallo_2_10_26"
    tennis.write_json(root / "valid.json", document)
    with pytest.raises(ValueError, match="three/one/one"):
        training_config(root, "skeleton", tmp_path / "bad-split")


def test_dallas_spotting_prediction_paths_and_fast_nms_preserve_ties(tmp_path):
    from tools.train.train_tennis_dallas_spotting import prepare_predictions
    from opensportslib.metrics.localization_metric import non_maximum_supression

    document = {"test": {"data": [{"inputs": [{"path": "test.mp4"}]}]}}
    raw = {"data": [{"inputs": [{"path": "test.mp4"}], "events": [
        {"head": "action", "label": "swing", "frame": i, "position_ms": i*40, "confidence": p}
        for i, p in enumerate([.8, .9, .9, .2, .7, .6, .8])]}]}
    expected = non_maximum_supression(raw["data"], 2)[0]["events"]
    result = prepare_predictions(raw, tmp_path, document, apply_nms=True)
    assert result["data"][0]["inputs"][0]["path"] == str(tmp_path / "test.mp4")
    assert [e["frame"] for e in result["data"][0]["events"]] == [e["frame"] for e in expected]
    assert all(e["head"] == "swing" for e in result["data"][0]["events"])
    assert len(raw["data"][0]["events"]) == 7  # No mutation of model results.
    with pytest.raises(ValueError, match="exactly once"):
        prepare_predictions({"data": []}, tmp_path, document, apply_nms=False)


def test_dallas_rgb_cache_keeps_clip_labels_and_uses_existing_video_dataset(dallas_dataset, tmp_path):
    from tools.convert.dallas_to_classification import build
    from tools.train.train_tennis_dallas import prepare_video_frames, training_config
    from opensportslib.datasets.builder import build_dataset
    from opensportslib.core.utils.config import resolve_config_omega

    root = tmp_path / "classification"
    build(dallas_dataset, root, tmp_path / "localization")
    frames = prepare_video_frames(root)
    pretrained = tmp_path / "pretrained"
    pretrained.mkdir()
    (pretrained / "model.safetensors").write_bytes(b"not loaded during configuration/data checks")
    config = training_config(root, "video", tmp_path / "pilot", pretrained_path=pretrained)
    payload = config.get_config()
    assert payload["TRAIN"]["execution"]["use_amp"] is True
    assert payload["MODEL"]["components"]["video_encoder"]["params"]["unfreeze_last_n_layers"] == 4
    dataset = build_dataset(resolve_config_omega(payload), str(frames / "valid.json"), None, split="valid")
    item = dataset[0]
    assert item["pixel_values"].shape == (16, 224, 224, 3)
    original = json.loads((root / "valid.json").read_text())["data"][0]
    assert item["id"] == original["id"]
    assert int(item["labels"]) == ["background", "swing"].index(original["labels"]["action"]["label"])


def test_dallas_publisher_check_only_writes_cards_and_never_contacts_hub(dallas_dataset, tmp_path, monkeypatch):
    from tools.convert.dallas_to_classification import build
    from tools.upload.upload_tennis_dallas import publish
    import huggingface_hub

    def forbidden(*args, **kwargs):
        raise AssertionError("check-only must be offline")

    root, localization = tmp_path / "classification", tmp_path / "localization"
    build(dallas_dataset, root, localization)
    monkeypatch.setattr(huggingface_hub, "HfApi", forbidden)
    publish(root, localization, check_only=True)
    assert len((root / "checksums.sha256").read_text().splitlines()) == 20
    card = (root / "README.md").read_text()
    assert "not manually verified negatives" in card
    assert "20 short video clips" in card and "10 annotated swings" in card
    assert "train/metadata.parquet" in card and "license:" not in card
    assert "only original swing events" in (localization / "README.md").read_text()


@pytest.fixture
def tennis_source_dataset(tmp_path):
    import os

    movie = tmp_path / "tennis.mp4"
    rate, tb = Fraction(30000, 1001), Fraction(1, 90000)
    with av.open(str(movie), "w") as output:
        encoder = output.add_stream("libx264", rate=rate)
        encoder.width, encoder.height = 64, 48
        encoder.pix_fmt = "yuv420p"
        encoder.time_base = encoder.codec_context.time_base = tb
        for index in range(48):
            frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), index * 3, np.uint8), format="rgb24")
            frame.pts, frame.time_base = 90000 + index * 3003, tb
            for packet in encoder.encode(frame):
                output.mux(packet)
        for packet in encoder.encode():
            output.mux(packet)
    root = tmp_path / "tennis-localization"
    for split, names in (("train", ("a", "b")), ("validation", ("a", "b")), ("test", ("t",))):
        directory = root / split
        directory.mkdir(parents=True)
        records = []
        for name in names:
            filename = f"{split}_{name}.mp4"
            os.link(movie, directory / filename)
            records.append({"id": f"{split}_{name}", "metadata": {"base_video": name},
                            "inputs": [{"type": "video", "path": filename}],
                            "events": [{"head": "tennis_action", "label": label, "position_ms": position}
                                       for label, position in zip(tennis.CLASSES, (0, 400, 800, 1500))]
                                      + [{"head": "tennis_action", "label": "far_court_bounce", "position_ms": 200}]})
        tennis.write_json(directory / f"{split}.json", {"task": "action_spotting",
                          "labels": {"tennis_action": {"labels": [*tennis.CLASSES, "far_court_bounce"]}}, "data": records})
    return root


def test_tennis_event_clips_preserve_timestamps_labels_and_match_splits(tennis_source_dataset, tmp_path):
    from opensportslib.core.utils.load_annotations import load_annotations

    output = tmp_path / "classification"
    report = tennis.build(tennis_source_dataset, output)
    assert report["status"] == "complete" and report["expected_clips"] == 20
    assert report["split_counts"] == {"train": 8, "valid": 8, "test": 4}
    assert not set(report["matches"]["train"]) & set(report["matches"]["valid"])
    assert not set(report["matches"]["test"]) & (set(report["matches"]["train"]) | set(report["matches"]["valid"]))
    for split in ("train", "valid", "test"):
        document = json.loads((output / f"{split}.json").read_text())
        assert document["labels"]["action"]["labels"] == list(tennis.CLASSES)
        loaded, labels = load_annotations(output / f"{split}.json", task_key="action")
        assert len(loaded) == report["split_counts"][split] and len(labels) == 4
        for sample in document["data"]:
            metadata = sample["metadata"]["conversion"]
            assert metadata["source_start_pts"] >= 90000
            assert Fraction(metadata["source_fps"]) == Fraction(30000, 1001)
            assert 0 <= metadata["event_offset_ms"] < metadata["duration_seconds"] * 1000
            assert metadata["duration_seconds"] <= 1.001
            assert metadata["width"] == 64 and metadata["height"] == 48
            with av.open(str(output / sample["inputs"][0]["path"])) as movie:
                times = [float(frame.pts * frame.time_base) for frame in movie.decode(video=0)]
                assert times[0] == 0 and len(times) == metadata["frame_count"]
                np.testing.assert_allclose(np.diff(times), float(Fraction(1001, 30000)), atol=2e-5, rtol=0)
            for other in (0, .4, .8, 1.5):
                if other != metadata["event_position_ms"] / 1000:
                    assert not metadata["source_start_seconds"] <= other < metadata["source_end_seconds"]


def test_tennis_conversion_resume_repairs_only_corrupt_clips(tennis_source_dataset, tmp_path):
    output = tmp_path / "classification"
    first = tennis.build(tennis_source_dataset, output)
    paths = sorted((output / "clips").glob("*/*.mp4"))
    mtimes = {path: path.stat().st_mtime_ns for path in paths}
    resumed = tennis.build(tennis_source_dataset, output)
    assert resumed["encoded_clips"] == 0 and resumed["reused_clips"] == first["expected_clips"]
    assert all(path.stat().st_mtime_ns == mtimes[path] for path in paths)
    paths[0].write_bytes(b"interrupted/corrupt video")
    repaired = tennis.build(tennis_source_dataset, output)
    assert repaired["encoded_clips"] == 1 and repaired["reused_clips"] == len(paths) - 1
    assert all(path.stat().st_mtime_ns == mtimes[path] for path in paths[1:])
    with pytest.raises(ValueError, match="another conversion"):
        tennis.build(tennis_source_dataset, output, before_seconds=.25)


def test_tennis_conversion_rejects_missing_media_and_invalid_targets(tennis_source_dataset, tmp_path):
    jobs, _ = tennis.load_sources(tennis_source_dataset)
    job = jobs[0]
    info = tennis.probe(job["source"])
    bad = copy.deepcopy(job)
    bad["events"][-1]["time"] = info["duration"] + 1
    with pytest.raises(ValueError, match="outside video"):
        tennis.plan_windows(bad, info, .5, .5, True)
    with pytest.raises(ValueError, match="separate"):
        tennis.build(tennis_source_dataset, tennis_source_dataset / "output")
    Path(job["source"]).unlink()
    with pytest.raises(ValueError, match="missing video"):
        tennis.load_sources(tennis_source_dataset)


def test_tennis_conversion_reports_explicitly_excluded_out_of_video_events(tennis_source_dataset, tmp_path):
    manifest = tennis_source_dataset / "test" / "test.json"
    document = json.loads(manifest.read_text())
    truncated = copy.deepcopy(document["data"][0])
    truncated["id"] = "truncated_source"
    truncated["events"] = [{"head": "tennis_action", "label": label, "position_ms": position}
                           for label, position in zip(tennis.CLASSES[:3], (2000, 2500, 3000))]
    document["data"].append(truncated)
    tennis.write_json(manifest, document)
    with pytest.raises(ValueError, match="outside video"):
        tennis.build(tennis_source_dataset, tmp_path / "strict")
    output = tmp_path / "classification"
    report = tennis.build(tennis_source_dataset, output, invalid_events="skip")
    assert report["status"] == "complete"
    assert report["requested_events"] == 23 and report["expected_clips"] == 20
    assert report["excluded_events"] == 3 and report["split_counts"]["test"] == 4
    excluded = json.loads((output / "reports" / "excluded_events.json").read_text())
    assert [event["event"]["position_ms"] for event in excluded] == [2000, 2500, 3000]
    assert all(event["source_record_id"] == "truncated_source"
               and event["reason"] == "event_outside_decoded_video"
               and event["event"]["position_ms"] > event["source_duration_seconds"] * 1000
               and len(event["source_video_sha256"]) == 64 for event in excluded)
    resumed = tennis.build(tennis_source_dataset, output, invalid_events="skip")
    assert resumed["encoded_clips"] == 0 and resumed["reused_clips"] == 20
    assert resumed["excluded_events"] == 3


def test_tennis_training_launcher_validates_converted_data_without_pose(tennis_source_dataset, tmp_path):
    output = tmp_path / "classification"
    tennis.build(tennis_source_dataset, output)
    spec = importlib.util.spec_from_file_location(
        "tennis_training", Path(__file__).parents[3] / "tools/train/train_tennis_skeleton.py"
    )
    training = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(training)
    config = training.training_config(output, epochs=1, cache_dir=tmp_path / "pose-cache").get_config()
    assert config["DATA"]["common"]["classes"] == list(tennis.CLASSES)
    assert config["MODEL"]["components"]["task_head"]["params"]["num_classes"] == 4
    assert config["DATA"]["common"]["splits"]["train"]["annotation_path"] == str(output / "train.json")
    assert config["TRAIN"]["epochs"] == 1 and not config["TRAIN"]["sampling"]["use_weighted_sampler"]
    assert not (tmp_path / "pose-cache").exists()
    document = json.loads((output / "valid.json").read_text())
    document["labels"]["action"]["labels"].reverse()
    tennis.write_json(output / "valid.json", document)
    with pytest.raises(ValueError, match="vocabulary"):
        training.training_config(output)


@pytest.fixture
def tennis_publisher(tennis_source_dataset, tmp_path, monkeypatch):
    from types import SimpleNamespace
    import opensportslib.tools.hf_transfer as transfer

    root = tmp_path / "classification"
    tennis.build(tennis_source_dataset, root)
    spec = importlib.util.spec_from_file_location(
        "tennis_publisher", Path(__file__).parents[3] / "tools/upload/upload_tennis_classification.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    remote = tmp_path / "remote"
    remote.mkdir()
    calls, failures = [], set()

    class FakeApi:
        def __init__(self, **kwargs):
            pass

        def create_repo(self, repo_id, **kwargs):
            calls.append(("repo", repo_id))
            return repo_id

        def list_repo_files(self, *args, **kwargs):
            return [path.relative_to(remote).as_posix() for path in remote.rglob("*") if path.is_file()]

        def create_commit(self, *, operations, **kwargs):
            names = [operation.path_in_repo for operation in operations]
            calls.append(("commit", names))
            if any(name.split("/")[0] in failures for name in names):
                raise RuntimeError("Upload interrupted")
            for operation in operations:
                path = remote / operation.path_in_repo
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(Path(operation.path_or_fileobj).read_bytes())
            return SimpleNamespace(oid=f"commit-{len(calls)}")

    monkeypatch.setattr(module, "HfApi", FakeApi)
    monkeypatch.setattr(transfer, "_import_hf_hub", lambda: (FakeApi, None, None))
    return module, root, remote, calls, failures


def test_tennis_publish_check_only_is_offline_and_renders_card(tennis_publisher):
    import yaml

    module, root, remote, calls, _ = tennis_publisher
    assert module.main(["--data-root", str(root), "--check-only"]) == 0
    assert calls == [] and not list(remote.iterdir())
    readme = (root / "README.md").read_text()
    header = yaml.safe_load(readme.split("---", 2)[1])
    assert header["license"] == "agpl-3.0"
    assert header["configs"][0]["data_files"][1] == {"split": "validation", "path": "valid/metadata.parquet"}
    assert "20 short broadcast-video clips" in readme and "**0 selected events**" in readme
    assert "| 0 | `far_court_serve` | 2 | 2 | 1 |" in readme


@pytest.mark.parametrize("failure", ["unfinished", "missing", "corrupt", "receipt", "exclusions"])
def test_tennis_publish_invalid_release_never_contacts_hub(tennis_publisher, failure):
    module, root, _, calls, _ = tennis_publisher
    if failure == "unfinished":
        report = json.loads((root / "reports/build.json").read_text())
        report["status"] = "running"
        tennis.write_json(root / "reports/build.json", report)
    elif failure == "exclusions":
        tennis.write_json(root / "reports/excluded_events.json", [{"unexpected": "event"}])
    elif failure == "receipt":
        path = next((root / "receipts").glob("*.json"))
        receipt = json.loads(path.read_text())
        receipt["output_sha256"] = "changed"
        tennis.write_json(path, receipt)
    else:
        path = next((root / "clips").glob("*/*.mp4"))
        if failure == "missing":
            path.unlink()
        else:
            path.write_bytes(b"corrupt MP4")
    assert module.main(["--data-root", str(root)]) == 1
    assert calls == []


def test_tennis_publish_uses_real_osl_shards_then_publishes_card(tennis_publisher):
    from opensportslib.tools import convert_parquet_to_json

    module, root, remote, calls, _ = tennis_publisher
    assert module.main(["--data-root", str(root), "--shard-size", "100KB"]) == 0
    assert calls[0] == ("repo", module.REPO_ID)
    assert [call[1][0].split("/")[0] for call in calls[1:]] == ["train", "valid", "test", "README.md"]
    assert (remote / "README.md").read_text() == (root / "README.md").read_text()
    report = json.loads((remote / "reports/conversion.json").read_text())
    assert report["source_commit"] == module.SOURCE_COMMIT and "source_root" not in report
    assert len((remote / "checksums.sha256").read_text().splitlines()) == 20
    assert not list(remote.rglob("receipts"))
    for split in module.SPLITS:
        output = root.parent / "roundtrip" / split
        output.mkdir(parents=True)
        manifest = output / f"{split}.json"
        convert_parquet_to_json(remote / split, manifest, extract_media=True, output_media_root=output)
        original = json.loads((root / f"{split}.json").read_text())
        restored = json.loads(manifest.read_text())
        assert restored["data"] == original["data"]
        for sample in restored["data"]:
            path = sample["inputs"][0]["path"]
            assert tennis.digest(output / path) == tennis.digest(root / path)


def test_tennis_publish_failure_does_not_publish_readme(tennis_publisher):
    module, root, remote, calls, failures = tennis_publisher
    failures.add("valid")
    assert module.main(["--data-root", str(root)]) == 1
    assert [call[1][0].split("/")[0] for call in calls[1:]] == ["train", "valid"]
    assert not (remote / "README.md").exists() and not (remote / "test").exists()


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
