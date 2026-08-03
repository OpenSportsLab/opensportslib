import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from omegaconf import OmegaConf

from opensportslib.datasets.localization_dataset import LocalizationDataset
from opensportslib.models.base.rule_based import (
    H5HeaderDistanceAngleSpotter,
    H5HeaderDistanceSpeedAngleSpotter,
    H5HeaderDistanceSpeedSpotter,
    H5HeaderDistanceSpotter,
    H5HeaderSpotter,
)
from opensportslib.models.builder import build_model
from opensportslib.core.trainer.localization_trainer import build_inferer
from opensportslib.apis.localization import LocalizationModel


def _bytes(values):
    return np.asarray([value.encode("utf-8") for value in values], dtype="S26")


def _write_ball(path: Path):
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "timestamp_utc",
            data=_bytes([
                "2026-01-01 00:00:01.000000",
                "2026-01-01 00:00:00.000000",
                "2026-01-01 00:00:00.020000",
            ]),
        )
        f.create_dataset("x", data=np.asarray([10.0, 0.0, 0.05]))
        f.create_dataset("y", data=np.asarray([10.0, 0.0, 0.0]))
        f.create_dataset("z", data=np.asarray([10.0, 0.0, 0.0]))


def _write_ball_from_rows(path: Path, rows):
    with h5py.File(path, "w") as f:
        f.create_dataset("timestamp_utc", data=_bytes([row[0] for row in rows]))
        f.create_dataset("x", data=np.asarray([row[1] for row in rows], dtype=float))
        f.create_dataset("y", data=np.asarray([row[2] for row in rows], dtype=float))
        f.create_dataset("z", data=np.asarray([row[3] for row in rows], dtype=float))


def _write_joints(path: Path):
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "timestamp_utc",
            data=_bytes([
                "2026-01-01 00:00:00.000000",
                "2026-01-01 00:00:00.020000",
                "2026-01-01 00:00:02.000000",
            ]),
        )
        f.create_dataset("player_id", data=_bytes(["p1", "p1", "p1"]))
        f.create_dataset("jersey_number", data=_bytes(["9", "9", "9"]))
        f.create_dataset("team_id", data=_bytes(["home", "home", "home"]))
        f.create_dataset("is_home", data=np.asarray([1, 1, 1]))
        f.create_dataset("nose_x", data=np.asarray([0.0, 0.10, 0.30]))
        f.create_dataset("nose_y", data=np.asarray([0.0, 0.0, 0.0]))
        f.create_dataset("nose_z", data=np.asarray([0.0, 0.0, 0.0]))
        f.create_dataset("neck_x", data=np.asarray([2.0, 2.0, 2.0]))
        f.create_dataset("neck_y", data=np.asarray([2.0, 2.0, 2.0]))
        f.create_dataset("neck_z", data=np.asarray([2.0, 2.0, 2.0]))


def _write_unsorted_joints(path: Path):
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "timestamp_utc",
            data=_bytes([
                "2026-01-01 00:00:02.000000",
                "2026-01-01 00:00:00.000000",
                "2026-01-01 00:00:00.020000",
            ]),
        )
        f.create_dataset("player_id", data=_bytes(["p1", "p1", "p1"]))
        f.create_dataset("jersey_number", data=_bytes(["9", "9", "9"]))
        f.create_dataset("team_id", data=_bytes(["home", "home", "home"]))
        f.create_dataset("is_home", data=np.asarray([1, 1, 1]))
        f.create_dataset("nose_x", data=np.asarray([0.30, 0.0, 0.10]))
        f.create_dataset("nose_y", data=np.asarray([0.0, 0.0, 0.0]))
        f.create_dataset("nose_z", data=np.asarray([0.0, 0.0, 0.0]))
        f.create_dataset("neck_x", data=np.asarray([2.0, 2.0, 2.0]))
        f.create_dataset("neck_y", data=np.asarray([2.0, 2.0, 2.0]))
        f.create_dataset("neck_z", data=np.asarray([2.0, 2.0, 2.0]))


def _write_manifest(path: Path, input_type="player_joints_h5", metadata=None):
    payload = {
        "version": "2.0",
        "modalities": [input_type],
        "labels": {"action": {"type": "single_label", "labels": ["header"]}},
        "data": [
            {
                "id": "sample",
                "inputs": [
                    {
                        "type": input_type,
                        "path": "joints.h5",
                        "ball_path": "ball.h5",
                    }
                ],
                "metadata": metadata or {},
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _config(
    root: Path,
    manifest: Path,
    *,
    include_diagnostics=True,
    input_type="player_joints_h5",
    rule_overrides=None,
    rule_name="h5_header_distance",
):
    rule_params = {
        "label": "header",
        "head_name": "action",
        "distance_threshold_m": 0.5,
        "min_confidence": 0.5,
        "confidence_mode": "linear_inverse_distance",
        "confidence_power": 1.0,
        "nms_window_ms": 1000,
        "nms_scope": "sample",
        "ball_tolerance_ms": 60,
        "chunk_size": 2,
        "head_joints": ["nose", "neck"],
        "required_input_type": "player_joints_h5",
        "ball_path_field": "ball_path",
        "timestamp_field": "timestamp_utc",
        "output_task": "action_spotting",
        "include_diagnostics": include_diagnostics,
        "sideline_filter_enabled": False,
        "pitch_half_width_m": 50.0,
        "sideline_exclusion_m": 1.0,
        "sideline_reference": "ball_y",
        "trajectory_filter_enabled": False,
        "trajectory_change_mode": "either_angle_or_speed",
        "trajectory_pre_window_ms": 200,
        "trajectory_post_window_ms": 200,
        "trajectory_min_angle_deg": 25.0,
        "trajectory_min_speed_delta_ratio": 0.25,
        "trajectory_min_vector_norm_m": 0.05,
        "trajectory_use_xy_only": False,
        "confidence_output_key": "confidence_score",
    }
    if rule_overrides:
        rule_params.update(rule_overrides)
    return OmegaConf.create(
        {
            "TASK": "localization",
            "VERSION": 2,
            "SYSTEM": {"device": "cpu", "gpu": {"count": 0}, "paths": {"work_dir": str(root)}},
            "DATA": {
                "common": {
                    "dataset_name": "h5_headers",
                    "classes": ["header"],
                    "splits": {
                        "test": {
                            "type": "H5OSLJsonSpotting",
                            "annotation_path": str(manifest),
                            "source_path": str(root),
                            "dataloader": {
                                "batch_size": 1,
                                "shuffle": False,
                                "num_workers": 0,
                                "pin_memory": False,
                            },
                        }
                    },
                },
                "inputs": {
                    "tracking": {
                        "modality": input_type,
                        "representation": "raw",
                        "source": {"format": "h5"},
                        "sampling": {},
                        "transform": {},
                        "augmentations": {},
                        "params": {},
                    }
                },
            },
            "MODEL": {
                "metadata": {
                    "family": "RuleBased",
                    "runner": {"type": "runner_h5_header_rule"},
                },
                "components": {
                    "rule": {
                        "kind": "algorithm",
                        "source": {
                            "provider": "opensportslib",
                            "registry": "rule_based",
                            "name": rule_name,
                        },
                        "params": rule_params,
                    }
                },
                "topology": [],
            },
            "TRAIN": {
                "trainer": {"type": "trainer_rule_based"},
                "execution": {"enabled": False},
            },
        }
    )


def test_rule_based_yaml_route_builds_model_and_inferer(tmp_path):
    _write_ball(tmp_path / "ball.h5")
    _write_joints(tmp_path / "joints.h5")
    manifest = tmp_path / "h5.json"
    _write_manifest(manifest)
    cfg = _config(tmp_path, manifest)

    model = build_model(cfg, device=None)
    dataset_obj = LocalizationDataset(cfg, split="test")
    dataset = dataset_obj.building_dataset(dataset_obj.cfg)
    dataloader = dataset_obj.building_dataloader(dataset, dataset_obj.cfg.dataloader, gpu=0, dali=False)
    predictions = build_inferer(cfg, model).infer(cfg, dataset, dataloader)

    assert isinstance(model, H5HeaderSpotter)
    assert dataloader is None
    assert predictions["task"] == "action_spotting"
    assert predictions["labels"]["action"]["labels"] == ["header"]


def test_rule_based_model_variants_are_selected_from_yaml_name(tmp_path):
    manifest = tmp_path / "h5.json"
    _write_manifest(manifest)

    variants = {
        "h5_header_distance": (H5HeaderDistanceSpotter, False, "either_angle_or_speed"),
        "h5_header_distance_speed": (H5HeaderDistanceSpeedSpotter, True, "speed"),
        "h5_header_distance_angle": (H5HeaderDistanceAngleSpotter, True, "angle"),
        "h5_header_distance_speed_angle": (
            H5HeaderDistanceSpeedAngleSpotter,
            True,
            "both_angle_and_speed",
        ),
    }
    for rule_name, (expected_cls, trajectory_enabled, trajectory_mode) in variants.items():
        model = build_model(_config(tmp_path, manifest, rule_name=rule_name), device=None)
        assert isinstance(model, expected_cls)
        assert model.params["trajectory_filter_enabled"] is trajectory_enabled
        assert model.params["trajectory_change_mode"] == trajectory_mode


def test_header_rule_filters_confidence_applies_nms_and_exports_osl_json(tmp_path):
    _write_ball(tmp_path / "ball.h5")
    _write_joints(tmp_path / "joints.h5")
    manifest = tmp_path / "h5.json"
    _write_manifest(manifest)
    cfg = _config(tmp_path, manifest)

    dataset_obj = LocalizationDataset(cfg, split="test")
    dataset = dataset_obj.building_dataset(dataset_obj.cfg)
    predictions = H5HeaderSpotter(cfg).predict(dataset)

    events = predictions["data"][0]["events"]
    assert len(events) == 1
    assert events[0]["label"] == "header"
    assert events[0]["confidence_score"] == 1.0
    assert "confidence" not in events[0]
    assert events[0]["position_ms"] == 0
    assert "metadata" in events[0]
    assert events[0]["metadata"]["joint"] == "nose"


def test_header_rule_diagnostics_can_be_disabled(tmp_path):
    _write_ball(tmp_path / "ball.h5")
    _write_joints(tmp_path / "joints.h5")
    manifest = tmp_path / "h5.json"
    _write_manifest(manifest)
    cfg = _config(tmp_path, manifest, include_diagnostics=False)

    dataset_obj = LocalizationDataset(cfg, split="test")
    dataset = dataset_obj.building_dataset(dataset_obj.cfg)
    event = H5HeaderSpotter(cfg).predict(dataset)["data"][0]["events"][0]

    assert "metadata" not in event


def test_header_rule_skips_centroid_only_samples(tmp_path):
    _write_ball(tmp_path / "ball.h5")
    _write_joints(tmp_path / "joints.h5")
    manifest = tmp_path / "h5.json"
    _write_manifest(manifest, input_type="player_centroids_h5")
    cfg = _config(tmp_path, manifest, input_type="player_centroids_h5")

    dataset_obj = LocalizationDataset(cfg, split="test")
    dataset = dataset_obj.building_dataset(dataset_obj.cfg)
    predictions = H5HeaderSpotter(cfg).predict(dataset)

    assert predictions["data"][0]["events"] == []


def test_position_ms_is_relative_to_joint_h5_start_not_metadata_start(tmp_path):
    _write_ball(tmp_path / "ball.h5")
    _write_joints(tmp_path / "joints.h5")
    manifest = tmp_path / "h5.json"
    _write_manifest(
        manifest,
        metadata={
            "start_utc": "2026-01-01 00:00:00.020000",
            "end_utc": "2026-01-01 00:00:00.020000",
        },
    )
    cfg = _config(tmp_path, manifest)

    dataset_obj = LocalizationDataset(cfg, split="test")
    dataset = dataset_obj.building_dataset(dataset_obj.cfg)
    predictions = H5HeaderSpotter(cfg).predict(dataset)

    events = predictions["data"][0]["events"]
    assert len(events) == 1
    assert events[0]["timestamp_utc"] == "2026-01-01 00:00:00.020000"
    assert events[0]["position_ms"] == 20


def test_position_ms_uses_earliest_joint_h5_timestamp_when_rows_are_unsorted(tmp_path):
    _write_ball(tmp_path / "ball.h5")
    _write_unsorted_joints(tmp_path / "joints.h5")
    manifest = tmp_path / "h5.json"
    _write_manifest(
        manifest,
        metadata={
            "start_utc": "2026-01-01 00:00:00.020000",
            "end_utc": "2026-01-01 00:00:00.020000",
        },
    )
    cfg = _config(tmp_path, manifest)

    dataset_obj = LocalizationDataset(cfg, split="test")
    dataset = dataset_obj.building_dataset(dataset_obj.cfg)
    predictions = H5HeaderSpotter(cfg).predict(dataset)

    events = predictions["data"][0]["events"]
    assert len(events) == 1
    assert events[0]["timestamp_utc"] == "2026-01-01 00:00:00.020000"
    assert events[0]["position_ms"] == 20
    assert events[0]["position_ms"] >= 0


def test_sideline_filter_excludes_candidate_using_ball_y(tmp_path):
    _write_ball_from_rows(
        tmp_path / "ball.h5",
        [
            ("2026-01-01 00:00:00.000000", 0.0, 0.0, 0.0),
            ("2026-01-01 00:00:00.020000", 0.10, 49.5, 0.0),
            ("2026-01-01 00:00:00.040000", 0.2, 0.0, 0.0),
        ],
    )
    _write_joints(tmp_path / "joints.h5")
    manifest = tmp_path / "h5.json"
    _write_manifest(
        manifest,
        metadata={
            "start_utc": "2026-01-01 00:00:00.020000",
            "end_utc": "2026-01-01 00:00:00.020000",
        },
    )
    cfg = _config(
        tmp_path,
        manifest,
        rule_overrides={
            "sideline_filter_enabled": True,
            "trajectory_filter_enabled": False,
        },
    )

    dataset_obj = LocalizationDataset(cfg, split="test")
    dataset = dataset_obj.building_dataset(dataset_obj.cfg)
    predictions = H5HeaderSpotter(cfg).predict(dataset)

    assert predictions["data"][0]["events"] == []


def test_angle_change_trajectory_passes_and_adds_diagnostics(tmp_path):
    _write_ball_from_rows(
        tmp_path / "ball.h5",
        [
            ("2026-01-01 00:00:00.000000", -0.20, 0.0, 0.0),
            ("2026-01-01 00:00:00.020000", 0.05, 0.0, 0.0),
            ("2026-01-01 00:00:00.040000", 0.05, 0.30, 0.0),
        ],
    )
    _write_joints(tmp_path / "joints.h5")
    manifest = tmp_path / "h5.json"
    _write_manifest(
        manifest,
        metadata={
            "start_utc": "2026-01-01 00:00:00.020000",
            "end_utc": "2026-01-01 00:00:00.020000",
        },
    )
    cfg = _config(
        tmp_path,
        manifest,
        rule_name="h5_header_distance_angle",
        rule_overrides={
            "sideline_filter_enabled": True,
            "trajectory_min_angle_deg": 25.0,
            "trajectory_min_speed_delta_ratio": 10.0,
            "trajectory_min_vector_norm_m": 0.01,
        },
    )

    dataset_obj = LocalizationDataset(cfg, split="test")
    dataset = dataset_obj.building_dataset(dataset_obj.cfg)
    event = H5HeaderSpotter(cfg).predict(dataset)["data"][0]["events"][0]

    assert event["confidence_score"] > 0.5
    metadata = event["metadata"]
    assert metadata["ball_y"] == 0.0
    assert metadata["sideline_distance_m"] == 50.0
    assert metadata["trajectory_passed"] is True
    assert metadata["trajectory_angle_deg"] >= 25.0


def test_speed_change_trajectory_passes_when_angle_does_not(tmp_path):
    _write_ball_from_rows(
        tmp_path / "ball.h5",
        [
            ("2026-01-01 00:00:00.000000", -0.05, 0.0, 0.0),
            ("2026-01-01 00:00:00.020000", 0.05, 0.0, 0.0),
            ("2026-01-01 00:00:00.040000", 0.45, 0.0, 0.0),
        ],
    )
    _write_joints(tmp_path / "joints.h5")
    manifest = tmp_path / "h5.json"
    _write_manifest(
        manifest,
        metadata={
            "start_utc": "2026-01-01 00:00:00.020000",
            "end_utc": "2026-01-01 00:00:00.020000",
        },
    )
    cfg = _config(
        tmp_path,
        manifest,
        rule_name="h5_header_distance_speed",
        rule_overrides={
            "trajectory_min_angle_deg": 25.0,
            "trajectory_min_speed_delta_ratio": 0.25,
            "trajectory_min_vector_norm_m": 0.01,
        },
    )

    dataset_obj = LocalizationDataset(cfg, split="test")
    dataset = dataset_obj.building_dataset(dataset_obj.cfg)
    event = H5HeaderSpotter(cfg).predict(dataset)["data"][0]["events"][0]

    assert event["metadata"]["trajectory_angle_deg"] == 0.0
    assert event["metadata"]["trajectory_speed_delta_ratio"] >= 0.25
    assert event["metadata"]["trajectory_passed"] is True


def test_speed_angle_variant_requires_both_trajectory_changes(tmp_path):
    _write_ball_from_rows(
        tmp_path / "ball.h5",
        [
            ("2026-01-01 00:00:00.000000", -0.10, 0.0, 0.0),
            ("2026-01-01 00:00:00.020000", 0.05, 0.0, 0.0),
            ("2026-01-01 00:00:00.040000", 0.05, 0.30, 0.0),
        ],
    )
    _write_joints(tmp_path / "joints.h5")
    manifest = tmp_path / "h5.json"
    _write_manifest(
        manifest,
        metadata={
            "start_utc": "2026-01-01 00:00:00.020000",
            "end_utc": "2026-01-01 00:00:00.020000",
        },
    )
    cfg = _config(
        tmp_path,
        manifest,
        rule_name="h5_header_distance_speed_angle",
        rule_overrides={
            "trajectory_min_angle_deg": 25.0,
            "trajectory_min_speed_delta_ratio": 0.25,
            "trajectory_min_vector_norm_m": 0.01,
        },
    )

    dataset_obj = LocalizationDataset(cfg, split="test")
    dataset = dataset_obj.building_dataset(dataset_obj.cfg)
    event = H5HeaderSpotter(cfg).predict(dataset)["data"][0]["events"][0]

    assert event["metadata"]["trajectory_angle_deg"] >= 25.0
    assert event["metadata"]["trajectory_speed_delta_ratio"] >= 0.25
    assert event["metadata"]["trajectory_passed"] is True


def test_trajectory_filter_rejects_when_context_is_insufficient(tmp_path):
    _write_ball_from_rows(
        tmp_path / "ball.h5",
        [
            ("2026-01-01 00:00:00.020000", 0.05, 0.0, 0.0),
        ],
    )
    _write_joints(tmp_path / "joints.h5")
    manifest = tmp_path / "h5.json"
    _write_manifest(
        manifest,
        metadata={
            "start_utc": "2026-01-01 00:00:00.020000",
            "end_utc": "2026-01-01 00:00:00.020000",
        },
    )
    cfg = _config(tmp_path, manifest, rule_name="h5_header_distance_speed")

    dataset_obj = LocalizationDataset(cfg, split="test")
    dataset = dataset_obj.building_dataset(dataset_obj.cfg)
    predictions = H5HeaderSpotter(cfg).predict(dataset)

    assert predictions["data"][0]["events"] == []


def test_rule_based_train_is_inference_only(tmp_path):
    manifest = tmp_path / "h5.json"
    _write_manifest(manifest)
    cfg_path = tmp_path / "config.yaml"
    OmegaConf.save(_config(tmp_path, manifest), cfg_path)

    api = LocalizationModel(config=str(cfg_path))

    with pytest.raises(NotImplementedError, match="inference-only"):
        api.train(use_wandb=False)
