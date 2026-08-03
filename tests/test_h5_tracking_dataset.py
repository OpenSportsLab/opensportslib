import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from opensportslib.datasets.classification_dataset import H5TrackingDataset
from opensportslib.datasets.utils.h5_tracking import H5TrackingReader, H5_FEATURE_DIM


def _bytes(values):
    return np.asarray([value.encode("utf-8") for value in values], dtype="S26")


def _write_centroids(path: Path):
    timestamps = [
        "2026-01-01 00:00:00.000000",
        "2026-01-01 00:00:00.000000",
        "2026-01-01 00:00:00.020000",
        "2026-01-01 00:00:00.020000",
        "2026-01-01 00:00:00.040000",
        "2026-01-01 00:00:00.040000",
    ]
    with h5py.File(path, "w") as f:
        f.create_dataset("timestamp_utc", data=_bytes(timestamps))
        f.create_dataset("x", data=np.asarray([0, 10, 1, 11, 2, 12], dtype=np.float64))
        f.create_dataset("y", data=np.asarray([0, 20, 1, 21, 2, 22], dtype=np.float64))
        f.create_dataset("player_id", data=_bytes(["p1", "p2", "p1", "p2", "p1", "p2"]))
        f.create_dataset("jersey_number", data=_bytes(["1", "2", "1", "2", "1", "2"]))
        f.create_dataset("is_home", data=np.asarray([1, 0, 1, 0, 1, 0], dtype=np.int64))
        f.create_dataset("role_name", data=_bytes(["MID", "DEF", "MID", "DEF", "MID", "DEF"]))


def _write_joints(path: Path):
    timestamps = [
        "2026-01-01 00:00:00.000000",
        "2026-01-01 00:00:00.020000",
    ]
    with h5py.File(path, "w") as f:
        f.create_dataset("timestamp_utc", data=_bytes(timestamps))
        f.create_dataset("player_id", data=_bytes(["p1", "p1"]))
        f.create_dataset("jersey_number", data=_bytes(["1", "1"]))
        f.create_dataset("is_home", data=np.asarray([1, 1], dtype=np.int64))
        f.create_dataset("role_name", data=_bytes(["MID", "MID"]))
        for joint in ["nose", "neck"]:
            f.create_dataset(f"{joint}_x", data=np.asarray([1.0, 2.0]))
            f.create_dataset(f"{joint}_y", data=np.asarray([3.0, 4.0]))
            f.create_dataset(f"{joint}_z", data=np.asarray([5.0, 6.0]))


def _write_ball(path: Path, *, offset_ms=0):
    base = np.datetime64("2026-01-01T00:00:00.000000", "us")
    timestamps = [
        str((base + np.timedelta64(offset_ms + i * 20, "ms")).astype("datetime64[us]")).replace("T", " ")
        for i in range(3)
    ]
    with h5py.File(path, "w") as f:
        f.create_dataset("timestamp_utc", data=_bytes(timestamps))
        f.create_dataset("x", data=np.asarray([100, 101, 102], dtype=np.float64))
        f.create_dataset("y", data=np.asarray([200, 201, 202], dtype=np.float64))
        f.create_dataset("z", data=np.asarray([3, 4, 5], dtype=np.float64))


def _write_annotation(path: Path, input_type: str, input_path: str, ball_path: str | None = None):
    input_obj = {"type": input_type, "path": input_path}
    if ball_path:
        input_obj["ball_path"] = ball_path
    payload = {
        "labels": {"action": {"type": "single_label", "labels": ["HEADER"]}},
        "data": [
            {
                "id": "sample_h5",
                "inputs": [input_obj],
                "metadata": {
                    "start_utc": "2026-01-01 00:00:00.020000",
                    "end_utc": "2026-01-01 00:00:00.040000",
                },
                "labels": {"action": {"label": "HEADER"}},
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _config(root: Path, modality="player_centroids_h5", num_frames=None, tolerance_ms=25.0):
    return SimpleNamespace(
        DATA=SimpleNamespace(
            common=SimpleNamespace(
                splits=SimpleNamespace(train=SimpleNamespace(source_path=str(root))),
            ),
            inputs=SimpleNamespace(
                tracking=SimpleNamespace(
                    modality=modality,
                    sampling=SimpleNamespace(num_frames=num_frames),
                    transform=SimpleNamespace(normalize=False),
                    augmentations=SimpleNamespace(),
                    params=SimpleNamespace(timestamp_tolerance_ms=tolerance_ms),
                )
            ),
        ),
        MODEL=SimpleNamespace(
            components=SimpleNamespace(
                encoder=SimpleNamespace(
                    kind="encoder",
                    source=SimpleNamespace(name="graph_conv"),
                    params=SimpleNamespace(edge_type="none"),
                    overrides=SimpleNamespace(),
                )
            )
        ),
    )


def test_h5_reader_clips_by_utc_and_aligns_ball(tmp_path):
    centroids = tmp_path / "centroids.h5"
    ball = tmp_path / "ball.h5"
    _write_centroids(centroids)
    _write_ball(ball)

    reader = H5TrackingReader(centroids, "player_centroids_h5", ball_path=ball, timestamp_tolerance_ms=5)
    timestamps = reader.select_timestamps(
        start_utc="2026-01-01 00:00:00.020000",
        end_utc="2026-01-01 00:00:00.040000",
    )
    frames = reader.read_sequence(timestamps)

    assert [str(ts) for ts in timestamps] == [
        "2026-01-01T00:00:00.020000",
        "2026-01-01T00:00:00.040000",
    ]
    assert frames[0].features.shape == (3, H5_FEATURE_DIM)
    assert frames[0].features[0, :3].tolist() == [101.0, 201.0, 4.0]
    assert frames[1].features[1, 6] == 1.0


def test_h5_reader_out_of_tolerance_ball_uses_missing_node(tmp_path):
    centroids = tmp_path / "centroids.h5"
    ball = tmp_path / "ball.h5"
    _write_centroids(centroids)
    _write_ball(ball, offset_ms=1000)

    reader = H5TrackingReader(centroids, "player_centroids_h5", ball_path=ball, timestamp_tolerance_ms=5)
    timestamps = reader.select_timestamps(
        start_utc="2026-01-01 00:00:00.000000",
        end_utc="2026-01-01 00:00:00.000000",
    )
    frames = reader.read_sequence(timestamps)

    assert frames[0].entity_ids[0] == "BALL"
    assert frames[0].features[0, 0] == -200.0


def test_h5_dataset_resolves_ball_path_and_returns_graphs(tmp_path, monkeypatch):
    centroids = tmp_path / "centroids.h5"
    ball = tmp_path / "ball.h5"
    annotation = tmp_path / "annotations.json"
    _write_centroids(centroids)
    _write_ball(ball)
    _write_annotation(annotation, "player_centroids_h5", "centroids.h5", "ball.h5")

    data_module = types.ModuleType("torch_geometric.data")

    class Data:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    data_module.Data = Data
    torch_geometric_module = types.ModuleType("torch_geometric")
    torch_geometric_module.data = data_module
    monkeypatch.setitem(sys.modules, "torch_geometric", torch_geometric_module)
    monkeypatch.setitem(sys.modules, "torch_geometric.data", data_module)

    dataset = H5TrackingDataset(_config(tmp_path), str(annotation), split="train")
    sample = dataset[0]

    assert dataset.samples[0]["inputs"][0]["ball_path"] == "ball.h5"
    assert sample["id"] == "sample_h5"
    assert sample["seq_len"] == 2
    assert sample["graphs"][0].x.shape == (3, H5_FEATURE_DIM)
    assert sample["label"] == 0


def test_h5_reader_materializes_joint_nodes(tmp_path):
    joints = tmp_path / "joints.h5"
    _write_joints(joints)

    reader = H5TrackingReader(joints, "player_joints_h5")
    timestamps = reader.select_timestamps(num_frames=1)
    frames = reader.read_sequence(timestamps)

    assert frames[0].features.shape == (2, H5_FEATURE_DIM)
    assert all(entity_id.endswith(("neck", "nose")) for entity_id in frames[0].entity_ids)
    assert frames[0].features[:, 9].tolist() == [1.0, 1.0]
