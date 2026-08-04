"""Tests for the frame-joined skeleton header spotter.

The fixtures build a minimal aerial header: a ball arriving fast, deflecting
off a player's head at frame 100, with the player's hands kept well clear.
Each test then breaks exactly one precondition and asserts the corresponding
gate rejects it.
"""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from omegaconf import OmegaConf

from opensportslib.models.base.rule_based import (
    H5HeaderSkeletonSpotter,
    H5HeaderSpotterBase,
    build_rule_based_model,
)


FPS = 50
CONTACT_FRAME = 100
BALL_XYZ = (0.0, 0.0, 2.0)          # head height, mid-pitch
JOINT_COLUMNS = (
    "nose_x", "nose_y", "nose_z",
    "l_wrist_x", "l_wrist_y", "l_wrist_z",
    "r_wrist_x", "r_wrist_y", "r_wrist_z",
    "l_shoulder_x", "l_shoulder_y", "r_shoulder_x", "r_shoulder_y",
    "l_ankle_z", "r_ankle_z",
)


def _bytes(values):
    return np.asarray([value.encode("utf-8") for value in values], dtype="S26")


def _timestamp(frame):
    """Frame number -> UTC string at 50 fps, starting from a round second."""
    micros = int(round(frame * 1_000_000 / FPS))
    return f"2026-01-01 00:00:{micros // 1_000_000:02d}.{micros % 1_000_000:06d}"


def _write_ball(path: Path, *, frames, positions):
    with h5py.File(path, "w") as f:
        f.create_dataset("frame", data=np.asarray(frames, dtype=np.int64))
        f.create_dataset("timestamp_utc", data=_bytes([_timestamp(i) for i in frames]))
        arr = np.asarray(positions, dtype=float)
        for axis, name in enumerate("xyz"):
            f.create_dataset(name, data=arr[:, axis])


def _write_joints(path: Path, *, frames, rows):
    """`rows` maps a column name to a per-frame value (scalar broadcasts)."""
    count = len(frames)
    with h5py.File(path, "w") as f:
        f.create_dataset("frame", data=np.asarray(frames, dtype=np.int64))
        f.create_dataset("player_id", data=_bytes(["p1"] * count))
        f.create_dataset("is_home", data=np.ones(count, dtype=np.int64))
        for column in JOINT_COLUMNS:
            value = rows.get(column, 0.0)
            data = np.full(count, value, dtype=float) if np.isscalar(value) \
                else np.asarray(value, dtype=float)
            f.create_dataset(column, data=data)


def _deflected_ball(frames):
    """A headed ball: driven in along +x and falling, sent back along -x rising.

    The reversal at the contact frame gives every trajectory gate something to
    see — direction change, speed change and a z-acceleration change. The 0.3 m
    step keeps x off the -1.0 invalid-coordinate sentinel.
    """
    return [
        (BALL_XYZ[0] - 0.3 * abs(frame - CONTACT_FRAME),
         0.0,
         BALL_XYZ[2] + 0.3 * abs(frame - CONTACT_FRAME))
        for frame in frames
    ]


@pytest.fixture
def scene(tmp_path):
    """A clean header that the default parameters should detect."""
    frames = list(range(CONTACT_FRAME - 10, CONTACT_FRAME + 11))
    _write_ball(tmp_path / "ball.h5", frames=frames, positions=_deflected_ball(frames))

    # Nose 0.2 m from the ball at contact (inside the 0.4 m gate, so confidence
    # lands mid-range), parked far away on every other frame so the dwell filter
    # sees a single close sample.
    nose_x = [BALL_XYZ[0] + 0.2 if f == CONTACT_FRAME else 50.0 for f in frames]
    _write_joints(
        tmp_path / "joints.h5",
        frames=frames,
        rows={
            "nose_x": nose_x,
            "nose_y": 0.0,
            "nose_z": BALL_XYZ[2],
            # hands at pitch level, far below the ball
            "l_wrist_x": 5.0, "l_wrist_y": 5.0, "l_wrist_z": 0.5,
            "r_wrist_x": 5.0, "r_wrist_y": -5.0, "r_wrist_z": 0.5,
            # shoulder line across y, so the facing normal points along +x
            "l_shoulder_x": 0.0, "l_shoulder_y": -0.2,
            "r_shoulder_x": 0.0, "r_shoulder_y": 0.2,
            "l_ankle_z": 0.1, "r_ankle_z": 0.1,
        },
    )
    return tmp_path


def _manifest(directory: Path) -> Path:
    path = directory / "manifest.json"
    path.write_text(json.dumps({
        "version": "2.0",
        "data": [{
            "id": "game",
            "inputs": [{
                "type": "player_joints_h5",
                "path": "joints.h5",
                "ball_path": "ball.h5",
            }],
        }],
    }))
    return path


class _Dataset:
    """Minimal stand-in for H5OSLJsonSpottingDataset."""

    def __init__(self, directory: Path):
        self.source_path = str(directory)
        self.samples = json.loads(_manifest(directory).read_text())["data"]


def _config(**params):
    return OmegaConf.create({
        "MODEL": {"components": {"rule": {
            "kind": "algorithm",
            "source": {"name": "h5_header_skeleton"},
            "params": params,
        }}},
    })


def _events(directory: Path, **params):
    spotter = H5HeaderSkeletonSpotter(_config(**params))
    return spotter.predict(_Dataset(directory))["data"][0]["events"]


# --------------------------------------------------------------- wiring
def test_registry_builds_skeleton_spotter():
    model = build_rule_based_model(_config())
    assert isinstance(model, H5HeaderSkeletonSpotter)
    assert model.params["model_variant"] == "h5_header_skeleton"


def test_skeleton_spotter_is_sibling_of_distance_family():
    """It shares the output envelope, not the distance detection logic."""
    from opensportslib.models.base.rule_based import H5HeaderSpotter

    assert issubclass(H5HeaderSkeletonSpotter, H5HeaderSpotterBase)
    assert not issubclass(H5HeaderSkeletonSpotter, H5HeaderSpotter)


def test_config_params_override_defaults():
    model = H5HeaderSkeletonSpotter(_config(head_ball_distance_max_m=0.9))
    assert model.params["head_ball_distance_max_m"] == 0.9
    assert model.params["ball_height_min_m"] == 1.3  # untouched default


# --------------------------------------------------------------- detection
def test_clean_header_is_detected(scene):
    events = _events(scene)
    assert len(events) == 1
    event = events[0]
    assert event["label"] == "header"
    assert event["head"] == "action"
    # confidence is 1 - distance/threshold, i.e. 1 - 0.2/0.4
    assert event["confidence_score"] == pytest.approx(0.5)
    assert event["metadata"]["player_id"].startswith("p1")
    assert event["metadata"]["team"] == "home"


def test_event_position_is_relative_to_first_ball_sample(scene):
    """Contact is 10 frames after the first ball row, i.e. 200 ms at 50 fps."""
    assert _events(scene)[0]["position_ms"] == 200


def test_position_offset_shifts_events(scene):
    shifted = _events(scene, position_offset_ms=1500)[0]["position_ms"]
    assert shifted == 200 + 1500


def test_output_envelope_matches_other_variants(scene):
    payload = H5HeaderSkeletonSpotter(_config()).predict(_Dataset(scene))
    assert payload["task"] == "action_spotting"
    assert payload["labels"]["action"]["labels"] == ["header"]
    assert payload["metadata"]["created_by"] == "h5_header_skeleton_rule"


# --------------------------------------------------------------- gates
def test_ball_below_head_height_is_rejected(scene):
    assert _events(scene, ball_height_min_m=2.5) == []


def test_ball_above_head_height_is_rejected(scene):
    assert _events(scene, ball_height_max_m=1.5) == []


def test_head_further_than_threshold_is_rejected(scene):
    """Contact is 0.2 m from the nose, so a 0.1 m gate must reject it."""
    assert _events(scene, head_ball_distance_max_m=0.1) == []


def test_hand_closer_than_head_is_rejected(tmp_path):
    """A wrist nearer the ball than the nose means it was not a header."""
    frames = list(range(CONTACT_FRAME - 10, CONTACT_FRAME + 11))
    _write_ball(tmp_path / "ball.h5", frames=frames, positions=_deflected_ball(frames))
    _write_joints(
        tmp_path / "joints.h5",
        frames=frames,
        rows={
            "nose_x": [BALL_XYZ[0] + 0.3 if f == CONTACT_FRAME else 50.0 for f in frames],
            "nose_y": 0.0, "nose_z": BALL_XYZ[2],
            "l_wrist_x": 0.0, "l_wrist_y": 0.0, "l_wrist_z": BALL_XYZ[2],  # on the ball
            "r_wrist_x": 5.0, "r_wrist_y": -5.0, "r_wrist_z": 0.5,
            "l_shoulder_x": 0.0, "l_shoulder_y": -0.2,
            "r_shoulder_x": 0.0, "r_shoulder_y": 0.2,
            "l_ankle_z": 0.1, "r_ankle_z": 0.1,
        },
    )
    assert _events(tmp_path) == []


def test_raised_ankles_reject_acrobatic_pose(scene):
    assert _events(scene, ankle_height_max_m=0.05) == []


def test_straight_ball_path_is_rejected(tmp_path):
    """No direction or speed change at contact — the ball was never headed."""
    frames = list(range(CONTACT_FRAME - 10, CONTACT_FRAME + 11))
    straight = [(BALL_XYZ[0] + (f - CONTACT_FRAME) * 0.2, 0.0, BALL_XYZ[2])
                for f in frames]
    _write_ball(tmp_path / "ball.h5", frames=frames, positions=straight)
    _write_joints(
        tmp_path / "joints.h5",
        frames=frames,
        rows={
            "nose_x": [BALL_XYZ[0] if f == CONTACT_FRAME else 50.0 for f in frames],
            "nose_y": 0.0, "nose_z": BALL_XYZ[2],
            "l_wrist_x": 5.0, "l_wrist_y": 5.0, "l_wrist_z": 0.5,
            "r_wrist_x": 5.0, "r_wrist_y": -5.0, "r_wrist_z": 0.5,
            "l_shoulder_x": 0.0, "l_shoulder_y": -0.2,
            "r_shoulder_x": 0.0, "r_shoulder_y": 0.2,
            "l_ankle_z": 0.1, "r_ankle_z": 0.1,
        },
    )
    assert _events(tmp_path) == []


def test_dwell_filter_rejects_ball_held_near_head(tmp_path):
    """Nose glued to the ball across the whole dwell window — carried, not headed."""
    frames = list(range(CONTACT_FRAME - 10, CONTACT_FRAME + 11))
    positions = _deflected_ball(frames)
    _write_ball(tmp_path / "ball.h5", frames=frames, positions=positions)
    _write_joints(
        tmp_path / "joints.h5",
        frames=frames,
        rows={
            "nose_x": [p[0] for p in positions],
            "nose_y": [p[1] for p in positions],
            "nose_z": [p[2] for p in positions],
            "l_wrist_x": 5.0, "l_wrist_y": 5.0, "l_wrist_z": 0.5,
            "r_wrist_x": 5.0, "r_wrist_y": -5.0, "r_wrist_z": 0.5,
            "l_shoulder_x": 0.0, "l_shoulder_y": -0.2,
            "r_shoulder_x": 0.0, "r_shoulder_y": 0.2,
            "l_ankle_z": 0.1, "r_ankle_z": 0.1,
        },
    )
    assert _events(tmp_path) == []
    # the same scene passes once the dwell allowance is lifted
    assert len(_events(tmp_path, dwell_max_frames=99)) == 1


# --------------------------------------------------------------- windowing
def test_scan_window_excludes_contact_outside_it(scene):
    spotter = H5HeaderSkeletonSpotter(_config())
    sample = dict(_Dataset(scene).samples[0])
    sample["metadata"] = {
        "start_utc": "2026-01-01 00:00:00.000000",
        "end_utc": "2026-01-01 00:00:01.000000",  # contact is at 00:00:02
    }
    assert spotter._predict_sample(sample, str(scene)) == []


def test_missing_joint_column_raises(tmp_path):
    frames = [CONTACT_FRAME]
    _write_ball(tmp_path / "ball.h5", frames=frames, positions=[BALL_XYZ])
    with h5py.File(tmp_path / "joints.h5", "w") as f:
        f.create_dataset("frame", data=np.asarray(frames, dtype=np.int64))
        f.create_dataset("player_id", data=_bytes(["p1"]))
    with pytest.raises(ValueError, match="missing datasets"):
        _events(tmp_path)
