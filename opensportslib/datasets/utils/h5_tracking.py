"""Utilities for UTC-indexed H5 player and ball tracking inputs."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

from opensportslib.datasets.utils.tracking import MISSING_VALUE, MISSING_VALUE_NORMALIZED


JOINTS_FILENAME = "live_joints.h5"
BALL_FILENAME = "live_ball.h5"

H5_FEATURE_DIM = 10
H5_FEATURE_LAYOUT = (
    "x",
    "y",
    "z",
    "is_ball",
    "is_home",
    "is_away",
    "dx",
    "dy",
    "dz",
    "is_joint",
)


def parse_utc(value) -> np.datetime64:
    """Parse UTC-ish H5/JSON timestamp values into microsecond datetime64."""
    if isinstance(value, np.datetime64):
        return value.astype("datetime64[us]")
    if isinstance(value, (bytes, np.bytes_)):
        value = value.decode("utf-8")
    value = str(value).strip()
    if not value:
        raise ValueError("Empty UTC timestamp")
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return np.datetime64(dt, "us")


def _decode_scalar(value) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    return str(value)


def _read_column(file_obj: h5py.File, key: str, indices: np.ndarray, default=None):
    if key not in file_obj:
        return np.full(len(indices), default, dtype=object)
    return file_obj[key][indices]


def _build_timestamp_index(path: str | Path):
    with h5py.File(path, "r") as f:
        if "timestamp_utc" not in f:
            raise ValueError(f"H5 file is missing required dataset 'timestamp_utc': {path}")
        raw = f["timestamp_utc"][:]

    timestamps = np.array([parse_utc(value) for value in raw], dtype="datetime64[us]")
    index = defaultdict(list)
    for row_idx, timestamp in enumerate(timestamps):
        index[timestamp.astype("int64").item()].append(row_idx)
    compact_index = {
        timestamp_us: np.asarray(indices, dtype=np.int64)
        for timestamp_us, indices in index.items()
    }
    unique_timestamps = np.array(
        sorted(compact_index.keys()), dtype="datetime64[us]"
    )
    return timestamps, compact_index, unique_timestamps


@dataclass(frozen=True)
class H5Frame:
    timestamp: np.datetime64
    features: np.ndarray
    entity_ids: list[str]
    positions: list[str]


class H5TrackingReader:
    """Lazy reader for columnar player tracking H5 files keyed by UTC."""

    def __init__(
        self,
        player_path: str | Path,
        modality: str,
        ball_path: str | Path | None = None,
        *,
        missing_value: float = MISSING_VALUE,
        timestamp_tolerance_ms: float = 50.0,
    ):
        self.player_path = str(player_path)
        self.ball_path = str(ball_path) if ball_path else None
        self.modality = str(modality).lower()
        self.missing_value = float(missing_value)
        self.timestamp_tolerance = np.timedelta64(int(timestamp_tolerance_ms * 1000), "us")

        (
            self.player_timestamps,
            self.player_index,
            self.player_unique_timestamps,
        ) = _build_timestamp_index(self.player_path)

        self.ball_timestamps = None
        self.ball_index = {}
        self.ball_unique_timestamps = np.array([], dtype="datetime64[us]")
        if self.ball_path is not None:
            (
                self.ball_timestamps,
                self.ball_index,
                self.ball_unique_timestamps,
            ) = _build_timestamp_index(self.ball_path)

    def select_timestamps(
        self,
        *,
        start_utc=None,
        end_utc=None,
        num_frames: int | None = None,
        stride_ms: float | None = None,
        target_fps: float | None = None,
    ) -> np.ndarray:
        if self.player_unique_timestamps.size == 0:
            raise ValueError(f"Player H5 input has no timestamps: {self.player_path}")
        start = parse_utc(start_utc) if start_utc else self.player_unique_timestamps[0]
        end = parse_utc(end_utc) if end_utc else self.player_unique_timestamps[-1]

        if self.ball_path and self.ball_unique_timestamps.size:
            if start_utc is None:
                start = max(start, self.ball_unique_timestamps[0])
            if end_utc is None:
                end = min(end, self.ball_unique_timestamps[-1])

        candidates = self.player_unique_timestamps[
            (self.player_unique_timestamps >= start) & (self.player_unique_timestamps <= end)
        ]
        if candidates.size == 0:
            raise ValueError(
                f"No player H5 timestamps found in requested UTC range: {start} to {end}"
            )

        stride_us = None
        if stride_ms:
            stride_us = int(float(stride_ms) * 1000)
        elif target_fps:
            stride_us = int(1_000_000 / float(target_fps))

        if stride_us and stride_us > 0:
            selected = []
            next_allowed = candidates[0]
            stride = np.timedelta64(stride_us, "us")
            for timestamp in candidates:
                if timestamp >= next_allowed:
                    selected.append(timestamp)
                    next_allowed = timestamp + stride
            candidates = np.asarray(selected, dtype="datetime64[us]")

        if num_frames is not None and int(num_frames) > 0 and candidates.size > int(num_frames):
            if stride_us:
                candidates = candidates[: int(num_frames)]
            else:
                indices = np.linspace(0, candidates.size - 1, int(num_frames))
                candidates = candidates[np.round(indices).astype(np.int64)]

        return candidates

    def read_sequence(self, timestamps: Iterable[np.datetime64]) -> list[H5Frame]:
        frames = [self._read_player_frame(timestamp) for timestamp in timestamps]
        if self.ball_path is not None:
            frames = [self._append_ball(frame) for frame in frames]
        return _compute_deltas(frames, missing_value=self.missing_value)

    def _row_indices(self, timestamp: np.datetime64) -> np.ndarray:
        key = timestamp.astype("datetime64[us]").astype("int64").item()
        return self.player_index.get(key, np.asarray([], dtype=np.int64))

    def _read_player_frame(self, timestamp: np.datetime64) -> H5Frame:
        indices = self._row_indices(timestamp)
        if indices.size == 0:
            return H5Frame(timestamp, np.empty((0, H5_FEATURE_DIM), dtype=np.float32), [], [])

        with h5py.File(self.player_path, "r") as f:
            if self.modality == "player_joints_h5":
                features, entity_ids, positions = self._read_joint_nodes(f, indices)
            else:
                features, entity_ids, positions = self._read_centroid_nodes(f, indices)
        return H5Frame(timestamp, features, entity_ids, positions)

    def _read_centroid_nodes(self, f: h5py.File, indices: np.ndarray):
        features = np.full((len(indices), H5_FEATURE_DIM), self.missing_value, dtype=np.float32)
        features[:, 3] = 0.0
        is_home_values = _read_column(f, "is_home", indices, default=-1)

        features[:, 0] = _read_column(f, "x", indices, default=self.missing_value).astype(np.float32)
        features[:, 1] = _read_column(f, "y", indices, default=self.missing_value).astype(np.float32)
        features[:, 2] = self.missing_value
        features[:, 4] = np.asarray(is_home_values == 1, dtype=np.float32)
        features[:, 5] = np.asarray(is_home_values == 0, dtype=np.float32)
        features[:, 6:9] = 0.0
        features[:, 9] = 0.0

        player_ids = _read_column(f, "player_id", indices, default="")
        jersey = _read_column(f, "jersey_number", indices, default="")
        entity_ids = []
        for row_offset, (player_id, jersey_number, is_home) in enumerate(
            zip(player_ids, jersey, is_home_values)
        ):
            player_id = _decode_scalar(player_id)
            jersey_number = _decode_scalar(jersey_number)
            side = "home" if is_home == 1 else "away" if is_home == 0 else "unknown"
            entity_ids.append(player_id or f"{side}:{jersey_number}:{row_offset}")
        positions = [_decode_scalar(v) for v in _read_column(f, "role_name", indices, default="")]
        return features, entity_ids, positions

    def _read_joint_nodes(self, f: h5py.File, indices: np.ndarray):
        joint_names = sorted(
            key[:-2]
            for key in f.keys()
            if key.endswith("_x") and f"{key[:-2]}_y" in f and f"{key[:-2]}_z" in f
        )
        is_home_values = _read_column(f, "is_home", indices, default=-1)
        player_ids = _read_column(f, "player_id", indices, default="")
        jersey = _read_column(f, "jersey_number", indices, default="")
        role_names = _read_column(f, "role_name", indices, default="")

        node_count = len(indices) * len(joint_names)
        features = np.full((node_count, H5_FEATURE_DIM), self.missing_value, dtype=np.float32)
        entity_ids = []
        positions = []
        out_idx = 0
        for row_offset, row_idx in enumerate(indices):
            player_id = _decode_scalar(player_ids[row_offset])
            jersey_number = _decode_scalar(jersey[row_offset])
            is_home = is_home_values[row_offset]
            side = "home" if is_home == 1 else "away" if is_home == 0 else "unknown"
            player_key = player_id or f"{side}:{jersey_number}:{row_offset}"
            for joint_name in joint_names:
                features[out_idx, 0] = f[f"{joint_name}_x"][row_idx]
                features[out_idx, 1] = f[f"{joint_name}_y"][row_idx]
                features[out_idx, 2] = f[f"{joint_name}_z"][row_idx]
                features[out_idx, 3] = 0.0
                features[out_idx, 4] = 1.0 if is_home == 1 else 0.0
                features[out_idx, 5] = 1.0 if is_home == 0 else 0.0
                features[out_idx, 6:9] = 0.0
                features[out_idx, 9] = 1.0
                entity_ids.append(f"{player_key}:{joint_name}")
                positions.append(_decode_scalar(role_names[row_offset]))
                out_idx += 1
        return features, entity_ids, positions

    def _append_ball(self, frame: H5Frame) -> H5Frame:
        ball_features = np.full((1, H5_FEATURE_DIM), self.missing_value, dtype=np.float32)
        ball_features[0, 3] = 1.0
        ball_features[0, 4:6] = 0.0
        ball_features[0, 6:9] = 0.0
        ball_features[0, 9] = 0.0

        nearest = self._nearest_ball_timestamp(frame.timestamp)
        if nearest is not None:
            key = nearest.astype("datetime64[us]").astype("int64").item()
            row_idx = self.ball_index[key][0]
            with h5py.File(self.ball_path, "r") as f:
                ball_features[0, 0] = f["x"][row_idx] if "x" in f else self.missing_value
                ball_features[0, 1] = f["y"][row_idx] if "y" in f else self.missing_value
                ball_features[0, 2] = f["z"][row_idx] if "z" in f else self.missing_value

        features = np.concatenate([ball_features, frame.features], axis=0)
        return H5Frame(
            timestamp=frame.timestamp,
            features=features,
            entity_ids=["BALL"] + frame.entity_ids,
            positions=["BALL"] + frame.positions,
        )

    def _nearest_ball_timestamp(self, timestamp: np.datetime64) -> np.datetime64 | None:
        if self.ball_unique_timestamps.size == 0:
            return None
        pos = np.searchsorted(self.ball_unique_timestamps, timestamp)
        candidates = []
        if pos < self.ball_unique_timestamps.size:
            candidates.append(self.ball_unique_timestamps[pos])
        if pos > 0:
            candidates.append(self.ball_unique_timestamps[pos - 1])
        if not candidates:
            return None
        nearest = min(candidates, key=lambda candidate: abs(candidate - timestamp))
        if abs(nearest - timestamp) <= self.timestamp_tolerance:
            return nearest
        return None


def _compute_deltas(frames: list[H5Frame], *, missing_value: float) -> list[H5Frame]:
    previous = {}
    output = []
    for frame in frames:
        features = frame.features.copy()
        for idx, entity_id in enumerate(frame.entity_ids):
            xyz = features[idx, 0:3]
            valid = not np.any(np.isclose(xyz[:2], missing_value))
            if valid and entity_id in previous:
                delta = xyz - previous[entity_id]
                if np.isclose(xyz[2], missing_value) or np.isclose(previous[entity_id][2], missing_value):
                    delta[2] = 0.0
                features[idx, 6:9] = delta
            if valid:
                previous[entity_id] = xyz.copy()
        output.append(H5Frame(frame.timestamp, features, frame.entity_ids, frame.positions))
    return output


def normalize_h5_features(
    features: np.ndarray,
    *,
    pitch_half_length: float = 85.0,
    pitch_half_width: float = 50.0,
    max_height: float = 30.0,
    max_displacement: float = 110.0,
    missing_value: float = MISSING_VALUE,
) -> np.ndarray:
    """Normalize H5 XYZ graph features without mutating the input."""
    out = features.copy()
    valid_mask = out[:, :, 0] != missing_value
    out[valid_mask, 0] /= pitch_half_length
    out[valid_mask, 1] /= pitch_half_width
    out[valid_mask, 2] /= max_height
    out[valid_mask, 6:9] /= max_displacement
    for ch in (0, 1, 2, 6, 7, 8):
        out[~valid_mask, ch] = MISSING_VALUE_NORMALIZED
    return out


def find_h5_games(
    path,
    joints_filename: str = JOINTS_FILENAME,
    ball_filename: str = BALL_FILENAME,
) -> list[Path]:
    """List game directories holding both tracking files.

    `path` is either one game directory or a directory of game directories.
    """
    def holds_both(directory: Path) -> bool:
        try:
            return (directory / joints_filename).exists() and (directory / ball_filename).exists()
        except OSError:
            return False

    path = Path(path).resolve()
    if holds_both(path):
        return [path]
    return sorted(d for d in path.iterdir() if d.is_dir() and holds_both(d))


def write_h5_manifest(
    path,
    games: Iterable[Path],
    label: str = "header",
    head_name: str = "action",
    joints_filename: str = JOINTS_FILENAME,
    ball_filename: str = BALL_FILENAME,
) -> Path:
    """Write the OSL JSON manifest a rule-based H5 spotter reads its inputs from.

    Input paths are absolute, so a config published without the data beside it
    still resolves.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "version": "2.0",
        "task": "action_spotting",
        "labels": {head_name: {"type": "single_label", "labels": [label]}},
        "data": [
            {
                "id": game.name,
                "inputs": [{
                    "type": "player_joints_h5",
                    "path": str(game / joints_filename),
                    "ball_path": str(game / ball_filename),
                }],
            }
            for game in games
        ],
    }, indent=2))
    return path
