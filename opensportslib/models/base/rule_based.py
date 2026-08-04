"""Rule-based localization models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path

import h5py
import numpy as np

from opensportslib.core.config.accessors import (
    get_component_params_by_kind,
    get_data_params,
)
from opensportslib.datasets.utils.h5_tracking import parse_utc


DEFAULT_HEADER_RULE_PARAMS = {
    "label": "header",
    "head_name": "action",
    "distance_threshold_m": 0.5,
    "min_confidence": 0.5,
    "confidence_mode": "linear_inverse_distance",
    "confidence_power": 1.0,
    "nms_window_ms": 1000,
    "nms_scope": "sample",
    "ball_tolerance_ms": 60,
    "chunk_size": 100000,
    "head_joints": ["nose", "neck", "l_eye", "r_eye", "l_ear", "r_ear"],
    "required_input_type": "player_joints_h5",
    "ball_path_field": "ball_path",
    "timestamp_field": "timestamp_utc",
    "output_task": "action_spotting",
    "include_diagnostics": True,
    "created_by": "h5_header_distance_rule",
    "position_ms_origin": "joint_h5_start",
    "metadata_start_field": "start_utc",
    "metadata_end_field": "end_utc",
    "ball_coordinate_fields": ["x", "y", "z"],
    "joint_coordinate_suffixes": ["x", "y", "z"],
    "identity_fields": ["player_id", "jersey_number", "team_id", "is_home"],
    "invalid_coordinate_values": [-200.0],
    "sideline_filter_enabled": True,
    "pitch_half_width_m": 50.0,
    "sideline_exclusion_m": 1.0,
    "sideline_reference": "ball_y",
    "trajectory_filter_enabled": True,
    "trajectory_change_mode": "either_angle_or_speed",
    "trajectory_pre_window_ms": 200,
    "trajectory_post_window_ms": 200,
    "trajectory_min_angle_deg": 25.0,
    "trajectory_min_speed_delta_ratio": 0.25,
    "trajectory_min_vector_norm_m": 0.05,
    "trajectory_use_xy_only": False,
    "confidence_output_key": "confidence_score",
}


HEADER_RULE_VARIANTS = {
    "h5_header_distance": {
        "trajectory_filter_enabled": False,
        "trajectory_change_mode": "either_angle_or_speed",
        "created_by": "h5_header_distance_rule",
    },
    "h5_header_distance_speed": {
        "trajectory_filter_enabled": True,
        "trajectory_change_mode": "speed",
        "created_by": "h5_header_distance_speed_rule",
    },
    "h5_header_distance_angle": {
        "trajectory_filter_enabled": True,
        "trajectory_change_mode": "angle",
        "created_by": "h5_header_distance_angle_rule",
    },
    "h5_header_distance_speed_angle": {
        "trajectory_filter_enabled": True,
        "trajectory_change_mode": "both_angle_and_speed",
        "created_by": "h5_header_distance_speed_angle_rule",
    },
}


@dataclass
class _Candidate:
    timestamp: np.datetime64
    position_ms: int
    confidence: float
    distance_m: float
    joint: str
    player_id: str | None
    ball_timestamp: np.datetime64
    ball_index: int
    ball_xyz: np.ndarray
    ball_y: float
    sideline_distance_m: float
    trajectory: dict


class H5OSLJsonSpottingDataset:
    """Lightweight OSL JSON manifest wrapper for rule-based H5 spotting."""

    rule_based_no_dataloader = True

    def __init__(self, annotation_path, source_path=None):
        self.annotation_path = str(annotation_path)
        self.source_path = str(source_path or os.path.dirname(os.path.abspath(self.annotation_path)))
        with open(self.annotation_path, "r", encoding="utf-8") as f:
            self.payload = json.load(f)
        self.samples = list(self.payload.get("data", []))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class H5HeaderSpotter:
    """Rule-based H5 header spotter driven entirely by config parameters."""

    variant_name = "h5_header_distance"

    def __init__(self, config):
        rule_params = get_component_params_by_kind(config, "algorithm")
        if not rule_params:
            rule_params = get_data_params(config)
        params = dict(DEFAULT_HEADER_RULE_PARAMS)
        params.update(rule_params or {})
        configured_variant = params.pop("type", None) or self.variant_name
        params.update(HEADER_RULE_VARIANTS.get(configured_variant, {}))
        params["model_variant"] = configured_variant
        self.params = params

    def predict(self, dataset):
        data = []
        for sample in getattr(dataset, "samples", []):
            events = self._predict_sample(sample, dataset.source_path)
            data.append(
                {
                    "id": sample.get("id"),
                    "inputs": sample.get("inputs", []),
                    "events": events,
                }
            )

        label = self.params["label"]
        head_name = self.params["head_name"]
        return {
            "version": "2.0",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "task": self.params["output_task"],
            "metadata": {
                "type": "predictions",
                "created_by": self.params["created_by"],
            },
            "labels": {
                head_name: {
                    "type": "single_label",
                    "labels": [label],
                }
            },
            "data": data,
        }

    def _predict_sample(self, sample, source_path):
        inputs = sample.get("inputs", []) or []
        candidates = []
        for input_obj in inputs:
            if input_obj.get("type") != self.params["required_input_type"]:
                continue
            ball_path = input_obj.get(self.params["ball_path_field"])
            if not ball_path:
                continue
            candidates.extend(
                self._scan_joint_input(
                    joint_path=self._resolve_path(source_path, input_obj["path"]),
                    ball_path=self._resolve_path(source_path, ball_path),
                    sample_metadata=sample.get("metadata", {}) or {},
                )
            )

        filtered = [
            candidate
            for candidate in candidates
            if candidate.confidence > float(self.params["min_confidence"])
        ]
        return [self._candidate_to_event(candidate) for candidate in self._nms(filtered)]

    @staticmethod
    def _resolve_path(source_path, path):
        if os.path.isabs(path):
            return path
        return os.path.join(source_path, path)

    def _scan_joint_input(self, joint_path, ball_path, sample_metadata):
        ball = self._load_ball(ball_path)
        if ball["timestamps"].size == 0:
            return []

        start_utc = sample_metadata.get(self.params["metadata_start_field"])
        end_utc = sample_metadata.get(self.params["metadata_end_field"])
        start_ts = parse_utc(start_utc) if start_utc else None
        end_ts = parse_utc(end_utc) if end_utc else None

        candidates = []
        timestamp_field = self.params["timestamp_field"]
        chunk_size = int(self.params["chunk_size"])
        with h5py.File(joint_path, "r") as f:
            if timestamp_field not in f:
                raise ValueError(f"H5 file is missing required dataset '{timestamp_field}': {joint_path}")
            num_rows = len(f[timestamp_field])
            base_ts = self._min_timestamp(f[timestamp_field], chunk_size)
            for chunk_start in range(0, num_rows, chunk_size):
                chunk_end = min(chunk_start + chunk_size, num_rows)
                row_slice = slice(chunk_start, chunk_end)
                timestamps = np.asarray(
                    [parse_utc(value) for value in f[timestamp_field][row_slice]],
                    dtype="datetime64[us]",
                )
                mask = np.ones(timestamps.shape[0], dtype=bool)
                if start_ts is not None:
                    mask &= timestamps >= start_ts
                if end_ts is not None:
                    mask &= timestamps <= end_ts
                if not np.any(mask):
                    continue

                chunk_candidates = self._scan_chunk(
                    f=f,
                    row_slice=row_slice,
                    timestamps=timestamps,
                    mask=mask,
                    base_ts=base_ts,
                    ball=ball,
                )
                candidates.extend(chunk_candidates)
        return candidates

    @staticmethod
    def _min_timestamp(timestamp_dataset, chunk_size):
        min_ts = None
        num_rows = len(timestamp_dataset)
        for chunk_start in range(0, num_rows, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_rows)
            timestamps = [
                parse_utc(value)
                for value in timestamp_dataset[chunk_start:chunk_end]
            ]
            if not timestamps:
                continue
            chunk_min = min(timestamps)
            if min_ts is None or chunk_min < min_ts:
                min_ts = chunk_min
        if min_ts is None:
            raise ValueError("Joint H5 input has no timestamps.")
        return min_ts

    def _scan_chunk(self, *, f, row_slice, timestamps, mask, base_ts, ball):
        masked_timestamps = timestamps[mask]
        nearest_indices, nearest_valid = self._nearest_ball_indices(masked_timestamps, ball)
        if not np.any(nearest_valid):
            return []

        all_rows = np.arange(row_slice.start, row_slice.stop)
        selected_rows = all_rows[mask]
        identity = self._read_identity(f, selected_rows)
        candidates = []
        for joint in self.params["head_joints"]:
            columns = [f"{joint}_{suffix}" for suffix in self.params["joint_coordinate_suffixes"]]
            if not all(column in f for column in columns):
                continue
            coords = np.stack([f[column][selected_rows] for column in columns], axis=1).astype(float)
            valid_coords = self._valid_coordinates(coords)
            valid = nearest_valid & valid_coords
            if not np.any(valid):
                continue

            ball_xyz = ball["xyz"][nearest_indices[valid]]
            distances = np.linalg.norm(coords[valid] - ball_xyz, axis=1)
            within = distances < float(self.params["distance_threshold_m"])
            if not np.any(within):
                continue

            valid_positions = np.flatnonzero(valid)[within]
            for local_pos, distance in zip(valid_positions, distances[within]):
                confidence = self._confidence(float(distance))
                timestamp = masked_timestamps[local_pos]
                ball_index = int(nearest_indices[local_pos])
                ball_timestamp = ball["timestamps"][ball_index]
                ball_xyz_at_contact = ball["xyz"][ball_index]
                if not self._passes_sideline_filter(ball_xyz_at_contact):
                    continue
                trajectory = self._trajectory_diagnostics(ball, ball_index)
                if self.params["trajectory_filter_enabled"] and not trajectory["trajectory_passed"]:
                    continue
                candidates.append(
                    _Candidate(
                        timestamp=timestamp,
                        position_ms=self._position_ms(timestamp, base_ts),
                        confidence=confidence,
                        distance_m=float(distance),
                        joint=joint,
                        player_id=identity[local_pos],
                        ball_timestamp=ball_timestamp,
                        ball_index=ball_index,
                        ball_xyz=ball_xyz_at_contact,
                        ball_y=float(ball_xyz_at_contact[1]),
                        sideline_distance_m=self._sideline_distance(ball_xyz_at_contact),
                        trajectory=trajectory,
                    )
                )
        return candidates

    def _load_ball(self, ball_path):
        timestamp_field = self.params["timestamp_field"]
        coord_fields = self.params["ball_coordinate_fields"]
        with h5py.File(ball_path, "r") as f:
            if timestamp_field not in f:
                raise ValueError(f"H5 file is missing required dataset '{timestamp_field}': {ball_path}")
            if not all(field in f for field in coord_fields):
                missing = [field for field in coord_fields if field not in f]
                raise ValueError(f"Ball H5 is missing coordinate datasets {missing}: {ball_path}")
            timestamps = np.asarray(
                [parse_utc(value) for value in f[timestamp_field][:]],
                dtype="datetime64[us]",
            )
            xyz = np.stack([f[field][:] for field in coord_fields], axis=1).astype(float)

        valid = self._valid_coordinates(xyz)
        timestamps = timestamps[valid]
        xyz = xyz[valid]
        order = np.argsort(timestamps)
        return {"timestamps": timestamps[order], "xyz": xyz[order]}

    def _nearest_ball_indices(self, timestamps, ball):
        ball_ts = ball["timestamps"]
        positions = np.searchsorted(ball_ts, timestamps)
        nearest = np.zeros(timestamps.shape[0], dtype=np.int64)
        valid = np.zeros(timestamps.shape[0], dtype=bool)
        tolerance = np.timedelta64(int(float(self.params["ball_tolerance_ms"]) * 1000), "us")

        for idx, pos in enumerate(positions):
            choices = []
            if pos < ball_ts.size:
                choices.append(pos)
            if pos > 0:
                choices.append(pos - 1)
            if not choices:
                continue
            best = min(choices, key=lambda choice: abs(ball_ts[choice] - timestamps[idx]))
            if abs(ball_ts[best] - timestamps[idx]) <= tolerance:
                nearest[idx] = best
                valid[idx] = True
        return nearest, valid

    def _valid_coordinates(self, coords):
        valid = np.isfinite(coords).all(axis=1)
        for invalid_value in self.params.get("invalid_coordinate_values", []):
            valid &= ~np.isclose(coords, float(invalid_value)).any(axis=1)
        return valid

    def _read_identity(self, f, rows):
        fields = [field for field in self.params["identity_fields"] if field in f]
        if not fields:
            return [None] * len(rows)

        values_by_field = {field: f[field][rows] for field in fields}
        identities = []
        for idx in range(len(rows)):
            parts = []
            for field in fields:
                value = values_by_field[field][idx]
                if isinstance(value, (bytes, np.bytes_)):
                    value = value.decode("utf-8")
                value = str(value)
                if value:
                    parts.append(f"{field}={value}")
            identities.append(";".join(parts) if parts else None)
        return identities

    def _confidence(self, distance):
        threshold = float(self.params["distance_threshold_m"])
        if self.params["confidence_mode"] != "linear_inverse_distance":
            raise ValueError(f"Unsupported confidence_mode: {self.params['confidence_mode']}")
        raw = 1.0 - (distance / threshold)
        confidence = max(0.0, min(1.0, raw))
        return confidence ** float(self.params["confidence_power"])

    def _passes_sideline_filter(self, ball_xyz):
        if not self.params["sideline_filter_enabled"]:
            return True
        if self.params["sideline_reference"] != "ball_y":
            raise ValueError(f"Unsupported sideline_reference: {self.params['sideline_reference']}")
        return self._sideline_distance(ball_xyz) > float(self.params["sideline_exclusion_m"])

    def _sideline_distance(self, ball_xyz):
        return float(self.params["pitch_half_width_m"]) - abs(float(ball_xyz[1]))

    def _trajectory_diagnostics(self, ball, ball_index):
        default = {
            "trajectory_angle_deg": None,
            "trajectory_speed_before_mps": None,
            "trajectory_speed_after_mps": None,
            "trajectory_speed_delta_ratio": None,
            "trajectory_passed": not self.params["trajectory_filter_enabled"],
        }
        pre_idx = self._window_endpoint_index(
            ball,
            ball_index,
            direction="pre",
            window_ms=float(self.params["trajectory_pre_window_ms"]),
        )
        post_idx = self._window_endpoint_index(
            ball,
            ball_index,
            direction="post",
            window_ms=float(self.params["trajectory_post_window_ms"]),
        )
        if pre_idx is None or post_idx is None:
            return default

        contact_xyz = ball["xyz"][ball_index]
        pre_vec = contact_xyz - ball["xyz"][pre_idx]
        post_vec = ball["xyz"][post_idx] - contact_xyz
        if self.params["trajectory_use_xy_only"]:
            pre_vec = pre_vec[:2]
            post_vec = post_vec[:2]

        pre_norm = float(np.linalg.norm(pre_vec))
        post_norm = float(np.linalg.norm(post_vec))
        min_norm = float(self.params["trajectory_min_vector_norm_m"])
        if pre_norm < min_norm or post_norm < min_norm:
            return default

        cos_angle = float(np.dot(pre_vec, post_vec) / (pre_norm * post_norm))
        cos_angle = max(-1.0, min(1.0, cos_angle))
        angle_deg = float(np.degrees(np.arccos(cos_angle)))
        pre_dt = self._elapsed_seconds(ball["timestamps"][pre_idx], ball["timestamps"][ball_index])
        post_dt = self._elapsed_seconds(ball["timestamps"][ball_index], ball["timestamps"][post_idx])
        if pre_dt <= 0 or post_dt <= 0:
            return default

        speed_before = pre_norm / pre_dt
        speed_after = post_norm / post_dt
        denom = max(speed_before, 1e-9)
        speed_delta_ratio = abs(speed_after - speed_before) / denom

        angle_passed = angle_deg >= float(self.params["trajectory_min_angle_deg"])
        speed_passed = speed_delta_ratio >= float(self.params["trajectory_min_speed_delta_ratio"])
        mode = self.params["trajectory_change_mode"]
        if mode == "either_angle_or_speed":
            trajectory_passed = angle_passed or speed_passed
        elif mode == "angle":
            trajectory_passed = angle_passed
        elif mode == "speed":
            trajectory_passed = speed_passed
        elif mode == "both_angle_and_speed":
            trajectory_passed = angle_passed and speed_passed
        else:
            raise ValueError(f"Unsupported trajectory_change_mode: {mode}")

        return {
            "trajectory_angle_deg": angle_deg,
            "trajectory_speed_before_mps": speed_before,
            "trajectory_speed_after_mps": speed_after,
            "trajectory_speed_delta_ratio": speed_delta_ratio,
            "trajectory_passed": bool(trajectory_passed),
        }

    @staticmethod
    def _elapsed_seconds(start, end):
        delta_us = (end - start).astype("timedelta64[us]").astype(np.int64)
        return float(delta_us) / 1_000_000.0

    @staticmethod
    def _window_endpoint_index(ball, ball_index, *, direction, window_ms):
        timestamps = ball["timestamps"]
        contact_ts = timestamps[ball_index]
        window = np.timedelta64(int(window_ms * 1000), "us")
        if direction == "pre":
            start_ts = contact_ts - window
            left = int(np.searchsorted(timestamps, start_ts, side="left"))
            if left >= ball_index:
                return None
            return left
        if direction == "post":
            end_ts = contact_ts + window
            right = int(np.searchsorted(timestamps, end_ts, side="right")) - 1
            if right <= ball_index:
                return None
            return right
        raise ValueError(f"Unsupported trajectory endpoint direction: {direction}")

    @staticmethod
    def _position_ms(timestamp, base_ts):
        delta_us = (timestamp - base_ts).astype("timedelta64[us]").astype(np.int64)
        return int(round(delta_us / 1000.0))

    def _nms(self, candidates):
        if self.params["nms_scope"] != "sample":
            raise ValueError(f"Unsupported nms_scope: {self.params['nms_scope']}")
        window_us = int(float(self.params["nms_window_ms"]) * 1000)
        selected = []
        for candidate in sorted(candidates, key=lambda item: item.confidence, reverse=True):
            timestamp_us = candidate.timestamp.astype("datetime64[us]").astype(np.int64)
            duplicate = False
            for kept in selected:
                kept_us = kept.timestamp.astype("datetime64[us]").astype(np.int64)
                if abs(timestamp_us - kept_us) <= window_us:
                    duplicate = True
                    break
            if not duplicate:
                selected.append(candidate)
        return sorted(selected, key=lambda item: item.timestamp)

    def _candidate_to_event(self, candidate):
        event = {
            "head": self.params["head_name"],
            "label": self.params["label"],
            "position_ms": candidate.position_ms,
            "timestamp_utc": str(candidate.timestamp).replace("T", " "),
            self.params["confidence_output_key"]: float(candidate.confidence),
        }
        if self.params["include_diagnostics"]:
            event["metadata"] = {
                "distance_m": candidate.distance_m,
                "joint": candidate.joint,
                "player_id": candidate.player_id,
                "ball_timestamp_utc": str(candidate.ball_timestamp).replace("T", " "),
                "ball_y": candidate.ball_y,
                "sideline_distance_m": candidate.sideline_distance_m,
                **candidate.trajectory,
            }
        return event


class H5HeaderDistanceSpotter(H5HeaderSpotter):
    """Header spotting from head-ball distance only."""

    variant_name = "h5_header_distance"


class H5HeaderDistanceSpeedSpotter(H5HeaderSpotter):
    """Header spotting from head-ball distance plus ball speed change."""

    variant_name = "h5_header_distance_speed"


class H5HeaderDistanceAngleSpotter(H5HeaderSpotter):
    """Header spotting from head-ball distance plus ball trajectory angle change."""

    variant_name = "h5_header_distance_angle"


class H5HeaderDistanceSpeedAngleSpotter(H5HeaderSpotter):
    """Header spotting from distance, speed change, and angle change."""

    variant_name = "h5_header_distance_speed_angle"


def build_rule_based_model(config):
    rule_params = get_component_params_by_kind(config, "algorithm")
    variant = (rule_params or {}).get("type", "h5_header_distance")
    registry = {
        "h5_header_distance": H5HeaderDistanceSpotter,
        "h5_header_distance_speed": H5HeaderDistanceSpeedSpotter,
        "h5_header_distance_angle": H5HeaderDistanceAngleSpotter,
        "h5_header_distance_speed_angle": H5HeaderDistanceSpeedAngleSpotter,
    }
    try:
        return registry[variant](config)
    except KeyError as exc:
        raise ValueError(
            f"Unsupported rule-based model: {variant}. "
            f"Expected one of: {', '.join(sorted(registry))}."
        ) from exc
