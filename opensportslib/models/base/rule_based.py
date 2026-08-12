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


class H5HeaderSpotterBase:
    """Shared plumbing for H5 header spotters.

    Holds only what every spotting algorithm needs regardless of how it finds
    contacts: the OSL JSON envelope, manifest path resolution and the
    position_ms conversion. Subclasses supply ``_predict_sample`` and their own
    parameter defaults.
    """

    variant_name = ""

    def predict(self, dataset):
        """Spot events for every sample of a dataset.

        Args:
            dataset (H5OSLJsonSpottingDataset): Manifest wrapper exposing
                `samples` and `source_path`.

        Returns:
            predictions (dict): OSL JSON v2 payload with one entry per sample
                under `data`, each holding the spotted `events`.
        """
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
        """Spot events for a single manifest sample.

        Args:
            sample (dict): Manifest entry with `inputs` and optional `metadata`.
            source_path (str): Directory relative input paths resolve against.

        Returns:
            events (List[dict]): Spotted events in chronological order.
        """
        raise NotImplementedError

    @staticmethod
    def _resolve_path(source_path, path):
        """Resolve a manifest input path against the dataset source directory.

        Args:
            source_path (str): Directory relative paths resolve against.
            path (str): Absolute or relative input path.

        Returns:
            resolved (str): Absolute path.
        """
        if os.path.isabs(path):
            return path
        return os.path.join(source_path, path)

    @staticmethod
    def _position_ms(timestamp, base_ts):
        """Convert a timestamp to milliseconds elapsed since a reference.

        Args:
            timestamp (np.datetime64): Instant to convert.
            base_ts (np.datetime64): Reference instant, i.e. position 0.

        Returns:
            position_ms (int): Milliseconds between the two instants.
        """
        delta_us = (timestamp - base_ts).astype("timedelta64[us]").astype(np.int64)
        return int(round(delta_us / 1000.0))


class H5HeaderSpotter(H5HeaderSpotterBase):
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


DEFAULT_SKELETON_RULE_PARAMS = {
    "label": "header",
    "head_name": "action",
    # ball gates
    "ball_height_min_m": 1.3,
    "ball_height_max_m": 3.0,
    "invalid_value": -1.0,
    # ball trajectory gates (velocity from +/-1 frame, acceleration from +/-2)
    "velocity_change_min_mps": 2.0,
    "velocity_mag_min_mps": 1.0,
    "angle_change_min_deg": 10.0,
    "accel_z_change_min_mps2": 8.0,
    "incoming_speed_min_mps": 4.0,
    # player gates
    "head_joints": ["nose"],
    "head_ball_distance_max_m": 0.4,
    "facing_dot_min": -0.5,
    "ankle_height_max_m": 1.2,
    "hand_check_enabled": True,
    # dwell filter (rejects the ball resting near a head)
    "dwell_window_frames": 3,
    "dwell_distance_factor": 1.2,
    "dwell_distance_m": None,
    "dwell_max_frames": 5,
    # de-duplication
    "nms_window_frames": 25,
    "fps": 50.0,
    # io / output
    "required_input_type": "player_joints_h5",
    "ball_path_field": "ball_path",
    "timestamp_field": "timestamp_utc",
    "output_task": "action_spotting",
    "include_diagnostics": True,
    "created_by": "h5_header_skeleton_rule",
    # position_ms is measured from the first ball-track timestamp in the file
    "metadata_start_field": "start_utc",
    "metadata_end_field": "end_utc",
    "confidence_output_key": "confidence_score",
    "position_offset_ms": 0.0,
}


SKELETON_RULE_VARIANTS = {
    "h5_header_skeleton": {},
    # Recall-first: keep only the gates that cost (almost) no true headers.
    # The trajectory gates and the narrow height band together reject about a
    # tenth of real headers -- flick-ons and glancing contacts barely disturb
    # the ball -- so they are dropped. The hand check is kept because it is
    # nearly free in recall and removes many arm and keeper contacts, and the
    # wider suppression window merges the extra detections each duel produces.
    "h5_header_skeleton_recall": {
        # Any tracked head joint counts, not just the nose: a nose is often
        # untracked while an ear or the neck is, and skipping those players
        # loses contacts outright.
        "head_joints": ["nose", "neck", "l_ear", "r_ear", "l_eye", "r_eye"],
        "velocity_change_min_mps": 0.0,
        "velocity_mag_min_mps": 0.0,
        # A 10 degree bend is the one trajectory test worth keeping here. It
        # costs about 3 points of recall and returns 8 of precision, measured
        # by sweeping every gate: 93.3/66.0 against 96.2/58.5 without it.
        # Set to 0.0 for maximum recall at the cost of far more false alarms.
        "angle_change_min_deg": 10.0,
        "accel_z_change_min_mps2": 0.0,
        "incoming_speed_min_mps": 0.0,
        "ball_height_min_m": 0.5,
        "ball_height_max_m": 8.0,
        "ankle_height_max_m": 99.0,
        "dwell_max_frames": 999999,
        "nms_window_frames": 40,
        "created_by": "h5_header_skeleton_recall_rule",
    },
}


class H5HeaderSkeletonSpotter(H5HeaderSpotterBase):
    """Frame-joined skeleton heuristic for header spotting.

    Logic: ball height and trajectory gates (velocity change, direction change, 
    z-acceleration change, incoming speed), then per-player checks (head-ball distance,
    facing the ball, hands farther than the head, no acrobatic pose) and a dwell filter, 
    joined on ``frame``/``player_id`` rather than timestamps.
    Events carry ``timestamp_utc`` from the matched ball sample, so the output 
    schema matches the other H5 header variants.

    A sibling of :class:`H5HeaderSpotter` rather than a subclass: the two share
    the output envelope but no detection logic, and this one deliberately does 
    not participate in ``HEADER_RULE_VARIANTS``.
    """

    variant_name = "h5_header_skeleton"

    def __init__(self, config):
        """Build the spotter from a canonical config.

        Args:
            config: Config whose `algorithm` component params override
                `DEFAULT_SKELETON_RULE_PARAMS`.
        """
        rule_params = get_component_params_by_kind(config, "algorithm")
        if not rule_params:
            rule_params = get_data_params(config)
        params = dict(DEFAULT_SKELETON_RULE_PARAMS)
        params.update(rule_params or {})
        configured_variant = params.pop("type", None) or self.variant_name
        params.update(SKELETON_RULE_VARIANTS.get(configured_variant, {}))
        params["model_variant"] = configured_variant
        self.params = params

    def _predict_sample(self, sample, source_path):
        """Spot headers across every joints/ball input of a sample.

        Args:
            sample (dict): Manifest entry with `inputs` and optional `metadata`.
            source_path (str): Directory relative input paths resolve against.

        Returns:
            events (List[dict]): Header events ordered by position.
        """
        inputs = sample.get("inputs", []) or []
        events = []
        for input_obj in inputs:
            if input_obj.get("type") != self.params["required_input_type"]:
                continue
            ball_path = input_obj.get(self.params["ball_path_field"])
            if not ball_path:
                continue
            events.extend(
                self._detect(
                    joint_path=self._resolve_path(source_path, input_obj["path"]),
                    ball_path=self._resolve_path(source_path, ball_path),
                    sample_metadata=sample.get("metadata", {}) or {},
                )
            )
        events.sort(key=lambda event: event["position_ms"])
        return events

    # ------------------------------------------------------------- ball
    def _load_ball_track(self, ball_path):
        """Load the ball track, sorted by frame.

        Args:
            ball_path (str): Path to the ball H5 file.

        Returns:
            frames (np.ndarray): Frame number per sample, ascending.
            xyz (np.ndarray): Ball positions of shape (num_samples, 3).
            timestamps (np.ndarray): UTC instant per sample, datetime64[us].
        """
        timestamp_field = self.params["timestamp_field"]
        with h5py.File(ball_path, "r") as f:
            for field in ("frame", "x", "y", "z", timestamp_field):
                if field not in f:
                    raise ValueError(f"Ball H5 is missing dataset '{field}': {ball_path}")
            frames = f["frame"][:]
            order = np.argsort(frames, kind="stable")
            frames = frames[order]
            xyz = np.stack([f["x"][:], f["y"][:], f["z"][:]], axis=1).astype(float)[order]
            timestamps = np.asarray(
                [parse_utc(value) for value in f[timestamp_field][:]],
                dtype="datetime64[us]",
            )[order]
        return frames, xyz, timestamps

    def _ball_velocities(self, frames, xyz):
        """Compute per-sample ball velocity from the preceding sample.

        Velocity is left at zero wherever the pair of samples is unusable:
        an invalid coordinate, a non-positive gap, or a gap long enough that
        the ball was untracked in between.

        Args:
            frames (np.ndarray): Frame number per sample, ascending.
            xyz (np.ndarray): Ball positions of shape (num_samples, 3).

        Returns:
            velocities (np.ndarray): Velocity in m/s, shape (num_samples, 3).
        """
        invalid = float(self.params["invalid_value"])
        fps = float(self.params["fps"])
        vel = np.zeros_like(xyz)
        dt = np.diff(frames) / fps
        pair_valid = (
            (xyz[1:, 0] != invalid)
            & (xyz[:-1, 0] != invalid)
            & (dt > 0)
            & (dt < 0.1)
        )
        safe_dt = np.where(dt > 0, dt, 1.0)
        step = np.diff(xyz, axis=0) / safe_dt[:, None]
        vel[1:][pair_valid] = step[pair_valid]
        return vel

    def _trajectory_pass(self, idx, n, vel):
        """Test whether the ball path around a sample looks like an impact.

        Applies the velocity-change, direction-change, z-acceleration and
        incoming-speed gates. Samples too close to either end of the track skip
        the gates that cannot be computed there.

        Args:
            idx (int): Index of the contact sample.
            n (int): Number of ball samples.
            vel (np.ndarray): Velocities of shape (num_samples, 3).

        Returns:
            passed (bool): Whether every applicable gate accepted the sample.
            diagnostics (dict): Measured angle, speeds and acceleration change,
                empty when a gate rejected the sample or none could be applied.
        """
        p = self.params
        if not (2 < idx < n - 2):
            return True, {}
        vel_before = vel[idx - 1]
        vel_after = vel[idx + 1]
        vel_change = float(np.linalg.norm(vel_after - vel_before))
        if vel_change < float(p["velocity_change_min_mps"]):
            return False, {}
        mag_before = float(np.linalg.norm(vel_before))
        mag_after = float(np.linalg.norm(vel_after))
        if mag_after < float(p["velocity_mag_min_mps"]) or mag_before < float(p["velocity_mag_min_mps"]):
            return False, {}
        cos_angle = float(np.dot(vel_before, vel_after) / (mag_before * mag_after))
        cos_angle = max(-1.0, min(1.0, cos_angle))
        angle_change = float(np.degrees(np.arccos(cos_angle)))
        if angle_change < float(p["angle_change_min_deg"]):
            return False, {}
        diagnostics = {
            "velocity_change_mps": vel_change,
            "angle_change_deg": angle_change,
            "incoming_speed_mps": mag_before,
        }
        if 3 < idx < n - 3:
            half_step = 2.0 / float(p["fps"])
            accel_before = (vel[idx, 2] - vel[idx - 2, 2]) / half_step
            accel_after = (vel[idx + 2, 2] - vel[idx, 2]) / half_step
            if abs(accel_after - accel_before) < float(p["accel_z_change_min_mps2"]):
                return False, {}
            if mag_before < float(p["incoming_speed_min_mps"]):
                return False, {}
            diagnostics["accel_z_change_mps2"] = float(abs(accel_after - accel_before))
        return True, diagnostics

    # ------------------------------------------------------------ joints
    _BODY_COLUMNS = (
        "l_wrist_x", "l_wrist_y", "l_wrist_z",
        "r_wrist_x", "r_wrist_y", "r_wrist_z",
        "l_shoulder_x", "l_shoulder_y", "r_shoulder_x", "r_shoulder_y",
        "l_ankle_z", "r_ankle_z",
    )

    @property
    def _JOINT_COLUMNS(self):
        """Columns this run needs: the configured head joints plus the body."""
        head = tuple(f"{joint}_{axis}"
                     for joint in self.params["head_joints"] for axis in "xyz")
        return head + self._BODY_COLUMNS

    def _heads_within(self, joints, rows, ball_pos, max_dist):
        """Find players at a frame whose head is inside the contact radius.

        Vectorised across players and head joints; a frame usually has many
        players and almost none of them near the ball.

        Args:
            joints (dict): Loaded joint columns.
            rows (range): Row indexes for this frame.
            ball_pos (np.ndarray): Ball position of shape (3,).
            max_dist (float): Contact radius in metres.

        Returns:
            rows (List[int]): Row indexes inside the radius, nearest first.
            distances (List[float]): Head-ball distance for each row.
            joints_used (List[str]): Which head joint was nearest, per row.
        """
        lo, hi = rows.start, rows.stop
        if lo >= hi:
            return [], [], []

        names = list(self.params["head_joints"])
        invalid = float(self.params["invalid_value"])
        points = np.stack(
            [np.stack([joints[f"{n}_x"][lo:hi],
                       joints[f"{n}_y"][lo:hi],
                       joints[f"{n}_z"][lo:hi]], axis=1) for n in names],
            axis=1,
        )
        distances = np.linalg.norm(points - ball_pos, axis=2)
        untracked = (points[:, :, 0] == invalid) | ~np.isfinite(points).all(axis=2)
        distances[untracked] = np.inf

        nearest_joint = distances.argmin(axis=1)
        nearest = distances.min(axis=1)
        inside = np.flatnonzero(nearest < max_dist)
        inside = inside[np.argsort(nearest[inside])]
        return (
            [lo + int(i) for i in inside],
            [float(nearest[i]) for i in inside],
            [names[int(nearest_joint[i])] for i in inside],
        )

    def _closest_head(self, joints, row, ball_pos):
        """Find the tracked head joint nearest the ball for one player row.

        Args:
            joints (dict): Loaded joint columns.
            row (int): Row index for the player at this frame.
            ball_pos (np.ndarray): Ball position of shape (3,).

        Returns:
            distance (float): Distance to the nearest tracked head joint, or
                None when no configured head joint is tracked on this row.
            position (np.ndarray): That joint's position, or None.
            name (str): That joint's name, or None.
        """
        invalid = float(self.params["invalid_value"])
        best = (None, None, None)
        for joint in self.params["head_joints"]:
            point = np.array([
                joints[f"{joint}_x"][row],
                joints[f"{joint}_y"][row],
                joints[f"{joint}_z"][row],
            ])
            if point[0] == invalid or not np.isfinite(point).all():
                continue
            distance = float(np.linalg.norm(point - ball_pos))
            if best[0] is None or distance < best[0]:
                best = (distance, point, joint)
        return best

    def _load_joints(self, joint_path, wanted_frames=None):
        """Load joint rows, sorted by frame.

        Restricting the read matters: the joint table holds millions of rows
        while a scan touches only the frames near ball candidates, and the
        variable-length `player_id` column dominates a full read.

        Args:
            joint_path (str): Path to the player joints H5 file.
            wanted_frames (np.ndarray): Frame numbers to read rows for. Reads
                every row when None.
                Default: None.

        Returns:
            joints (dict): Column name to values, ordered by frame. `is_home`
                is None when the file does not carry it.
        """
        with h5py.File(joint_path, "r") as f:
            missing = [c for c in ("frame", "player_id", *self._JOINT_COLUMNS) if c not in f]
            if missing:
                raise ValueError(f"Joint H5 is missing datasets {missing}: {joint_path}")
            frames = f["frame"][:]
            if wanted_frames is None:
                rows = np.arange(len(frames))
            else:
                rows = np.flatnonzero(np.isin(frames, wanted_frames))
            # h5py fancy selection needs ascending indices; reorder to frame
            # order in memory afterwards.
            rows = np.sort(rows)
            row_frames = frames[rows]
            order = np.argsort(row_frames, kind="stable")

            joints = {"frame": row_frames[order]}
            for column in ("player_id", *self._JOINT_COLUMNS, "is_home"):
                if column not in f:
                    joints[column] = None
                    continue
                values = f[column][rows] if len(rows) else f[column][:0]
                if column in self._JOINT_COLUMNS:
                    values = values.astype(float, copy=False)
                joints[column] = values[order]
        return joints

    @staticmethod
    def _rows_at_frame(joints, frame):
        """Locate the joint rows recorded at a frame.

        Args:
            joints (dict): Loaded joint columns, ordered by frame.
            frame (int): Frame number to look up.

        Returns:
            rows (range): Row indexes for that frame, one per tracked player.
        """
        lo = int(np.searchsorted(joints["frame"], frame, side="left"))
        hi = int(np.searchsorted(joints["frame"], frame, side="right"))
        return range(lo, hi)

    def _row_for_player(self, joints, frame, player_id):
        """Locate one player's joint row at a frame.

        Args:
            joints (dict): Loaded joint columns, ordered by frame.
            frame (int): Frame number to look up.
            player_id: Player identifier as stored in the H5 file.

        Returns:
            row (int): Row index, or None when that player is untracked there.
        """
        for row in self._rows_at_frame(joints, frame):
            if joints["player_id"][row] == player_id:
                return row
        return None

    def _dwell_frames(self, joints, ball_frames, ball_xyz, ball_idx, player_id):
        """Count how long the ball stays beside a player's head.

        A header is a brief contact, so a ball that lingers near the head
        across the window was carried or held rather than headed.

        Args:
            joints (dict): Loaded joint columns, ordered by frame.
            ball_frames (np.ndarray): Frame number per ball sample.
            ball_xyz (np.ndarray): Ball positions of shape (num_samples, 3).
            ball_idx (int): Index of the contact sample.
            player_id: Player identifier as stored in the H5 file.

        Returns:
            count (int): Samples in the window with the head close to the ball.
        """
        p = self.params
        # Defaults to a multiple of the contact threshold; set dwell_distance_m
        # to keep the dwell radius fixed when widening that threshold.
        near = (
            float(p["dwell_distance_m"]) if p.get("dwell_distance_m") is not None
            else float(p["head_ball_distance_max_m"]) * float(p["dwell_distance_factor"])
        )
        window = int(p["dwell_window_frames"])
        count = 0
        for offset in range(-window, window + 1):
            check_idx = ball_idx + offset
            if check_idx < 0 or check_idx >= len(ball_frames):
                continue
            row = self._row_for_player(joints, ball_frames[check_idx], player_id)
            if row is None:
                continue
            distance, _, _ = self._closest_head(joints, row, ball_xyz[check_idx])
            if distance is not None and distance < near:
                count += 1
        return count

    # ------------------------------------------------------------ detect
    def _detect(self, joint_path, ball_path, sample_metadata):
        """Spot headers in one joints/ball pair.

        Ball samples at head height start as candidates, survive the trajectory
        gates, then face the per-player checks (head distance, facing the ball,
        hands clear of the head, no acrobatic pose) and the dwell filter.
        Survivors are de-duplicated so each contact yields one event.

        Args:
            joint_path (str): Path to the player joints H5 file.
            ball_path (str): Path to the ball H5 file.
            sample_metadata (dict): Manifest metadata; `start_utc` and
                `end_utc` bound the scan window when present.

        Returns:
            events (List[dict]): Spotted header events for this input pair.
        """
        p = self.params
        invalid = float(p["invalid_value"])
        frames, xyz, timestamps = self._load_ball_track(ball_path)
        n = len(frames)
        if n == 0:
            return []
        vel = self._ball_velocities(frames, xyz)

        candidate_mask = (
            (xyz[:, 2] >= float(p["ball_height_min_m"]))
            & (xyz[:, 2] <= float(p["ball_height_max_m"]))
            & (xyz[:, 0] != invalid)
        )
        start_utc = sample_metadata.get(p["metadata_start_field"])
        end_utc = sample_metadata.get(p["metadata_end_field"])
        if start_utc:
            candidate_mask &= timestamps >= parse_utc(start_utc)
        if end_utc:
            candidate_mask &= timestamps <= parse_utc(end_utc)
        candidate_indices = np.flatnonzero(candidate_mask)
        if candidate_indices.size == 0:
            return []

        # The scan reads joints at candidate frames, and the dwell filter looks
        # a few ball samples either side of those, so that neighbourhood is the
        # whole set of frames this sample can touch.
        dwell_window = int(p["dwell_window_frames"])
        neighbourhood = np.clip(
            candidate_indices[:, None] + np.arange(-dwell_window, dwell_window + 1),
            0,
            n - 1,
        )
        joints = self._load_joints(joint_path, wanted_frames=np.unique(frames[neighbourhood]))
        base_ts = timestamps.min()
        max_dist = float(p["head_ball_distance_max_m"])

        detections = []
        for ball_idx in candidate_indices:
            passed, trajectory = self._trajectory_pass(int(ball_idx), n, vel)
            if not passed:
                continue
            ball_pos = xyz[ball_idx]
            # Screen every player at this frame at once; only those with a head
            # inside the contact radius are worth the per-player checks below.
            rows = self._rows_at_frame(joints, frames[ball_idx])
            near_rows, head_dists, head_joints = self._heads_within(
                joints, rows, ball_pos, max_dist)
            for row, head_dist, head_joint in zip(near_rows, head_dists, head_joints):
                nose = np.array([
                    joints[f"{head_joint}_x"][row],
                    joints[f"{head_joint}_y"][row],
                    joints[f"{head_joint}_z"][row],
                ])
                # facing: shoulder-line normal vs. direction to the ball (2-D)
                facing = np.array([
                    -(joints["r_shoulder_y"][row] - joints["l_shoulder_y"][row]),
                    joints["r_shoulder_x"][row] - joints["l_shoulder_x"][row],
                ])
                to_ball = ball_pos[:2] - nose[:2]
                if float(np.dot(facing, to_ball)) < float(p["facing_dot_min"]):
                    continue
                # hands: a hand nearer the ball than the head is not a header
                l_hand = np.array([
                    joints["l_wrist_x"][row], joints["l_wrist_y"][row], joints["l_wrist_z"][row],
                ])
                r_hand = np.array([
                    joints["r_wrist_x"][row], joints["r_wrist_y"][row], joints["r_wrist_z"][row],
                ])
                hand_dist = min(
                    float(np.linalg.norm(l_hand - ball_pos)),
                    float(np.linalg.norm(r_hand - ball_pos)),
                )
                if p["hand_check_enabled"] and hand_dist < head_dist:
                    continue
                # acrobatic pose: both feet must be below ankle_height_max
                if (
                    joints["l_ankle_z"][row] > float(p["ankle_height_max_m"])
                    or joints["r_ankle_z"][row] > float(p["ankle_height_max_m"])
                ):
                    continue
                player_id = joints["player_id"][row]
                dwell = self._dwell_frames(joints, frames, xyz, int(ball_idx), player_id)
                if dwell > int(p["dwell_max_frames"]):
                    continue
                detections.append({
                    "ball_idx": int(ball_idx),
                    "frame": int(frames[ball_idx]),
                    "head_dist": head_dist,
                    "joint": head_joint,
                    "player_id": player_id,
                    "is_home": (
                        int(joints["is_home"][row]) if joints["is_home"] is not None else None
                    ),
                    "ball_z": float(ball_pos[2]),
                    "dwell": dwell,
                    "trajectory": trajectory,
                })

        return [
            self._detection_to_event(det, timestamps, base_ts)
            for det in self._nms_frames(detections)
        ]

    def _nms_frames(self, detections):
        """Reduce each cluster of detections to its closest contact.

        One aerial duel produces detections on many neighbouring frames; the
        one with the smallest head-ball distance represents the contact.

        Args:
            detections (List[dict]): Raw detections carrying `frame` and
                `head_dist`.

        Returns:
            kept (List[dict]): One detection per cluster, ordered by frame.
        """
        window = int(self.params["nms_window_frames"])
        detections = sorted(detections, key=lambda det: det["frame"])
        kept, processed = [], []
        for det in detections:
            if any(abs(det["frame"] - frame) <= window for frame in processed):
                continue
            cluster = [det] + [
                other for other in detections
                if other is not det and abs(other["frame"] - det["frame"]) <= window
            ]
            best = min(cluster, key=lambda item: item["head_dist"])
            kept.append(best)
            processed.append(best["frame"])
        return sorted(kept, key=lambda det: det["frame"])

    def _detection_to_event(self, det, timestamps, base_ts):
        """Convert a detection into an OSL JSON event.

        Args:
            det (dict): Detection produced by `_detect`.
            timestamps (np.ndarray): UTC instant per ball sample.
            base_ts (np.datetime64): Reference instant for `position_ms`.

        Returns:
            event (dict): Event with label, position, timestamp, confidence and
                optional diagnostics metadata.
        """
        p = self.params
        timestamp = timestamps[det["ball_idx"]]
        position_ms = self._position_ms(timestamp, base_ts) + int(round(float(p["position_offset_ms"])))
        confidence = max(0.0, min(1.0, 1.0 - det["head_dist"] / float(p["head_ball_distance_max_m"])))
        player_id = det["player_id"]
        if isinstance(player_id, (bytes, np.bytes_)):
            player_id = player_id.decode("utf-8")
        event = {
            "head": p["head_name"],
            "label": p["label"],
            "position_ms": position_ms,
            "timestamp_utc": str(timestamp).replace("T", " "),
            p["confidence_output_key"]: float(confidence),
        }
        if p["include_diagnostics"]:
            event["metadata"] = {
                "distance_m": det["head_dist"],
                "joint": det["joint"],
                "player_id": str(player_id) if player_id else None,
                "team": (
                    None if det["is_home"] is None
                    else ("home" if det["is_home"] == 1 else "away")
                ),
                "ball_z_m": det["ball_z"],
                "dwell_frames": det["dwell"],
                **det["trajectory"],
            }
        return event


class H5HeaderSkeletonRecallSpotter(H5HeaderSkeletonSpotter):
    """Skeleton header spotting tuned for recall over precision.

    Drops the trajectory and pose gates that reject real headers, keeping the
    hand and facing checks. Finds nearly every header the tracking supports at
    the cost of more false positives; use it to build a candidate set for
    review or for a downstream classifier rather than as a final answer.
    """

    variant_name = "h5_header_skeleton_recall"


def build_rule_based_model(config):
    rule_params = get_component_params_by_kind(config, "algorithm")
    variant = (rule_params or {}).get("type", "h5_header_distance")
    registry = {
        "h5_header_distance": H5HeaderDistanceSpotter,
        "h5_header_distance_speed": H5HeaderDistanceSpeedSpotter,
        "h5_header_distance_angle": H5HeaderDistanceAngleSpotter,
        "h5_header_distance_speed_angle": H5HeaderDistanceSpeedAngleSpotter,
        "h5_header_skeleton": H5HeaderSkeletonSpotter,
        "h5_header_skeleton_recall": H5HeaderSkeletonRecallSpotter,
    }
    try:
        return registry[variant](config)
    except KeyError as exc:
        raise ValueError(
            f"Unsupported rule-based model: {variant}. "
            f"Expected one of: {', '.join(sorted(registry))}."
        ) from exc
