"""Versioned, backend-independent COCO-17 skeleton observations."""
from dataclasses import dataclass, fields

import numpy as np

SCHEMA_VERSION = 1
FEATURE_DIM = 73
JOINT_NAMES = (
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
    'right_knee', 'left_ankle', 'right_ankle',
)


@dataclass
class SkeletonFrame:
    joints: np.ndarray
    confidence: np.ndarray
    joint_valid: np.ndarray
    boxes: np.ndarray
    player_valid: np.ndarray
    track_ids: np.ndarray
    timestamp: float
    sample_time: float
    frame_index: int
    pts: int
    time_base: np.ndarray
    shot_id: int
    image_size: np.ndarray  # width, height
    view_valid: bool
    roles: np.ndarray  # -1 unknown, 0 far, 1 near

    @classmethod
    def empty(cls, players, *, timestamp, sample_time, frame_index, pts,
              time_base, shot_id, image_size, view_valid=False):
        return cls(
            np.zeros((players, 17, 2), np.float32),
            np.zeros((players, 17), np.float32),
            np.zeros((players, 17), bool), np.zeros((players, 4), np.float32),
            np.zeros(players, bool), np.full(players, -1, np.int64),
            float(timestamp), float(sample_time), int(frame_index), int(pts),
            np.asarray(time_base, np.int64), int(shot_id),
            np.asarray(image_size, np.int32), bool(view_valid),
            np.full(players, -1, np.int8),
        )

    def validate(self):
        p = len(self.player_valid)
        expected = {'joints': (p, 17, 2), 'confidence': (p, 17),
                    'joint_valid': (p, 17), 'boxes': (p, 4), 'track_ids': (p,),
                    'image_size': (2,), 'time_base': (2,), 'roles': (p,)}
        for name, shape in expected.items():
            if np.shape(getattr(self, name)) != shape:
                raise ValueError(f'Invalid skeleton {name}: expected {shape}')
        for name in ('joint_valid', 'player_valid'):
            if getattr(self, name).dtype != np.bool_:
                raise ValueError(f'Skeleton {name} must be boolean')
        if len(np.unique(self.track_ids[self.player_valid])) != int(self.player_valid.sum()):
            raise ValueError('Valid player IDs must be unique within a frame')
        if np.any(self.track_ids[self.player_valid] < 0):
            raise ValueError('Valid player IDs must be nonnegative')
        if np.any(self.boxes[self.player_valid, 2:] <= self.boxes[self.player_valid, :2]):
            raise ValueError('Valid player boxes must have positive width and height')
        if not np.all(self.image_size > 0) or self.time_base[1] <= 0:
            raise ValueError('Invalid image size or timestamp time base')
        if not np.isfinite([self.timestamp, self.sample_time]).all():
            raise ValueError('Skeleton timestamps must be finite')
        for name in ('joints', 'boxes', 'confidence'):
            if not np.isfinite(getattr(self, name)).all():
                raise ValueError(f'Skeleton {name} must be finite')
        if np.any(self.joint_valid & ~self.player_valid[:, None]):
            raise ValueError('Missing players cannot have valid joints')
        if np.any(self.track_ids[~self.player_valid] != -1):
            raise ValueError('Missing player observations must use track ID -1')
        return self


def stack_frames(frames):
    if not frames:
        raise ValueError('Cannot stack an empty skeleton chunk')
    for frame in frames:
        frame.validate()
    return {f.name: np.stack([getattr(frame, f.name) for frame in frames])
            for f in fields(SkeletonFrame)}


def player_features(frame):
    """Return [P,73] with global positions and masked body-relative joints."""
    frame.validate()
    boxes = frame.boxes
    size = np.maximum(boxes[:, 2:] - boxes[:, :2], 0)
    center = (boxes[:, :2] + boxes[:, 2:]) / 2
    scale = np.maximum(size.max(axis=1), 1)[:, None, None]
    relative = (frame.joints - center[:, None, :]) / scale
    relative = np.where(frame.joint_valid[..., None], relative, 0)
    joints = np.concatenate((relative, frame.confidence[..., None],
                             frame.joint_valid[..., None]), axis=-1)
    features = np.concatenate((center / frame.image_size, size / frame.image_size,
                               joints.reshape(len(boxes), -1),
                               frame.player_valid[:, None]), axis=1).astype(np.float32)
    features[~frame.player_valid] = 0
    return features


def frame_graph(frame):
    """Build PyG data lazily; optional pose/PyG dependencies stay isolated."""
    import torch
    from torch_geometric.data import Data
    valid = np.flatnonzero(frame.player_valid)
    edges = np.asarray([(i, j) for i in valid for j in valid if i != j], np.int64)
    edges = edges.reshape(-1, 2).T.copy()
    return Data(x=torch.from_numpy(player_features(frame)),
                edge_index=torch.from_numpy(edges),
                node_mask=torch.from_numpy(frame.player_valid.copy()))
