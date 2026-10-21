"""Explicit export of observations consumed by a model-owned pose frontend."""
from dataclasses import fields
import json
from pathlib import Path

import numpy as np

from .records import SkeletonFrame, player_features


class PoseTraceWriter:
    """Synchronous observer; keeps only one request/JSON record in memory.

    Repeated sampled frames remain repeated in the export because they occupy
    separate positions in the model input. Coordinates are source-image pixels.
    """
    def __init__(self, path):
        self.path = Path(path)
        self.handle = None
        self.requests = 0
        self.observations = 0

    def __enter__(self):
        self.handle = self.path.open('x', encoding='utf-8')
        return self

    def __call__(self, records, metadata):
        if self.handle is None:
            raise RuntimeError('Pose trace writer is not open')
        for offset, frame in enumerate(records):
            skeleton = {}
            for field in fields(SkeletonFrame):
                value = getattr(frame, field.name)
                skeleton[field.name] = value.tolist() if isinstance(value, np.ndarray) else value
            row = dict(request_index=self.requests, sample_index=offset,
                       source=metadata['source'], decoder=metadata.get('decoder'),
                       input_frame=metadata['frames'][offset], skeleton=skeleton,
                       player_features=player_features(frame).tolist())
            self.handle.write(json.dumps(row, allow_nan=False) + '\n')
            self.observations += 1
        self.requests += 1

    def __exit__(self, *args):
        if self.handle is not None:
            self.handle.close()
            self.handle = None


def trace_frames(path, video_path):
    """Read chronological, distinct source frames without retaining the trace."""
    source = Path(video_path).resolve()
    previous = None
    with Path(path).open(encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            if Path(row['source']).resolve() != source:
                raise ValueError('Pose trace belongs to a different source video')
            frame = row['skeleton']
            if frame['frame_index'] < 0:
                continue
            if previous is not None and frame['frame_index'] < previous['frame_index']:
                raise ValueError('Pose trace must be chronological for single-video rendering')
            if previous is not None and frame['frame_index'] != previous['frame_index']:
                yield previous
            previous = frame
        if previous is not None:
            yield previous
