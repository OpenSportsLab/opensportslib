"""Bounded shot-local windows and overlap-add; shared by live and cache paths."""
from dataclasses import dataclass

import numpy as np


@dataclass
class SkeletonWindow:
    frames: list
    indices: list
    valid: np.ndarray


def iter_windows(frames, length=64, overlap=32, include_boundaries=False):
    if length < 1 or not 0 <= overlap < length:
        raise ValueError('Require length > 0 and 0 <= overlap < length')
    step = length - overlap
    pending, indices, shot = [], [], None
    has_full = False

    def window():
        count = len(pending)
        return SkeletonWindow(pending + [pending[-1]]*(length-count),
                              indices.copy(), np.arange(length) < count)

    for index, frame in enumerate(frames):
        if shot is not None and frame.shot_id != shot:
            # The retained overlap has already been predicted. Only create
            # a tail window when genuinely new observations remain.
            if pending and (not has_full or len(pending) > overlap):
                yield window()
            if include_boundaries:
                yield None  # finalize old-shot scores immediately at the cut
            pending, indices, has_full = [], [], False
        shot = frame.shot_id
        pending.append(frame)
        indices.append(index)
        if len(pending) == length:
            yield window()
            pending, indices = pending[step:], indices[step:]
            has_full = True
    if pending and (not has_full or len(pending) > overlap):
        yield window()


class RollingScores:
    """Accumulate only unfinished frame scores, in chronological order."""
    def __init__(self):
        self.pending = {}

    def add(self, window, scores):
        scores = np.asarray(scores)
        if scores.ndim != 2 or len(scores) != len(window.valid):
            raise ValueError('Spotting model must return [T,C] scores')
        for offset, index in enumerate(window.indices):
            if index in self.pending:
                frame, total, count = self.pending[index]
                self.pending[index] = frame, total+scores[offset], count+1
            else:
                self.pending[index] = window.frames[offset], scores[offset].copy(), 1

    def finalize_before(self, index):
        for key in sorted(k for k in self.pending if k < index):
            frame, total, count = self.pending.pop(key)
            yield key, frame, total/count

    def finish(self):
        yield from self.finalize_before(float('inf'))


def sequence_batch(window):
    from opensportslib.core.utils.data import _flatten_graphs_to_batch
    from .records import frame_graph
    import torch
    batch = _flatten_graphs_to_batch([[frame_graph(f) for f in window.frames]])
    batch.time_mask = torch.as_tensor(window.valid[None])
    return batch
