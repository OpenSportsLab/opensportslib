"""Small action-task orchestration helpers; no alternative model framework."""
from collections import deque
from pathlib import Path
import json

import numpy as np

from opensportslib.core.config.accessors import (
    get_data_sampling, get_data_classes, get_split_cfg,
)
from opensportslib.preprocessing.windows import iter_windows, RollingScores, sequence_batch


def manifest_sources(config, annotation_path):
    """Resolve video sources from ordinary action annotations."""
    root = Path(getattr(get_split_cfg(config, 'test'), 'source_path', '.') or '.')
    document = json.loads(Path(annotation_path).read_text())
    from opensportslib.core.utils.load_annotations import expand_localization_intervals
    for record in document['data']:
        for segment in expand_localization_intervals(record):
            segment['output_path'] = str(root / segment['logical_path'])
            yield segment['id'], root / segment['source_path'], segment


class EventFinalizer:
    """Bounded symmetric local-max suppression, matching existing NMS ties.

    Keep raw neighboring scores even when their own event was suppressed.
    This matches non_maximum_supression rather than greedy peak removal.
    """
    def __init__(self, classes, fps, radius=0, threshold=.01, observer=None, offset_seconds=0.):
        self.observer = observer
        self.offset_seconds = offset_seconds
        self.classes, self.fps = classes, fps
        self.radius, self.threshold = int(radius), threshold
        self.pending, self.history = deque(), deque()

    def _index(self, frame):
        return int((frame.timestamp-self.offset_seconds)*self.fps+1e-7)

    def push(self, frame, scores):
        self.pending.append((frame, scores))
        current = self._index(frame)
        while self.pending and current >= self._index(self.pending[0][0])+self.radius:
            yield from self._pop()

    def _pop(self):
        from opensportslib.metrics.localization_metric import frame_score_events
        frame, scores = self.pending.popleft()
        index = self._index(frame)
        while self.history and self._index(self.history[0][0]) < index-self.radius:
            self.history.popleft()
        neighbors = list(self.history) + list(self.pending)
        row = np.asarray(scores).copy()
        for other, other_scores in neighbors:
            other_index = self._index(other)
            if other_index != index and abs(other_index-index) <= self.radius:
                row[np.asarray(other_scores) > scores] = -np.inf
        events = frame_score_events(row, self.classes, int((frame.timestamp-self.offset_seconds)*1000+1e-7), index, self.threshold)
        self.history.append((frame, scores))
        if self.observer is not None:
            self.observer(frame, scores)
        yield from events

    def finish(self):
        while self.pending:
            yield from self._pop()
        self.history.clear()


def spotting_events(model, config, frames, observer=None, *, fps=None, offset_seconds=0.):
    """Yield finalized event batches; retain O(window + suppression radius)."""
    sampling = get_data_sampling(config)
    length = int(sampling.get('clip_len', 64))
    split = get_split_cfg(config, 'test')
    overlap = int(getattr(split, 'overlap_len', length//2))
    fps = float(fps if fps is not None else sampling.get('extract_fps', 25))
    rolling = RollingScores()
    finalizer = EventFinalizer(get_data_classes(config), fps,
                               getattr(split, 'nms_window', 0) or 0, observer=observer, offset_seconds=offset_seconds)
    shot = None

    def finalized(rows):
        nonlocal shot
        events = []
        for _, frame, scores in rows:
            if shot is not None and frame.shot_id != shot:
                events.extend(finalizer.finish())
            shot = frame.shot_id
            events.extend(finalizer.push(frame, scores))
        return events

    source = iter_windows(frames, length, overlap, include_boundaries=True)
    try:
        for window in source:
            if window is None:
                events = finalized(rolling.finish())
                events.extend(finalizer.finish())
                shot = None
                if events:
                    yield events
                continue
            events = finalized(rolling.finalize_before(window.indices[0]))
            batch = sequence_batch(window)
            _, scores = model.predict(batch, use_amp=False)
            rolling.add(window, scores[0])
            if events:
                yield events
        events = finalized(rolling.finish())
        events.extend(finalizer.finish())
        if events:
            yield events
    finally:
        close = getattr(frames, 'close', None)
        if close:
            close()


def evaluate_spotting(config, annotation_path, predictions):
    """Use OSL mAP/F1 implementations directly on in-memory predictions."""
    from opensportslib.metrics.localization_metric import compute_mAPs_E2E, ForegroundF1
    fps = float(get_data_sampling(config).get('extract_fps', 25))
    classes = {name: i+1 for i, name in enumerate(get_data_classes(config))}
    document = json.loads(Path(annotation_path).read_text())
    if not any(record.get('events') for record in document['data']):
        return None
    sources = list(manifest_sources(config, annotation_path))
    truth = []
    expected = {}
    for identity, path, segment in sources:
        if segment['annotation_status'] != 'verified':
            continue
        events = [dict(label=e['label'], frame=int(float(e['position_ms'])*fps/1000))
                  for e in segment['events']]
        name = segment['output_path']
        truth.append({'path': name, 'events': events})
        expected[name] = {e['frame']: classes[e['label']] for e in events}
    pred = []
    f1 = ForegroundF1()
    for record in predictions['data']:
        path = record['inputs'][0]['path']
        if path not in expected:
            continue
        # A common evaluation clock supports videos with different effective FPS.
        events = [{**e, 'frame': int(float(e['position_ms'])*fps/1000)} for e in record['events']]
        pred.append({'video': path, 'events': events})
        # Existing event JSON can contain multiple classes at a frame; choose
        # the strongest emitted event for exact-frame foreground F1.
        strongest = {}
        for event in sorted(events, key=lambda e:e['confidence']):
            strongest[event['frame']] = classes[event['label']]
        frames = sorted(set(expected[path]) | set(strongest))
        for index in frames:
            f1.update(expected[path].get(index,0), strongest.get(index,0))
    tolerances = [int(round(s*fps)) for s in (0,1,2,3,4)]
    maps, _ = compute_mAPs_E2E(truth, pred, tolerances=tolerances)
    return {'mAP': float(np.mean(maps)), 'mAP_by_tolerance_seconds': dict(zip((0,1,2,3,4),maps)),
            'event_foreground_f1': f1.get(None)}
