"""Chronological extraction sessions with bounded frame storage."""
from fractions import Fraction
from pathlib import Path
import math

import numpy as np

from .config import validate_preprocessing
from .records import SkeletonFrame
from .selection import PlayerAssociation, TennisSelector


class SkeletonSession:
    """Per-video temporal state around a reusable frozen backend.

    push consumes one BGR frame and returns zero or one observation. Call it
    for every decoded frame so camera cuts are detected before sampling.
    """
    def __init__(self, backend, settings):
        validate_preprocessing(settings)
        self.backend = backend
        self.settings = settings
        self.fps = float(settings.get('sample_fps', 25))
        self.players = settings.get('max_players', 2)
        self.selector = TennisSelector()
        self.association = PlayerAssociation(self.players, settings.get('track_retention_seconds', .5))
        self.shot_id = 0
        self.next_time = 0.
        self.previous = None
        self.previous_time = None
        self.closed = False
        self.finished = False

    def push(self, image, *, timestamp, frame_index, pts, time_base):
        import cv2
        if self.closed or self.finished:
            raise RuntimeError('Skeleton session is finished or closed')
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError('Video timestamps must be finite and nonnegative')
        if self.previous_time is not None and timestamp <= self.previous_time:
            raise ValueError('Video presentation timestamps must increase strictly')
        small = cv2.resize(image, (64, 36)).astype(np.float32)/255
        cut = self.previous is not None and float(np.abs(small-self.previous).mean()) > self.settings.get('cut_threshold', .32)
        if self.previous_time is not None and timestamp-self.previous_time > 1:
            cut = True
        self.previous, self.previous_time = small, timestamp
        if cut:
            self.shot_id += 1
            self.selector.reset()
            self.association.reset()
        if timestamp + 1e-9 < self.next_time:
            return []
        sample_time = math.floor((timestamp+1e-9)*self.fps)/self.fps
        self.next_time = sample_time + 1/self.fps
        top_down = hasattr(self.backend, 'detect') and not getattr(self.backend, 'one_stage', False)
        if top_down:
            boxes = self.backend.detect(image)
            if self.settings.get('selection', 'tennis_conservative') == 'tennis_conservative':
                selected, view_valid = self.selector.select(image, boxes, self.association.tracks)
            else:
                selected, view_valid = [(i,-1) for i in range(min(len(boxes),self.players))], True
            # Court eligibility is considered before the candidate cap. Large
            # spectator boxes must not evict the small far-court competitor.
            selected = selected[:min(self.players, self.settings.get('max_candidates',8))]
            roles = [role for _,role in selected]
            boxes = np.asarray([boxes[i] for i,_ in selected],np.float32).reshape(-1,4)
            joints, scores = self.backend.estimate(image, boxes)
            selected = list(enumerate(roles))
        else:
            boxes, joints, scores = self.backend.predict(image)
            if self.settings.get('selection', 'tennis_conservative') == 'tennis_conservative':
                selected, view_valid = self.selector.select(image, boxes, self.association.tracks)
            else:
                selected, view_valid = [(i,-1) for i in range(min(len(boxes),self.players))], True
        boxes = np.asarray(boxes, np.float32)
        joints, scores = np.asarray(joints, np.float32), np.asarray(scores, np.float32)
        if boxes.shape != (len(boxes), 4) or joints.shape != (len(boxes), 17, 2) or scores.shape != (len(boxes), 17):
            raise ValueError('Pose backend must return boxes[N,4], joints[N,17,2], scores[N,17]')
        if not all(np.isfinite(a).all() for a in (boxes, joints, scores)):
            raise ValueError('Pose backend returned non-finite values')
        limit = self.settings.get('max_candidates', 8)
        boxes, joints, scores = boxes[:limit], joints[:limit], scores[:limit]
        selected = [(i,role) for i,role in selected if i < limit]
        valid = scores >= self.settings.get('joint_threshold', .3)
        selected = [(i, role) for i, role in selected if valid[i].sum() >= 3]
        h, w = image.shape[:2]
        frame = SkeletonFrame.empty(self.players, timestamp=timestamp, sample_time=sample_time,
            frame_index=frame_index, pts=pts, time_base=time_base, shot_id=self.shot_id,
            image_size=(w, h), view_valid=view_valid)
        for slot, index, identity, role in self.association.update(
                boxes, joints, valid, selected, timestamp, (w, h)):
            frame.boxes[slot], frame.joints[slot] = boxes[index], joints[index]
            frame.confidence[slot], frame.joint_valid[slot] = scores[index], valid[index]
            frame.player_valid[slot], frame.track_ids[slot], frame.roles[slot] = True, identity, role
        return [frame.validate()]

    def finish(self):
        if self.closed:
            raise RuntimeError('Skeleton session is closed')
        self.finished = True
        return []

    def close(self):
        self.association.reset()
        self.selector.reset()
        self.previous = None
        self.closed = True


def decode_video(path):
    """Yield BGR frames with exact source PTS, retaining one decoded frame.

    No full-video reads or codec-frame worker queues are used. Thread count
    is one; this is stricter than the four-frame prefetch budget.
    """
    import av
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_count = 1
        origin = None
        for index, frame in enumerate(container.decode(stream)):
            if frame.pts is None or frame.time_base is None:
                raise ValueError('Video frames require presentation timestamps')
            clock = Fraction(frame.pts) * frame.time_base
            if origin is None:
                origin = clock
            yield (frame.to_ndarray(format='bgr24'), float(clock-origin), index,
                   frame.pts, (frame.time_base.numerator, frame.time_base.denominator))


def iter_skeletons(video_path, settings, backend=None):
    from .backend import RTMLibBackend
    from .cache import SkeletonCache, file_sha256
    cache_cfg = settings.get('cache', {})
    if cache_cfg.get('mode', 'off') == 'read':
        cache = SkeletonCache(cache_cfg['path'])
        identity = cache.manifest['identity']
        if identity.get('video_sha256') != file_sha256(video_path):
            raise ValueError('Skeleton cache video mismatch')
        expected = {k: v for k, v in settings.items() if k not in {'cache', 'model_dir'}}
        if identity.get('settings') != expected:
            raise ValueError('Skeleton cache extraction settings mismatch')
        recorded = {a['name']: a['sha256'] for a in identity.get('models', {}).get('artifacts', [])}
        for name, artifact in settings.get('artifacts', {}).items():
            expected_hash = artifact.get('sha256')
            if artifact.get('path'):
                expected_hash = file_sha256(Path(artifact['path']).expanduser())
            if expected_hash and recorded.get(name) != expected_hash:
                raise ValueError(f'Skeleton cache model checksum mismatch for {name}')
        yield from cache
        return
    owned = backend is None
    if owned:
        backend = RTMLibBackend(settings)
    session = SkeletonSession(backend, settings)
    try:
        for image, timestamp, index, pts, time_base in decode_video(video_path):
            yield from session.push(image, timestamp=timestamp, frame_index=index,
                                    pts=pts, time_base=time_base)
        yield from session.finish()
    finally:
        session.close()
        if owned:
            backend.close()


def extract(video_path, destination, settings, backend=None, chunk_size=256):
    """Explicit pre-extraction; writes cache only at the requested destination."""
    from .backend import RTMLibBackend
    from .cache import cache_identity, write_cache
    if settings.get('cache', {}).get('mode', 'off') != 'off':
        raise ValueError('extract() requires cache.mode=off')
    owned = backend is None
    backend = RTMLibBackend(settings) if owned else backend
    try:
        clean = {k: v for k, v in settings.items() if k not in {'cache', 'model_dir'}}
        identity = cache_identity(video_path, clean, backend.provenance)
        return write_cache(destination, iter_skeletons(video_path, settings, backend), identity, chunk_size)
    finally:
        if owned:
            backend.close()
