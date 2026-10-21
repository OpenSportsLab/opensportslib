"""Bounded timestamp-uniform sampling for clip action inputs."""
import numpy as np
from .cache import SkeletonCache
from .session import decode_video, iter_skeletons


def classification_frames(path, settings, count, cached=False):
    """Two-pass timestamp-uniform sampling with O(count) skeleton storage."""
    if cached or settings.get('cache', {}).get('mode') == 'read':
        cache = SkeletonCache(path if cached else settings['cache']['path'])
        first, last = cache[0].timestamp, cache[-1].timestamp
    else:
        # Exact bounds without materializing video/skeleton arrays. Decode is
        # included in benchmarks; no pose inference on this first pass.
        first, last, next_time = None, None, 0.
        fps = settings.get('sample_fps', 25)
        for _, timestamp, _, _, _ in decode_video(path):
            if timestamp + 1e-9 >= next_time:
                first = timestamp if first is None else first
                last = timestamp
                next_time = np.floor((timestamp+1e-9)*fps)/fps + 1/fps
        if first is None:
            raise ValueError('Classification video has no frames')
    if count < 1:
        raise ValueError('num_frames must be positive')
    targets = np.linspace(first, last, count)
    selected = []
    source = iter(SkeletonCache(path)) if cached else iter_skeletons(path, settings)
    try:
        for frame in source:
            while len(selected) < count and frame.timestamp + 1e-9 >= targets[len(selected)]:
                selected.append(frame)
            if len(selected) == count:
                break
    finally:
        close = getattr(source, 'close', None)
        if close:
            close()
    if len(selected) != count:
        raise ValueError('Video ended before classification sampling targets')
    return selected

