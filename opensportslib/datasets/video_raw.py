"""Raw video contracts shared by ordinary video datasets and model frontends.

No pose models are constructed here. PyAV supplies exact source clocks on the
OpenCV loader route; only selected images are retained. DALI is lazy/optional.
"""
from functools import lru_cache
from pathlib import Path
import math

import numpy as np
import torch


@lru_cache(maxsize=2)
def decoder_versions(decoder):
    import av
    import cv2
    versions = {'av': av.__version__, 'opencv': cv2.__version__, 'sampling': 1}
    if decoder == 'dali':
        import importlib.metadata
        versions['dali'] = 'unavailable'
        for package in ('nvidia-dali-cuda120', 'nvidia-dali-cuda130'):
            try:
                versions['dali'] = importlib.metadata.version(package)
                break
            except importlib.metadata.PackageNotFoundError:
                continue
    return versions


@lru_cache(maxsize=64)
def _probe(path, size, modified):
    import av
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_count = 1
        decoded = container.decode(stream)
        first = next(decoded, None)
        if first is None or first.pts is None or first.time_base is None:
            raise ValueError(f'Video requires timestamped frames: {path}')
        count = stream.frames
        if not count:
            count = 1 + sum(1 for _ in decoded)
        return dict(count=count, fps=float(stream.average_rate or stream.guessed_rate),
                    width=first.width, height=first.height, origin=first.pts,
                    time_base=[first.time_base.numerator, first.time_base.denominator])


def probe_video(path):
    path = Path(path).resolve()
    stat = path.stat()
    return _probe(str(path), stat.st_size, stat.st_mtime_ns).copy()


def uniform_indices(start, end, count):
    if end <= start or count < 1:
        raise ValueError('Uniform sampling requires a nonempty interval and count > 0')
    # Match the DALI reader's std::round, including halfway cases.
    return np.floor(np.linspace(start, end - 1, count) + .5).astype(np.int64).tolist()


def read_raw_clip(path, indices, *, sample_times=None, resize=None, decoder='opencv', device_id=None):
    """Return uint8 [T,C,H,W] and serializable, source-space frame metadata."""
    import av
    import cv2
    path = str(Path(path).resolve())
    info = probe_video(path)
    indices = [int(i) for i in indices]
    wanted = {i for i in indices if 0 <= i < info['count']}
    images, clocks = {}, {}
    last = max(wanted, default=-1)
    # Retain at most T images and clock records, irrespective of video duration.
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_count = 1
        for index, frame in enumerate(container.decode(stream)):
            if index > last:
                break
            if index not in wanted:
                continue
            if frame.pts is None:
                raise ValueError('Video frames require presentation timestamps')
            clock = float(frame.pts * frame.time_base)
            origin = info['origin'] * info['time_base'][0] / info['time_base'][1]
            clocks[index] = (int(frame.pts), [frame.time_base.numerator, frame.time_base.denominator], clock-origin)
            if decoder != 'dali':
                images[index] = frame.to_ndarray(format='rgb24')
    if wanted - clocks.keys():
        raise ValueError('Video ended before requested source frames')
    if decoder == 'dali' and wanted:
        from .video_dali import decode_selected
        if device_id is None:
            device_id = torch.cuda.current_device() if torch.cuda.is_available() else 0
        images = decode_selected(path, sorted(wanted), clocks, info, device_id)
    elif decoder not in {'opencv', 'dali'}:
        raise ValueError(f'Unsupported raw video decoder: {decoder}')
    width, height = info['width'], info['height']
    if resize:
        height, width = int(resize.get('height') or height), int(resize.get('width') or width)
    output = []
    metadata = []
    for offset, index in enumerate(indices):
        valid = index in clocks
        if valid:
            image = images[index]
            if tuple(image.shape[:2]) != (height, width):
                image = cv2.resize(image, (width, height))
            pts, time_base, timestamp = clocks[index]
        else:
            image = np.zeros((height, width, 3), np.uint8)
            pts, time_base, timestamp = -1, info['time_base'], index/info['fps']
        output.append(torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1))
        metadata.append(dict(valid=valid, frame_index=index if valid else -1,
            pts=pts, time_base=time_base, timestamp=timestamp,
            sample_time=float(sample_times[offset]) if sample_times is not None else timestamp,
            image_size=[info['width'], info['height']], decoded_size=[width, height]))
    return torch.stack(output), dict(source=path, decoder=f'{decoder}:raw-v1',
                                    decoder_versions=decoder_versions(decoder), frames=metadata)


def read_spotting_clip(path, start, length, fps, *, source_start=None, source_end=None, **kwargs):
    info = probe_video(path)
    # Preserve the existing FrameReader/annotation stride policy.
    from opensportslib.core.utils.video_processing import get_stride
    stride = max(1, get_stride(info['fps'], min(fps, info['fps'])))
    first = int(source_start or 0)
    stop = int(source_end if source_end is not None else info['count'])
    count = math.ceil((stop-first)/stride)
    grid = range(start, start+length)
    indices = [first+i*stride if 0 <= i < count else -1 for i in grid]
    return read_raw_clip(path, indices, sample_times=[i*stride/info['fps'] for i in grid], **kwargs)


def effective_fps(path, requested):
    from opensportslib.core.utils.video_processing import get_stride
    native = probe_video(path)['fps']
    return native/max(1, get_stride(native, min(requested, native)))


def iter_raw_video(path, length, fps, *, resize=None, decoder='opencv', start_time_ms=0, end_time_ms=None):
    """Sequential version of the raw reader; retain only one action chunk."""
    import av
    import cv2
    from opensportslib.core.utils.video_processing import get_stride
    path = str(Path(path).resolve())
    info = probe_video(path)
    stride = max(1, get_stride(info['fps'], min(fps, info['fps'])))
    first = max(0, int(math.ceil(start_time_ms*info['fps']/1000-1e-9)))
    stop = info['count'] if end_time_ms is None else int(math.ceil(end_time_ms*info['fps']/1000-1e-9))
    if stop > info['count'] or stop <= first:
        raise ValueError('Invalid source video interval')
    if decoder == 'dali':
        from .video_dali import iter_decoded
        source = iter_decoded(path, length, stride, info, first, stop)
    else:
        def cpu_frames():
            with av.open(path) as container:
                stream = container.streams.video[0]
                stream.thread_count = 1
                origin = info['origin']*info['time_base'][0]/info['time_base'][1]
                for index, frame in enumerate(container.decode(stream)):
                    if index >= stop:
                        break
                    if index < first or (index-first) % stride:
                        continue
                    if frame.pts is None:
                        raise ValueError('Video frames require presentation timestamps')
                    yield (index, frame.to_ndarray(format='rgb24'), int(frame.pts),
                           [frame.time_base.numerator, frame.time_base.denominator],
                           float(frame.pts*frame.time_base)-origin)
        source = cpu_frames()
    frames, metadata = [], []
    try:
        for index, image, pts, time_base, timestamp in source:
            height, width = image.shape[:2]
            if resize:
                height, width = int(resize.get('height') or height), int(resize.get('width') or width)
                image = cv2.resize(image, (width, height))
            frames.append(torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1))
            metadata.append(dict(valid=True, frame_index=index, pts=pts, time_base=time_base,
                timestamp=timestamp, sample_time=(index-first)/info['fps'],
                image_size=[info['width'],info['height']], decoded_size=[width,height]))
            if len(frames) == length:
                yield torch.stack(frames), dict(source=path, decoder=f'{decoder}:raw-v1',
                                               decoder_versions=decoder_versions(decoder), frames=metadata)
                frames, metadata = [], []
        if frames:
            yield torch.stack(frames), dict(source=path, decoder=f'{decoder}:raw-v1',
                                           decoder_versions=decoder_versions(decoder), frames=metadata)
    finally:
        source.close()


def video_collate(samples):
    """Pad time/space/views without changing coordinates; keep metadata nested."""
    from torch.utils.data._utils.collate import default_collate
    key = 'pixel_values' if 'pixel_values' in samples[0] else 'frame'
    classification = key == 'pixel_values'
    values = [s[key] for s in samples]
    # Classification samples: list of [T,C,H,W] views. Spotting: [T,C,H,W].
    views = values if classification else [[v] for v in values]
    nv = max(map(len, views))
    nt = max(v.shape[0] for row in views for v in row)
    nh = max(v.shape[-2] for row in views for v in row)
    nw = max(v.shape[-1] for row in views for v in row)
    target_device = views[0][0].device
    frames = torch.zeros((len(samples), nv, nt, 3, nh, nw), dtype=torch.uint8, device=target_device)
    view_mask = torch.zeros((len(samples), nv), dtype=torch.bool)
    time_mask = torch.zeros((len(samples), nv, nt), dtype=torch.bool)
    metas = []
    for b, row in enumerate(views):
        row_meta = samples[b]['video_meta'] if classification else [samples[b]['video_meta']]
        metas.append(row_meta)
        for v, image in enumerate(row):
            t, _, h, w = image.shape
            frames[b, v, :t, :, :h, :w] = image
            view_mask[b, v] = True
            time_mask[b, v, :t] = torch.tensor([f['valid'] for f in row_meta[v]['frames']])
    result = {key: frames.permute(0, 1, 3, 2, 4, 5) if classification else frames[:, 0],
              'video_meta': metas, 'view_mask': view_mask,
              'time_mask': time_mask if classification else time_mask[:, 0]}
    for name in samples[0]:
        if name not in {key, 'video_meta', 'time_mask', 'view_mask'}:
            result[name] = default_collate([s[name] for s in samples])
    return result


def build_video_loader(dataset, **kwargs):
    """Keep torch sampler semantics while keeping CUDA out of workers."""
    from torch.utils.data import DataLoader
    if getattr(dataset, 'raw_video', False):
        kwargs['collate_fn'] = video_collate
        if getattr(dataset, 'raw_decoder', 'opencv') == 'dali':
            for name in ('multiprocessing_context', 'prefetch_factor', 'persistent_workers'):
                kwargs.pop(name, None)
            kwargs['num_workers'] = 0
        elif kwargs.get('num_workers', 0):
            kwargs['prefetch_factor'] = 1
        if not kwargs.get('num_workers', 0):
            kwargs.pop('prefetch_factor', None)
            kwargs.pop('multiprocessing_context', None)
            kwargs['persistent_workers'] = False
    loader = DataLoader(dataset, **kwargs)
    return _DaliVideoLoader(loader) if getattr(dataset, 'raw_decoder', None) == 'dali' else loader


class _DaliVideoLoader:
    """Internal batch adapter; the public dataset stays Video/ActionSpotDataset."""
    def __init__(self, loader):
        self.loader = loader

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        return iter(self.loader)

    def __getattr__(self, name):
        loader = self.__dict__['loader']
        return getattr(loader, name) if hasattr(loader, name) else getattr(loader.dataset, name)

    def delete(self):
        pass  # Request-scoped DALI pipelines are released after every decode.

    def reset(self):
        pass  # torch samplers are iterated afresh each epoch.
