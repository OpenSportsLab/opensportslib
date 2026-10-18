"""Frozen model-owned video frontend, with explicit per-video session state."""
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
import os

import numpy as np
import torch
from torch import nn

from opensportslib.preprocessing.config import preprocessing_config, requires_raw_video
from opensportslib.preprocessing.records import FEATURE_DIM, SkeletonFrame, frame_graph
from opensportslib.preprocessing.session import SkeletonSession


@lru_cache(maxsize=32)
def _source_hash(path, size, modified):
    from opensportslib.preprocessing.cache import file_sha256
    return file_sha256(path)


class PoseTracking(nn.Module):
    """COCO-17 player graphs. Only the action model has torch parameters.

    forward is clip-local. open_session is the only way to retain association
    across chunks. ONNX sessions are not checkpoint state or worker state.
    """
    input_spec = {'modality': 'video', 'dtype': 'uint8', 'color': 'rgb', 'uncropped': True}
    output_spec = {'feature_dim': FEATURE_DIM, 'joint_order': 'coco17', 'nodes': 'players'}

    def __init__(self, settings):
        super().__init__()
        self.settings = dict(settings)
        from opensportslib.preprocessing.config import validate_preprocessing
        validate_preprocessing(self.settings)
        self._backends = {}
        self.register_buffer('_device_anchor', torch.empty(0), persistent=False)
        self.train(False)

    def train(self, mode=True):
        return super().train(False)

    @property
    def backend(self):
        device = str(self._device_anchor.device)
        key = (os.getpid(), device)
        if key not in self._backends:
            from opensportslib.preprocessing.backend import RTMLibBackend
            self._backends[key] = RTMLibBackend({**self.settings, 'device': device})
        return self._backends[key]

    def close(self):
        for backend in self._backends.values():
            backend.close()
        self._backends.clear()

    def open_session(self):
        return PoseVideoSession(self)

    def iter_video(self, path, config, *, start_time_ms=0, end_time_ms=None):
        """Chronological inference through the ordinary spotting video reader."""
        from opensportslib.datasets.video_raw import iter_raw_video
        from opensportslib.core.config.accessors import get_data_sampling, get_data_transform, get_loader_backend
        sampling = get_data_sampling(config)
        loader = iter_raw_video(path, int(sampling.get('clip_len',64)),
            sampling.get('extract_fps',25), decoder=get_loader_backend(config),
            resize=get_data_transform(config).get('resize'),
            start_time_ms=start_time_ms, end_time_ms=end_time_ms)
        session = self.open_session()
        try:
            for frames, meta in loader:
                for row in session.push(frames, meta):
                    if row.frame_index >= 0:
                        yield row
            session.finish()
        finally:
            session.close()
            loader.close()

    def _identity(self, meta):
        source = Path(meta['source'])
        stat = source.stat()
        settings = {k: v for k, v in self.settings.items() if k not in {'cache', 'device', 'model_dir'}}
        # Hashes/artifact provenance identify the actual backend, not an alias.
        # Resolve artifact checksums without importing RTMLib/ONNX Runtime or
        # constructing GPU sessions. CPU cache readers can use pinned hashes.
        if self._backends:
            provenance = next(iter(self._backends.values())).provenance
            models = [{k: a[k] for k in ('name', 'sha256')} for a in provenance.get('artifacts', [])]
            if not models:
                models = provenance
        else:
            from opensportslib.preprocessing.backend import resolve_artifact
            pose = settings.get('pose', 'rtmpose_s')
            names = [pose] if pose == 'rtmo_s' else [settings.get('detector', 'yolox_m'), pose]
            models = []
            for name in names:
                pin = settings.get('artifacts', {}).get(name, {})
                if pin.get('sha256'):
                    digest = pin['sha256']
                else:
                    _, _, artifact = resolve_artifact(name, settings)
                    digest = artifact['sha256']
                models.append({'name': name, 'sha256': digest})
        identity = dict(schema_version=1, video_sha256=_source_hash(str(source), stat.st_size, stat.st_mtime_ns),
                        settings=settings, models=models, selector_version=1, tracker_version=1,
                        frontend_version=2, request=meta, reset_policy='independent_clip_v1')
        return identity

    def observations(self, frames, meta):
        from opensportslib.preprocessing.cache import SkeletonCache, identity_digest, write_cache
        cache = self.settings.get('cache', {})
        mode = cache.get('mode', 'off')
        destination = identity = None
        if mode != 'off' and any(f['valid'] for f in meta['frames']):
            identity = self._identity(meta)
            destination = Path(cache['path'])/identity_digest(identity)
            if destination.exists():
                return list(SkeletonCache(destination, identity))
            if mode == 'read':
                raise FileNotFoundError(f'Exact-clip skeleton cache missing: {destination}')
        session = self.open_session()
        try:
            rows = session.push(frames, meta)
        finally:
            session.close()
        if destination is not None:
            try:
                write_cache(destination, rows, identity)
            except OSError as exc:
                # Another rank may have atomically published the same request.
                import errno
                if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                    raise
                SkeletonCache(destination, identity)
        return rows

    def forward(self, batch):
        key = 'pixel_values' if 'pixel_values' in batch else 'frame'
        images = batch[key]
        classification = key == 'pixel_values'
        rows, owners = [], []
        for b, metas in enumerate(batch['video_meta']):
            for v, meta in enumerate(metas):
                frames = images[b, v].permute(1, 0, 2, 3) if classification else images[b]
                rows.append(self.observations(frames, meta))
                owners.append(b)
        return graph_sequences(rows, self._device_anchor.device, owners=owners, batch_size=len(batch['video_meta']))


class PoseVideoSession:
    """A single source/view, bounded overlap reuse; never stored on the module."""
    def __init__(self, frontend):
        self.frontend = frontend
        self.session = None
        self.source = None
        self.recent = {}
        self.closed = False
        self.finished = False

    def push(self, frames, meta):
        if self.closed or self.finished:
            raise RuntimeError('Pose video session is finished or closed')
        source = meta['source']
        if self.source is not None and source != self.source:
            raise ValueError('Open a separate pose session for each video/view')
        self.source = source
        if frames.dtype != torch.uint8:
            raise ValueError('Pose frontend requires uint8, unnormalized RGB video')
        output = []
        for offset, item in enumerate(meta['frames']):
            if not item['valid']:
                output.append(SkeletonFrame.empty(self.frontend.settings.get('max_players', 2),
                    timestamp=item['timestamp'], sample_time=item['sample_time'], frame_index=-1,
                    pts=-1, time_base=item['time_base'], shot_id=-1, image_size=item['image_size']))
                continue
            index = item['frame_index']
            if index in self.recent:
                output.append(replace(self.recent[index], sample_time=item['sample_time']))
                continue
            w, h = item['decoded_size']
            image = frames[offset, :, :h, :w].detach().cpu().permute(1, 2, 0).numpy()
            if image.dtype != np.uint8:
                raise ValueError('Pose frontend requires uint8, unnormalized RGB video')
            image = np.ascontiguousarray(image[..., ::-1])
            if self.session is None:
                self.session = SkeletonSession(self.frontend.backend, self.frontend.settings)
            row = self.session.push(image, timestamp=item['timestamp'], frame_index=index,
                pts=item['pts'], time_base=item['time_base'], sampled=True,
                sample_time=item['sample_time'])[0]
            ow, oh = item['image_size']
            scale = np.array([ow/w, oh/h], np.float32)
            row.joints *= scale
            row.boxes *= np.tile(scale, 2)
            row.image_size = np.asarray([ow, oh], np.int32)
            row.validate()
            self.recent[index] = row
            output.append(row)
        # Enough for repeated frames and the next overlapping action window.
        limit = max(1, 2*len(meta['frames']))
        while len(self.recent) > limit:
            self.recent.pop(next(iter(self.recent)))
        return output

    def finish(self):
        if self.closed:
            raise RuntimeError('Pose video session is closed')
        self.finished = True
        if self.session is not None:
            self.session.finish()
        return []

    def close(self):
        if self.session is not None:
            self.session.close()
        self.recent.clear()
        self.closed = True


def graph_sequences(rows, device, *, owners=None, batch_size=None):
    from opensportslib.core.utils.data import _flatten_graphs_to_batch
    length = max(map(len, rows))
    graphs, masks, shots = [], [], []
    for sequence in rows:
        padded = sequence + [sequence[-1]]*(length-len(sequence))
        graphs.append([frame_graph(f) for f in padded])
        masks.append([f.frame_index >= 0 for f in sequence]+[False]*(length-len(sequence)))
        shots.append([f.shot_id for f in sequence]+[-1]*(length-len(sequence)))
    result = _flatten_graphs_to_batch(graphs).to(device)
    result.time_mask = torch.tensor(masks, dtype=torch.bool, device=device)
    result.shot_ids = torch.tensor(shots, dtype=torch.long, device=device)
    result.view_owners = owners if owners is not None else list(range(len(rows)))
    result.video_batch_size = batch_size if batch_size is not None else len(rows)
    return result


def build_preprocessor(config):
    return PoseTracking(preprocessing_config(config)) if requires_raw_video(config) else None


def temporal_segments(mask, shots=None):
    """Contiguous valid ranges; neither padding nor camera cuts join state."""
    start, previous = None, None
    for index, valid in enumerate(mask.tolist()):
        shot = int(shots[index]) if shots is not None else 0
        if start is not None and (not valid or shot != previous):
            yield start, index
            start = None
        if valid and start is None:
            start = index
        previous = shot
    if start is not None:
        yield start, len(mask)
