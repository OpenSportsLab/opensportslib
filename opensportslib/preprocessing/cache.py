"""Explicit, immutable chunked skeleton caches. No pickle or implicit writes."""
from dataclasses import fields
from hashlib import sha256
from pathlib import Path
import json
import os
import shutil
import tempfile

import numpy as np

from .records import SCHEMA_VERSION, SkeletonFrame, stack_frames


def file_sha256(path):
    digest = sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def cache_identity(video_path, settings, provenance):
    return {'schema_version': SCHEMA_VERSION, 'video_sha256': file_sha256(video_path),
            'source_video_path': str(Path(video_path).resolve()),
            'settings': settings, 'models': provenance,
            'sampling_version': 1, 'selector_version': 1, 'tracker_version': 1}


def identity_digest(identity):
    return sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_cache(destination, frames, identity, chunk_size=256):
    """Publish a complete directory in one rename; never overwrite caches."""
    destination = Path(destination)
    if chunk_size < 1:
        raise ValueError('chunk_size must be positive')
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.skeleton-', dir=destination.parent))
    chunks, pending, total = [], [], 0

    def flush():
        nonlocal total
        directory = temporary / f'{len(chunks):06d}'
        directory.mkdir()
        for key, value in stack_frames(pending).items():
            np.save(directory / f'{key}.npy', value, allow_pickle=False)
        chunks.append({'directory': directory.name, 'count': len(pending)})
        total += len(pending)
        pending.clear()

    try:
        for frame in frames:
            pending.append(frame)
            if len(pending) == chunk_size:
                flush()
        if pending:
            flush()
        if not total:
            raise ValueError('Video contains no timestamped frames')
        manifest = {'schema_version': SCHEMA_VERSION, 'identity': identity,
                    'identity_sha256': identity_digest(identity), 'complete': True,
                    'count': total, 'chunks': chunks}
        (temporary / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        os.rename(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


class SkeletonCache:
    """Lazy reader holding at most one memory-mapped chunk."""
    def __init__(self, path, expected_identity=None):
        self.path = Path(path)
        try:
            self.manifest = json.loads((self.path / 'manifest.json').read_text())
        except (OSError, ValueError) as exc:
            raise ValueError(f'Incomplete skeleton cache: {path}') from exc
        m = self.manifest
        if not m.get('complete') or m.get('schema_version') != SCHEMA_VERSION:
            raise ValueError('Incomplete or unsupported skeleton cache')
        if identity_digest(m['identity']) != m.get('identity_sha256'):
            raise ValueError('Corrupt skeleton cache identity')
        if expected_identity is not None and m['identity'] != expected_identity:
            raise ValueError('Skeleton cache identity mismatch; extract a new cache')
        if m['count'] <= 0 or sum(c['count'] for c in m['chunks']) != m['count']:
            raise ValueError('Invalid skeleton cache length')
        self.ends = np.cumsum([c['count'] for c in m['chunks']])
        self._chunk_index = None
        self._arrays = None

    def __len__(self):
        return self.manifest['count']

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        chunk = int(np.searchsorted(self.ends, index, side='right'))
        if chunk != self._chunk_index:
            entry = self.manifest['chunks'][chunk]
            directory = self.path / entry['directory']
            if directory.parent != self.path or not directory.name.isdigit():
                raise ValueError('Invalid skeleton chunk path')
            self._arrays = {f.name: np.load(directory / f'{f.name}.npy', mmap_mode='r',
                                            allow_pickle=False) for f in fields(SkeletonFrame)}
            if any(len(a) != entry['count'] for a in self._arrays.values()):
                raise ValueError('Truncated skeleton cache chunk')
            self._chunk_index = chunk
        offset = index - (int(self.ends[chunk - 1]) if chunk else 0)
        # Copy keeps yielded records independent of mmap/chunk lifetime.
        values = {k: a[offset].copy() if isinstance(a[offset], np.ndarray) else a[offset].item()
                  for k, a in self._arrays.items()}
        return SkeletonFrame(**values).validate()

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]


def warm_clip_cache(config, destination, split='train', max_clips=None):
    """Explicit cache writes; annotations continue to reference source videos.

    Random spotting windows are a seeded sample, not exhaustive coverage. A
    read-only training run rejects any request that was not warmed exactly.
    """
    import random
    import torch
    from opensportslib.core.config.accessors import get_split_annotation_path, get_system_seed
    from opensportslib.core.utils.config import select_device
    from opensportslib.datasets.builder import build_dataset
    from opensportslib.datasets.video_raw import build_video_loader
    from opensportslib.models.preprocessors import PoseTracking
    from .config import preprocessing_config
    if max_clips is not None and max_clips < 1:
        raise ValueError('max_clips must be positive')
    seed=get_system_seed(config)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    dataset=build_dataset(config,get_split_annotation_path(config,split),split=split)
    if config.TASK.lower()=='localization':
        wrapper=dataset
        dataset=wrapper.building_dataset(wrapper.cfg,gpu=0,default_args=wrapper.default_args)
    if hasattr(dataset, 'loader'):
        dataset=dataset.dataset
    loader=build_video_loader(dataset,batch_size=1,num_workers=0,shuffle=False)
    settings=preprocessing_config(config)
    settings['cache']={'mode':'read_write','path':str(destination)}
    frontend=PoseTracking(settings).to(select_device(config.SYSTEM))
    count=0
    try:
        for batch in loader:
            frontend(batch)
            count+=len(batch['video_meta'])
            if max_clips is not None and count>=max_clips:
                break
    finally:
        frontend.close()
    return count
