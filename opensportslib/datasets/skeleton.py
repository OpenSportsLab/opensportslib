"""Task adapters over canonical skeleton caches; never create GPU sessions."""
from pathlib import Path
import json

import numpy as np
from torch.utils.data import Dataset

from opensportslib.core.config.accessors import get_data_sampling, get_data_classes, get_data_params
from opensportslib.core.utils.data import tracking_spotting_collate_fn, tracking_spotting_eval_collate_fn
from opensportslib.datasets.classification_dataset import ClassificationDataset
from opensportslib.preprocessing.cache import SkeletonCache
from opensportslib.preprocessing.records import frame_graph


def cache_source(record, root):
    sources = [i for i in record.get('inputs', []) if i.get('type') == 'skeleton_cache']
    if len(sources) != 1:
        raise ValueError('Cached action training requires exactly one skeleton_cache input per record; run extraction first')
    return SkeletonCache(Path(root or '.') / sources[0]['path'])


def validate_cached_sampling(cache, config):
    settings = cache.manifest['identity'].get('settings', {})
    expected = float(get_data_sampling(config).get('extract_fps', 25))
    if settings.get('sample_fps', expected) != expected:
        raise ValueError('Cached sample rate differs from action configuration; extract a new cache')
    from opensportslib.preprocessing.config import preprocessing_config
    configured = preprocessing_config(config)
    for key, value in settings.items():
        if key in configured and key not in {'cache', 'model_dir', 'device'} and configured[key] != value:
            raise ValueError(f'Cached extraction setting {key} differs from action configuration')
    players = get_data_params(config).get('objects', {}).get('num_objects', 2)
    if len(cache[0].player_valid) != players:
        raise ValueError('Cached player count differs from action configuration')


class SkeletonClassificationDataset(ClassificationDataset):
    def __init__(self, config, annotations_path, split='train'):
        super().__init__(config, annotations_path, None, split)
        self.count = int(get_data_sampling(config).get('num_frames', 64))

    def __getitem__(self, index):
        sample = self.samples[index]
        cache = SkeletonCache(Path(self.video_path or '.') / sample['video_paths'][0])
        validate_cached_sampling(cache, self.config)
        from opensportslib.preprocessing.sampling import classification_frames
        frames = classification_frames(cache.path, {}, self.count, cached=True)
        result = {'graphs': [frame_graph(f) for f in frames],
                  'seq_len': self.count, 'id': sample['id']}
        if sample.get('label') is not None:
            result['label'] = sample['label']
        return result


class SkeletonSpottingDataset(Dataset):
    """Deterministic shot-local windows, using the live window schedule."""
    def __init__(self, config, annotation_path, root, split, overlap=32):
        from opensportslib.preprocessing.windows import iter_windows
        self.length = int(get_data_sampling(config).get('clip_len', 64))
        self.fps = float(get_data_sampling(config).get('extract_fps', 25))
        self.eval_mode = split in {'test', 'valid_data_frames'}
        self.collate_fn = tracking_spotting_eval_collate_fn if self.eval_mode else tracking_spotting_collate_fn
        self.classes = {name: i+1 for i, name in enumerate(get_data_classes(config))}
        self.caches, self.windows, self._labels, self._dense = {}, [], [], {}
        self.frame_times = {}
        doc = json.loads(Path(annotation_path).read_text())
        dilation = int(get_data_params(config).get('dilate_len', 0))
        for record in doc['data']:
            cache = cache_source(record, root)
            validate_cached_sampling(cache, config)
            video = record.get('id', str(cache.path))
            if video in self.caches:
                raise ValueError(f'Duplicate skeleton video ID: {video}')
            self.caches[video] = cache
            times = np.array([f.timestamp*1000 for f in cache])
            self.frame_times[video] = times
            dense = np.zeros(len(cache), np.int64)
            events = []
            for event in record.get('events', []):
                if event['label'] not in self.classes:
                    raise ValueError(f'Unknown skeleton action label: {event["label"]}')
                timestamp = float(event['position_ms'])
                if timestamp < 0 or timestamp > times[-1] + 1000/self.fps:
                    raise ValueError('Action event is outside skeleton video timeline')
                index = min(int(np.searchsorted(times, timestamp)), len(cache)-1)
                dense[max(0,index-dilation):min(len(cache),index+dilation+1)] = self.classes[event['label']]
                events.append({'frame': index, 'eval_frame': int(self.fps*timestamp/1000), 'label': event['label']})
            self._dense[video] = dense
            self._labels.append({'video': video, 'path': video, 'num_frames': len(cache),
                                 'fps': self.fps, 'events': events, 'num_events': len(events)})
            for window in iter_windows(cache, self.length, overlap):
                self.windows.append((video, window.indices[0], len(window.indices)))
        if not self.windows:
            raise ValueError('No skeleton spotting windows')

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        video, start, count = self.windows[index]
        cache = self.caches[video]
        frames = [cache[i] for i in range(start, start+count)]
        frames += [frames[-1]]*(self.length-count)
        result = {'video': video, 'start': start, 'graphs': [frame_graph(f) for f in frames],
                  'time_mask': np.arange(self.length) < count}
        labels = np.zeros(self.length, np.int64)
        labels[:count] = self._dense[video][start:start+count]
        result.update(label=labels, contains_event=int(np.any(labels)))
        return result

    @staticmethod
    def load_frame_gpu(batch, device):
        return batch['frame'].to(device)

    def get_labels(self, video):
        return self._dense[video]

    @property
    def videos(self):
        return [(v['video'], v['num_frames'], v['fps']) for v in self._labels]

    @property
    def labels(self):
        return self._labels

    @property
    def augment(self):
        return False

    def print_info(self):
        print(f'Skeleton spotting: {len(self.caches)} videos, {len(self)} shot-local windows')
