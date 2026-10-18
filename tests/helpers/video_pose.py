"""Small real-video fixtures and an external pose backend for CPU tests."""
from fractions import Fraction
import json
import os

import numpy as np


class PoseBackend:
    provenance = {'artifacts': [{'name': 'yolox_m', 'sha256': 'detector-test'},
                                {'name': 'rtmpose_s', 'sha256': 'pose-test'}]}
    def __init__(self, settings=None):
        self.calls = 0
        self.absent = False

    def predict(self, image):
        self.calls += 1
        if self.absent:
            return np.empty((0, 4)), np.empty((0, 17, 2)), np.empty((0, 17))
        return (np.array([[10, 10, 30, 60]], np.float32),
                np.full((1, 17, 2), 20, np.float32), np.ones((1, 17), np.float32))

    def close(self):
        pass


def write_video(path, count=9, width=100, height=80, cut=None, rate=Fraction(25), frame_indices=None):
    import av
    with av.open(str(path), 'w') as output:
        stream = output.add_stream('mpeg4', rate=rate)
        stream.width, stream.height = width, height
        for i in range(count):
            image = np.full((height, width, 3), 255 if cut is not None and i >= cut else 0, np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format='rgb24')
            if frame_indices is not None:
                frame.pts = frame_indices[i]
                frame.time_base = 1/rate
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    return path


def pose_config(task, directory, *, count=9):
    from opensportslib.core.config.editable import Config
    payload = Config.from_file(f'examples/tennis/{task}_skeleton.yaml').get_config()
    payload['SYSTEM']['device'] = 'cpu'
    payload['SYSTEM']['paths'] = {k: str(directory/task/k) for k in ('save_dir', 'log_dir', 'work_dir')}
    payload['DATA']['common']['classes'] = ['swing', 'serve']
    for split in payload['DATA']['common']['splits'].values():
        split.update(source_path=str(directory), annotation_path=str(directory/'labels.json'), overlap_len=2)
        split['dataloader'].update(batch_size=1, num_workers=0)
    payload['DATA']['inputs']['video']['sampling'].update(num_frames=4, clip_len=4, epoch_num_frames=4)
    payload['MODEL']['components']['tracking_encoder']['params'].update(num_layers=1, dropout=0.)
    settings = payload['MODEL']['components']['pose_tracking']['params']
    settings.update(selection='all_people', artifacts={
        'yolox_m': {'sha256': 'detector-test'}, 'rtmpose_s': {'sha256': 'pose-test'}})
    if task == 'classification':
        payload['MODEL']['components']['task_head']['params'].update(num_classes=2, dropout=0.)
        payload['MODEL']['components']['tracking_adapter']['params']['dropout'] = 0.
    video = write_video(directory/'clip.mp4', count)
    record = {'id': 'clip', 'inputs': [{'type': 'video', 'path': video.name}]}
    if task == 'classification':
        record['labels'] = {'action': {'label': 'swing'}}
    else:
        record['events'] = [{'label': 'swing', 'position_ms': 80}]
    (directory/'labels.json').write_text(json.dumps({'version': '2.0', 'task': 'action_classification' if task == 'classification' else 'action_spotting',
        'labels': {'action': {'labels': ['swing', 'serve']}}, 'data': [record]}))
    return payload


def inject_backend(frontend, backend=None):
    backend = backend or PoseBackend()
    frontend._backends[(os.getpid(), str(frontend._device_anchor.device))] = backend
    return backend
