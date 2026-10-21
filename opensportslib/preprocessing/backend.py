"""Lazy frozen RTMLib ONNX backend and model provenance.

Network/model-cache access happens only when constructing this backend.
Training from skeleton caches never constructs it.
"""
from pathlib import Path
from functools import lru_cache
import importlib.metadata
import os
import tempfile
import urllib.request
import zipfile

import numpy as np

from .cache import file_sha256

_BASE = 'https://download.openmmlab.com/mmpose/v1/projects/'
MODEL_ARTIFACTS = {
    'yolox_m': ('rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip', (640, 640)),
    'yolox_tiny': ('rtmposev1/onnx_sdk/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip', (416, 416)),
    'rtmpose_s': ('rtmposev1/onnx_sdk/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip', (192, 256)),
    'rtmpose_m': ('rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip', (192, 256)),
    'rtmo_s': ('rtmo/onnx_sdk/rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.zip', (640, 640)),
}


@lru_cache(maxsize=16)
def _artifact_sha256(path, size, modified_ns, changed_ns):
    """Avoid re-reading frozen weights per cached clip; file changes invalidate."""
    return file_sha256(path)


def resolve_artifact(name, cfg):
    """Resolve an ONNX file, with optional deployment-supplied SHA-256 pin."""
    if name not in MODEL_ARTIFACTS:
        raise ValueError(f'Unknown pose frontend artifact: {name}')
    suffix, size = MODEL_ARTIFACTS[name]
    override = cfg.get('artifacts', {}).get(name, {})
    url = _BASE + suffix
    if override.get('path'):
        path = Path(override['path']).expanduser()
    else:
        root = Path(cfg.get('model_dir', '~/.cache/opensportslib/pose-models')).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        path = root / (Path(suffix).stem + '.onnx')
        if not path.exists():
            with tempfile.TemporaryDirectory(dir=root) as directory:
                archive = Path(directory) / 'model.zip'
                urllib.request.urlretrieve(url, archive)
                with zipfile.ZipFile(archive) as zipped:
                    models = [n for n in zipped.namelist() if n.endswith('.onnx')]
                    if len(models) != 1:
                        raise ValueError(f'Expected one ONNX model in {url}')
                    with zipped.open(models[0]) as source, open(Path(directory) / 'model.onnx', 'wb') as target:
                        import shutil
                        shutil.copyfileobj(source, target)
                os.replace(Path(directory) / 'model.onnx', path)
    path = path.resolve()
    stat = path.stat()
    digest = _artifact_sha256(str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if override.get('sha256') and digest != override['sha256']:
        raise ValueError(f'Weight checksum mismatch for {name}')
    return str(path), size, {
        'name': name, 'url': url, 'sha256': digest,
        'code_license': 'Apache-2.0',
        'code_source': 'https://github.com/Tau-J/rtmlib',
        'weight_license_evidence': 'https://huggingface.co/Tau-J/RTMPose',
        'weight_license_status': 'mirror declares Apache-2.0; verify exact artifact before redistribution',
    }


class RTMLibBackend:
    """Inference-only sessions; no torch modules or optimizer parameters."""
    def __init__(self, cfg):
        try:
            import onnxruntime as ort
            from rtmlib import YOLOX, RTMPose, RTMO
        except ImportError as exc:
            raise ImportError('Video pose extraction requires optional RTMLib and onnxruntime-gpu; '
                              'see docs/preprocessing/video-to-skeleton.md') from exc
        device = cfg.get('device', 'cuda:0')
        if not str(device).startswith('cuda'):
            raise ValueError('RTMLibBackend requires a CUDA device; cached skeletons support CPU training')
        if 'CUDAExecutionProvider' not in ort.get_available_providers():
            raise RuntimeError('ONNX Runtime CUDAExecutionProvider is unavailable; CPU fallback is disabled')
        # Load compatible CUDA/cuDNN libraries before creating ORT sessions.
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('Pose extraction requires working CUDA; the NVIDIA driver is unavailable')
        device_index = int(str(device).split(':')[-1]) if ':' in str(device) else 0
        if device_index >= torch.cuda.device_count() or device_index < 0:
            raise ValueError(f'CUDA pose device does not exist: {device}')
        if hasattr(ort, 'preload_dlls'):
            ort.preload_dlls()
        self.timings = {'detection_seconds': 0., 'pose_seconds': 0.}
        self.max_candidates = cfg.get('max_candidates', 8)
        self.provenance = {'rtmlib_version': importlib.metadata.version('rtmlib'),
                           'onnxruntime_version': ort.__version__, 'artifacts': []}
        self.one_stage = cfg.get('pose', 'rtmpose_s') == 'rtmo_s'
        self.detector = None
        if not self.one_stage:
            path, size, provenance = resolve_artifact(cfg.get('detector', 'yolox_m'), cfg)
            self.detector = YOLOX(path, model_input_size=size, backend='onnxruntime', device=device)
            self.provenance['artifacts'].append(provenance)
        path, size, provenance = resolve_artifact(cfg.get('pose', 'rtmpose_s'), cfg)
        self.pose = (RTMO if self.one_stage else RTMPose)(
            path, model_input_size=size, to_openpose=False, backend='onnxruntime', device=device)
        self.provenance['artifacts'].append(provenance)
        for model in (self.detector, self.pose):
            if model is not None and model.session.get_providers()[0] != 'CUDAExecutionProvider':
                raise RuntimeError('Requested CUDA execution but ONNX Runtime fell back to CPU')

    def detect(self, image):
        from time import perf_counter
        started = perf_counter()
        boxes = np.asarray(self.detector(image), np.float32).reshape(-1, 4)
        self.timings['detection_seconds'] += perf_counter()-started
        return boxes

    def estimate(self, image, boxes):
        from time import perf_counter
        if not len(boxes):
            return np.empty((0,17,2), np.float32), np.empty((0,17), np.float32)
        started = perf_counter()
        joints, scores = self.pose(image, bboxes=boxes)
        self.timings['pose_seconds'] += perf_counter()-started
        return np.asarray(joints, np.float32), np.asarray(scores, np.float32)

    def predict(self, image):
        from time import perf_counter
        started = perf_counter()
        if self.one_stage:
            joints, scores = self.pose(image)
            self.timings["pose_seconds"] += perf_counter()-started
            joints, scores = np.asarray(joints), np.asarray(scores)
            if not len(joints):
                return np.empty((0, 4), np.float32), np.empty((0, 17, 2), np.float32), np.empty((0, 17), np.float32)
            # RTMO convenience API returns keypoints, not boxes; derive boxes
            # from confident joints instead of inferring an absent detector API.
            boxes = []
            for points, confidence in zip(joints, scores):
                visible = points[confidence > .3]
                boxes.append(np.r_[visible.min(0), visible.max(0)] if len(visible) else np.zeros(4))
            order = np.argsort(-scores.mean(1), kind='stable')[:self.max_candidates]
            return np.asarray(boxes, np.float32)[order], joints[order], scores[order]
        boxes = np.asarray(self.detector(image), np.float32).reshape(-1, 4)
        self.timings["detection_seconds"] += perf_counter()-started
        if not len(boxes):  # RTMPose([]) means full-image pose, not no people.
            return boxes, np.empty((0, 17, 2), np.float32), np.empty((0, 17), np.float32)
        if len(boxes) > self.max_candidates:
            area = np.prod(np.maximum(boxes[:, 2:] - boxes[:, :2], 0), axis=1)
            boxes = boxes[np.argsort(-area, kind='stable')[:self.max_candidates]]
        started = perf_counter()
        joints, scores = self.pose(image, bboxes=boxes)
        self.timings["pose_seconds"] += perf_counter()-started
        return boxes, np.asarray(joints, np.float32), np.asarray(scores, np.float32)

    def close(self):
        self.detector = None
        self.pose = None
