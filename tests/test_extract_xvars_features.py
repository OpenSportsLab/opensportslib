from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch


def _load_module():
    path = Path("tools/convert/extract_xvars_clip_features.py")
    spec = importlib.util.spec_from_file_location("extract_xvars_clip_features", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_dual_mode_extractors_return_expected_shapes():
    module = _load_module()

    class StrictStub:
        def extract(self, video_path, *, max_frames):
            del video_path, max_frames
            return np.zeros((300, 1024), dtype=np.float16)

    class ClipStub:
        def extract(self, video_path, *, max_frames):
            del video_path, max_frames
            return np.zeros((356, 1024), dtype=np.float16)

    strict = module.extract_feature_for_video(
        Path("/tmp/fake.mp4"),
        mode="strict_xvars",
        strict_extractor=StrictStub(),
    )
    compat = module.extract_feature_for_video(
        Path("/tmp/fake.mp4"),
        mode="clip_compat",
        clip_extractor=ClipStub(),
    )

    assert strict.shape == (300, 1024)
    assert compat.shape == (356, 1024)


def test_strict_xvars_normalization_preserves_upstream_vision_tower_keys():
    module = _load_module()
    weight = torch.zeros(1024)

    normalized = module.normalize_strict_xvars_state_dict(
        {"vision_tower.vision_model.embeddings.class_embedding": weight}
    )

    assert normalized == {"vision_tower.vision_model.embeddings.class_embedding": weight}


def test_strict_xvars_normalization_unwraps_module_prefix():
    module = _load_module()
    weight = torch.zeros(1024)

    normalized = module.normalize_strict_xvars_state_dict(
        {"module.vision_tower.vision_model.embeddings.class_embedding": weight}
    )

    assert normalized == {"vision_tower.vision_model.embeddings.class_embedding": weight}


def test_strict_xvars_normalization_maps_bare_vision_model_keys():
    module = _load_module()
    weight = torch.zeros(1024)

    normalized = module.normalize_strict_xvars_state_dict(
        {"vision_model.embeddings.class_embedding": weight}
    )

    assert normalized == {"vision_tower.vision_model.embeddings.class_embedding": weight}


def test_strict_window_crop_uses_original_bounds():
    module = _load_module()
    frames = list(range(100))
    cropped = module.crop_strict_xvars_window(frames)

    assert frames[module.STRICT_START_FRAME] in cropped
    assert frames[module.STRICT_END_FRAME - 1] in cropped
    assert len(cropped) == module.STRICT_TARGET_FPS


def test_strict_window_crop_accepts_cli_style_overrides():
    module = _load_module()
    frames = list(range(40))
    cropped = module.crop_strict_xvars_window(
        frames,
        start_frame=10,
        end_frame=20,
        target_fps=5,
        source_fps=10,
    )

    assert frames[10] in cropped
    assert frames[19] in cropped
    assert len(cropped) == 5
