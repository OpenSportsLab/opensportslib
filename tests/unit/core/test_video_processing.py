"""Boundary contracts for deterministic video sampling utilities."""

from __future__ import annotations

from opensportslib.core.utils.video_processing import (
    distribute_elements,
    get_num_frames,
    get_remaining,
    get_stride,
    resample_video_idx,
)
from opensportslib.core.utils.direct_video import (
    build_direct_video_manifest,
    direct_video_manifest,
    validate_direct_video_config,
)
from opensportslib.core.config.runtime_adapter import dict_to_namespace
import pytest


def test_video_rate_helpers_return_expected_sampling_values():
    assert get_stride(25, 5) == 5
    assert get_num_frames(100, 25, 5) == 20
    assert resample_video_idx(10, 10, 5) == slice(None, None, 2)


def test_batch_distribution_and_remainder_cover_boundaries():
    assert distribute_elements(8, 3) == [3, 3, 2]
    assert get_remaining(10, 4) == 2
    assert get_remaining(8, 4) == 0


def test_non_positive_requested_rate_keeps_every_frame():
    assert get_stride(25, 0) == 1


def test_direct_video_manifest_supports_classification_and_localization(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not-decoded-in-this-unit-test")
    config = dict_to_namespace({
        "DATA": {
            "common": {"classes": ["PASS", "SHOT"]},
            "inputs": {"video": {"modality": "video"}},
        }
    })

    classification = build_direct_video_manifest(config, video, "classification")
    localization = build_direct_video_manifest(config, video, "localization")
    assert classification["task"] == "action_classification"
    assert classification["data"][0]["inputs"][0]["path"] == str(video.resolve())
    assert localization["task"] == "action_spotting"
    assert localization["data"][0]["events"] == []

    with direct_video_manifest(config, video, "classification") as manifest_path:
        assert open(manifest_path, encoding="utf-8").read().startswith("{")
    with pytest.raises(FileNotFoundError, match="Video file not found"):
        with direct_video_manifest(config, tmp_path / "missing.mp4", "classification"):
            pass


def test_direct_video_manifest_rejects_non_video_configs():
    config = dict_to_namespace(
        {"DATA": {"common": {"classes": []}, "inputs": {"tracking": {"modality": "tracking"}}}}
    )
    with pytest.raises(ValueError, match="video-based config"):
        validate_direct_video_config(config)
