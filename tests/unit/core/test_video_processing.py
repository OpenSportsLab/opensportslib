"""Boundary contracts for deterministic video sampling utilities."""

from __future__ import annotations

from opensportslib.core.utils.video_processing import (
    distribute_elements,
    get_num_frames,
    get_remaining,
    get_stride,
    resample_video_idx,
)


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
