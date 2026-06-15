import pytest

from opensportslib.datasets.localization_dataset import (
    _build_dali_filenames_and_labels,
    _count_dali_video_samples,
    _dali_frame_num_to_local_frame,
    _pad_dali_iterator_size,
    _resolve_dali_video_sample,
)


def test_build_dali_filenames_preserves_spaced_paths():
    labels = [
        {"video": "/tmp/with spaces/game one.mp4"},
        {"video": "/tmp/plain/game-two.mp4"},
    ]

    filenames, label_indices = _build_dali_filenames_and_labels(labels)

    assert filenames == [
        "/tmp/with spaces/game one.mp4",
        "/tmp/plain/game-two.mp4",
    ]
    assert label_indices == [0, 1]


def test_dali_frame_numbers_translate_to_one_based_local_frames():
    assert _dali_frame_num_to_local_frame(0, 12) == 1
    assert _dali_frame_num_to_local_frame(12, 12) == 2
    assert _dali_frame_num_to_local_frame(24, 12) == 3


def test_count_dali_video_samples_matches_previous_manifest_schedule():
    assert _count_dali_video_samples(10, 4, 1) == len(list(range(1, 10, 3)))
    assert _count_dali_video_samples(101, 100, 50) == len(list(range(1, 101, 50)))


def test_count_dali_video_samples_rejects_non_positive_step():
    with pytest.raises(ValueError):
        _count_dali_video_samples(10, 4, 4)


def test_pad_dali_iterator_size_rounds_up_to_full_batch():
    assert _pad_dali_iterator_size(10, 4) == 12
    assert _pad_dali_iterator_size(12, 4) == 12


def test_resolve_dali_video_sample_uses_relative_path_and_one_based_start():
    labels = [
        {
            "path": "train/match one.mp4",
            "video": "/abs/with spaces/train/match one.mp4",
        }
    ]

    video_name, start = _resolve_dali_video_sample(labels, 0, 24, 12)

    assert video_name == "train/match one.mp4"
    assert start == 3