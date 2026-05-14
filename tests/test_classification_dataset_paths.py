import json
from pathlib import Path
from types import SimpleNamespace

import opensportslib.datasets.classification_dataset as classification_dataset


def _write_classification_annotation(path: Path) -> str:
    payload = {
        "labels": {
            "action": {"labels": ["PASS"]},
        },
        "data": [
            {
                "id": "sample_00000",
                "inputs": [
                    {
                        "type": "video",
                        "path": "clips/video_00000.mp4",
                    }
                ],
                "labels": {
                    "action": {"label": "PASS"},
                },
            }
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _make_config(data_dir: Path, valid_video_root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        DATA=SimpleNamespace(
            data_dir=str(data_dir),
            data_modality="video",
            view_type="single",
            num_frames=16,
            input_fps=25,
            target_fps=17,
            start_frame=0,
            end_frame=15,
            train=SimpleNamespace(video_path=str(data_dir / "train_root")),
            valid=SimpleNamespace(video_path=str(valid_video_root)),
            test=SimpleNamespace(video_path=str(data_dir / "test_root")),
        ),
        MODEL=SimpleNamespace(type="custom"),
    )


def test_video_dataset_resolves_relative_paths_from_selected_split_root(
    tmp_path,
    monkeypatch,
):
    annotation_path = _write_classification_annotation(tmp_path / "annotations" / "valid.json")
    data_dir = tmp_path / "dataset_root"
    valid_video_root = tmp_path / "separate_valid_root"
    config = _make_config(data_dir, valid_video_root)

    monkeypatch.setattr(classification_dataset, "build_transform", lambda config, mode: None)

    dataset = classification_dataset.VideoDataset(
        config,
        annotation_path,
        processor=None,
        split="valid",
    )

    resolved_path = Path(dataset.samples[0]["video_paths"][0])

    assert dataset.split == "valid"
    assert resolved_path.is_absolute()
    assert resolved_path == valid_video_root / "clips" / "video_00000.mp4"
