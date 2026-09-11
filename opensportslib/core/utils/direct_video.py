"""Build temporary OSL manifests for direct single-video inference."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date
import json
from pathlib import Path
import tempfile
from typing import Any, Iterator

from opensportslib.core.config.accessors import get_data_classes, get_data_modality, get_data_num_classes


def validate_direct_video_config(config) -> None:
    modality = str(get_data_modality(config)).lower()
    if modality != "video":
        raise ValueError(
            f"Direct video inference requires a video-based config; configured modality is {modality!r}. "
            "Use infer(test_set=...) for feature, frame, or tracking inputs."
        )


def build_direct_video_manifest(
    config,
    video_path: str | Path,
    task_type: str,
    *,
    generated_by: str = "opensportslib",
) -> dict[str, Any]:
    """Return an unlabeled one-video OSL v2 manifest."""
    validate_direct_video_config(config)
    source = Path(video_path).expanduser().resolve()
    classes = get_data_classes(config)
    if not classes:
        classes = [f"class_{index}" for index in range(get_data_num_classes(config))]
    sample: dict[str, Any] = {
        "id": source.stem,
        "inputs": [{"type": "video", "path": str(source)}],
        "metadata": {},
    }
    if task_type == "localization":
        sample["events"] = []
    return {
        "version": "2.0",
        "date": str(date.today()),
        "task": "action_spotting" if task_type == "localization" else "action_classification",
        "dataset_name": f"{task_type}-direct-video",
        "description": f"Generated {task_type} direct-video inference manifest.",
        "modalities": ["video"],
        "metadata": {"split": "test", "generated_by": generated_by},
        "labels": {"action": {"type": "single_label", "labels": classes}},
        "data": [sample],
    }


@contextmanager
def direct_video_manifest(config, video_path: str | Path, task_type: str) -> Iterator[str]:
    source = Path(video_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Video file not found: {source}")
    manifest = build_direct_video_manifest(config, source, task_type)
    with tempfile.TemporaryDirectory(prefix=f"opensportslib-{task_type}-video-") as directory:
        destination = Path(directory) / "request.json"
        destination.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        yield str(destination)
