from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any


def build_classification_manifest(
    working_dir: Path, video_path: str, task_options: dict[str, Any], config=None
) -> Path:
    manifest = _manifest_for_request("classification", video_path, task_options, config)
    destination = working_dir / "classification_request.json"
    destination.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return destination


def build_localization_manifest(
    working_dir: Path, video_path: str, task_options: dict[str, Any], config=None
) -> Path:
    manifest = _manifest_for_request("localization", video_path, task_options, config)
    destination = working_dir / "localization_request.json"
    destination.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return destination


def _manifest_for_request(task_type: str, video_path: str, task_options: dict[str, Any], config=None):
    if config is None:
        manifest = _base_manifest(task_type, video_path, task_options)
        if task_type == "localization":
            manifest["data"][0]["events"] = []
        return manifest

    from opensportslib.core.utils.direct_video import build_direct_video_manifest

    manifest = build_direct_video_manifest(
        config,
        video_path,
        task_type,
        generated_by="opensportslib-server",
    )
    if task_options.get("label_schema"):
        manifest["labels"] = task_options["label_schema"]
    if task_options.get("sample_id"):
        manifest["data"][0]["id"] = task_options["sample_id"]
    if task_options.get("sample_metadata"):
        manifest["data"][0]["metadata"] = task_options["sample_metadata"]
    return manifest


def _base_manifest(task_type: str, video_path: str, task_options: dict[str, Any]) -> dict[str, Any]:
    label_schema = task_options.get("label_schema") or {
        "action": {
            "type": "single_label",
            "labels": task_options.get("label_space") or [],
        }
    }
    sample_id = task_options.get("sample_id") or Path(video_path).stem
    fps = float(task_options.get("fps", 25.0))
    return {
        "version": "2.0",
        "date": str(date.today()),
        "dataset_name": f"{task_type}-request",
        "description": f"Generated {task_type} inference request manifest.",
        "modalities": ["video"],
        "metadata": {
            "split": "test",
            "generated_by": "opensportslib-server",
        },
        "labels": label_schema,
        "data": [
            {
                "id": sample_id,
                "inputs": [
                    {
                        "type": "video",
                        "path": video_path,
                        "fps": fps,
                    }
                ],
                "metadata": task_options.get("sample_metadata", {}),
            }
        ],
    }
