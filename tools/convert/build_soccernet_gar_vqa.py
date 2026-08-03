"""Build a test-only VQA manifest for SoccerNet-GAR frames_npy clips."""

from __future__ import annotations

import argparse
import json
import os


QUESTION = (
    "What is the action class in this soccer clip? Answer with exactly one label from: "
    "PASS, HEADER, HIGH PASS, OUT, CROSS, THROW IN, SHOT, PLAYER SUCCESSFUL TACKLE, FREE KICK, GOAL."
)


def _load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_manifest(dataset_root: str, annotations_path: str) -> dict:
    payload = _load_json(annotations_path)
    action_spec = ((payload.get("labels") or {}).get("action") or {})
    allowed_labels = [str(label).strip() for label in action_spec.get("labels", []) if str(label).strip()]

    rows = []
    for item in payload.get("data", []):
        sample_id = item.get("id")
        inputs = item.get("inputs", [])
        clip_rel_path = None
        for inp in inputs:
            if str(inp.get("type", "")).lower() in {"frames_npy", "frames", "video"} and inp.get("path"):
                clip_rel_path = str(inp["path"])
                break
        if not sample_id or not clip_rel_path:
            continue

        clip_abs_path = (
            clip_rel_path
            if os.path.isabs(clip_rel_path)
            else os.path.abspath(os.path.join(dataset_root, clip_rel_path))
        )
        if not os.path.isfile(clip_abs_path):
            raise FileNotFoundError(f"Missing clip for sample '{sample_id}': {clip_abs_path}")

        ground_truth_label = str((((item.get("labels") or {}).get("action") or {}).get("label")) or "").strip()
        rows.append(
            {
                "id": sample_id,
                "question": QUESTION,
                "references": [ground_truth_label] if ground_truth_label else [],
                "video_path": clip_abs_path,
                "ground_truth_label": ground_truth_label or None,
                "allowed_labels": list(allowed_labels),
                "inputs": [{"type": "frames_npy", "path": clip_rel_path}],
                "metadata": dict(item.get("metadata") or {}),
                "labels": dict(item.get("labels") or {}),
            }
        )

    return {
        "task": "vqa",
        "dataset_name": f"{payload.get('dataset_name', 'sngar-frames')}-vqa-test",
        "source_annotation": os.path.abspath(annotations_path),
        "labels": {
            "action": {
                "type": "single_label",
                "labels": allowed_labels,
            }
        },
        "data": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, help="Root directory for the sngar-frames dataset.")
    parser.add_argument("--annotations", required=True, help="Path to annotations_test.json.")
    parser.add_argument("--output", required=True, help="Path to write the derived VQA test.json.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_manifest(
        dataset_root=os.path.abspath(os.path.expanduser(args.dataset_root)),
        annotations_path=os.path.abspath(os.path.expanduser(args.annotations)),
    )
    output_path = os.path.abspath(os.path.expanduser(args.output))
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {output_path} ({len(manifest['data'])} rows)")


if __name__ == "__main__":
    main()
