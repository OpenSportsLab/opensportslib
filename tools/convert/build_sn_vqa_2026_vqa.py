"""Convert SoccerNet SN-VQA-2026 test rows into native-Qwen-runnable OSL JSON."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
DEFAULT_CONVERSION_DATE = "2026-07-30"


@dataclass
class MaterialInfo:
    path: Path
    relative_path: str
    kind: str


def _require_media_runtime():
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "SN-VQA-2026 native-Qwen conversion requires both 'numpy' and 'opencv-python'."
        ) from exc
    return cv2, np


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _normalize_answer_text(text: str) -> str:
    trimmed = str(text or "").strip()
    cleaned = trimmed.strip(" \t\r\n.,;:!?\"'`()[]{}")
    cleaned = re.sub(r"\s+", " ", cleaned).strip().upper()
    return cleaned


def _normalize_answer_alias(text: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", _normalize_answer_text(text))


def _find_unique_substring_label(answer_text: str, allowed_labels: list[str]) -> str | None:
    normalized_answer = _normalize_answer_text(answer_text)
    if not normalized_answer:
        return None
    hits: list[tuple[int, str]] = []
    for label in allowed_labels:
        normalized_label = _normalize_answer_text(label)
        if not normalized_label:
            continue
        if normalized_label in normalized_answer:
            hits.append((len(normalized_label), label))
            continue
        alias = _normalize_answer_alias(label)
        if alias and alias in _normalize_answer_alias(answer_text):
            hits.append((len(alias), label))
    if not hits:
        return None
    hits.sort(reverse=True)
    best_len = hits[0][0]
    best = {label for length, label in hits if length == best_len}
    if len(best) == 1:
        return next(iter(best))
    return None


def _find_unique_token_cover_label(answer_text: str, allowed_labels: list[str]) -> str | None:
    answer_tokens = set(re.findall(r"[A-Z0-9]+", _normalize_answer_text(answer_text)))
    if not answer_tokens:
        return None
    matches: list[tuple[int, str]] = []
    for label in allowed_labels:
        label_tokens = re.findall(r"[A-Z0-9]+", _normalize_answer_text(label))
        if not label_tokens:
            continue
        if len(label_tokens) > 5:
            continue
        if all(token in answer_tokens for token in label_tokens):
            matches.append((len(label_tokens), label))
    if not matches:
        return None
    matches.sort(reverse=True)
    best_len = matches[0][0]
    best = {label for length, label in matches if length == best_len}
    if len(best) == 1:
        return next(iter(best))
    return None


def normalize_prediction_to_option(answer_text: str, allowed_labels: list[str]) -> str | None:
    labels = [str(label).strip() for label in (allowed_labels or []) if str(label).strip()]
    if not labels:
        return None
    normalized = _normalize_answer_text(answer_text)
    if not normalized:
        return None
    exact_map = {_normalize_answer_text(label): label for label in labels}
    predicted = exact_map.get(normalized)
    if predicted is not None:
        return predicted
    alias_map = {_normalize_answer_alias(label): label for label in labels}
    predicted = alias_map.get(_normalize_answer_alias(answer_text))
    if predicted is not None:
        return predicted
    predicted = _find_unique_substring_label(answer_text, labels)
    if predicted is not None:
        return predicted
    return _find_unique_token_cover_label(answer_text, labels)


def _classify_material(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in VIDEO_SUFFIXES:
        return "video"
    return "unknown"


def _sample_video_frames(video_path: Path, num_frames: int) -> list[Any]:
    cv2, np = _require_media_runtime()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0:
            raise ValueError(f"Video has no readable frames: {video_path}")
        indices = np.linspace(0, max(frame_count - 1, 0), num=max(1, int(num_frames)), dtype=int)
        frames: list[np.ndarray] = []
        for idx in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not frames:
            raise ValueError(f"Could not sample frames from video: {video_path}")
        return frames
    finally:
        capture.release()


def _load_image(path: Path) -> Any:
    cv2, _ = _require_media_runtime()
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _resolve_materials(
    materials: Any,
    *,
    dataset_root: Path,
    split: str,
) -> tuple[str, list[MaterialInfo], list[str]]:
    if materials is None:
        return "null", [], []
    if not isinstance(materials, list) or not materials:
        return "empty", [], []

    infos: list[MaterialInfo] = []
    missing: list[str] = []
    has_image = False
    has_video = False
    for rel in materials:
        rel_path = str(rel)
        full_path = dataset_root / split / rel_path
        if not full_path.exists():
            missing.append(rel_path)
            continue
        kind = _classify_material(rel_path)
        if kind == "image":
            has_image = True
        elif kind == "video":
            has_video = True
        infos.append(MaterialInfo(path=full_path, relative_path=f"{split}/{rel_path}", kind=kind))

    if has_image and has_video:
        category = "mixed_media"
    elif has_video and len(infos) == 1:
        category = "single_video"
    elif has_video:
        category = "multi_video"
    elif has_image and len(infos) == 1:
        category = "single_image"
    elif has_image:
        category = "multi_image"
    else:
        category = "unknown"
    return category, infos, missing


def _build_choice_map(row: dict[str, Any]) -> dict[str, str]:
    out = {}
    for key in ("O1", "O2", "O3", "O4"):
        value = str(row.get(key) or "").strip()
        if value:
            out[key] = value
    return out


def _fuse_materials(
    sample_id: str,
    materials: list[MaterialInfo],
    *,
    fused_dir: Path,
    frames_per_video: int,
) -> str:
    _, np = _require_media_runtime()
    frames: list[Any] = []
    for item in materials:
        if item.kind == "video":
            frames.extend(_sample_video_frames(item.path, frames_per_video))
        elif item.kind == "image":
            frames.append(_load_image(item.path))
    if not frames:
        raise ValueError(f"No usable frames produced for sample '{sample_id}'")
    fused_dir.mkdir(parents=True, exist_ok=True)
    array = np.stack(frames, axis=0).astype(np.uint8, copy=False)
    fused_path = fused_dir / f"{sample_id}.npy"
    np.save(fused_path, array)
    return str(fused_path)


def convert_sn_vqa_2026_to_osl(
    *,
    dataset_root: str,
    split: str = "test",
    output_manifest: str | None = None,
    output_report: str | None = None,
    fused_dir: str | None = None,
    frames_per_video: int = 4,
    conversion_date: str = DEFAULT_CONVERSION_DATE,
) -> dict[str, Any]:
    root = Path(dataset_root).expanduser().resolve()
    split_dir = root / split
    source_json = split_dir / f"{split}.json"
    rows = _load_json(source_json)
    if not isinstance(rows, list):
        raise ValueError(f"Expected list payload in {source_json}")

    fused_root = Path(fused_dir).expanduser().resolve() if fused_dir else (split_dir / "fused_qwen_native")
    manifest_path = Path(output_manifest).expanduser().resolve() if output_manifest else (root / f"{split}_osl_qwen_native.json")
    report_path = Path(output_report).expanduser().resolve() if output_report else (root / f"{split}_conversion_report.json")

    data: list[dict[str, Any]] = []
    excluded_ids: dict[str, list[str]] = defaultdict(list)
    category_counts: Counter[str] = Counter()
    fused_artifact_count = 0

    for row in rows:
        sample_id = str(row.get("id"))
        question = str(row.get("Q") or "").strip()
        open_answer = str(row.get("openA") or "").strip()
        choice_map = _build_choice_map(row)
        allowed_labels = [choice_map[key] for key in ("O1", "O2", "O3", "O4") if key in choice_map]
        correct_key = str(row.get("closeA") or "").strip()
        correct_option_text = choice_map.get(correct_key, "")

        category, materials, missing = _resolve_materials(row.get("materials"), dataset_root=root, split=split)
        category_counts[category] += 1
        if category == "null":
            excluded_ids["null_materials"].append(sample_id)
            continue
        if category == "empty":
            excluded_ids["empty_materials"].append(sample_id)
            continue
        if missing:
            excluded_ids["missing_files"].append(sample_id)
            continue
        if not question:
            excluded_ids["missing_question"].append(sample_id)
            continue
        if not allowed_labels:
            excluded_ids["missing_options"].append(sample_id)
            continue
        if any(item.kind == "unknown" for item in materials):
            excluded_ids["unsupported_material_type"].append(sample_id)
            continue

        sample = {
            "id": sample_id,
            "question": question,
            "references": [open_answer] if open_answer else [],
            "ground_truth_label": open_answer,
            "allowed_labels": allowed_labels,
            "metadata": {
                "source_dataset": "SN-VQA-2026",
                "source_row_id": row.get("id"),
                "correct_option_key": correct_key,
                "correct_option_text": correct_option_text,
                "choice_map": choice_map,
                "original_materials": row.get("materials"),
            },
        }

        try:
            if category == "single_image":
                sample["frame_paths"] = [materials[0].relative_path]
            elif category == "multi_image":
                sample["frame_paths"] = [item.relative_path for item in materials]
            elif category == "single_video":
                sample["video_path"] = materials[0].relative_path
            elif category in {"multi_video", "mixed_media"}:
                fused_path = _fuse_materials(sample_id, materials, fused_dir=fused_root, frames_per_video=frames_per_video)
                sample["video_path"] = os.path.relpath(fused_path, root)
                fused_artifact_count += 1
            else:
                excluded_ids["unsupported_material_layout"].append(sample_id)
                continue
        except Exception:
            excluded_ids["fusion_or_decode_error"].append(sample_id)
            continue

        data.append(sample)

    manifest = {
        "version": "2.0",
        "date": conversion_date,
        "dataset_name": "SN-VQA-2026-test-qwen-native",
        "description": "Runnable SN-VQA-2026 test conversion for OpenSportsLib native Qwen VL inference.",
        "modalities": ["image", "video"],
        "metadata": {
            "source_dataset": "SN-VQA-2026",
            "split": split,
            "conversion_date": conversion_date,
            "source_annotation": str(source_json),
            "frames_per_video_for_fusion": int(frames_per_video),
        },
        "data": data,
    }
    report = {
        "dataset_name": "SN-VQA-2026-test-qwen-native",
        "split": split,
        "conversion_date": conversion_date,
        "source_row_count": len(rows),
        "runnable_row_count": len(data),
        "excluded_row_count": len(rows) - len(data),
        "excluded_ids_by_reason": {
            key: sorted(value, key=lambda item: int(item) if str(item).isdigit() else str(item))
            for key, value in sorted(excluded_ids.items())
        },
        "material_category_counts": dict(category_counts),
        "fused_artifact_count": fused_artifact_count,
        "generated_manifest": str(manifest_path),
        "generated_fused_dir": str(fused_root),
    }

    _dump_json(manifest_path, manifest)
    _dump_json(report_path, report)
    return {
        "manifest_path": str(manifest_path),
        "report_path": str(report_path),
        "fused_dir": str(fused_root),
        "manifest": manifest,
        "report": report,
    }


def evaluate_sn_vqa_predictions(
    *,
    manifest_path: str,
    predictions_path: str,
    output_path: str | None = None,
) -> dict[str, Any]:
    manifest = _load_json(Path(manifest_path).expanduser().resolve())
    predictions = _load_json(Path(predictions_path).expanduser().resolve())
    manifest_rows = {str(row.get("id")): row for row in manifest.get("data", [])}
    prediction_rows = list(predictions.get("data", []))

    evaluated_rows = []
    matched_count = 0
    option_key_correct = 0
    text_exact_match = 0
    unresolved = 0

    for row in prediction_rows:
        sample_id = str(row.get("id"))
        source = manifest_rows.get(sample_id, {})
        allowed_labels = list(source.get("allowed_labels") or [])
        metadata = dict(source.get("metadata") or {})
        choice_map = dict(metadata.get("choice_map") or {})
        answer_text = str(row.get("answer_text") or "")
        predicted_label = str(row.get("predicted_label") or "").strip() or normalize_prediction_to_option(answer_text, allowed_labels)
        predicted_option_key = None
        for key, value in choice_map.items():
            if predicted_label and str(value).strip() == predicted_label:
                predicted_option_key = key
                break
        correct_option_key = str(metadata.get("correct_option_key") or "").strip() or None
        correct_text = str(source.get("ground_truth_label") or "").strip()
        is_unresolved = predicted_option_key is None
        unresolved += int(is_unresolved)
        if predicted_label:
            matched_count += 1
        if predicted_option_key and correct_option_key and predicted_option_key == correct_option_key:
            option_key_correct += 1
        if _normalize_answer_text(answer_text) == _normalize_answer_text(correct_text):
            text_exact_match += 1
        enriched = dict(row)
        enriched["predicted_label"] = predicted_label
        enriched["predicted_option_key"] = predicted_option_key
        enriched["correct_option_key"] = correct_option_key
        enriched["correct_option_text"] = metadata.get("correct_option_text")
        enriched["text_exact_match_openA"] = _normalize_answer_text(answer_text) == _normalize_answer_text(correct_text)
        enriched["option_key_correct_closeA"] = bool(
            predicted_option_key and correct_option_key and predicted_option_key == correct_option_key
        )
        enriched["is_unresolved"] = is_unresolved
        evaluated_rows.append(enriched)

    total = len(prediction_rows)
    result = {
        "dataset_name": manifest.get("dataset_name"),
        "manifest_path": str(Path(manifest_path).expanduser().resolve()),
        "predictions_path": str(Path(predictions_path).expanduser().resolve()),
        "evaluated_at": date.today().isoformat(),
        "summary": {
            "prediction_count": total,
            "resolved_prediction_count": matched_count,
            "unresolved_prediction_count": unresolved,
            "unresolved_rate": (unresolved / total) if total else 0.0,
            "text_exact_match_openA": (text_exact_match / total) if total else 0.0,
            "option_key_accuracy_closeA": (option_key_correct / total) if total else 0.0,
        },
        "data": evaluated_rows,
    }
    if output_path:
        _dump_json(Path(output_path).expanduser().resolve(), result)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert_parser = subparsers.add_parser("convert", help="Convert SoccerNet test rows into runnable OSL JSON.")
    convert_parser.add_argument("--dataset-root", required=True, help="Path to the SN-VQA-2026 dataset root.")
    convert_parser.add_argument("--split", default="test", help="Dataset split to convert.")
    convert_parser.add_argument("--output-manifest", default=None, help="Output OSL manifest path.")
    convert_parser.add_argument("--output-report", default=None, help="Output conversion report path.")
    convert_parser.add_argument("--fused-dir", default=None, help="Directory for fused .npy assets.")
    convert_parser.add_argument(
        "--frames-per-video",
        type=int,
        default=4,
        help="Frames sampled from each video during fusion.",
    )
    convert_parser.add_argument(
        "--conversion-date",
        default=DEFAULT_CONVERSION_DATE,
        help="Date string to record in metadata.",
    )

    eval_parser = subparsers.add_parser(
        "evaluate-predictions",
        help="Map predictions back to SoccerNet option keys.",
    )
    eval_parser.add_argument("--manifest", required=True, help="Converted runnable manifest path.")
    eval_parser.add_argument(
        "--predictions",
        required=True,
        help="Prediction JSON from VQAModel/Trainer_VQA inference.",
    )
    eval_parser.add_argument("--output", default=None, help="Optional evaluation output JSON.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "convert":
        result = convert_sn_vqa_2026_to_osl(
            dataset_root=args.dataset_root,
            split=args.split,
            output_manifest=args.output_manifest,
            output_report=args.output_report,
            fused_dir=args.fused_dir,
            frames_per_video=args.frames_per_video,
            conversion_date=args.conversion_date,
        )
        print(
            json.dumps(
                {
                    "manifest_path": result["manifest_path"],
                    "report_path": result["report_path"],
                    "runnable_row_count": result["report"]["runnable_row_count"],
                    "excluded_row_count": result["report"]["excluded_row_count"],
                    "fused_artifact_count": result["report"]["fused_artifact_count"],
                },
                indent=2,
            )
        )
        return

    result = evaluate_sn_vqa_predictions(
        manifest_path=args.manifest,
        predictions_path=args.predictions,
        output_path=args.output,
    )
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
