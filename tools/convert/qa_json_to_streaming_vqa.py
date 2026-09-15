#!/usr/bin/env python3
"""Convert multiple-choice QA annotations to canonical streaming-VQA JSON."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import tempfile
from collections import OrderedDict
from datetime import date
from pathlib import Path
from typing import Any


TIMESTAMP_RE = re.compile(r"^\s*([12])\s*-\s*(\d+):(\d{2})\s*$")
VIDEO_HALF_RE = re.compile(r"^([12])_720p(?:[-_.]|$)", re.IGNORECASE)
OPTION_KEYS = ("A", "B", "C", "D")


class ConversionError(ValueError):
    """Raised after all invalid source rows have been collected."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("Streaming-VQA conversion failed:\n- " + "\n- ".join(errors))


def _answer_text(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if value is None:
        return ""
    return str(value).strip()


def _normalized_answer(value: Any) -> str:
    return " ".join(_answer_text(value).casefold().split())


def _parse_timestamp(value: Any) -> tuple[int, int] | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    match = TIMESTAMP_RE.fullmatch(str(value))
    if match is None:
        raise ValueError("expected '1 - MM:SS' or '2 - MM:SS'")
    half, minutes, seconds = (int(part) for part in match.groups())
    if seconds >= 60:
        raise ValueError("seconds must be between 00 and 59")
    return half, (minutes * 60 + seconds) * 1000


def _video_half(video_file: Any) -> int | None:
    if not isinstance(video_file, str) or not video_file.strip():
        return None
    match = VIDEO_HALF_RE.match(Path(video_file).name)
    return int(match.group(1)) if match else None


def _convert_options(options: Any) -> tuple[list[dict[str, str]], dict[str, str]]:
    if not isinstance(options, dict) or not options:
        raise ValueError("options must be a non-empty object")

    keys = [key for key in OPTION_KEYS if key in options]
    if set(options) != set(keys) or keys != list(OPTION_KEYS[: len(keys)]):
        raise ValueError("option keys must be contiguous labels starting at A and ending by D")
    if len(keys) < 2:
        raise ValueError("at least two options are required")

    converted: list[dict[str, str]] = []
    option_ids: dict[str, str] = {}
    normalized_texts: set[str] = set()
    for index, key in enumerate(keys, start=1):
        text = _answer_text(options[key])
        if not text:
            raise ValueError(f"option {key} has empty text")
        normalized = _normalized_answer(text)
        if normalized in normalized_texts:
            raise ValueError("option texts must be unique")
        normalized_texts.add(normalized)
        option_id = f"o{index}"
        option_ids[key] = option_id
        converted.append({"id": option_id, "text": text})
    return converted, option_ids


def _resolve_correct_option(row: dict[str, Any], option_ids: dict[str, str]) -> tuple[str, str | None]:
    marked_key = row.get("closeA")
    if marked_key is not None and str(marked_key).strip():
        key = str(marked_key).strip().upper()
        if key not in option_ids:
            raise ValueError(f"closeA '{marked_key}' does not identify an available option")
        return option_ids[key], key

    if "answer" not in row or not _answer_text(row.get("answer")):
        raise ValueError("closeA is absent and answer cannot be used to infer the correct option")
    target = _normalized_answer(row["answer"])
    hits = [
        key
        for key in option_ids
        if _normalized_answer(row["options"][key]) == target
    ]
    if len(hits) != 1:
        raise ValueError(
            "closeA is absent and answer does not match exactly one normalized option"
        )
    return option_ids[hits[0]], hits[0]


def _question_metadata(
    row: dict[str, Any],
    *,
    row_index: int,
    source_file: str,
    correct_option_key: str | None,
) -> dict[str, Any]:
    metadata = {
        key: copy.deepcopy(value)
        for key, value in row.items()
        if key not in {"question", "options", "closeA"}
    }
    metadata.update(
        {
            "source_question": row["question"],
            "source_options": copy.deepcopy(row["options"]),
            "correct_option_key": correct_option_key,
            "source_file": source_file,
            "source_row_index": row_index,
        }
    )
    return metadata


def convert_rows(
    rows: Any,
    *,
    dataset_name: str,
    source_file: str,
    conversion_date: str,
) -> tuple[dict[str, Any], list[int]]:
    """Convert source rows and return ``(manifest, missing_timestamp_rows)``."""
    if not isinstance(rows, list):
        raise ConversionError(["input root must be a JSON list"])

    errors: list[str] = []
    prepared: list[dict[str, Any]] = []
    missing_timestamps: list[int] = []

    for zero_index, raw_row in enumerate(rows):
        row_number = zero_index + 1
        row_errors: list[str] = []
        if not isinstance(raw_row, dict):
            errors.append(f"row {row_number}: expected an object")
            continue
        row = raw_row

        question = row.get("question")
        match_name = row.get("match_name")
        video_file = row.get("video_file")
        if not isinstance(question, str) or not question.strip():
            row_errors.append("question must be a non-empty string")
        if not isinstance(match_name, str) or not match_name.strip():
            row_errors.append("match_name must be a non-empty string")
        if not isinstance(video_file, str) or not video_file.strip():
            row_errors.append("video_file must be a non-empty string")

        parsed_timestamp: tuple[int, int] | None = None
        try:
            parsed_timestamp = _parse_timestamp(row.get("timestamp"))
        except ValueError as exc:
            row_errors.append(f"invalid timestamp: {exc}")

        filename_half = _video_half(video_file)
        timestamp_half = parsed_timestamp[0] if parsed_timestamp else None
        if timestamp_half is not None and filename_half is not None and timestamp_half != filename_half:
            row_errors.append(
                f"timestamp half {timestamp_half} conflicts with video_file half {filename_half}"
            )
        half = timestamp_half or filename_half
        if half is None:
            row_errors.append("could not infer half from timestamp or video_file")
        if parsed_timestamp is None and not any(error.startswith("invalid timestamp") for error in row_errors):
            missing_timestamps.append(row_number)

        options: list[dict[str, str]] | None = None
        option_ids: dict[str, str] | None = None
        try:
            options, option_ids = _convert_options(row.get("options"))
        except ValueError as exc:
            row_errors.append(str(exc))

        correct_option_id: str | None = None
        correct_option_key: str | None = None
        if option_ids is not None:
            try:
                correct_option_id, correct_option_key = _resolve_correct_option(row, option_ids)
            except ValueError as exc:
                row_errors.append(str(exc))

        if row_errors:
            errors.extend(f"row {row_number}: {message}" for message in row_errors)
            continue

        assert isinstance(question, str)
        assert isinstance(match_name, str)
        assert isinstance(video_file, str)
        assert half is not None
        assert options is not None
        assert correct_option_id is not None
        prepared.append(
            {
                "source_index": zero_index,
                "match_name": match_name.strip(),
                "half": half,
                "position_ms": parsed_timestamp[1] if parsed_timestamp else None,
                "question": question.strip(),
                "options": options,
                "correct_option_id": correct_option_id,
                "metadata": _question_metadata(
                    row,
                    row_index=row_number,
                    source_file=source_file,
                    correct_option_key=correct_option_key,
                ),
            }
        )

    if errors:
        raise ConversionError(errors)

    match_numbers: dict[str, int] = {}
    grouped: OrderedDict[tuple[str, int], list[dict[str, Any]]] = OrderedDict()
    for item in prepared:
        match_name = item["match_name"]
        if match_name not in match_numbers:
            match_numbers[match_name] = len(match_numbers) + 1
        grouped.setdefault((match_name, item["half"]), []).append(item)

    data: list[dict[str, Any]] = []
    for (match_name, half), questions in grouped.items():
        questions.sort(
            key=lambda item: (
                item["position_ms"] is None,
                item["position_ms"] if item["position_ms"] is not None else 0,
                item["source_index"],
            )
        )
        streaming_vqa = []
        for question_number, item in enumerate(questions, start=1):
            streaming_vqa.append(
                {
                    "id": f"q{question_number}",
                    "position_ms": item["position_ms"],
                    "question": item["question"],
                    "options": item["options"],
                    "correct_option_id": item["correct_option_id"],
                    "metadata": item["metadata"],
                }
            )
        data.append(
            {
                "id": f"match_{match_numbers[match_name]:03d}_half_{half}",
                "inputs": [
                    {
                        "type": "video",
                        "path": f"{match_name}/{half}_720p.mkv",
                    }
                ],
                "metadata": {"match_name": match_name, "half": half},
                "streaming_vqa": streaming_vqa,
            }
        )

    manifest = {
        "version": "2.0",
        "date": conversion_date,
        "task": "streaming_vqa",
        "dataset_name": dataset_name,
        "modalities": ["video"],
        "metadata": {
            "source_file": source_file,
            "source_row_count": len(rows),
            "missing_timestamp_count": len(missing_timestamps),
        },
        "data": data,
    }
    return manifest, missing_timestamps


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            json.dump(payload, temporary, ensure_ascii=False, indent=2)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
    except Exception:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
        raise


def _resolve_manifest_video_paths(
    manifest: dict[str, Any],
    *,
    video_root: Path,
    output_directory: Path,
) -> None:
    """Validate long videos and make their paths relative to the manifest."""
    errors: list[str] = []
    resolved_paths: dict[str, str] = {}
    for sample in manifest["data"]:
        logical_path = str(sample["inputs"][0]["path"])
        video_path = video_root / Path(logical_path)
        if not video_path.is_file():
            errors.append(f"missing long video for sample {sample['id']}: {video_path}")
            continue
        resolved_paths[sample["id"]] = os.path.relpath(video_path, output_directory).replace(
            os.sep, "/"
        )
    if errors:
        raise ConversionError(errors)
    for sample in manifest["data"]:
        sample["inputs"][0]["path"] = resolved_paths[sample["id"]]
    manifest["metadata"]["video_root"] = str(video_root)


def convert_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    dataset_name: str | None = None,
    conversion_date: str | None = None,
    video_root: str | Path | None = None,
) -> dict[str, Any]:
    source = Path(input_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    chosen_date = conversion_date or date.today().isoformat()
    try:
        date.fromisoformat(chosen_date)
    except ValueError as exc:
        raise ConversionError([f"conversion date must use YYYY-MM-DD: {chosen_date}"]) from exc

    with source.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    manifest, missing_timestamps = convert_rows(
        rows,
        dataset_name=dataset_name or source.stem,
        source_file=source.name,
        conversion_date=chosen_date,
    )
    resolved_video_root = (
        Path(video_root).expanduser().resolve() if video_root is not None else destination.parent
    )
    _resolve_manifest_video_paths(
        manifest,
        video_root=resolved_video_root,
        output_directory=destination.parent,
    )
    _write_json_atomic(destination, manifest)
    if missing_timestamps:
        row_list = ", ".join(str(row) for row in missing_timestamps)
        print(
            f"warning: {len(missing_timestamps)} row(s) have no timestamp; "
            f"position_ms was set to null (rows: {row_list})",
            file=sys.stderr,
        )
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Source QA JSON list.")
    parser.add_argument("output", help="Destination OSL streaming-VQA JSON file.")
    parser.add_argument("--dataset-name", help="Dataset name; defaults to the input filename stem.")
    parser.add_argument("--conversion-date", help="ISO date for the manifest; defaults to today.")
    parser.add_argument(
        "--video-root",
        help=(
            "Directory containing <match_name>/1_720p.mkv and 2_720p.mkv; "
            "defaults to the output JSON directory. Referenced videos must exist."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = convert_file(
            args.input,
            args.output,
            dataset_name=args.dataset_name,
            conversion_date=args.conversion_date,
            video_root=args.video_root,
        )
    except (ConversionError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    question_count = sum(len(item["streaming_vqa"]) for item in manifest["data"])
    print(f"wrote {Path(args.output).expanduser().resolve()} ({question_count} questions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
