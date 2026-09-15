from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest


def _load_module():
    path = Path("tools/convert/qa_json_to_streaming_vqa.py")
    spec = importlib.util.spec_from_file_location("qa_json_to_streaming_vqa", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _row(**overrides):
    row = {
        "question": "Who made the pass?",
        "answer": "Player 10",
        "options": {"A": "Player 7", "B": "Player 10", "C": "Player 9", "D": "Player 3"},
        "closeA": "B",
        "match_name": "game one",
        "timestamp": "1 - 02:00",
        "video_file": "1_720p-Scene-010.mp4",
    }
    row.update(overrides)
    return row


def test_groups_by_match_half_sorts_questions_and_builds_long_video_paths():
    module = _load_module()
    rows = [
        _row(question="Later", timestamp="1 - 46:03"),
        _row(question="Second half", timestamp="2 - 00:05", video_file="2_720p-Scene-001.mp4"),
        _row(question="Earlier", timestamp="1 - 01:02"),
    ]

    manifest, warnings = module.convert_rows(
        rows, dataset_name="demo", source_file="input.json", conversion_date="2026-09-14"
    )

    assert warnings == []
    assert manifest["task"] == "streaming_vqa"
    assert [item["id"] for item in manifest["data"]] == ["match_001_half_1", "match_001_half_2"]
    first_half = manifest["data"][0]
    assert first_half["inputs"] == [{"type": "video", "path": "game one/1_720p.mkv"}]
    assert [item["question"] for item in first_half["streaming_vqa"]] == ["Earlier", "Later"]
    assert [item["position_ms"] for item in first_half["streaming_vqa"]] == [62_000, 2_763_000]
    assert [item["id"] for item in first_half["streaming_vqa"]] == ["q1", "q2"]


def test_converts_options_and_uses_marked_option_as_authoritative():
    module = _load_module()
    row = _row(answer="short free-form wording", referenced_players=[{"id": "p1"}])

    manifest, _ = module.convert_rows(
        [row], dataset_name="demo", source_file="input.json", conversion_date="2026-09-14"
    )
    question = manifest["data"][0]["streaming_vqa"][0]

    assert question["options"] == [
        {"id": "o1", "text": "Player 7"},
        {"id": "o2", "text": "Player 10"},
        {"id": "o3", "text": "Player 9"},
        {"id": "o4", "text": "Player 3"},
    ]
    assert question["correct_option_id"] == "o2"
    assert question["metadata"]["answer"] == "short free-form wording"
    assert question["metadata"]["referenced_players"] == [{"id": "p1"}]
    assert question["metadata"]["correct_option_key"] == "B"


@pytest.mark.parametrize(("answer", "expected"), [(True, "o1"), (False, "o2"), ("YES", "o1")])
def test_infers_yes_no_option_when_close_answer_is_absent(answer, expected):
    module = _load_module()
    row = _row(answer=answer, options={"A": "yes", "B": "no"}, closeA=None)

    manifest, _ = module.convert_rows(
        [row], dataset_name="demo", source_file="input.json", conversion_date="2026-09-14"
    )

    assert manifest["data"][0]["streaming_vqa"][0]["correct_option_id"] == expected


def test_preserves_reason_and_dimension_metadata():
    module = _load_module()
    row = _row(reason="Because the attack needed more width.", dimension="dim4")
    row.pop("answer")

    manifest, _ = module.convert_rows(
        [row], dataset_name="demo", source_file="dim4.json", conversion_date="2026-09-14"
    )
    metadata = manifest["data"][0]["streaming_vqa"][0]["metadata"]

    assert metadata["reason"] == "Because the attack needed more width."
    assert metadata["dimension"] == "dim4"
    assert metadata["source_question"] == "Who made the pass?"
    assert metadata["source_options"]["B"] == "Player 10"


def test_missing_timestamp_warns_and_writes_null_position(tmp_path, capsys):
    module = _load_module()
    input_path = tmp_path / "action_en.json"
    output_path = tmp_path / "converted.json"
    input_path.write_text(
        json.dumps([_row(timestamp=None, video_file="2_720p-Scene-002_tracked.mp4")]),
        encoding="utf-8",
    )
    video = tmp_path / "game one" / "2_720p.mkv"
    video.parent.mkdir()
    video.touch()

    manifest = module.convert_file(input_path, output_path, conversion_date="2026-09-14")

    assert output_path.exists()
    assert manifest["data"][0]["id"] == "match_001_half_2"
    assert manifest["data"][0]["streaming_vqa"][0]["position_ms"] is None
    warning = capsys.readouterr().err
    assert "1 row(s) have no timestamp" in warning
    assert "rows: 1" in warning


def test_video_root_validates_media_and_writes_manifest_relative_path(tmp_path):
    module = _load_module()
    input_path = tmp_path / "source" / "dim1.json"
    output_path = tmp_path / "manifests" / "converted.json"
    video_root = tmp_path / "media"
    input_path.parent.mkdir()
    output_path.parent.mkdir()
    video = video_root / "game one" / "1_720p.mkv"
    video.parent.mkdir(parents=True)
    video.touch()
    input_path.write_text(json.dumps([_row()]), encoding="utf-8")

    manifest = module.convert_file(
        input_path,
        output_path,
        video_root=video_root,
        conversion_date="2026-09-14",
    )

    expected = Path(os.path.relpath(video, output_path.parent)).as_posix()
    assert manifest["data"][0]["inputs"][0]["path"] == expected
    assert manifest["metadata"]["video_root"] == str(video_root.resolve())


def test_missing_long_video_fails_without_writing_output(tmp_path):
    module = _load_module()
    input_path = tmp_path / "source.json"
    output_path = tmp_path / "converted.json"
    input_path.write_text(json.dumps([_row()]), encoding="utf-8")

    with pytest.raises(module.ConversionError, match="missing long video"):
        module.convert_file(input_path, output_path, conversion_date="2026-09-14")

    assert not output_path.exists()


def test_null_positions_sort_after_timed_questions_stably():
    module = _load_module()
    rows = [
        _row(question="Unknown one", timestamp=None),
        _row(question="Known", timestamp="1 - 00:03"),
        _row(question="Unknown two", timestamp=None),
    ]

    manifest, warnings = module.convert_rows(
        rows, dataset_name="demo", source_file="input.json", conversion_date="2026-09-14"
    )

    assert warnings == [1, 3]
    questions = manifest["data"][0]["streaming_vqa"]
    assert [item["question"] for item in questions] == ["Known", "Unknown one", "Unknown two"]
    assert [item["position_ms"] for item in questions] == [3_000, None, None]


def test_aggregates_validation_errors_and_does_not_write_output(tmp_path):
    module = _load_module()
    input_path = tmp_path / "bad.json"
    output_path = tmp_path / "should_not_exist.json"
    rows = [
        _row(question="", closeA="Z"),
        _row(timestamp="3 - 10:00", options={"A": "same", "B": "same"}),
    ]
    input_path.write_text(json.dumps(rows), encoding="utf-8")

    with pytest.raises(module.ConversionError) as exc_info:
        module.convert_file(input_path, output_path, conversion_date="2026-09-14")

    message = str(exc_info.value)
    assert "row 1: question must be a non-empty string" in message
    assert "row 1: closeA 'Z' does not identify an available option" in message
    assert "row 2: invalid timestamp" in message
    assert "row 2: option texts must be unique" in message
    assert not output_path.exists()
