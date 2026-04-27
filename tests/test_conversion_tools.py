import json

import pandas as pd
import pytest

from opensportslib.tools import convert_json_to_parquet, convert_parquet_to_json, parse_shard_size


def test_json_to_parquet_and_back_supports_non_video_inputs(tmp_path):
    tracking_path = tmp_path / "test" / "clip_000000.parquet"
    video_path = tmp_path / "test" / "clip_000000.mp4"
    tracking_path.parent.mkdir(parents=True)
    tracking_path.write_bytes(b"tracking-bytes")
    video_path.write_bytes(b"video-bytes")

    payload = {
        "version": "2.0",
        "date": "2026-03-08",
        "task": "action_classification",
        "modalities": ["tracking_parquet", "video"],
        "dataset_name": "mixed_modalities_test",
        "metadata": {"split": "test"},
        "labels": {"action": {"type": "single_label", "labels": ["PASS"]}},
        "data": [
            {
                "id": "sample_000000",
                "inputs": [
                    {"type": "tracking_parquet", "path": "test/clip_000000.parquet"},
                    {"type": "video", "path": "test/clip_000000.mp4", "fps": 25.0},
                ],
                "labels": {"action": {"label": "PASS"}},
                "metadata": {"game_id": "3850"},
            }
        ],
    }
    json_path = tmp_path / "annotations_test.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    parquet_dataset_dir = tmp_path / "converted_dataset"
    forward_result = convert_json_to_parquet(
        json_path=json_path,
        media_root=tmp_path,
        output_dir=parquet_dataset_dir,
        overwrite=True,
    )

    assert forward_result["input_files_added"] == 2

    metadata_df = pd.read_parquet(parquet_dataset_dir / "metadata.parquet")
    row = metadata_df.iloc[0]
    assert "header" in metadata_df.columns
    assert json.loads(row["header"]) == {k: v for k, v in payload.items() if k != "data"}
    assert json.loads(row["sample_payload"]) == payload["data"][0]

    restored_json_path = tmp_path / "restored" / "annotations_test.json"
    restored_media_root = tmp_path / "restored"
    backward_result = convert_parquet_to_json(
        dataset_dir=parquet_dataset_dir,
        output_json_path=restored_json_path,
        extract_media=True,
        output_media_root=restored_media_root,
    )

    restored_payload = json.loads(restored_json_path.read_text(encoding="utf-8"))
    assert restored_payload["modalities"] == payload["modalities"]
    assert restored_payload["data"][0]["inputs"] == payload["data"][0]["inputs"]
    assert backward_result["extracted_input_files"] == 2
    assert (restored_media_root / "test" / "clip_000000.parquet").is_file()
    assert (restored_media_root / "test" / "clip_000000.mp4").is_file()


def test_json_to_parquet_missing_policy_skip_and_raise(tmp_path):
    existing = tmp_path / "clips" / "clip_ok.mp4"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"ok")

    payload = {
        "version": "2.0",
        "task": "action_classification",
        "labels": {},
        "data": [
            {
                "id": "sample_0",
                "inputs": [
                    {"type": "video", "path": "clips/clip_ok.mp4"},
                    {"type": "video", "path": "clips/clip_missing.mp4"},
                ],
            }
        ],
    }
    json_path = tmp_path / "annotations.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    skip_out = tmp_path / "out_skip"
    skip_result = convert_json_to_parquet(
        json_path=json_path,
        media_root=tmp_path,
        output_dir=skip_out,
        missing_policy="skip",
        overwrite=True,
    )
    assert skip_result["input_files_added"] == 1
    assert skip_result["missing_input_files"] == 1

    with pytest.raises(FileNotFoundError):
        convert_json_to_parquet(
            json_path=json_path,
            media_root=tmp_path,
            output_dir=tmp_path / "out_raise",
            missing_policy="raise",
            overwrite=True,
        )


def test_json_to_parquet_absolute_paths_rewrites_sample_payload(tmp_path):
    clip_path = tmp_path / "clips" / "clip_0.mp4"
    clip_path.parent.mkdir(parents=True)
    clip_path.write_bytes(b"clip")

    payload = {
        "version": "2.0",
        "task": "action_classification",
        "labels": {},
        "data": [
            {
                "id": "sample_0",
                "inputs": [{"type": "video", "path": "clips/clip_0.mp4"}],
            }
        ],
    }
    json_path = tmp_path / "annotations.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    out_dir = tmp_path / "out_abs"
    convert_json_to_parquet(
        json_path=json_path,
        media_root=tmp_path,
        output_dir=out_dir,
        keep_relative_paths_in_parquet=False,
        overwrite=True,
    )

    df = pd.read_parquet(out_dir / "metadata.parquet")
    payload_from_parquet = json.loads(df.iloc[0]["sample_payload"])
    assert payload_from_parquet["inputs"][0]["path"] == str(clip_path)


def test_json_to_parquet_size_mode_splits_on_target_tar_size(tmp_path):
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    data = []
    for idx in range(3):
        clip_path = clips_dir / f"clip_{idx}.bin"
        clip_path.write_bytes(b"x" * 3_000)
        data.append({
            "id": f"sample_{idx}",
            "inputs": [{"type": "binary", "path": f"clips/clip_{idx}.bin"}],
        })

    json_path = tmp_path / "annotations.json"
    json_path.write_text(json.dumps({"labels": {}, "data": data}), encoding="utf-8")

    out_dir = tmp_path / "out_size"
    result = convert_json_to_parquet(
        json_path=json_path,
        media_root=tmp_path,
        output_dir=out_dir,
        shard_size=15_000,
        overwrite=True,
    )

    assert result["shard_mode"] == "size"
    assert result["shard_size"] == 15_000
    assert result["num_shards"] == 2
    metadata_df = pd.read_parquet(out_dir / "metadata.parquet")
    assert metadata_df["shard_name"].tolist() == [
        "shard-000000.tar",
        "shard-000000.tar",
        "shard-000001.tar",
    ]


def test_json_to_parquet_oversized_sample_gets_single_shard(tmp_path):
    clip_path = tmp_path / "clips" / "large.bin"
    clip_path.parent.mkdir()
    clip_path.write_bytes(b"x" * 5_000)
    payload = {
        "labels": {},
        "data": [
            {"id": "large", "inputs": [{"type": "binary", "path": "clips/large.bin"}]},
            {"id": "small", "inputs": []},
        ],
    }
    json_path = tmp_path / "annotations.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    out_dir = tmp_path / "out_oversized"
    result = convert_json_to_parquet(
        json_path=json_path,
        media_root=tmp_path,
        output_dir=out_dir,
        shard_size=2_000,
        overwrite=True,
    )

    assert result["num_shards"] == 2
    metadata_df = pd.read_parquet(out_dir / "metadata.parquet")
    assert metadata_df["shard_name"].tolist() == ["shard-000000.tar", "shard-000001.tar"]


def test_json_to_parquet_samples_mode_preserves_count_based_sharding(tmp_path):
    payload = {
        "labels": {},
        "data": [
            {"id": f"sample_{idx}", "inputs": []}
            for idx in range(5)
        ],
    }
    json_path = tmp_path / "annotations.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    out_dir = tmp_path / "out_samples"
    result = convert_json_to_parquet(
        json_path=json_path,
        media_root=tmp_path,
        output_dir=out_dir,
        shard_mode="samples",
        samples_per_shard=2,
        overwrite=True,
    )

    assert result["shard_mode"] == "samples"
    assert result["samples_per_shard"] == 2
    assert result["num_shards"] == 3
    metadata_df = pd.read_parquet(out_dir / "metadata.parquet")
    assert metadata_df["shard_name"].tolist() == [
        "shard-000000.tar",
        "shard-000000.tar",
        "shard-000001.tar",
        "shard-000001.tar",
        "shard-000002.tar",
    ]


def test_parse_shard_size_supports_human_readable_units():
    assert parse_shard_size("500MB") == 500_000_000
    assert parse_shard_size("1GB") == 1_000_000_000
    assert parse_shard_size("1024MiB") == 1024 * 1024**2
    assert parse_shard_size("42") == 42


def test_parquet_to_json_rejects_legacy_schema_without_header(tmp_path):
    dataset_dir = tmp_path / "legacy_dataset"
    shards_dir = dataset_dir / "shards"
    shards_dir.mkdir(parents=True)

    legacy_df = pd.DataFrame(
        [
            {
                "sample_id": "sample_0",
                "sample_index": 0,
                "shard_name": "shard-000000.tar",
                "sample_inputs": json.dumps([{"type": "video", "path": "clips/clip.mp4"}]),
            }
        ]
    )
    legacy_df.to_parquet(dataset_dir / "metadata.parquet", index=False)

    with pytest.raises(ValueError, match="Legacy schemas are no longer supported"):
        convert_parquet_to_json(
            dataset_dir=dataset_dir,
            output_json_path=tmp_path / "reconstructed.json",
        )
