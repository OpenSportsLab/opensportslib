import json
import shutil
import tarfile
from pathlib import Path

import pandas as pd
import pytest
from fsspec.callbacks import TqdmCallback

import opensportslib.tools.hf_transfer as hf_transfer_module

from opensportslib.tools.hf_transfer import (
    HF_BRANCH_KEY,
    HF_COMMIT_KEY,
    HF_FORMAT_KEY,
    HF_REPO_ID_KEY,
    HF_SPLIT_KEY,
    HfTransferCancelled,
    MissingDatasetInputsError,
    create_dataset_branch_on_hf,
    create_dataset_repo_on_hf,
    dataset_repo_exists_on_hf,
    download_dataset_split_from_hf,
    download_dataset_sample_inputs_from_hf,
    download_dataset_missing_inputs_from_hf,
    download_dataset_splits_from_hf,
    extract_local_input_upload_entries_from_json,
    extract_repo_paths_from_json,
    find_missing_dataset_inputs,
    is_hf_download_url_not_found_error,
    is_hf_repo_not_found_error,
    is_hf_revision_not_found_error,
    list_dataset_branches_on_hf,
    list_dataset_splits_on_hf,
    read_hf_source_metadata_from_dataset,
    upload_dataset_as_parquet_to_hf,
    upload_dataset_inputs_from_json_to_hf,
    write_hf_source_metadata_to_dataset_json,
)


def _copying_hf_download(remote_root: Path, downloaded: list[str]):
    def _download(**kwargs):
        filename = kwargs["filename"]
        downloaded.append(filename)
        source = remote_root / filename
        destination = Path(kwargs["local_dir"]) / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return str(destination)

    return _download


def test_download_hf_file_reports_transferred_and_total_bytes(monkeypatch, tmp_path):
    class _FakeFileSystem:
        def __init__(self, token=None):
            assert token == "hf_test"

        def get_file(self, remote_path, local_path, callback, revision):
            assert remote_path == "datasets/OpenSportsLab/repo/clips/large.mp4"
            assert revision == "pinned"
            callback.set_size(6)
            Path(local_path).write_bytes(b"abcdef")
            callback.tqdm.update(2)
            callback.tqdm.update(4)

    monkeypatch.setattr(
        hf_transfer_module,
        "_import_hf_file_system",
        lambda: (_FakeFileSystem, TqdmCallback),
    )
    monkeypatch.setattr(hf_transfer_module, "_import_hf_xet_download", lambda: None)
    progress = []

    result = hf_transfer_module._download_hf_file(
        pytest.fail,
        repo_id="OpenSportsLab/repo",
        filename="clips/large.mp4",
        revision="pinned",
        local_dir=str(tmp_path),
        token="hf_test",
        byte_progress_cb=lambda filename, current, total: progress.append(
            (filename, current, total)
        ),
    )

    assert result == str(tmp_path / "clips" / "large.mp4")
    assert Path(result).read_bytes() == b"abcdef"
    assert progress[0] == ("clips/large.mp4", 0, 6)
    assert ("clips/large.mp4", 2, 6) in progress
    assert progress[-1] == ("clips/large.mp4", 6, 6)
    assert [current for _, current, _ in progress] == sorted(
        current for _, current, _ in progress
    )


def test_download_hf_file_cancels_during_transfer_and_removes_partial(
    monkeypatch, tmp_path
):
    cancelled = {"value": False}

    class _FakeFileSystem:
        def __init__(self, token=None):
            del token

        def get_file(self, remote_path, local_path, callback, revision):
            del remote_path, revision
            callback.set_size(10)
            Path(local_path).write_bytes(b"partial")
            cancelled["value"] = True
            callback.tqdm.update(7)

    monkeypatch.setattr(
        hf_transfer_module,
        "_import_hf_file_system",
        lambda: (_FakeFileSystem, TqdmCallback),
    )
    monkeypatch.setattr(hf_transfer_module, "_import_hf_xet_download", lambda: None)

    with pytest.raises(HfTransferCancelled):
        hf_transfer_module._download_hf_file(
            pytest.fail,
            repo_id="OpenSportsLab/repo",
            filename="clips/large.mp4",
            revision="pinned",
            local_dir=str(tmp_path),
            token=None,
            byte_progress_cb=lambda *_args: None,
            is_cancelled=lambda: cancelled["value"],
        )

    assert not (tmp_path / "clips" / "large.mp4").exists()
    assert list((tmp_path / "clips").glob("*.part")) == []


def test_download_hf_file_reports_bytes_while_using_xet(monkeypatch, tmp_path):
    xet_file_data = object()

    class _Metadata:
        size = 6

    _Metadata.xet_file_data = xet_file_data

    def _fake_xet_get(**kwargs):
        assert kwargs["xet_file_data"] is xet_file_data
        assert kwargs["headers"] == {"authorization": "Bearer hf_test"}
        assert kwargs["expected_size"] == 6
        assert kwargs["displayed_filename"] == "clips/large.mp4"
        Path(kwargs["incomplete_path"]).write_bytes(b"abcdef")
        kwargs["_tqdm_bar"].update(2)
        kwargs["_tqdm_bar"].update(4)

    monkeypatch.setattr(
        hf_transfer_module,
        "_import_hf_xet_download",
        lambda: (
            lambda url, token: _Metadata(),
            lambda **kwargs: "https://huggingface.test/file",
            _fake_xet_get,
            lambda token: {"authorization": f"Bearer {token}"},
            lambda: True,
        ),
    )
    monkeypatch.setattr(
        hf_transfer_module,
        "_import_hf_file_system",
        lambda: pytest.fail("classic HTTP fallback should not be used"),
    )
    progress = []

    result = hf_transfer_module._download_hf_file(
        pytest.fail,
        repo_id="OpenSportsLab/repo",
        filename="clips/large.mp4",
        revision="pinned",
        local_dir=str(tmp_path),
        token="hf_test",
        byte_progress_cb=lambda filename, current, total: progress.append(
            (filename, current, total)
        ),
    )

    assert result == str(tmp_path / "clips" / "large.mp4")
    assert Path(result).read_bytes() == b"abcdef"
    assert progress == [
        ("clips/large.mp4", 0, 6),
        ("clips/large.mp4", 2, 6),
        ("clips/large.mp4", 6, 6),
    ]


def test_download_hf_file_rejects_unsafe_destination(tmp_path):
    with pytest.raises(ValueError, match="Unsafe Hugging Face file path"):
        hf_transfer_module._download_hf_file(
            pytest.fail,
            repo_id="OpenSportsLab/repo",
            filename="../escape.mp4",
            revision="pinned",
            local_dir=str(tmp_path),
            token=None,
            byte_progress_cb=lambda *_args: None,
        )


def test_find_missing_dataset_inputs_includes_primary_and_ball_paths(tmp_path):
    present_path = tmp_path / "clips" / "present.mp4"
    present_path.parent.mkdir()
    present_path.write_bytes(b"present")
    json_path = tmp_path / "test.json"
    json_path.write_text(
        json.dumps(
            {
                "data": [
                    {
                        "id": "sample-1",
                        "inputs": [
                            {"path": "clips/present.mp4", "type": "video"},
                            {
                                "path": "tracking/joints.h5",
                                "ball_path": "tracking/ball.h5",
                                "type": "player_joints_h5",
                            },
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    missing = find_missing_dataset_inputs(str(json_path))

    assert [(item["role"], item["path"]) for item in missing] == [
        ("primary", "tracking/joints.h5"),
        ("ball", "tracking/ball.h5"),
    ]
    assert all(item["sample_id"] == "sample-1" for item in missing)


def test_download_dataset_missing_inputs_revalidates_after_hydration(
    monkeypatch, tmp_path
):
    json_path = tmp_path / "test.json"
    json_path.write_text(
        json.dumps(
            {
                "data": [
                    {
                        "id": "sample-1",
                        "inputs": [{"path": "clips/one.mp4", "type": "video"}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    calls = []

    def _hydrate(dataset_json_path, sample_id, **kwargs):
        calls.append((dataset_json_path, sample_id, kwargs))
        destination = tmp_path / "clips" / "one.mp4"
        destination.parent.mkdir()
        destination.write_bytes(b"video")
        return {
            "requested_downloaded_count": 1,
            "opportunistic_downloaded_count": 0,
            "failed_count": 0,
        }

    monkeypatch.setattr(
        hf_transfer_module,
        "download_dataset_sample_inputs_from_hf",
        _hydrate,
    )

    result = download_dataset_missing_inputs_from_hf(str(json_path))

    assert [sample_id for _, sample_id, _ in calls] == ["sample-1"]
    assert result["initial_missing_count"] == 1
    assert result["remaining_missing_count"] == 0
    assert result["requested_downloaded_count"] == 1


def test_extract_repo_paths_from_json_supports_legacy_and_osl_v2():
    payload = {
        "videos": [{"path": "legacy/a.mp4"}],
        "data": [
            {
                "id": "sample_1",
                "inputs": [
                    {"path": "v2/b.mp4", "type": "video"},
                    {"path": "v2/c.json", "type": "captions"},
                ],
            }
        ],
    }
    paths = extract_repo_paths_from_json(payload)
    assert set(paths) == {"legacy/a.mp4", "v2/b.mp4", "v2/c.json"}


def test_extract_repo_paths_from_json_returns_all_input_paths_by_default():
    payload = {
        "data": [
            {
                "id": "sample_1",
                "inputs": [
                    {"path": "test/a.mp4", "type": "video"},
                    {"path": "test/a.txt", "type": "captions"},
                ],
            }
        ]
    }

    assert extract_repo_paths_from_json(payload) == ["test/a.mp4", "test/a.txt"]


def test_extract_local_input_upload_entries_from_json_uses_paths_from_inputs(tmp_path):
    clip_a = tmp_path / "train" / "action_0" / "clip_0.mp4"
    clip_b = tmp_path / "valid" / "action_1" / "clip_1.mp4"
    clip_a.parent.mkdir(parents=True)
    clip_b.parent.mkdir(parents=True)
    clip_a.write_bytes(b"a")
    clip_b.write_bytes(b"b")

    payload = {
        "data": [
            {
                "id": "sample_1",
                "inputs": [
                    {"path": "train/action_0/clip_0.mp4", "type": "video"},
                    {"path": "valid/action_1/clip_1.mp4", "type": "video"},
                ],
            }
        ]
    }
    json_path = tmp_path / "annotations.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    entries = extract_local_input_upload_entries_from_json(str(json_path))
    assert len(entries) == 2
    assert entries[0]["path_in_repo"] == "train/action_0/clip_0.mp4"
    assert entries[1]["path_in_repo"] == "valid/action_1/clip_1.mp4"


def test_extract_local_input_upload_entries_from_json_raises_for_missing_local_file(tmp_path):
    payload = {
        "data": [
            {
                "id": "sample_1",
                "inputs": [
                    {"path": "train/action_0/missing.mp4", "type": "video"},
                ],
            }
        ]
    }
    json_path = tmp_path / "annotations.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        extract_local_input_upload_entries_from_json(str(json_path))


def test_is_hf_repo_not_found_error_detects_hf_repo_missing_message():
    message = (
        "404 Client Error. Repository Not Found for url: "
        "https://huggingface.co/api/datasets/OpenSportsLab/OSL-test-auto-upload/preupload/main."
    )
    assert is_hf_repo_not_found_error(message) is True
    assert is_hf_repo_not_found_error("HTTP 500 Internal Server Error") is False


def test_is_hf_revision_not_found_error_detects_missing_branch_message():
    message = (
        "404 Client Error. Revision Not Found for url: "
        "https://huggingface.co/api/datasets/OpenSportsLab/repo/preupload/feature-branch."
    )
    assert is_hf_revision_not_found_error(message) is True
    assert is_hf_revision_not_found_error("404 Client Error. Repository Not Found for url: ...") is False


def test_is_hf_download_url_not_found_error_detects_missing_url():
    message = (
        "404 Client Error. Entry Not Found for url: "
        "https://huggingface.co/datasets/OpenSportsLab/repo/resolve/main/missing.json."
    )
    assert is_hf_download_url_not_found_error(message) is True
    assert is_hf_download_url_not_found_error("Repository Not Found without 404 marker") is False


def test_create_dataset_repo_on_hf_calls_hf_api_create_repo(monkeypatch):
    calls = {}

    class _FakeApi:
        def __init__(self, token=None):
            calls["token"] = token

        def create_repo(self, **kwargs):
            calls["kwargs"] = kwargs
            return "https://huggingface.co/datasets/OpenSportsLab/new-repo"

    monkeypatch.setattr(
        "opensportslib.tools.hf_transfer._import_hf_hub",
        lambda: (_FakeApi, object(), object()),
    )

    result = create_dataset_repo_on_hf("OpenSportsLab/new-repo", token="hf_token", private=True)

    assert calls["token"] == "hf_token"
    assert calls["kwargs"]["repo_id"] == "OpenSportsLab/new-repo"
    assert calls["kwargs"]["repo_type"] == "dataset"
    assert calls["kwargs"]["private"] is True
    assert calls["kwargs"]["exist_ok"] is True
    assert result["repo_id"] == "OpenSportsLab/new-repo"
    assert result["repo_type"] == "dataset"


def test_create_dataset_branch_on_hf_calls_hf_api_create_branch(monkeypatch):
    calls = {}

    class _FakeApi:
        def __init__(self, token=None):
            calls["token"] = token

        def list_repo_commits(self, repo_id, repo_type=None):
            calls["list_repo_commits"] = {"repo_id": repo_id, "repo_type": repo_type}
            commit = type("_Commit", (), {"commit_id": "initsha"})()
            return [commit]

        def create_branch(self, **kwargs):
            calls["kwargs"] = kwargs

    monkeypatch.setattr(
        "opensportslib.tools.hf_transfer._import_hf_hub",
        lambda: (_FakeApi, object(), object()),
    )

    result = create_dataset_branch_on_hf(
        "OpenSportsLab/new-repo",
        "feature-x",
        source_revision="main",
        token="hf_token",
    )

    assert calls["token"] == "hf_token"
    assert calls["kwargs"]["repo_id"] == "OpenSportsLab/new-repo"
    assert calls["kwargs"]["repo_type"] == "dataset"
    assert calls["kwargs"]["branch"] == "feature-x"
    assert calls["kwargs"]["revision"] == "initsha"
    assert calls["kwargs"]["exist_ok"] is True
    assert result["repo_id"] == "OpenSportsLab/new-repo"
    assert result["branch"] == "feature-x"


def test_dataset_repo_exists_on_hf_returns_true_when_repo_info_succeeds(monkeypatch):
    class _FakeApi:
        def __init__(self, token=None):
            pass

        def repo_info(self, **kwargs):
            return {"id": kwargs.get("repo_id")}

    monkeypatch.setattr(
        "opensportslib.tools.hf_transfer._import_hf_hub",
        lambda: (_FakeApi, object(), object()),
    )

    assert dataset_repo_exists_on_hf("OpenSportsLab/existing-repo", token="hf_token") is True


def test_dataset_repo_exists_on_hf_returns_false_for_repo_not_found(monkeypatch):
    class _FakeApi:
        def __init__(self, token=None):
            pass

        def repo_info(self, **kwargs):
            raise RuntimeError("404 Client Error. Repository Not Found for url: https://huggingface.co/api/datasets/...")

    monkeypatch.setattr(
        "opensportslib.tools.hf_transfer._import_hf_hub",
        lambda: (_FakeApi, object(), object()),
    )

    assert dataset_repo_exists_on_hf("OpenSportsLab/missing-repo", token="hf_token") is False


def test_list_dataset_branches_on_hf_puts_main_first_then_alphabetical(monkeypatch):
    class _FakeRefs:
        branches = [
            type("_Ref", (), {"name": "zeta"})(),
            type("_Ref", (), {"name": "main"})(),
            type("_Ref", (), {"name": "alpha"})(),
        ]

    class _FakeApi:
        def __init__(self, token=None):
            pass

        def list_repo_refs(self, repo_id, repo_type=None):
            return _FakeRefs()

    monkeypatch.setattr(
        "opensportslib.tools.hf_transfer._import_hf_hub",
        lambda: (_FakeApi, object(), object()),
    )

    assert list_dataset_branches_on_hf("OpenSportsLab/repo") == ["main", "alpha", "zeta"]


def test_list_dataset_splits_on_hf_detects_parquet_layout(monkeypatch):
    class _FakeApi:
        def __init__(self, token=None):
            pass

        def list_repo_files(self, repo_id, revision=None, repo_type=None):
            return [
                "README.md",
                "train/metadata.parquet",
                "train/shard_manifest.parquet",
                "train/shards/shard-000000.tar",
                "test/metadata.parquet",
                "test/shards/shard-000000.tar",
            ]

    monkeypatch.setattr(
        "opensportslib.tools.hf_transfer._import_hf_hub",
        lambda: (_FakeApi, object(), object()),
    )

    result = list_dataset_splits_on_hf("OpenSportsLab/repo", "main")

    assert result == {"format": "parquet", "splits": ["train", "test"]}


def test_list_dataset_splits_on_hf_treats_json_dataset_with_parquet_media_as_json(monkeypatch):
    """
    A JSON-format dataset can reference arbitrary media files (e.g. tensor-encoded
    videos serialized as .parquet) inside a folder that happens to share a split's
    name. That must not be misdetected as the canonical Parquet+WebDataset export
    layout, which requires `{split}/metadata.parquet` + `{split}/shards/*.tar`.
    Regression test for OpenSportsLab/SNGAR-Action-Spotting-Tracking.
    """

    class _FakeApi:
        def __init__(self, token=None):
            pass

        def list_repo_files(self, repo_id, revision=None, repo_type=None):
            return [
                "README.md",
                "train.json",
                "valid.json",
                "test.json",
                "train/videos/10502.parquet",
                "train/videos/10503.parquet",
                "valid/videos/3841.parquet",
                "test/videos/3850.parquet",
            ]

    monkeypatch.setattr(
        "opensportslib.tools.hf_transfer._import_hf_hub",
        lambda: (_FakeApi, object(), object()),
    )

    result = list_dataset_splits_on_hf("OpenSportsLab/repo", "main")

    assert result == {"format": "json", "splits": ["train", "valid", "test"]}


def test_list_dataset_splits_on_hf_detects_json_layout(monkeypatch):
    class _FakeApi:
        def __init__(self, token=None):
            pass

        def list_repo_files(self, repo_id, revision=None, repo_type=None):
            return [
                "dataset_infos.json",
                "train.json",
                "challenge.json",
                "valid.json",
            ]

    monkeypatch.setattr(
        "opensportslib.tools.hf_transfer._import_hf_hub",
        lambda: (_FakeApi, object(), object()),
    )

    result = list_dataset_splits_on_hf("OpenSportsLab/repo", "main")

    assert result == {"format": "json", "splits": ["train", "valid", "challenge"]}


def test_list_dataset_splits_on_hf_returns_none_format_when_nothing_matches(monkeypatch):
    class _FakeApi:
        def __init__(self, token=None):
            pass

        def list_repo_files(self, repo_id, revision=None, repo_type=None):
            return ["README.md", "dataset_infos.json"]

    monkeypatch.setattr(
        "opensportslib.tools.hf_transfer._import_hf_hub",
        lambda: (_FakeApi, object(), object()),
    )

    assert list_dataset_splits_on_hf("OpenSportsLab/repo", "main") == {"format": None, "splits": []}


def test_download_dataset_splits_from_hf_downloads_each_split_with_prefixed_progress(monkeypatch, tmp_path):
    progress_messages = []
    calls = []

    def _fake_download_dataset_split_from_hf(repo_id, revision, split, output_dir, **kwargs):
        calls.append((repo_id, revision, split, output_dir))
        kwargs["progress_cb"](f"working on {split}")
        return {"split": split, "json_path": str(tmp_path / f"{split}.json")}

    monkeypatch.setattr(
        "opensportslib.tools.hf_transfer.download_dataset_split_from_hf",
        _fake_download_dataset_split_from_hf,
    )

    results = download_dataset_splits_from_hf(
        "OpenSportsLab/repo",
        "main",
        ["train", "valid"],
        str(tmp_path),
        download_format="json",
        progress_cb=progress_messages.append,
    )

    assert [call[2] for call in calls] == ["train", "valid"]
    assert [result["split"] for result in results] == ["train", "valid"]
    assert progress_messages == [
        "[1/2] train: working on train",
        "[2/2] valid: working on valid",
    ]


def test_download_dataset_splits_from_hf_requires_at_least_one_split():
    with pytest.raises(ValueError):
        download_dataset_splits_from_hf("OpenSportsLab/repo", "main", [], "/tmp/out")


def test_upload_dataset_inputs_from_json_to_hf_uploads_inputs_and_json(monkeypatch, tmp_path):
    clip_path = tmp_path / "train" / "clip_0.mp4"
    clip_path.parent.mkdir(parents=True)
    clip_path.write_bytes(b"video")

    payload = {
        "data": [
            {
                "id": "sample_1",
                "inputs": [{"path": "train/clip_0.mp4", "type": "video"}],
            }
        ]
    }
    json_path = tmp_path / "annotations.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    commit_calls = []

    class _FakeCommitOperationAdd:
        def __init__(self, *, path_in_repo, path_or_fileobj):
            self.path_in_repo = path_in_repo
            self.path_or_fileobj = path_or_fileobj

    class _FakeApi:
        def __init__(self, token=None):
            pass

        def create_commit(self, **kwargs):
            commit_calls.append(kwargs)
            return type("_CommitInfo", (), {"oid": "abc123"})()

    monkeypatch.setattr(
        "opensportslib.tools.hf_transfer._import_hf_hub",
        lambda: (_FakeApi, object(), object()),
    )
    monkeypatch.setattr(
        "opensportslib.tools.hf_transfer._import_hf_commit_operation_add",
        lambda: _FakeCommitOperationAdd,
    )

    result = upload_dataset_inputs_from_json_to_hf(
        repo_id="OpenSportsLab/test-repo",
        json_path=str(json_path),
        revision="dev-branch",
        split="test",
        commit_message="Upload test",
        token="hf_token",
    )

    assert len(commit_calls) == 1
    commit_kwargs = commit_calls[0]
    assert commit_kwargs["repo_id"] == "OpenSportsLab/test-repo"
    assert commit_kwargs["repo_type"] == "dataset"
    assert commit_kwargs["revision"] == "dev-branch"
    assert commit_kwargs["commit_message"] == "Upload test"
    operations = commit_kwargs["operations"]
    assert len(operations) == 2
    assert operations[0].path_in_repo == "test.json"
    assert operations[0].path_or_fileobj == str(json_path)
    assert operations[1].path_in_repo == "train/clip_0.mp4"
    assert result["input_file_count"] == 1
    assert result["unique_input_file_count"] == 1
    assert result["uploaded_file_count"] == 2
    assert result["split"] == "test"
    assert result["json_path_in_repo"] == "test.json"
    assert result["revision"] == "dev-branch"
    assert result["commit_ref"] == "abc123"


def test_upload_dataset_inputs_from_json_to_hf_skips_missing_inputs(monkeypatch, tmp_path):
    clip_path = tmp_path / "train" / "present.mp4"
    clip_path.parent.mkdir(parents=True)
    clip_path.write_bytes(b"video")
    json_path = tmp_path / "annotations.json"
    json_path.write_text(
        json.dumps(
            {
                "data": [
                    {
                        "id": "sample_1",
                        "inputs": [
                            {"path": "train/present.mp4", "type": "video"},
                            {"path": "train/missing.mp4", "type": "video"},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    commit_calls = []

    class _FakeCommitOperationAdd:
        def __init__(self, *, path_in_repo, path_or_fileobj):
            self.path_in_repo = path_in_repo
            self.path_or_fileobj = path_or_fileobj

    class _FakeApi:
        def __init__(self, token=None):
            pass

        def create_commit(self, **kwargs):
            commit_calls.append(kwargs)
            return type("_CommitInfo", (), {"oid": "abc123"})()

    monkeypatch.setattr(
        "opensportslib.tools.hf_transfer._import_hf_hub",
        lambda: (_FakeApi, object(), object()),
    )
    monkeypatch.setattr(
        "opensportslib.tools.hf_transfer._import_hf_commit_operation_add",
        lambda: _FakeCommitOperationAdd,
    )
    progress_messages = []

    result = upload_dataset_inputs_from_json_to_hf(
        repo_id="OpenSportsLab/test-repo",
        json_path=str(json_path),
        split="test",
        progress_cb=progress_messages.append,
    )

    operations = commit_calls[0]["operations"]
    assert [operation.path_in_repo for operation in operations] == [
        "test.json",
        "train/present.mp4",
    ]
    assert result["input_file_count"] == 1
    assert result["uploaded_file_count"] == 2
    assert result["skipped_missing_input_count"] == 1
    assert result["skipped_missing_input_paths"] == ["train/missing.mp4"]
    assert progress_messages[0] == (
        "Skipping 1 referenced input files that are not available locally."
    )


def test_upload_dataset_inputs_from_json_to_hf_can_upload_json_when_all_inputs_missing(
    monkeypatch, tmp_path
):
    json_path = tmp_path / "annotations.json"
    json_path.write_text(
        json.dumps(
            {
                "data": [
                    {
                        "id": "sample_1",
                        "inputs": [{"path": "train/missing.mp4", "type": "video"}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    commit_calls = []

    class _FakeCommitOperationAdd:
        def __init__(self, *, path_in_repo, path_or_fileobj):
            self.path_in_repo = path_in_repo
            self.path_or_fileobj = path_or_fileobj

    class _FakeApi:
        def __init__(self, token=None):
            pass

        def create_commit(self, **kwargs):
            commit_calls.append(kwargs)
            return type("_CommitInfo", (), {"oid": "abc123"})()

    monkeypatch.setattr(
        "opensportslib.tools.hf_transfer._import_hf_hub",
        lambda: (_FakeApi, object(), object()),
    )
    monkeypatch.setattr(
        "opensportslib.tools.hf_transfer._import_hf_commit_operation_add",
        lambda: _FakeCommitOperationAdd,
    )

    result = upload_dataset_inputs_from_json_to_hf(
        repo_id="OpenSportsLab/test-repo",
        json_path=str(json_path),
        split="test",
    )

    assert [operation.path_in_repo for operation in commit_calls[0]["operations"]] == [
        "test.json"
    ]
    assert result["input_file_count"] == 0
    assert result["uploaded_file_count"] == 1
    assert result["skipped_missing_input_count"] == 1


def test_upload_dataset_as_parquet_to_hf_blocks_missing_inputs_before_conversion(
    monkeypatch, tmp_path
):
    json_path = tmp_path / "annotations.json"
    json_path.write_text(
        json.dumps(
            {
                "data": [
                    {
                        "id": "sample-1",
                        "inputs": [{"path": "clips/missing.mp4", "type": "video"}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        hf_transfer_module,
        "convert_json_to_parquet",
        lambda **_kwargs: pytest.fail("Conversion must not start with missing inputs"),
    )

    with pytest.raises(MissingDatasetInputsError) as error:
        upload_dataset_as_parquet_to_hf(
            repo_id="OpenSportsLab/test-repo",
            json_path=str(json_path),
        )

    assert error.value.missing_inputs[0]["path"] == "clips/missing.mp4"


def test_upload_dataset_as_parquet_to_hf_uploads_all_generated_files_in_one_commit(monkeypatch, tmp_path):
    json_path = tmp_path / "annotations.json"
    json_path.write_text(json.dumps({"data": []}), encoding="utf-8")

    commit_calls = []
    convert_calls = []

    class _FakeCommitOperationAdd:
        def __init__(self, *, path_in_repo, path_or_fileobj):
            self.path_in_repo = path_in_repo
            self.path_or_fileobj = path_or_fileobj

    class _FakeApi:
        def __init__(self, token=None):
            pass

        def list_repo_files(self, *args, **kwargs):
            return [
                "test/shards/shard-999999.tar",
                "test/notes.txt",
                "other/shards/shard-000000.tar",
            ]

        def create_commit(self, **kwargs):
            commit_calls.append(kwargs)
            return type("_CommitInfo", (), {"oid": "parquetsha"})()

    def _fake_convert_json_to_parquet(**kwargs):
        convert_calls.append(kwargs)
        output_dir = kwargs["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "dataset.parquet").write_bytes(b"parquet")
        shard_dir = output_dir / "samples"
        shard_dir.mkdir(parents=True, exist_ok=True)
        (shard_dir / "shard-00000.tar").write_bytes(b"tar0")
        (shard_dir / "shard-00001.tar").write_bytes(b"tar1")
        return {"num_samples": 2, "input_files_added": 2}

    monkeypatch.setattr(
        "opensportslib.tools.hf_transfer._import_hf_hub",
        lambda: (_FakeApi, object(), object()),
    )
    monkeypatch.setattr(
        "opensportslib.tools.hf_transfer._import_hf_commit_operation_add",
        lambda: _FakeCommitOperationAdd,
    )
    monkeypatch.setattr(
        "opensportslib.tools.hf_transfer.convert_json_to_parquet",
        _fake_convert_json_to_parquet,
    )

    result = upload_dataset_as_parquet_to_hf(
        repo_id="OpenSportsLab/test-repo",
        json_path=str(json_path),
        revision="dev-branch",
        split="test",
        commit_message="Upload parquet test",
        token="hf_token",
    )

    assert len(commit_calls) == 1
    commit_kwargs = commit_calls[0]
    assert commit_kwargs["repo_id"] == "OpenSportsLab/test-repo"
    assert commit_kwargs["repo_type"] == "dataset"
    assert commit_kwargs["revision"] == "dev-branch"
    assert commit_kwargs["commit_message"] == "Upload parquet test"
    operations = commit_kwargs["operations"]
    assert len(operations) == 4
    assert [op.path_in_repo for op in operations] == [
        "test/shards/shard-999999.tar",
        "test/dataset.parquet",
        "test/samples/shard-00000.tar",
        "test/samples/shard-00001.tar",
    ]
    assert result["upload_kind"] == "parquet"
    assert result["split"] == "test"
    assert result["folder_name"] == "test"
    assert result["uploaded_file_count"] == 3
    assert result["deleted_file_count"] == 1
    assert result["num_samples"] == 2
    assert result["shard_mode"] == "size"
    assert result["shard_size"] == 1_000_000_000
    assert result["samples_per_shard"] == 100
    assert result["num_shards"] == 0
    assert result["input_file_count"] == 2
    assert "video_file_count" not in result
    assert result["commit_ref"] == "parquetsha"
    assert len(convert_calls) == 1
    assert convert_calls[0]["shard_mode"] == "size"
    assert convert_calls[0]["shard_size"] == 1_000_000_000
    assert convert_calls[0]["samples_per_shard"] == 100
    assert convert_calls[0]["missing_policy"] == "raise"


def test_upload_dataset_as_parquet_to_hf_forwards_custom_shard_size(monkeypatch, tmp_path):
    json_path = tmp_path / "annotations.json"
    json_path.write_text(json.dumps({"data": []}), encoding="utf-8")

    convert_calls = []

    class _FakeCommitOperationAdd:
        def __init__(self, *, path_in_repo, path_or_fileobj):
            self.path_in_repo = path_in_repo
            self.path_or_fileobj = path_or_fileobj

    class _FakeApi:
        def __init__(self, token=None):
            pass

        def list_repo_files(self, *args, **kwargs):
            return []

        def create_commit(self, **kwargs):
            return type("_CommitInfo", (), {"oid": "parquetsha"})()

    def _fake_convert_json_to_parquet(**kwargs):
        convert_calls.append(kwargs)
        output_dir = kwargs["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "dataset.parquet").write_bytes(b"parquet")
        return {"num_samples": 0, "input_files_added": 0}

    monkeypatch.setattr(
        "opensportslib.tools.hf_transfer._import_hf_hub",
        lambda: (_FakeApi, object(), object()),
    )
    monkeypatch.setattr(
        "opensportslib.tools.hf_transfer._import_hf_commit_operation_add",
        lambda: _FakeCommitOperationAdd,
    )
    monkeypatch.setattr(
        "opensportslib.tools.hf_transfer.convert_json_to_parquet",
        _fake_convert_json_to_parquet,
    )

    result = upload_dataset_as_parquet_to_hf(
        repo_id="OpenSportsLab/test-repo",
        json_path=str(json_path),
        shard_size=123_000_000,
    )

    assert len(convert_calls) == 1
    assert convert_calls[0]["shard_mode"] == "size"
    assert convert_calls[0]["shard_size"] == 123_000_000
    assert result["shard_size"] == 123_000_000


def test_upload_dataset_as_parquet_to_hf_forwards_sample_mode(monkeypatch, tmp_path):
    json_path = tmp_path / "annotations.json"
    json_path.write_text(json.dumps({"data": []}), encoding="utf-8")

    convert_calls = []

    class _FakeCommitOperationAdd:
        def __init__(self, *, path_in_repo, path_or_fileobj):
            self.path_in_repo = path_in_repo
            self.path_or_fileobj = path_or_fileobj

    class _FakeApi:
        def __init__(self, token=None):
            pass

        def list_repo_files(self, *args, **kwargs):
            return []

        def create_commit(self, **kwargs):
            return type("_CommitInfo", (), {"oid": "parquetsha"})()

    def _fake_convert_json_to_parquet(**kwargs):
        convert_calls.append(kwargs)
        output_dir = kwargs["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "dataset.parquet").write_bytes(b"parquet")
        return {"num_samples": 0, "input_files_added": 0, "num_shards": 1}

    monkeypatch.setattr(
        "opensportslib.tools.hf_transfer._import_hf_hub",
        lambda: (_FakeApi, object(), object()),
    )
    monkeypatch.setattr(
        "opensportslib.tools.hf_transfer._import_hf_commit_operation_add",
        lambda: _FakeCommitOperationAdd,
    )
    monkeypatch.setattr(
        "opensportslib.tools.hf_transfer.convert_json_to_parquet",
        _fake_convert_json_to_parquet,
    )

    result = upload_dataset_as_parquet_to_hf(
        repo_id="OpenSportsLab/test-repo",
        json_path=str(json_path),
        shard_mode="samples",
        samples_per_shard=7,
    )

    assert len(convert_calls) == 1
    assert convert_calls[0]["shard_mode"] == "samples"
    assert convert_calls[0]["samples_per_shard"] == 7
    assert result["shard_mode"] == "samples"
    assert result["samples_per_shard"] == 7


def test_download_dataset_split_from_hf_json_can_be_cancelled_before_network(monkeypatch, tmp_path):
    called = {"hf_hub_download": 0}

    class _FakeApi:
        def __init__(self, token=None):
            pass

    def _fake_hf_hub_download(**kwargs):
        called["hf_hub_download"] += 1
        raise AssertionError("hf_hub_download should not be called when cancelled early")

    monkeypatch.setattr(
        "opensportslib.tools.hf_transfer._import_hf_hub",
        lambda: (_FakeApi, _fake_hf_hub_download, object()),
    )

    with pytest.raises(HfTransferCancelled):
        download_dataset_split_from_hf(
            "OpenSportsLab/repo",
            "main",
            "test",
            str(tmp_path),
            download_format="json",
            is_cancelled=lambda: True,
        )

    assert called["hf_hub_download"] == 0


def test_download_dataset_split_from_hf_json_downloads_split_json_and_all_inputs(monkeypatch, tmp_path):
    payload = {
        "data": [
            {
                "id": "sample_1",
                "inputs": [
                    {"path": "test/clip_0.mp4", "type": "video"},
                    {"path": "test/captions.json", "type": "captions"},
                ],
            }
        ]
    }
    downloaded = []
    planned = []
    completed = []
    json_ready = []

    class _FakeApi:
        def __init__(self, token=None):
            pass

    def _fake_hf_hub_download(**kwargs):
        downloaded.append(kwargs["filename"])
        local_dir = Path(kwargs["local_dir"])
        if kwargs["filename"] == "test.json":
            json_path = local_dir / "test.json"
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(json.dumps(payload), encoding="utf-8")
            return str(json_path)
        local_path = local_dir / kwargs["filename"]
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(b"data")
        return str(local_path)

    monkeypatch.setattr(
        "opensportslib.tools.hf_transfer._import_hf_hub",
        lambda: (_FakeApi, _fake_hf_hub_download, object()),
    )

    result = download_dataset_split_from_hf(
        "OpenSportsLab/repo",
        "dev",
        "test",
        str(tmp_path),
        download_format="json",
        file_plan_cb=planned.append,
        file_completed_cb=lambda filename, path: completed.append(
            (filename, path)
        ),
        json_ready_cb=lambda split, path: json_ready.append((split, path)),
    )

    assert downloaded == ["test.json", "test/captions.json", "test/clip_0.mp4"]
    expected_output_dir = tmp_path / "dev" / "test"
    written_payload = json.loads((expected_output_dir / "test.json").read_text(encoding="utf-8"))
    assert written_payload[HF_REPO_ID_KEY] == "OpenSportsLab/repo"
    assert written_payload[HF_BRANCH_KEY] == "dev"
    assert written_payload[HF_SPLIT_KEY] == "test"
    assert result["split"] == "test"
    assert result["output_dir"] == str(expected_output_dir)
    assert result["json_path"] == str(expected_output_dir / "test.json")
    assert result["downloaded_file_count"] == 2
    assert planned == [
        ["test.json"],
        ["test/captions.json", "test/clip_0.mp4"],
    ]
    assert [filename for filename, _path in completed] == downloaded
    assert json_ready == [("test", str(expected_output_dir / "test.json"))]


def test_download_dataset_split_from_hf_parquet_downloads_split_folder(monkeypatch, tmp_path):
    calls = {}

    def _fake_snapshot_download(**kwargs):
        calls["snapshot"] = kwargs
        dataset_dir = Path(kwargs["local_dir"]) / "test"
        dataset_dir.mkdir(parents=True)
        return str(kwargs["local_dir"])

    def _fake_convert_parquet_to_json(**kwargs):
        calls["convert"] = kwargs
        output_json_path = kwargs["output_json_path"]
        output_json_path.write_text(json.dumps({"data": []}), encoding="utf-8")
        return {"num_samples": 3, "extracted_media_files": 2}

    monkeypatch.setattr(
        "opensportslib.tools.hf_transfer._import_hf_hub",
        lambda: (object(), object(), _fake_snapshot_download),
    )
    monkeypatch.setattr(
        "opensportslib.tools.hf_transfer.convert_parquet_to_json",
        _fake_convert_parquet_to_json,
    )

    result = download_dataset_split_from_hf(
        "OpenSportsLab/repo",
        "dev",
        "test",
        str(tmp_path),
        download_format="parquet",
    )

    assert calls["snapshot"]["repo_id"] == "OpenSportsLab/repo"
    assert calls["snapshot"]["revision"] == "dev"
    assert calls["snapshot"]["allow_patterns"] == ["test/*"]
    assert calls["convert"]["dataset_dir"].as_posix().endswith("/test")
    assert result["split"] == "test"
    assert result["folder_path"] == "test"
    assert result["output_dir"] == str(tmp_path / "dev" / "test")
    assert result["json_path"] == str(tmp_path / "dev" / "test" / "test.json")
    assert result["num_samples"] == 3
    assert result["download_skipped"] is False


def test_parquet_byte_download_reports_file_count_progress(monkeypatch, tmp_path):
    progress_messages = []
    downloaded = []
    planned = []
    completed = []
    json_ready = []

    class _FakeApi:
        def __init__(self, token=None):
            pass

        def repo_info(self, **kwargs):
            return type("_Info", (), {"sha": "pinned"})()

        def list_repo_files(self, *args, **kwargs):
            return [
                "test/metadata.parquet",
                "test/shards/shard-000000.tar",
            ]

    def _fake_download_file(_hf_hub_download, **kwargs):
        downloaded.append(kwargs["filename"])
        return str(Path(kwargs["local_dir"]) / kwargs["filename"])

    def _fake_conversion(**kwargs):
        kwargs["output_json_path"].write_text(
            json.dumps({"data": []}), encoding="utf-8"
        )
        return {"num_samples": 0, "extracted_media_files": 0}

    monkeypatch.setattr(
        "opensportslib.tools.hf_transfer._import_hf_hub",
        lambda: (_FakeApi, object(), object()),
    )
    monkeypatch.setattr(
        "opensportslib.tools.hf_transfer._download_hf_file",
        _fake_download_file,
    )
    monkeypatch.setattr(
        "opensportslib.tools.hf_transfer.convert_parquet_to_json",
        _fake_conversion,
    )

    download_dataset_split_from_hf(
        "OpenSportsLab/repo",
        "main",
        "test",
        str(tmp_path),
        download_format="parquet",
        progress_cb=progress_messages.append,
        byte_progress_cb=lambda *_args: None,
        file_plan_cb=planned.append,
        file_completed_cb=lambda filename, path: completed.append(
            (filename, path)
        ),
        json_ready_cb=lambda split, path: json_ready.append((split, path)),
    )

    assert downloaded == [
        "test/metadata.parquet",
        "test/shards/shard-000000.tar",
    ]
    assert "[1/2] Downloading test/metadata.parquet" in progress_messages
    assert "[2/2] Downloading test/shards/shard-000000.tar" in progress_messages
    assert planned == [[
        "test/metadata.parquet",
        "test/shards/shard-000000.tar",
    ]]
    assert [filename for filename, _path in completed] == downloaded
    assert json_ready == [("test", str(tmp_path / "main" / "test" / "test.json"))]


def test_download_dataset_split_from_hf_parquet_completes_existing_json(
    monkeypatch, tmp_path
):
    output_json_path = tmp_path / "dev" / "test" / "test.json"
    output_json_path.parent.mkdir(parents=True)
    output_json_path.write_text(json.dumps({"data": []}), encoding="utf-8")
    progress_messages = []

    calls = {}

    class _FakeApi:
        def __init__(self, token=None):
            pass

        def repo_info(self, **kwargs):
            return type("_Info", (), {"sha": "abc123"})()

    def _fake_snapshot_download(**kwargs):
        calls["snapshot"] = kwargs
        (Path(kwargs["local_dir"]) / "test" / "shards").mkdir(parents=True)

    def _fake_conversion(**kwargs):
        calls["conversion"] = kwargs
        kwargs["output_json_path"].write_text(json.dumps({"data": []}), encoding="utf-8")
        return {"num_samples": 0, "extracted_media_files": 0}

    monkeypatch.setattr(
        "opensportslib.tools.hf_transfer._import_hf_hub",
        lambda: (_FakeApi, object(), _fake_snapshot_download),
    )
    monkeypatch.setattr("opensportslib.tools.hf_transfer.convert_parquet_to_json", _fake_conversion)

    result = download_dataset_split_from_hf(
        "OpenSportsLab/repo",
        "dev",
        "test",
        str(tmp_path),
        download_format="parquet",
        progress_cb=progress_messages.append,
    )

    assert result["json_path"] == str(output_json_path)
    assert result["download_skipped"] is False
    assert calls["snapshot"]["revision"] == "abc123"
    assert calls["conversion"]
    assert progress_messages


def test_json_annotations_only_downloads_json_and_persists_pinned_source(monkeypatch, tmp_path):
    remote_root = tmp_path / "remote"
    remote_root.mkdir()
    (remote_root / "test.json").write_text(
        json.dumps({"data": [{"id": "one", "inputs": [{"path": "clips/one.mp4"}]}]}),
        encoding="utf-8",
    )
    downloaded = []

    class _FakeApi:
        def __init__(self, token=None):
            pass

        def repo_info(self, **kwargs):
            return type("_Info", (), {"sha": "pinned-json"})()

    monkeypatch.setattr(
        "opensportslib.tools.hf_transfer._import_hf_hub",
        lambda: (_FakeApi, _copying_hf_download(remote_root, downloaded), object()),
    )
    result = download_dataset_split_from_hf(
        "OpenSportsLab/repo",
        "main",
        "test",
        str(tmp_path / "output"),
        download_format="json",
        annotations_only=True,
    )

    assert downloaded == ["test.json"]
    payload = json.loads(Path(result["json_path"]).read_text(encoding="utf-8"))
    assert payload[HF_FORMAT_KEY] == "json"
    assert payload[HF_COMMIT_KEY] == "pinned-json"
    assert result["annotations_only"] is True
    assert result["downloaded_file_count"] == 0


def test_parquet_annotations_only_reconstructs_without_shards(monkeypatch, tmp_path):
    remote_root = tmp_path / "remote"
    metadata_path = remote_root / "test" / "metadata.parquet"
    metadata_path.parent.mkdir(parents=True)
    sample = {"id": "one", "inputs": [{"path": "clips/one.mp4", "type": "video"}]}
    pd.DataFrame(
        [{
            "sample_id": "one",
            "sample_index": 0,
            "shard_name": "shard-000000.tar",
            "header": json.dumps({"version": "2.0", "data": []}),
            "sample_payload": json.dumps(sample),
        }]
    ).to_parquet(metadata_path, index=False)
    downloaded = []

    class _FakeApi:
        def __init__(self, token=None):
            pass

        def repo_info(self, **kwargs):
            return type("_Info", (), {"sha": "pinned-parquet"})()

    monkeypatch.setattr(
        "opensportslib.tools.hf_transfer._import_hf_hub",
        lambda: (_FakeApi, _copying_hf_download(remote_root, downloaded), object()),
    )
    result = download_dataset_split_from_hf(
        "OpenSportsLab/repo",
        "main",
        "test",
        str(tmp_path / "output"),
        download_format="parquet",
        annotations_only=True,
    )

    assert downloaded == ["test/metadata.parquet"]
    payload = json.loads(Path(result["json_path"]).read_text(encoding="utf-8"))
    assert payload["data"] == [sample]
    assert payload[HF_FORMAT_KEY] == "parquet"
    assert payload[HF_COMMIT_KEY] == "pinned-parquet"
    assert not list((tmp_path / "output").rglob("*.parquet"))


def test_parquet_annotations_only_rejects_legacy_metadata_without_fetching_shards(
    monkeypatch, tmp_path
):
    remote_root = tmp_path / "remote"
    split_root = remote_root / "test"
    split_root.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "sample_id": "legacy",
                "sample_index": 0,
                "shard_name": "shard-000000.tar",
                "header": json.dumps({"data": []}),
            }
        ]
    ).to_parquet(split_root / "metadata.parquet", index=False)
    downloaded = []

    class _FakeApi:
        def __init__(self, token=None):
            pass

        def repo_info(self, **kwargs):
            return type("_Info", (), {"sha": "pinned"})()

    monkeypatch.setattr(
        "opensportslib.tools.hf_transfer._import_hf_hub",
        lambda: (_FakeApi, _copying_hf_download(remote_root, downloaded), object()),
    )

    with pytest.raises(ValueError, match="missing required columns: sample_payload"):
        download_dataset_split_from_hf(
            "OpenSportsLab/repo",
            "main",
            "test",
            str(tmp_path / "output"),
            download_format="parquet",
            annotations_only=True,
        )

    assert downloaded == ["test/metadata.parquet"]
    assert not list((tmp_path / "output").rglob("*.tar"))


def test_parquet_selective_download_extracts_all_missing_shard_assets(monkeypatch, tmp_path):
    remote_root = tmp_path / "remote"
    split_root = remote_root / "test"
    (split_root / "shards").mkdir(parents=True)
    samples = [
        {"id": "requested", "inputs": [{"path": "clips/requested.mp4"}]},
        {"id": "existing", "inputs": [{"path": "clips/existing.mp4"}]},
        {"id": "missing", "inputs": [{"path": "clips/missing.mp4"}]},
    ]
    pd.DataFrame(
        [{
            "sample_id": sample["id"],
            "sample_index": index,
            "shard_name": "shard-000000.tar",
            "header": json.dumps({"data": []}),
            "sample_payload": json.dumps(sample),
        } for index, sample in enumerate(samples)]
    ).to_parquet(split_root / "metadata.parquet", index=False)
    pd.DataFrame(
        [{
            "sample_id": sample["id"],
            "shard_name": "shard-000000.tar",
            "input_index": 0,
            "file_role": "primary",
            "relative_path": sample["inputs"][0]["path"],
            "status": "ok",
            "wds_member": f"{index:09d}.0.mp4",
        } for index, sample in enumerate(samples)]
    ).to_parquet(split_root / "shard_manifest.parquet", index=False)
    with tarfile.open(split_root / "shards" / "shard-000000.tar", "w") as archive:
        for index, sample in enumerate(samples):
            source = tmp_path / f"source-{index}.mp4"
            source.write_bytes(sample["id"].encode())
            archive.add(source, arcname=f"{index:09d}.0.mp4")

    local_root = tmp_path / "local"
    local_root.mkdir()
    dataset_path = local_root / "test.json"
    dataset_path.write_text(
        json.dumps({
            "hf_repo_id": "OpenSportsLab/repo",
            "hf_branch": "main",
            "hf_split": "test",
            "hf_format": "parquet",
            "hf_commit": "pinned",
            "data": samples,
        }),
        encoding="utf-8",
    )
    existing_path = local_root / "clips" / "existing.mp4"
    existing_path.parent.mkdir(parents=True)
    existing_path.write_bytes(b"keep-me")
    downloaded = []
    monkeypatch.setattr(
        "opensportslib.tools.hf_transfer._import_hf_hub",
        lambda: (object(), _copying_hf_download(remote_root, downloaded), object()),
    )

    result = download_dataset_sample_inputs_from_hf(str(dataset_path), "requested")

    assert downloaded == [
        "test/metadata.parquet",
        "test/shard_manifest.parquet",
        "test/shards/shard-000000.tar",
    ]
    assert (local_root / "clips" / "requested.mp4").read_bytes() == b"requested"
    assert existing_path.read_bytes() == b"keep-me"
    assert (local_root / "clips" / "missing.mp4").read_bytes() == b"missing"
    assert result["requested_downloaded_count"] == 1
    assert result["opportunistic_downloaded_count"] == 1
    assert result["collateral_skipped_count"] == 1
    assert not list(local_root.rglob("*.tar"))


def test_json_selective_input_download_includes_ball_and_overwrites_only_requested(
    monkeypatch, tmp_path
):
    remote_root = tmp_path / "remote"
    (remote_root / "tracking").mkdir(parents=True)
    (remote_root / "tracking" / "players.h5").write_bytes(b"new-players")
    (remote_root / "tracking" / "ball.h5").write_bytes(b"new-ball")
    local_root = tmp_path / "local"
    (local_root / "tracking").mkdir(parents=True)
    (local_root / "tracking" / "players.h5").write_bytes(b"old-players")
    dataset_path = local_root / "test.json"
    dataset_path.write_text(
        json.dumps(
            {
                "hf_repo_id": "OpenSportsLab/repo",
                "hf_branch": "main",
                "hf_split": "test",
                "hf_format": "json",
                "hf_commit": "pinned",
                "data": [
                    {
                        "id": "sample-1",
                        "inputs": [
                            {
                                "path": "tracking/players.h5",
                                "ball_path": "tracking/ball.h5",
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    downloaded = []
    monkeypatch.setattr(
        "opensportslib.tools.hf_transfer._import_hf_hub",
        lambda: (object(), _copying_hf_download(remote_root, downloaded), object()),
    )

    result = download_dataset_sample_inputs_from_hf(
        str(dataset_path),
        "sample-1",
        input_path="tracking/players.h5",
        overwrite=True,
    )

    assert downloaded == ["tracking/players.h5", "tracking/ball.h5"]
    assert (local_root / "tracking" / "players.h5").read_bytes() == b"new-players"
    assert (local_root / "tracking" / "ball.h5").read_bytes() == b"new-ball"
    assert result["requested_overwritten_paths"] == ["tracking/players.h5"]
    assert result["requested_downloaded_paths"] == ["tracking/ball.h5"]


def test_parquet_selective_overwrite_preserves_existing_collateral(monkeypatch, tmp_path):
    remote_root = tmp_path / "remote"
    split_root = remote_root / "test"
    (split_root / "shards").mkdir(parents=True)
    samples = [
        {"id": "requested", "inputs": [{"path": "clips/requested.mp4"}]},
        {"id": "collateral", "inputs": [{"path": "clips/collateral.mp4"}]},
    ]
    pd.DataFrame(
        [
            {
                "sample_id": sample["id"],
                "sample_index": index,
                "shard_name": "shard-000000.tar",
                "header": json.dumps({"data": []}),
                "sample_payload": json.dumps(sample),
            }
            for index, sample in enumerate(samples)
        ]
    ).to_parquet(split_root / "metadata.parquet", index=False)
    pd.DataFrame(
        [
            {
                "sample_id": sample["id"],
                "shard_name": "shard-000000.tar",
                "input_index": 0,
                "file_role": "primary",
                "relative_path": sample["inputs"][0]["path"],
                "status": "ok",
                "wds_member": f"{index:09d}.0.mp4",
            }
            for index, sample in enumerate(samples)
        ]
    ).to_parquet(split_root / "shard_manifest.parquet", index=False)
    with tarfile.open(split_root / "shards" / "shard-000000.tar", "w") as archive:
        for index, sample in enumerate(samples):
            source = tmp_path / f"payload-{index}.mp4"
            source.write_bytes(f"remote-{sample['id']}".encode())
            archive.add(source, arcname=f"{index:09d}.0.mp4")

    local_root = tmp_path / "local"
    (local_root / "clips").mkdir(parents=True)
    (local_root / "clips" / "requested.mp4").write_bytes(b"old-requested")
    (local_root / "clips" / "collateral.mp4").write_bytes(b"keep-collateral")
    dataset_path = local_root / "test.json"
    dataset_path.write_text(
        json.dumps(
            {
                "hf_repo_id": "OpenSportsLab/repo",
                "hf_branch": "main",
                "hf_split": "test",
                "hf_format": "parquet",
                "hf_commit": "pinned",
                "data": samples,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "opensportslib.tools.hf_transfer._import_hf_hub",
        lambda: (object(), _copying_hf_download(remote_root, []), object()),
    )

    result = download_dataset_sample_inputs_from_hf(
        str(dataset_path), "requested", overwrite=True
    )

    assert (local_root / "clips" / "requested.mp4").read_bytes() == b"remote-requested"
    assert (local_root / "clips" / "collateral.mp4").read_bytes() == b"keep-collateral"
    assert result["requested_overwritten_count"] == 1
    assert result["collateral_skipped_count"] == 1


def test_selective_download_rejects_unsafe_requested_destination_before_network(
    monkeypatch, tmp_path
):
    dataset_path = tmp_path / "test.json"
    dataset_path.write_text(
        json.dumps(
            {
                "hf_repo_id": "OpenSportsLab/repo",
                "hf_branch": "main",
                "hf_split": "test",
                "hf_format": "json",
                "hf_commit": "pinned",
                "data": [{"id": "sample-1", "inputs": [{"path": "../escape.mp4"}]}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "opensportslib.tools.hf_transfer._import_hf_hub",
        lambda: pytest.fail("Network must not be used for an unsafe path"),
    )

    with pytest.raises(ValueError, match="Unsafe dataset asset path"):
        download_dataset_sample_inputs_from_hf(str(dataset_path), "sample-1")


def test_parquet_selective_cancellation_removes_downloaded_shard(monkeypatch, tmp_path):
    remote_root = tmp_path / "remote"
    split_root = remote_root / "test"
    (split_root / "shards").mkdir(parents=True)
    sample = {"id": "sample-1", "inputs": [{"path": "clips/one.mp4"}]}
    pd.DataFrame(
        [
            {
                "sample_id": "sample-1",
                "sample_index": 0,
                "shard_name": "shard-000000.tar",
                "header": json.dumps({"data": []}),
                "sample_payload": json.dumps(sample),
            }
        ]
    ).to_parquet(split_root / "metadata.parquet", index=False)
    pd.DataFrame(
        [
            {
                "sample_id": "sample-1",
                "shard_name": "shard-000000.tar",
                "input_index": 0,
                "file_role": "primary",
                "relative_path": "clips/one.mp4",
                "status": "ok",
                "wds_member": "000000000.0.mp4",
            }
        ]
    ).to_parquet(split_root / "shard_manifest.parquet", index=False)
    with tarfile.open(split_root / "shards" / "shard-000000.tar", "w") as archive:
        source = tmp_path / "source.mp4"
        source.write_bytes(b"video")
        archive.add(source, arcname="000000000.0.mp4")

    local_root = tmp_path / "local"
    local_root.mkdir()
    dataset_path = local_root / "test.json"
    dataset_path.write_text(
        json.dumps(
            {
                "hf_repo_id": "OpenSportsLab/repo",
                "hf_branch": "main",
                "hf_split": "test",
                "hf_format": "parquet",
                "hf_commit": "pinned",
                "data": [sample],
            }
        ),
        encoding="utf-8",
    )
    downloaded = []
    temporary_dir = tmp_path / "selective-temp"

    def _make_temp_dir(prefix):
        assert prefix == "hf_selective_parquet_"
        temporary_dir.mkdir()
        return str(temporary_dir)

    monkeypatch.setattr(
        "opensportslib.tools.hf_transfer._import_hf_hub",
        lambda: (object(), _copying_hf_download(remote_root, downloaded), object()),
    )
    monkeypatch.setattr(
        "opensportslib.tools.hf_transfer.tempfile.mkdtemp", _make_temp_dir
    )

    with pytest.raises(HfTransferCancelled):
        download_dataset_sample_inputs_from_hf(
            str(dataset_path),
            "sample-1",
            is_cancelled=lambda: len(downloaded) >= 3,
        )

    assert downloaded[-1] == "test/shards/shard-000000.tar"
    assert not temporary_dir.exists()
    assert not (local_root / "clips" / "one.mp4").exists()


def test_download_dataset_split_from_hf_json_writes_hf_metadata_on_non_dry_run(monkeypatch, tmp_path):
    payload = {
        "data": [
            {
                "id": "sample_1",
                "inputs": [{"path": "train/clip_0.mp4", "type": "video"}],
            }
        ]
    }
    class _FakeApi:
        def __init__(self, token=None):
            pass

    def _fake_hf_hub_download(**kwargs):
        filename = kwargs.get("filename")
        local_dir = Path(kwargs["local_dir"])
        if filename == "test.json":
            json_path = local_dir / "test.json"
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(json.dumps(payload), encoding="utf-8")
            return str(json_path)
        local_path = local_dir / filename
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(b"video")
        return str(local_path)

    monkeypatch.setattr(
        "opensportslib.tools.hf_transfer._import_hf_hub",
        lambda: (_FakeApi, _fake_hf_hub_download, object()),
    )

    result = download_dataset_split_from_hf(
        "OpenSportsLab/repo",
        "main",
        "test",
        str(tmp_path),
        download_format="json",
    )

    written_payload = json.loads((tmp_path / "main" / "test" / "test.json").read_text(encoding="utf-8"))
    assert written_payload[HF_REPO_ID_KEY] == "OpenSportsLab/repo"
    assert written_payload[HF_BRANCH_KEY] == "main"
    assert written_payload[HF_SPLIT_KEY] == "test"
    assert result["downloaded_file_count"] == 1
    assert result["hf_source_metadata"]["repo_id"] == "OpenSportsLab/repo"


def test_download_dataset_split_from_hf_json_dry_run_does_not_write_hf_metadata(monkeypatch, tmp_path):
    payload = {
        "data": [
            {
                "id": "sample_1",
                "inputs": [{"path": "train/clip_0.mp4", "type": "video"}],
            }
        ]
    }
    class _FakeApi:
        def __init__(self, token=None):
            pass

        def repo_info(self, **kwargs):
            sibling = type("_Sibling", (), {"rfilename": "train/clip_0.mp4", "size": 12})()
            return type("_Info", (), {"siblings": [sibling]})()

    def _fake_hf_hub_download(**kwargs):
        filename = kwargs.get("filename")
        if filename == "test.json":
            json_path = Path(kwargs["local_dir"]) / "test.json"
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(json.dumps(payload), encoding="utf-8")
            return str(json_path)
        raise AssertionError("Unexpected file download in dry-run mode")

    monkeypatch.setattr(
        "opensportslib.tools.hf_transfer._import_hf_hub",
        lambda: (_FakeApi, _fake_hf_hub_download, object()),
    )

    result = download_dataset_split_from_hf(
        "OpenSportsLab/repo",
        "main",
        "test",
        str(tmp_path),
        download_format="json",
        dry_run=True,
    )

    expected_output_dir = tmp_path / "main" / "test"
    written_payload = json.loads((expected_output_dir / "test.json").read_text(encoding="utf-8"))
    assert HF_REPO_ID_KEY not in written_payload
    assert HF_BRANCH_KEY not in written_payload
    assert "hf_source_metadata" not in result
    assert result["output_dir"] == str(expected_output_dir)


def test_read_hf_source_metadata_from_dataset_reads_split_keys():
    metadata = read_hf_source_metadata_from_dataset(
        {
            HF_REPO_ID_KEY: "OpenSportsLab/repo",
            HF_BRANCH_KEY: "main",
            HF_SPLIT_KEY: "test",
        }
    )

    assert metadata["repo_id"] == "OpenSportsLab/repo"
    assert metadata["branch"] == "main"
    assert metadata["split"] == "test"


def test_write_hf_source_metadata_to_dataset_json_persists_top_level_keys(tmp_path):
    json_path = tmp_path / "dataset.json"
    json_path.write_text(json.dumps({"data": []}), encoding="utf-8")

    write_hf_source_metadata_to_dataset_json(
        str(json_path),
        repo_id="OpenSportsLab/repo",
        branch="main",
        split="test",
    )
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload[HF_REPO_ID_KEY] == "OpenSportsLab/repo"
    assert payload[HF_BRANCH_KEY] == "main"
    assert payload[HF_SPLIT_KEY] == "test"


def test_upload_dataset_inputs_from_json_to_hf_can_be_cancelled_before_upload(monkeypatch, tmp_path):
    clip_path = tmp_path / "train" / "clip_0.mp4"
    clip_path.parent.mkdir(parents=True)
    clip_path.write_bytes(b"video")
    json_path = tmp_path / "annotations.json"
    json_path.write_text(
        json.dumps(
            {
                "data": [
                    {
                        "id": "sample_1",
                        "inputs": [{"path": "train/clip_0.mp4", "type": "video"}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    calls = {"create_commit": 0}

    class _FakeCommitOperationAdd:
        def __init__(self, *, path_in_repo, path_or_fileobj):
            self.path_in_repo = path_in_repo
            self.path_or_fileobj = path_or_fileobj

    class _FakeApi:
        def __init__(self, token=None):
            pass

        def create_commit(self, **kwargs):
            calls["create_commit"] += 1
            return type("_CommitInfo", (), {"oid": "abc123"})()

    monkeypatch.setattr(
        "opensportslib.tools.hf_transfer._import_hf_hub",
        lambda: (_FakeApi, object(), object()),
    )
    monkeypatch.setattr(
        "opensportslib.tools.hf_transfer._import_hf_commit_operation_add",
        lambda: _FakeCommitOperationAdd,
    )

    with pytest.raises(HfTransferCancelled):
        upload_dataset_inputs_from_json_to_hf(
            repo_id="OpenSportsLab/test-repo",
            json_path=str(json_path),
            revision="main",
            is_cancelled=lambda: True,
        )

    assert calls["create_commit"] == 0
