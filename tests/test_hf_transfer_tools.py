import json

import pytest

from opensportslib.tools.hf_transfer import (
    HF_BRANCH_KEY,
    HF_REPO_ID_KEY,
    HF_SOURCE_URL_KEY,
    HfTransferCancelled,
    create_dataset_branch_on_hf,
    create_dataset_repo_on_hf,
    dataset_repo_exists_on_hf,
    download_dataset_from_hf,
    extract_local_input_upload_entries_from_json,
    extract_repo_paths_from_json,
    fix_hf_url,
    is_hf_download_url_not_found_error,
    is_hf_repo_not_found_error,
    is_hf_revision_not_found_error,
    parse_hf_url,
    parse_types_arg,
    read_hf_source_metadata_from_dataset,
    upload_dataset_as_parquet_to_hf,
    upload_dataset_inputs_from_json_to_hf,
    write_hf_source_metadata_to_dataset_json,
)


def test_fix_hf_url_converts_blob_to_resolve():
    blob_url = "https://huggingface.co/datasets/OpenSportsLab/repo/blob/main/annotations.json"
    assert fix_hf_url(blob_url) == "https://huggingface.co/datasets/OpenSportsLab/repo/resolve/main/annotations.json"


@pytest.mark.parametrize(
    "url",
    [
        "https://huggingface.co/datasets/OpenSportsLab/repo/blob/main/annotations.json",
        "https://huggingface.co/datasets/OpenSportsLab/repo/resolve/main/annotations.json",
    ],
)
def test_parse_hf_url_supports_blob_and_resolve(url):
    repo_id, revision, path_in_repo = parse_hf_url(url)
    assert repo_id == "OpenSportsLab/repo"
    assert revision == "main"
    assert path_in_repo == "annotations.json"


def test_parse_hf_url_rejects_invalid_url_shape():
    with pytest.raises(ValueError):
        parse_hf_url("https://huggingface.co/datasets/OpenSportsLab/repo/main/annotations.json")


def test_parse_types_arg_handles_all_and_comma_list():
    assert parse_types_arg("all") == "all"
    assert parse_types_arg("*") == "all"
    assert parse_types_arg("video,captions,features") == {"video", "captions", "features"}


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
    paths = extract_repo_paths_from_json(payload, "all")
    assert set(paths) == {"legacy/a.mp4", "v2/b.mp4", "v2/c.json"}


def test_extract_repo_paths_from_json_filters_by_requested_types():
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

    video_only = extract_repo_paths_from_json(payload, {"video"})
    assert video_only == ["test/a.mp4"]

    with pytest.raises(ValueError):
        extract_repo_paths_from_json(payload, {"features"})


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
    assert operations[0].path_in_repo == "annotations.json"
    assert operations[0].path_or_fileobj == str(json_path)
    assert operations[1].path_in_repo == "train/clip_0.mp4"
    assert result["input_file_count"] == 1
    assert result["unique_input_file_count"] == 1
    assert result["uploaded_file_count"] == 2
    assert result["json_path_in_repo"] == "annotations.json"
    assert result["revision"] == "dev-branch"
    assert result["commit_ref"] == "abc123"


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
    assert len(operations) == 3
    assert [op.path_in_repo for op in operations] == [
        "annotations/dataset.parquet",
        "annotations/samples/shard-00000.tar",
        "annotations/samples/shard-00001.tar",
    ]
    assert result["upload_kind"] == "parquet"
    assert result["uploaded_file_count"] == 3
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


def test_download_dataset_from_hf_can_be_cancelled_before_network(monkeypatch, tmp_path):
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
        download_dataset_from_hf(
            "https://huggingface.co/datasets/OpenSportsLab/repo/blob/main/annotations.json",
            str(tmp_path),
            is_cancelled=lambda: True,
        )

    assert called["hf_hub_download"] == 0


def test_download_dataset_from_hf_writes_hf_source_metadata_on_non_dry_run(monkeypatch, tmp_path):
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

    class _FakeApi:
        def __init__(self, token=None):
            pass

    def _fake_hf_hub_download(**kwargs):
        filename = kwargs.get("filename")
        if filename == "annotations.json":
            return str(json_path)
        local_path = tmp_path / filename
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(b"video")
        return str(local_path)

    monkeypatch.setattr(
        "opensportslib.tools.hf_transfer._import_hf_hub",
        lambda: (_FakeApi, _fake_hf_hub_download, object()),
    )

    source_url = "https://huggingface.co/datasets/OpenSportsLab/repo/blob/main/annotations.json"
    result = download_dataset_from_hf(source_url, str(tmp_path))

    written_payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert written_payload[HF_SOURCE_URL_KEY] == source_url
    assert written_payload[HF_REPO_ID_KEY] == "OpenSportsLab/repo"
    assert written_payload[HF_BRANCH_KEY] == "main"
    assert result["downloaded_file_count"] == 1
    assert result["hf_source_metadata"]["repo_id"] == "OpenSportsLab/repo"


def test_download_dataset_from_hf_dry_run_does_not_write_hf_source_metadata(monkeypatch, tmp_path):
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

    class _FakeApi:
        def __init__(self, token=None):
            pass

        def repo_info(self, **kwargs):
            sibling = type("_Sibling", (), {"rfilename": "train/clip_0.mp4", "size": 12})()
            return type("_Info", (), {"siblings": [sibling]})()

    def _fake_hf_hub_download(**kwargs):
        filename = kwargs.get("filename")
        if filename == "annotations.json":
            return str(json_path)
        raise AssertionError("Unexpected file download in dry-run mode")

    monkeypatch.setattr(
        "opensportslib.tools.hf_transfer._import_hf_hub",
        lambda: (_FakeApi, _fake_hf_hub_download, object()),
    )

    source_url = "https://huggingface.co/datasets/OpenSportsLab/repo/blob/main/annotations.json"
    result = download_dataset_from_hf(source_url, str(tmp_path), dry_run=True)

    written_payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert HF_SOURCE_URL_KEY not in written_payload
    assert HF_REPO_ID_KEY not in written_payload
    assert HF_BRANCH_KEY not in written_payload
    assert "hf_source_metadata" not in result


def test_read_hf_source_metadata_from_dataset_normalizes_and_backfills_from_url():
    source_url = "https://huggingface.co/datasets/OpenSportsLab/repo/blob/main/annotations.json"
    metadata = read_hf_source_metadata_from_dataset(
        {
            HF_SOURCE_URL_KEY: source_url,
            HF_REPO_ID_KEY: "",
            HF_BRANCH_KEY: "",
        }
    )

    assert metadata["source_url"] == source_url
    assert metadata["repo_id"] == "OpenSportsLab/repo"
    assert metadata["branch"] == "main"


def test_write_hf_source_metadata_to_dataset_json_persists_top_level_keys(tmp_path):
    json_path = tmp_path / "dataset.json"
    json_path.write_text(json.dumps({"data": []}), encoding="utf-8")

    write_hf_source_metadata_to_dataset_json(
        str(json_path),
        source_url="https://huggingface.co/datasets/OpenSportsLab/repo/blob/main/annotations.json",
        repo_id="OpenSportsLab/repo",
        branch="main",
    )
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload[HF_SOURCE_URL_KEY] == "https://huggingface.co/datasets/OpenSportsLab/repo/blob/main/annotations.json"
    assert payload[HF_REPO_ID_KEY] == "OpenSportsLab/repo"
    assert payload[HF_BRANCH_KEY] == "main"


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
