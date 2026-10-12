import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


SERVER_ROOT = Path(__file__).resolve().parents[3] / "server"
sys.path.insert(0, str(SERVER_ROOT))

from app.schemas import JobStatus, PredictRequest  # noqa: E402
from storage.jobs import JobStore  # noqa: E402


def store(tmp_path):
    runtime = tmp_path / "runtime"
    settings = SimpleNamespace(
        runtime_dir=runtime,
        tmp_dir=runtime / "tmp",
        jobs_dir=runtime / "jobs",
        results_dir=runtime / "results",
    )
    settings.jobs_dir.mkdir(parents=True)
    settings.tmp_dir.mkdir(parents=True)
    settings.results_dir.mkdir(parents=True)
    return JobStore(settings)


def request():
    return PredictRequest(task_type="classification", video_path="/tmp/clip.mp4")


def test_metadata_is_valid_json_after_create(tmp_path):
    jobs = store(tmp_path)
    metadata = jobs.create_job(request(), "model")

    loaded = jobs.load_metadata(metadata.job_id)
    assert loaded is not None
    assert loaded.status is JobStatus.QUEUED
    assert jobs.metadata_path(metadata.job_id).read_text(encoding="utf-8").strip()


def test_metadata_replacement_never_leaves_partial_file(tmp_path):
    jobs = store(tmp_path)
    metadata = jobs.create_job(request(), "model")

    for status in (JobStatus.RUNNING, JobStatus.SUCCEEDED, JobStatus.FAILED):
        metadata.status = status
        jobs.save_metadata(metadata)
        loaded = jobs.load_metadata(metadata.job_id)
        assert loaded is not None
        assert loaded.status is status
        assert not list(jobs.settings.jobs_dir.glob(f".{metadata.job_id}.*.tmp"))


def test_invalid_metadata_is_treated_as_missing(tmp_path):
    jobs = store(tmp_path)
    path = jobs.metadata_path("broken")
    path.write_text("", encoding="utf-8")

    assert jobs.load_metadata("broken") is None
    with pytest.raises(FileNotFoundError, match="broken"):
        jobs.mark_running("broken")


def test_job_lifecycle_persists_request_result_and_clears_failure(tmp_path):
    import json

    jobs = store(tmp_path)
    metadata = jobs.create_job(request(), "model", session_id="session", input_source="uploaded.mp4")
    job_id = metadata.job_id
    assert jobs.load_result(job_id) is None
    assert jobs.mark_running(job_id).status is JobStatus.RUNNING
    assert jobs.mark_failed(job_id, "temporary failure").error == "temporary failure"
    assert jobs.load_metadata(job_id).status is JobStatus.FAILED
    payload = {"predictions": [{"label": "PASS", "confidence": .9}]}
    succeeded = jobs.mark_succeeded(job_id, payload)
    assert succeeded.status is JobStatus.SUCCEEDED and succeeded.error is None
    assert jobs.load_result(job_id) == payload
    assert jobs.load_metadata(job_id).session_id == "session"
    assert jobs.load_metadata(job_id).input_source == "uploaded.mp4"
    saved = json.loads((jobs.settings.tmp_dir / job_id / "request.json").read_text())
    assert saved["video_path"] == "/tmp/clip.mp4"


@pytest.mark.parametrize("source", ["video_path", "media_url", "test_set_path"])
def test_updating_request_updates_persisted_input_source(tmp_path, source):
    import json

    jobs = store(tmp_path)
    metadata = jobs.create_job(request(), "model")
    replacement = PredictRequest(task_type="classification", **{source: "new-input"})
    jobs.update_request_payload(metadata.job_id, replacement)
    assert jobs.load_metadata(metadata.job_id).input_source == "new-input"
    saved = json.loads((jobs.settings.tmp_dir / metadata.job_id / "request.json").read_text())
    assert saved[source] == "new-input"


def test_atomic_metadata_failure_preserves_last_valid_record(tmp_path, monkeypatch):
    import storage.jobs as module

    jobs = store(tmp_path)
    metadata = jobs.create_job(request(), "model")
    previous = jobs.metadata_path(metadata.job_id).read_bytes()
    def fail_replace(*args):
        raise OSError("disk replacement failed")
    monkeypatch.setattr(module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replacement failed"):
        jobs.mark_running(metadata.job_id)
    assert jobs.metadata_path(metadata.job_id).read_bytes() == previous
    assert not list(jobs.settings.jobs_dir.glob("*.tmp"))
    assert jobs.load_metadata(metadata.job_id).status is JobStatus.QUEUED


@pytest.mark.parametrize("operation", ["mark_running", "mark_failed", "mark_succeeded"])
def test_updates_to_missing_job_fail_without_creating_results(tmp_path, operation):
    jobs = store(tmp_path)
    args = {"mark_running": (), "mark_failed": ("failure",), "mark_succeeded": ({},)}[operation]
    with pytest.raises(FileNotFoundError, match="metadata missing"):
        getattr(jobs, operation)("missing", *args)
    assert jobs.load_result("missing") is None
