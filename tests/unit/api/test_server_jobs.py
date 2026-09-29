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
