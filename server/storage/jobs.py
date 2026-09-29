from __future__ import annotations

import json
import logging
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.schemas import JobMetadata, JobStatus, PredictRequest, TaskType
from config.settings import Settings


logger = logging.getLogger("osl.storage.jobs")


class JobStore:
    def __init__(self, settings: Settings):
        self.settings = settings

    def create_job(
        self,
        request: PredictRequest,
        model_id: str,
        *,
        session_id: str | None = None,
        input_source: str | None = None,
    ) -> JobMetadata:
        job_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc)
        working_dir = self.settings.tmp_dir / job_id
        working_dir.mkdir(parents=True, exist_ok=True)

        metadata = JobMetadata(
            job_id=job_id,
            session_id=session_id,
            task_type=request.task_type,
            model_id=model_id,
            config_overrides=(request.task_options or {}).get("config_overrides"),
            status=JobStatus.QUEUED,
            created_at=created_at,
            updated_at=created_at,
            input_source=input_source if input_source is not None else (request.video_path or request.media_url or request.test_set_path or ""),
            runtime_dir=str(working_dir),
        )
        self.save_metadata(metadata)
        request_path = working_dir / "request.json"
        request_path.write_text(request.model_dump_json(indent=2), encoding="utf-8")
        return metadata

    def update_request_payload(self, job_id: str, request: PredictRequest) -> None:
        request_path = self.settings.tmp_dir / job_id / "request.json"
        request_path.write_text(request.model_dump_json(indent=2), encoding="utf-8")
        metadata = self._require_metadata(job_id)
        metadata.input_source = request.video_path or request.media_url or request.test_set_path or metadata.input_source
        self.save_metadata(metadata)

    def metadata_path(self, job_id: str) -> Path:
        return self.settings.jobs_dir / f"{job_id}.json"

    def result_path(self, job_id: str) -> Path:
        return self.settings.results_dir / f"{job_id}.json"

    def save_metadata(self, metadata: JobMetadata) -> None:
        metadata.updated_at = datetime.now(timezone.utc)
        target = self.metadata_path(metadata.job_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = metadata.model_dump_json(indent=2)
        # The worker can consume a job immediately after enqueueing. Write to a
        # sibling file and replace the target so readers never observe a
        # partially-written JSON document (including on Docker bind mounts).
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{metadata.job_id}.", suffix=".tmp", dir=target.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as temporary:
                temporary.write(payload)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_name, target)
        except Exception:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise

    def load_metadata(self, job_id: str) -> JobMetadata | None:
        path = self.metadata_path(job_id)
        if not path.is_file():
            return None
        try:
            return JobMetadata.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, ValidationError) as exc:
            logger.warning("Invalid job metadata | job_id=%s path=%s error=%s", job_id, path, exc)
            return None

    def mark_running(self, job_id: str) -> JobMetadata:
        metadata = self._require_metadata(job_id)
        metadata.status = JobStatus.RUNNING
        self.save_metadata(metadata)
        return metadata

    def mark_succeeded(self, job_id: str, result: dict[str, Any]) -> JobMetadata:
        metadata = self._require_metadata(job_id)
        result_path = self.result_path(job_id)
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        metadata.status = JobStatus.SUCCEEDED
        metadata.result_path = str(result_path)
        metadata.error = None
        self.save_metadata(metadata)
        return metadata

    def mark_failed(self, job_id: str, error: str) -> JobMetadata:
        metadata = self._require_metadata(job_id)
        metadata.status = JobStatus.FAILED
        metadata.error = error
        self.save_metadata(metadata)
        return metadata

    def load_result(self, job_id: str) -> dict[str, Any] | None:
        result_path = self.result_path(job_id)
        if not result_path.is_file():
            return None
        return json.loads(result_path.read_text(encoding="utf-8"))

    def _require_metadata(self, job_id: str) -> JobMetadata:
        metadata = self.load_metadata(job_id)
        if metadata is None:
            raise FileNotFoundError(f"Job metadata missing for job `{job_id}`.")
        return metadata
