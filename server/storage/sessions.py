from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from app.schemas import JobStatus, SessionJobEntry, SessionRecord, TaskType, VQATurn
from config.settings import Settings


class SessionStore:
    def __init__(self, settings: Settings):
        self.settings = settings

    def generate_session_id(self) -> str:
        return str(uuid.uuid4())

    def session_path(self, session_id: str) -> Path:
        return self.settings.sessions_dir / f"{session_id}.json"

    def load_session(self, session_id: str) -> SessionRecord | None:
        path = self.session_path(session_id)
        if not path.is_file():
            return None
        return SessionRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def list_sessions(self) -> list[SessionRecord]:
        sessions: list[SessionRecord] = []
        for path in sorted(self.settings.sessions_dir.glob("*.json")):
            if not path.is_file():
                continue
            sessions.append(SessionRecord.model_validate_json(path.read_text(encoding="utf-8")))
        return sessions

    def save_session(self, session: SessionRecord) -> SessionRecord:
        session.updated_at = datetime.now(timezone.utc)
        session.expires_at = session.updated_at + timedelta(seconds=self.settings.session_ttl_seconds)
        self.session_path(session.session_id).write_text(
            session.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return session

    def is_expired(self, session: SessionRecord) -> bool:
        return session.expires_at <= datetime.now(timezone.utc)

    def delete_session(self, session_id: str) -> None:
        path = self.session_path(session_id)
        if path.exists():
            path.unlink()

    def create_session(
        self,
        session_id: str,
        task_type: TaskType,
        model_id: str,
        input_source: str,
        video_path: str | None = None,
        media_url: str | None = None,
    ) -> SessionRecord:
        now = datetime.now(timezone.utc)
        session = SessionRecord(
            session_id=session_id,
            task_type=task_type,
            model_id=model_id,
            created_at=now,
            updated_at=now,
            expires_at=now + timedelta(seconds=self.settings.session_ttl_seconds),
            input_source=input_source,
            video_path=video_path,
            media_url=media_url,
        )
        return self.save_session(session)

    def touch_session(self, session: SessionRecord) -> SessionRecord:
        return self.save_session(session)

    def set_resolved_video_path(
        self,
        session_id: str,
        video_path: str,
        *,
        input_source: str | None = None,
        media_url: str | None = None,
    ) -> SessionRecord:
        session = self._require_session(session_id)
        session.video_path = video_path
        session.media_url = media_url
        if input_source is not None:
            session.input_source = input_source
        return self.save_session(session)

    def set_model_id(self, session_id: str, model_id: str) -> SessionRecord:
        session = self._require_session(session_id)
        session.model_id = model_id
        return self.save_session(session)

    def add_job(
        self,
        session_id: str,
        *,
        job_id: str,
        model_id: str,
        request_summary: dict[str, Any],
    ) -> SessionRecord:
        session = self._require_session(session_id)
        now = datetime.now(timezone.utc)
        session.job_history.append(
            SessionJobEntry(
                job_id=job_id,
                status=JobStatus.QUEUED,
                model_id=model_id,
                created_at=now,
                updated_at=now,
                request_summary=request_summary,
            )
        )
        return self.save_session(session)

    def update_job(
        self,
        session_id: str,
        *,
        job_id: str,
        status: JobStatus,
        result_path: str | None = None,
        error: str | None = None,
        result_summary: dict[str, Any] | None = None,
    ) -> SessionRecord:
        session = self._require_session(session_id)
        now = datetime.now(timezone.utc)
        for job in session.job_history:
            if job.job_id == job_id:
                job.status = status
                job.updated_at = now
                job.result_path = result_path
                job.error = error
                if result_summary is not None:
                    job.result_summary = result_summary
                break
        return self.save_session(session)

    def add_vqa_turn(
        self,
        session_id: str,
        *,
        job_id: str,
        model_id: str,
        question: str,
    ) -> SessionRecord:
        session = self._require_session(session_id)
        now = datetime.now(timezone.utc)
        session.vqa_turns.append(
            VQATurn(
                job_id=job_id,
                model_id=model_id,
                question=question,
                status=JobStatus.QUEUED,
                created_at=now,
                updated_at=now,
            )
        )
        return self.save_session(session)

    def complete_vqa_turn(
        self,
        session_id: str,
        *,
        job_id: str,
        answer_text: str | None,
        history_mode: str | None,
        result_summary: dict[str, Any] | None = None,
    ) -> SessionRecord:
        session = self._require_session(session_id)
        now = datetime.now(timezone.utc)
        for turn in session.vqa_turns:
            if turn.job_id == job_id:
                turn.answer_text = answer_text
                turn.status = JobStatus.SUCCEEDED
                turn.updated_at = now
                turn.history_mode = history_mode
                if result_summary is not None:
                    turn.result_summary = result_summary
                break
        return self.save_session(session)

    def fail_vqa_turn(self, session_id: str, *, job_id: str, error: str) -> SessionRecord:
        session = self._require_session(session_id)
        now = datetime.now(timezone.utc)
        for turn in session.vqa_turns:
            if turn.job_id == job_id:
                turn.status = JobStatus.FAILED
                turn.error = error
                turn.updated_at = now
                break
        return self.save_session(session)

    def successful_vqa_history(self, session_id: str, *, exclude_job_id: str | None = None) -> list[dict[str, Any]]:
        session = self._require_session(session_id)
        history = []
        for turn in session.vqa_turns:
            if turn.status != JobStatus.SUCCEEDED:
                continue
            if exclude_job_id is not None and turn.job_id == exclude_job_id:
                continue
            history.append(
                {
                    "job_id": turn.job_id,
                    "model_id": turn.model_id,
                    "question": turn.question,
                    "answer_text": turn.answer_text,
                    "history_mode": turn.history_mode,
                }
            )
        return history

    def latest_successful_job(self, session_id: str) -> SessionJobEntry | None:
        session = self._require_session(session_id)
        successful = [job for job in session.job_history if job.status == JobStatus.SUCCEEDED]
        if not successful:
            return None
        successful.sort(key=lambda job: job.updated_at)
        return successful[-1]

    def _require_session(self, session_id: str) -> SessionRecord:
        session = self.load_session(session_id)
        if session is None:
            raise FileNotFoundError(f"Session `{session_id}` not found.")
        return session
