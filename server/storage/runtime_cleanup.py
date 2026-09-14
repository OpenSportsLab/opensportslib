from __future__ import annotations

import logging
import shutil
from pathlib import Path

from app.schemas import JobStatus, SessionRecord
from config.settings import Settings
from storage.jobs import JobStore
from storage.sessions import SessionStore


logger = logging.getLogger("osl.runtime_cleanup")


class RuntimeCleaner:
    def __init__(self, settings: Settings, session_store: SessionStore, job_store: JobStore):
        self.settings = settings
        self.session_store = session_store
        self.job_store = job_store

    def prune_expired_sessions(self) -> list[str]:
        removed: list[str] = []
        for session in self.session_store.list_sessions():
            if not self.session_store.is_expired(session):
                continue
            if self._session_has_active_jobs(session):
                logger.info(
                    "Skipping expired session cleanup because jobs are still active | session_id=%s",
                    session.session_id,
                )
                continue
            self.cleanup_session(session)
            removed.append(session.session_id)
        if removed:
            logger.info("Expired session cleanup complete | removed_sessions=%s", ",".join(removed))
        return removed

    def cleanup_session_by_id(self, session_id: str) -> bool:
        session = self.session_store.load_session(session_id)
        if session is None:
            return False
        if self._session_has_active_jobs(session):
            logger.info(
                "Skipping session cleanup because jobs are still active | session_id=%s",
                session.session_id,
            )
            return False
        self.cleanup_session(session)
        return True

    def cleanup_session(self, session: SessionRecord) -> None:
        logger.info("Cleaning session runtime artifacts | session_id=%s task_type=%s", session.session_id, session.task_type.value)
        for job in session.job_history:
            self._remove_path(self.settings.tmp_dir / job.job_id)
            self._remove_path(self.job_store.metadata_path(job.job_id))
            self._remove_path(self.job_store.result_path(job.job_id))
        self._remove_extra_session_paths(session)
        self.session_store.delete_session(session.session_id)

    def _remove_extra_session_paths(self, session: SessionRecord) -> None:
        candidates: set[Path] = set()
        if session.video_path:
            candidates.add(Path(session.video_path))
        if session.input_source and session.input_source.startswith(str(self.settings.runtime_dir)):
            candidates.add(Path(session.input_source))

        for path in candidates:
            if not self._is_within_runtime(path):
                continue
            if path.is_file():
                self._remove_path(path)
                parent = path.parent
                if parent != self.settings.tmp_dir:
                    self._remove_path(parent)
            elif path.is_dir():
                self._remove_path(path)

    def _is_within_runtime(self, path: Path) -> bool:
        try:
            path.resolve().relative_to(self.settings.runtime_dir.resolve())
            return True
        except Exception:
            return False

    def _remove_path(self, path: Path) -> None:
        try:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=False)
            elif path.exists():
                path.unlink()
        except FileNotFoundError:
            return
        except Exception:
            logger.exception("Failed to remove runtime path | path=%s", path)

    def _session_has_active_jobs(self, session: SessionRecord) -> bool:
        active_statuses = {JobStatus.QUEUED, JobStatus.RUNNING}
        return any(job.status in active_statuses for job in session.job_history)
