from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

from rq import Queue
from rq.exceptions import NoSuchJobError

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
        from redis import Redis
        self.queue = Queue(settings.queue_name, connection=Redis.from_url(settings.redis_url))

    def prune_expired_sessions(self) -> list[str]:
        self.reconcile_jobs()
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

    def reconcile_jobs(self, *, dry_run: bool = False) -> dict:
        """Reconcile active session metadata with the authoritative RQ state."""
        summary = {"inspected": 0, "active": 0, "recovered": 0, "cleaned": 0, "jobs": [], "sessions": []}
        now = datetime.now(timezone.utc)
        stale_after = self.settings.job_timeout_seconds + self.settings.job_stale_grace_seconds
        for session in self.session_store.list_sessions():
            original_expiry = session.expires_at
            for entry in session.job_history:
                if entry.status not in {JobStatus.QUEUED, JobStatus.RUNNING}:
                    continue
                summary["inspected"] += 1
                metadata = self.job_store.load_metadata(entry.job_id)
                queue_job = None
                if metadata is not None and metadata.queue_job_id:
                    try:
                        queue_job = self.queue.fetch_job(metadata.queue_job_id)
                    except NoSuchJobError:
                        queue_job = None
                reason = None
                age = (now - entry.updated_at).total_seconds()
                if metadata is None or not metadata.queue_job_id:
                    if age <= stale_after:
                        summary["active"] += 1
                        continue
                    reason = "RQ job reference missing; recovered stale session state"
                elif queue_job is None:
                    reason = "RQ job missing; recovered stale session state"
                else:
                    try:
                        rq_status = queue_job.get_status(refresh=True)
                    except NoSuchJobError:
                        rq_status = "missing"
                    if rq_status == "missing":
                        reason = "RQ job missing; recovered stale session state"
                        rq_status = None
                    if rq_status is None:
                        pass
                    elif rq_status in {"queued", "started", "deferred", "scheduled"}:
                        if age <= stale_after:
                            summary["active"] += 1
                            continue
                        reason = "RQ job exceeded stale timeout"
                    else:
                        reason = "RQ job finished without synchronized session update"
                detail = {"session_id": session.session_id, "job_id": entry.job_id,
                          "queue_job_id": metadata.queue_job_id if metadata else None, "reason": reason}
                summary["jobs"].append(detail)
                if dry_run:
                    continue
                if metadata is not None and metadata.status in {JobStatus.QUEUED, JobStatus.RUNNING}:
                    try:
                        self.job_store.mark_failed(entry.job_id, reason)
                    except FileNotFoundError:
                        pass
                self.session_store.update_job(session.session_id, job_id=entry.job_id,
                                               status=JobStatus.FAILED, error=reason)
                if session.task_type.value == "vqa":
                    self.session_store.fail_vqa_turn(session.session_id, job_id=entry.job_id, error=reason)
                # Failed-job bookkeeping must not extend an already expired session.
                refreshed = self.session_store.load_session(session.session_id)
                if refreshed is not None:
                    refreshed.expires_at = original_expiry
                    self.session_store.save_session(refreshed)
                summary["recovered"] += 1
        if not dry_run:
            for session in self.session_store.list_sessions():
                if self.session_store.is_expired(session) and not self._session_has_active_jobs(session):
                    self.cleanup_session(session)
                    summary["cleaned"] += 1
                    summary["sessions"].append(session.session_id)
        return summary

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
