from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_repo_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


@dataclass(frozen=True)
class ModelSettings:
    task_type: str
    model_id: str
    enabled: bool
    config_path: str | None
    weights: str | None
    source_mode: str = "checkpoint"


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    redis_url: str
    queue_name: str
    runtime_dir: Path
    jobs_dir: Path
    results_dir: Path
    sessions_dir: Path
    tmp_dir: Path
    log_level: str
    worker_execution_mode: str
    session_ttl_seconds: int
    worker_idle_unload_seconds: int
    worker_heartbeat_ttl_seconds: int
    worker_heartbeat_interval_seconds: int
    worker_name: str
    job_timeout_seconds: int
    job_stale_grace_seconds: int
    max_upload_archive_bytes: int
    max_upload_extracted_bytes: int
    max_upload_file_count: int
    allowed_upload_extensions: tuple[str, ...]
    model_root: Path
    api_key: str
    model_operation_timeout_seconds: int

    def ensure_runtime_dirs(self) -> None:
        for path in (self.runtime_dir, self.jobs_dir, self.results_dir, self.sessions_dir, self.tmp_dir):
            path.mkdir(parents=True, exist_ok=True)

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    runtime_dir = _resolve_repo_path(os.getenv("OSL_RUNTIME_DIR", "./runtime"))
    settings = Settings(
        host=os.getenv("OSL_SERVER_HOST", "0.0.0.0"),
        port=int(os.getenv("OSL_SERVER_PORT", "8000")),
        redis_url=os.getenv("OSL_REDIS_URL", "redis://127.0.0.1:6379/0"),
        queue_name=os.getenv("OSL_QUEUE_NAME", "osl_inference"),
        runtime_dir=runtime_dir,
        jobs_dir=runtime_dir / "jobs",
        results_dir=runtime_dir / "results",
        sessions_dir=runtime_dir / "sessions",
        tmp_dir=runtime_dir / "tmp",
        log_level=os.getenv("OSL_LOG_LEVEL", "INFO"),
        worker_execution_mode=os.getenv("OSL_WORKER_EXECUTION_MODE", "simple").strip().lower(),
        session_ttl_seconds=int(os.getenv("OSL_SESSION_TTL_SECONDS", "1800")),
        worker_idle_unload_seconds=int(os.getenv("OSL_WORKER_IDLE_UNLOAD_SECONDS", "600")),
        worker_heartbeat_ttl_seconds=int(os.getenv("OSL_WORKER_HEARTBEAT_TTL_SECONDS", "30")),
        worker_heartbeat_interval_seconds=int(os.getenv("OSL_WORKER_HEARTBEAT_INTERVAL_SECONDS", "10")),
        worker_name=os.getenv("OSL_WORKER_NAME", "osl-worker-1"),
        job_timeout_seconds=int(os.getenv("OSL_JOB_TIMEOUT_SECONDS", "7200")),
        job_stale_grace_seconds=int(os.getenv("OSL_JOB_STALE_GRACE_SECONDS", "60")),
        max_upload_archive_bytes=int(os.getenv("OSL_MAX_UPLOAD_ARCHIVE_BYTES", str(10 * 1024 * 1024 * 1024))),
        max_upload_extracted_bytes=int(os.getenv("OSL_MAX_UPLOAD_EXTRACTED_BYTES", str(50 * 1024 * 1024 * 1024))),
        max_upload_file_count=int(os.getenv("OSL_MAX_UPLOAD_FILE_COUNT", "10000")),
        allowed_upload_extensions=tuple(
            item.strip().lower()
            for item in os.getenv(
                "OSL_ALLOWED_UPLOAD_EXTENSIONS",
                ".mp4,.avi,.mov,.mkv,.webm,.h5,.hdf5,.npy,.npz,.jpg,.jpeg,.png",
            ).split(",")
            if item.strip()
        ),
        model_root=_resolve_repo_path(os.getenv("OSL_MODEL_ROOT", "./models")),
        api_key=os.getenv("OSL_API_KEY", ""),
        model_operation_timeout_seconds=int(os.getenv("OSL_MODEL_OPERATION_TIMEOUT_SECONDS", "7200")),
    )
    settings.ensure_runtime_dirs()
    return settings
