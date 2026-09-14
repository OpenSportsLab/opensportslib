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
    max_upload_archive_bytes: int
    max_upload_extracted_bytes: int
    max_upload_file_count: int
    allowed_upload_extensions: tuple[str, ...]
    models: tuple[ModelSettings, ...]

    def ensure_runtime_dirs(self) -> None:
        for path in (self.runtime_dir, self.jobs_dir, self.results_dir, self.sessions_dir, self.tmp_dir):
            path.mkdir(parents=True, exist_ok=True)

    def model_settings(self) -> list[ModelSettings]:
        return list(self.models)

    def resolve_model_id(self, task_type: str, model_id: str | None) -> ModelSettings | None:
        matches = [
            item
            for item in self.model_settings()
            if item.task_type == task_type and item.enabled and item.config_path
        ]
        if model_id is None:
            return matches[0] if matches else None
        for item in matches:
            if item.model_id == model_id:
                return item
        return None


def _model_settings(prefix: str, task_type: str, default_model_id: str) -> ModelSettings:
    raw_config_path = os.getenv(f"{prefix}_CONFIG_PATH") or None
    weights = os.getenv(f"{prefix}_WEIGHTS") or None
    config_path = str(_resolve_repo_path(raw_config_path)) if raw_config_path else None
    enabled = _as_bool(os.getenv(f"{prefix}_MODEL_ENABLED"), default=bool(config_path))
    model_id = os.getenv(f"{prefix}_MODEL_ID", default_model_id)
    return ModelSettings(
        task_type=task_type,
        model_id=model_id,
        enabled=enabled,
        config_path=config_path,
        weights=weights,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    runtime_dir = _resolve_repo_path(os.getenv("OSL_RUNTIME_DIR", "./runtime"))
    models = (
        _model_settings("OSL_CLASSIFICATION", "classification", "OpenSportsLab/OSL-cls-action-mvitv2"),
        _model_settings("OSL_LOCALIZATION", "localization", "OpenSportsLab/OSL-loc-snbas-2025-e2e"),
        _model_settings("OSL_VQA_QWEN3", "vqa", "OpenSportsLab/OSL-VQA-XFOUL-qwen3-8B-VL-lora"),
        _model_settings("OSL_VQA_QWEN25", "vqa", "OpenSportsLab/OSL-VQA-XFOUL-qwen2.5-7B-VL-lora"),
        _model_settings("OSL_VQA_XVARS", "vqa", "OpenSportsLab/OSL-VQA-XFOUL-XVARS-lora"),
    )
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
        models=models,
    )
    settings.ensure_runtime_dirs()
    return settings
