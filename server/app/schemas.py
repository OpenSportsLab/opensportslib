from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class TaskType(str, Enum):
    VQA = "vqa"
    CLASSIFICATION = "classification"
    LOCALIZATION = "localization"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class PredictRequest(BaseModel):
    task_type: TaskType
    model_id: str | None = None
    session_id: str | None = None
    video_path: str | None = None
    media_url: str | None = None
    test_set_path: str | None = None
    media_archive_path: str | None = None
    question: str | None = None
    task_options: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_payload(self) -> "PredictRequest":
        source_count = int(bool(self.video_path)) + int(bool(self.media_url)) + int(bool(self.test_set_path))
        if source_count > 1:
            raise ValueError("Provide only one input source: `video_path`, `media_url`, or `test_set_path`.")
        if self.session_id is None and source_count != 1:
            raise ValueError("Provide exactly one of `video_path`, `media_url`, or `test_set_path`.")
        if self.task_type == TaskType.VQA and not self.test_set_path and not str(self.question or "").strip():
            raise ValueError("`question` is required for VQA requests.")
        return self


class PredictAccepted(BaseModel):
    job_id: str
    session_id: str | None = None
    status: JobStatus
    task_type: TaskType
    model_id: str
    created_at: datetime


class JobMetadata(BaseModel):
    job_id: str
    session_id: str | None = None
    task_type: TaskType
    model_id: str
    config_overrides: dict[str, Any] | None = None
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    input_source: str
    runtime_dir: str
    result_path: str | None = None
    error: str | None = None
    queue_job_id: str | None = None


class JobStatusResponse(BaseModel):
    job_id: str
    session_id: str | None = None
    status: JobStatus
    task_type: TaskType
    model_id: str
    created_at: datetime
    updated_at: datetime
    error: str | None = None


class PredictionResult(BaseModel):
    job_id: str
    session_id: str | None = None
    status: JobStatus
    task_type: TaskType
    model_id: str
    result: dict[str, Any]
    raw_result_path: str | None = None
    reused_result: bool = False


class SessionJobEntry(BaseModel):
    job_id: str
    status: JobStatus
    model_id: str
    created_at: datetime
    updated_at: datetime
    request_summary: dict[str, Any] = Field(default_factory=dict)
    result_summary: dict[str, Any] = Field(default_factory=dict)
    result_path: str | None = None
    error: str | None = None


class VQATurn(BaseModel):
    job_id: str
    model_id: str
    question: str
    answer_text: str | None = None
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    history_mode: str | None = None
    result_summary: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class SessionRecord(BaseModel):
    session_id: str
    task_type: TaskType
    model_id: str
    created_at: datetime
    updated_at: datetime
    expires_at: datetime
    input_source: str
    video_path: str | None = None
    media_url: str | None = None
    job_history: list[SessionJobEntry] = Field(default_factory=list)
    vqa_turns: list[VQATurn] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    api_time: datetime
    redis_reachable: bool
    queue_name: str
    worker_alive: bool
    worker_name: str | None = None
    worker_loaded_models: list[str] = Field(default_factory=list)
    worker_last_seen: datetime | None = None
    configured_models: list[str] = Field(default_factory=list)


class HuggingFaceModelSource(BaseModel):
    type: Literal["huggingface"]
    model_id: str


class LocalModelSource(BaseModel):
    type: Literal["local"]
    weights_path: str
    config_path: str | None = None


class ModelRegistrationRequest(BaseModel):
    task_type: TaskType
    model_id: str | None = None
    source: HuggingFaceModelSource | LocalModelSource = Field(discriminator="type")


class ModelDefaultRequest(BaseModel):
    model_id: str


class RuntimeReconcileRequest(BaseModel):
    dry_run: bool = True
    include_active: bool = False
