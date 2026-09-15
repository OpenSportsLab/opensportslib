from __future__ import annotations

import gc
import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path

from config.model_registry import FAILED, READY, UNREGISTERING, ModelRegistry
from config.settings import get_settings
from services.base import BaseTaskService
from storage.files import prepare_uploaded_test_set, prepare_video_source
from storage.jobs import JobStore
from storage.sessions import SessionStore
from worker.registry import SERVICE_BY_TASK


SETTINGS = get_settings()
MODEL_REGISTRY = ModelRegistry(SETTINGS)
JOB_STORE = JobStore(SETTINGS)
SESSION_STORE = SessionStore(SETTINGS)
ACTIVE_MODEL_ID: str | None = None
ACTIVE_SERVICE: BaseTaskService | None = None
LAST_ACTIVITY_AT: datetime | None = None
logger = logging.getLogger("osl.worker")


def get_active_model_ids() -> list[str]:
    return [ACTIVE_MODEL_ID] if ACTIVE_MODEL_ID else []


def preload_enabled_models() -> list[str]:
    return []


def touch_worker_activity() -> None:
    global LAST_ACTIVITY_AT
    LAST_ACTIVITY_AT = datetime.now(timezone.utc)


def maybe_unload_idle_model(idle_seconds: int) -> bool:
    global ACTIVE_MODEL_ID, ACTIVE_SERVICE

    if idle_seconds <= 0 or ACTIVE_SERVICE is None or ACTIVE_MODEL_ID is None or LAST_ACTIVITY_AT is None:
        return False

    idle_for = (datetime.now(timezone.utc) - LAST_ACTIVITY_AT).total_seconds()
    if idle_for < idle_seconds:
        return False

    logger.info(
        "Idle timeout reached, unloading active model | model_id=%s idle_for_s=%.1f threshold_s=%s",
        ACTIVE_MODEL_ID,
        idle_for,
        idle_seconds,
    )
    ACTIVE_SERVICE.unload()
    ACTIVE_SERVICE = None
    ACTIVE_MODEL_ID = None
    _clear_gpu_memory()
    touch_worker_activity()
    return True


def _clear_gpu_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _get_or_load_service(model_id: str, *, allow_registering: bool = False) -> BaseTaskService:
    global ACTIVE_MODEL_ID, ACTIVE_SERVICE

    if ACTIVE_MODEL_ID == model_id and ACTIVE_SERVICE is not None:
        logger.info("Reusing loaded model | model_id=%s", model_id)
        touch_worker_activity()
        return ACTIVE_SERVICE

    record = MODEL_REGISTRY.get(model_id)
    allowed_statuses = {READY, UNREGISTERING}
    if allow_registering:
        allowed_statuses.add("registering")
    if record is None or record["status"] not in allowed_statuses:
        raise ValueError(f"Model `{model_id}` is not ready.")
    model_settings = MODEL_REGISTRY.model_settings(record)

    if ACTIVE_SERVICE is not None:
        logger.info("Unloading active model | model_id=%s", ACTIVE_MODEL_ID)
        ACTIVE_SERVICE.unload()
        ACTIVE_SERVICE = None
        ACTIVE_MODEL_ID = None
        _clear_gpu_memory()

    service_cls = SERVICE_BY_TASK[model_settings.task_type]
    logger.info(
        "Loading model for inference | model_id=%s task_type=%s config_path=%s",
        model_id,
        model_settings.task_type,
        model_settings.config_path,
    )
    service = service_cls(model_settings)
    service.preload()
    ACTIVE_SERVICE = service
    ACTIVE_MODEL_ID = model_id
    touch_worker_activity()
    logger.info("Model loaded | model_id=%s", model_id)
    return service


def run_inference_job(
    job_id: str,
    request_payload: dict,
    model_id: str,
    model_generation: int | None = None,
) -> dict:
    metadata = JOB_STORE.mark_running(job_id)
    session = SESSION_STORE.load_session(metadata.session_id) if metadata.session_id else None
    try:
        record = MODEL_REGISTRY.get(model_id)
        if record is None or record["status"] not in {READY, UNREGISTERING}:
            raise ValueError(f"Model `{model_id}` is no longer available.")
        if model_generation is not None and record["generation"] != model_generation:
            raise ValueError(f"Model `{model_id}` registration changed after this job was accepted.")
        touch_worker_activity()
        logger.info(
            "Inference starting | job_id=%s session_id=%s task_type=%s model_id=%s input_source=%s",
            job_id,
            metadata.session_id,
            metadata.task_type.value,
            model_id,
            metadata.input_source,
        )
        working_dir = Path(metadata.runtime_dir)
        uploaded_manifest = request_payload.get("test_set_path")
        if uploaded_manifest:
            request_payload["resolved_test_set_path"] = str(
                prepare_uploaded_test_set(
                    manifest_path=uploaded_manifest,
                    archive_path=request_payload.get("media_archive_path") or "",
                    destination_dir=working_dir,
                    max_archive_bytes=SETTINGS.max_upload_archive_bytes,
                    max_extracted_bytes=SETTINGS.max_upload_extracted_bytes,
                    max_file_count=SETTINGS.max_upload_file_count,
                    allowed_extensions=SETTINGS.allowed_upload_extensions,
                )
            )
            logger.info("Test-set upload prepared | job_id=%s manifest=%s", job_id, request_payload["resolved_test_set_path"])
        else:
            source_path = request_payload.get("video_path")
            media_url = request_payload.get("media_url")
            if metadata.task_type.value == "vqa" and session is not None and not source_path and not media_url:
                source_path = session.video_path
                media_url = session.media_url
                logger.info(
                    "Using stored session video for VQA follow-up | session_id=%s video_path=%s media_url=%s",
                    session.session_id,
                    source_path,
                    media_url,
                )
            resolved_video_path = prepare_video_source(
                source_path=source_path,
                media_url=media_url,
                destination_dir=working_dir,
            )
            request_payload["resolved_video_path"] = str(resolved_video_path)
            logger.info(
                "Input prepared | job_id=%s session_id=%s resolved_video_path=%s",
                job_id,
                metadata.session_id,
                resolved_video_path,
            )
        if session is not None:
            if not uploaded_manifest:
                SESSION_STORE.set_resolved_video_path(
                    session.session_id,
                    str(resolved_video_path),
                    input_source=str(resolved_video_path),
                    media_url=None,
                )
            if metadata.task_type.value == "vqa":
                request_payload["conversation_history"] = SESSION_STORE.successful_vqa_history(
                    session.session_id,
                    exclude_job_id=job_id,
                )
        service = _get_or_load_service(model_id)
        result = service.predict(request_payload, working_dir)
        succeeded_metadata = JOB_STORE.mark_succeeded(job_id, result)
        if session is not None:
            SESSION_STORE.update_job(
                session.session_id,
                job_id=job_id,
                status=succeeded_metadata.status,
                result_path=succeeded_metadata.result_path,
                result_summary=result.get("summary") or {},
            )
            SESSION_STORE.set_model_id(session.session_id, model_id)
            if metadata.task_type.value == "vqa":
                SESSION_STORE.complete_vqa_turn(
                    session.session_id,
                    job_id=job_id,
                    answer_text=(result.get("summary") or {}).get("answer_text"),
                    history_mode=result.get("history_mode"),
                    result_summary=result.get("summary") or {},
                )
                logger.info(
                    "VQA history mode used | session_id=%s job_id=%s mode=%s",
                    session.session_id,
                    job_id,
                    result.get("history_mode"),
                )
        logger.info(
            "Inference completed | job_id=%s session_id=%s task_type=%s model_id=%s",
            job_id,
            metadata.session_id,
            metadata.task_type.value,
            model_id,
        )
        touch_worker_activity()
        return result
    except Exception as exc:
        error = f"{exc}\n\n{traceback.format_exc()}"
        failed_metadata = JOB_STORE.mark_failed(job_id, error)
        if session is not None:
            SESSION_STORE.update_job(
                session.session_id,
                job_id=job_id,
                status=failed_metadata.status,
                error=error,
            )
            if metadata.task_type.value == "vqa":
                SESSION_STORE.fail_vqa_turn(
                    session.session_id,
                    job_id=job_id,
                    error=error,
                )
        logger.exception(
            "Inference failed | job_id=%s session_id=%s task_type=%s model_id=%s error=%s",
            job_id,
            metadata.session_id,
            metadata.task_type.value,
            model_id,
            exc,
        )
        touch_worker_activity()
        raise


def register_model_job(operation_id: str, model_id: str) -> dict:
    operation = MODEL_REGISTRY.get_operation(operation_id)
    record = MODEL_REGISTRY.get(model_id)
    if operation is None or record is None:
        raise ValueError("Registration operation or model record no longer exists.")
    operation["status"] = "running"
    MODEL_REGISTRY.save_operation(operation)
    try:
        if record["source_type"] == "huggingface" and not record.get("config_path"):
            from opensportslib.core.utils.config import resolve_config_path

            record["config_path"] = str(resolve_config_path(record["source"]))
            MODEL_REGISTRY.put(record)
        _get_or_load_service(model_id, allow_registering=True)
        record["status"] = READY
        MODEL_REGISTRY.put(record)
        operation["status"] = "succeeded"
        MODEL_REGISTRY.save_operation(operation)
        return {"model_id": model_id, "status": READY}
    except Exception as exc:
        record["status"] = FAILED
        record["error"] = str(exc)
        MODEL_REGISTRY.put(record)
        operation["status"] = "failed"
        operation["error"] = str(exc)
        MODEL_REGISTRY.save_operation(operation)
        _unload_active_model(model_id)
        logger.exception("Model registration failed | model_id=%s", model_id)
        raise


def unregister_model_job(operation_id: str, model_id: str, generation: int) -> dict:
    operation = MODEL_REGISTRY.get_operation(operation_id)
    record = MODEL_REGISTRY.get(model_id)
    if operation is None:
        raise ValueError("Unregistration operation no longer exists.")
    operation["status"] = "running"
    MODEL_REGISTRY.save_operation(operation)
    try:
        if record is not None and record["generation"] == generation:
            _unload_active_model(model_id)
            MODEL_REGISTRY.delete(model_id)
        operation["status"] = "succeeded"
        MODEL_REGISTRY.save_operation(operation)
        return {"model_id": model_id, "status": "unregistered"}
    except Exception as exc:
        operation["status"] = "failed"
        operation["error"] = str(exc)
        MODEL_REGISTRY.save_operation(operation)
        if record is not None and record["generation"] == generation:
            record["status"] = READY
            MODEL_REGISTRY.put(record)
        logger.exception("Model unregistration failed | model_id=%s", model_id)
        raise


def _unload_active_model(model_id: str) -> None:
    global ACTIVE_MODEL_ID, ACTIVE_SERVICE
    if ACTIVE_MODEL_ID != model_id or ACTIVE_SERVICE is None:
        return
    ACTIVE_SERVICE.unload()
    ACTIVE_SERVICE = None
    ACTIVE_MODEL_ID = None
    _clear_gpu_memory()
    touch_worker_activity()
