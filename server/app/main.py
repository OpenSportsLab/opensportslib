from __future__ import annotations

import logging
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile, status
from pydantic import ValidationError

from app.schemas import (
    HealthResponse,
    JobStatus,
    JobStatusResponse,
    ModelDefaultRequest,
    ModelRegistrationRequest,
    PredictAccepted,
    PredictionResult,
    PredictRequest,
    SessionRecord,
    TaskType,
)
from config.model_registry import (
    FAILED,
    READY,
    REGISTERING,
    UNREGISTERING,
    ModelRegistry,
    generated_local_model_id,
    public_record,
    require_admin_token,
    resolve_local_source,
    utc_now,
    validate_local_model_id,
)
from config.settings import get_settings
from storage.files import save_uploaded_file
from storage.jobs import JobStore
from storage.runtime_cleanup import RuntimeCleaner
from storage.sessions import SessionStore
from worker.queueing import (
    enqueue_inference_job,
    enqueue_model_registration,
    enqueue_model_unregistration,
    get_queue,
    get_redis_connection,
    get_worker_heartbeat,
)


settings = get_settings()
job_store = JobStore(settings)
session_store = SessionStore(settings)
runtime_cleaner = RuntimeCleaner(settings, session_store, job_store)
model_registry = ModelRegistry(settings)
app = FastAPI(title="OSL Inference Server", version="0.1.0")
logger = logging.getLogger("osl.api")


@app.on_event("startup")
def bootstrap_model_registry() -> None:
    try:
        model_registry.bootstrap_legacy_models()
    except Exception:
        logger.exception("Could not bootstrap the model registry.")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    runtime_cleaner.prune_expired_sessions()
    redis_reachable = False
    heartbeat = {"alive": False}
    try:
        redis_reachable = bool(get_redis_connection(settings).ping())
        heartbeat = get_worker_heartbeat(settings)
        configured_models = [item["model_id"] for item in model_registry.list() if item["status"] == READY]
    except Exception:
        redis_reachable = False
        heartbeat = {"alive": False}
        configured_models = []
    return HealthResponse(
        status="ok" if redis_reachable else "degraded",
        api_time=datetime.now(timezone.utc),
        redis_reachable=redis_reachable,
        queue_name=settings.queue_name,
        worker_alive=heartbeat.get("alive", False),
        worker_name=heartbeat.get("worker_name"),
        worker_loaded_models=heartbeat.get("loaded_models", []),
        worker_last_seen=heartbeat.get("last_seen"),
        configured_models=configured_models,
    )


def _authorize_model_admin(authorization: str | None) -> None:
    try:
        require_admin_token(settings, authorization)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=401, detail=str(exc), headers={"WWW-Authenticate": "Bearer"}) from exc


@app.post("/models", status_code=status.HTTP_202_ACCEPTED)
def register_model(
    request: ModelRegistrationRequest,
    authorization: str | None = Header(None),
) -> dict[str, Any]:
    _authorize_model_admin(authorization)
    now = utc_now()
    if request.source.type == "huggingface":
        try:
            from huggingface_hub.utils import HFValidationError, validate_repo_id

            validate_repo_id(request.source.model_id)
        except (HFValidationError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=f"Invalid Hugging Face model ID: {exc}") from exc
        if request.model_id is not None and request.model_id != request.source.model_id:
            raise HTTPException(status_code=422, detail="Hugging Face models use their repository ID as model_id.")
        model_id = request.source.model_id
        source_type = "huggingface"
        source = request.source.model_id
        config_path = None
        source_mode = "pending"
    else:
        try:
            weights_path, resolved_config = resolve_local_source(
                settings, request.source.weights_path, request.source.config_path
            )
            model_id = validate_local_model_id(
                request.model_id or generated_local_model_id(weights_path)
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        source_type = "local"
        source = str(weights_path)
        config_path = str(resolved_config)
        source_mode = "checkpoint"

    generation = int(model_registry.redis.incr(f"osl:model-registry:generation:{model_id}"))
    record = {
        "model_id": model_id,
        "task_type": request.task_type.value,
        "source_type": source_type,
        "source": source,
        "config_path": config_path,
        "source_mode": source_mode,
        "status": REGISTERING,
        "generation": generation,
        "created_at": now,
        "updated_at": now,
        "error": None,
    }
    try:
        record, created = model_registry.reserve(record)
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not created:
        return {
            "operation_id": record.get("operation_id"),
            "model_id": model_id,
            "status": record["status"],
            "message": f"Model already registered. Use model_id `{model_id}` for inference when READY.",
        }

    operation = model_registry.create_operation("register", model_id)
    record["operation_id"] = operation["operation_id"]
    model_registry.put(record)
    try:
        enqueue_model_registration(
            get_queue(settings), operation["operation_id"], model_id,
            settings.model_operation_timeout_seconds,
        )
    except Exception as exc:
        record["status"] = FAILED
        record["error"] = "Failed to enqueue registration."
        model_registry.put(record)
        operation["status"] = "failed"
        operation["error"] = "Failed to enqueue registration."
        model_registry.save_operation(operation)
        raise HTTPException(status_code=500, detail="Failed to enqueue model registration.") from exc
    return {
        "operation_id": operation["operation_id"],
        "model_id": model_id,
        "status": REGISTERING,
        "message": f"Use model_id `{model_id}` for inference after registration reaches READY.",
    }


@app.get("/models")
def list_models() -> dict[str, list[dict[str, Any]]]:
    return {"models": [public_record(item) for item in model_registry.list()]}


@app.get("/models/status")
def get_model_status(model_id: str) -> dict[str, Any]:
    record = model_registry.get(model_id)
    if record is None:
        raise HTTPException(status_code=404, detail="MODEL_NOT_REGISTERED")
    return public_record(record)


@app.get("/model-operations/{operation_id}")
def get_model_operation(operation_id: str, authorization: str | None = Header(None)) -> dict[str, Any]:
    _authorize_model_admin(authorization)
    operation = model_registry.get_operation(operation_id)
    if operation is None:
        raise HTTPException(status_code=404, detail="Model operation not found.")
    return operation


@app.put("/models/defaults/{task_type}")
def set_default_model(
    task_type: TaskType,
    request: ModelDefaultRequest,
    authorization: str | None = Header(None),
) -> dict[str, str]:
    _authorize_model_admin(authorization)
    try:
        model_registry.set_default(task_type.value, request.model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="MODEL_NOT_REGISTERED") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"task_type": task_type.value, "model_id": request.model_id}


@app.delete("/models/{model_id:path}", status_code=status.HTTP_202_ACCEPTED)
def unregister_model(model_id: str, authorization: str | None = Header(None)) -> dict[str, Any]:
    _authorize_model_admin(authorization)
    record = model_registry.get(model_id)
    if record is None:
        raise HTTPException(status_code=404, detail="MODEL_NOT_REGISTERED")
    if record["status"] == UNREGISTERING:
        return {
            "operation_id": record.get("unregistration_operation_id"),
            "model_id": model_id,
            "status": UNREGISTERING,
        }
    record["status"] = UNREGISTERING
    model_registry.put(record)
    operation = model_registry.create_operation("unregister", model_id)
    record["unregistration_operation_id"] = operation["operation_id"]
    model_registry.put(record)
    try:
        enqueue_model_unregistration(
            get_queue(settings), operation["operation_id"], model_id,
            record["generation"], settings.model_operation_timeout_seconds,
        )
    except Exception as exc:
        record["status"] = READY
        model_registry.put(record)
        operation["status"] = "failed"
        operation["error"] = "Failed to enqueue unregistration."
        model_registry.save_operation(operation)
        raise HTTPException(status_code=500, detail="Failed to enqueue model unregistration.") from exc
    return {"operation_id": operation["operation_id"], "model_id": model_id, "status": UNREGISTERING}


def _resolve_prediction_model(task_type: str, model_id: str | None) -> dict[str, Any]:
    try:
        record = model_registry.resolve(task_type, model_id)
    except TypeError as exc:
        raise HTTPException(status_code=400, detail={"code": "MODEL_TASK_MISMATCH", "message": str(exc)}) from exc
    if record is None:
        raise HTTPException(status_code=404, detail={"code": "MODEL_NOT_REGISTERED", "model_id": model_id})
    if record["status"] == REGISTERING:
        raise HTTPException(status_code=409, detail={"code": "MODEL_NOT_READY", "model_id": record["model_id"]})
    if record["status"] == FAILED:
        raise HTTPException(status_code=422, detail={"code": "MODEL_REGISTRATION_FAILED", "model_id": record["model_id"], "error": record.get("error")})
    if record["status"] == UNREGISTERING:
        raise HTTPException(status_code=409, detail={"code": "MODEL_UNAVAILABLE", "model_id": record["model_id"]})
    if record["status"] != READY:
        raise HTTPException(status_code=409, detail={"code": "MODEL_NOT_READY", "model_id": record["model_id"]})
    return record


@app.get("/config-capabilities")
def config_capabilities(task_type: str, model_id: str | None = None):
    from services.configuration import capabilities, model_config

    try:
        record = model_registry.resolve(task_type, model_id or None)
    except TypeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if record is None or record["status"] != READY:
        raise HTTPException(status_code=404, detail="No ready registered model for this task")
    entry = model_registry.model_settings(record)
    try:
        return {"model_id": entry.model_id, **capabilities(model_config(entry))}
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/predict", response_model=PredictAccepted | PredictionResult)
async def predict(
    request: Request,
    task_type: str | None = Form(None),
    model_id: str | None = Form(None),
    session_id: str | None = Form(None),
    video_path: str | None = Form(None),
    media_url: str | None = Form(None),
    question: str | None = Form(None),
    task_options: str | None = Form(None),
    upload_file: UploadFile | None = File(None),
    test_set_file: UploadFile | None = File(None),
    media_archive: UploadFile | None = File(None),
) -> PredictAccepted:
    runtime_cleaner.prune_expired_sessions()
    predict_request = await _build_predict_request(
        request=request,
        task_type=task_type,
        model_id=model_id,
        session_id=session_id,
        video_path=video_path,
        media_url=media_url,
        question=question,
        task_options=task_options,
        upload_file=upload_file,
        test_set_file=test_set_file,
        media_archive=media_archive,
    )
    logger.info(
        "Predict request received | task_type=%s model_id=%s has_upload=%s has_test_set=%s video_path=%s media_url=%s question=%s",
        predict_request.task_type.value,
        predict_request.model_id,
        upload_file is not None,
        test_set_file is not None,
        predict_request.video_path,
        predict_request.media_url,
        bool(str(predict_request.question or "").strip()),
    )

    has_test_set_upload = test_set_file is not None or media_archive is not None
    if has_test_set_upload and (test_set_file is None or media_archive is None):
        raise HTTPException(status_code=422, detail="Provide both `test_set_file` and `media_archive`.")
    if has_test_set_upload and (upload_file is not None or video_path or media_url):
        raise HTTPException(status_code=422, detail="A test-set upload cannot be combined with a single-video input.")
    has_new_input = _has_new_input_source(predict_request, upload_file, test_set_file)
    current_session: SessionRecord | None = None

    if predict_request.session_id is not None:
        current_session = _load_active_session_or_raise(predict_request.session_id)
        logger.info(
            "Session-linked request received | session_id=%s task_type=%s model_id=%s",
            current_session.session_id,
            predict_request.task_type.value,
            predict_request.model_id,
        )
        if current_session.task_type != predict_request.task_type:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Session `{current_session.session_id}` belongs to task "
                    f"`{current_session.task_type.value}`, not `{predict_request.task_type.value}`."
                ),
            )
        if has_test_set_upload:
            raise HTTPException(status_code=400, detail="Test-set uploads cannot be attached to an existing session.")
        if predict_request.task_type == TaskType.VQA and has_new_input:
            raise HTTPException(
                status_code=400,
                detail="VQA follow-up requests must not provide `upload_file`, `video_path`, or `media_url`.",
            )
    else:
        if not has_new_input:
            raise HTTPException(
                status_code=422,
                detail="First-turn requests must provide a video source or `test_set_file` with `media_archive`.",
            )

    if (
        current_session is not None
        and current_session.task_type in {TaskType.CLASSIFICATION, TaskType.LOCALIZATION}
        and not has_new_input
        and "config_overrides" not in predict_request.task_options
    ):
        cached_result = _reuse_latest_session_result_or_raise(current_session)
        logger.info(
            "Returning cached session result | session_id=%s task_type=%s job_id=%s",
            current_session.session_id,
            current_session.task_type.value,
            cached_result.job_id,
        )
        session_store.touch_session(current_session)
        return cached_result

    effective_requested_model_id = predict_request.model_id or (current_session.model_id if current_session else None)
    registry_record = _resolve_prediction_model(
        predict_request.task_type.value, effective_requested_model_id
    )
    registry_entry = model_registry.model_settings(registry_record)

    envelope = predict_request.task_options.get("config_overrides")
    if envelope is not None:
        from services.configuration import model_config, validate_overrides
        try:
            validate_overrides(model_config(registry_entry), envelope)
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    active_session_id = predict_request.session_id or session_store.generate_session_id()
    initial_input_source = _initial_input_source(predict_request, current_session)
    metadata = job_store.create_job(
        predict_request,
        registry_entry.model_id,
        session_id=active_session_id,
        input_source=initial_input_source,
    )
    logger.info(
        "Job created | job_id=%s session_id=%s task_type=%s resolved_model_id=%s runtime_dir=%s",
        metadata.job_id,
        active_session_id,
        metadata.task_type.value,
        metadata.model_id,
        metadata.runtime_dir,
    )

    if upload_file is not None:
        saved_path = save_uploaded_file(
            filename=upload_file.filename,
            source_file=upload_file.file,
            destination_dir=Path(_runtime_dir_for_job(metadata.job_id)),
        )
        predict_request.video_path = str(saved_path)
        predict_request.media_url = None
        job_store.update_request_payload(metadata.job_id, predict_request)
        logger.info(
            "Uploaded file saved | job_id=%s session_id=%s filename=%s saved_path=%s",
            metadata.job_id,
            active_session_id,
            upload_file.filename,
            saved_path,
        )

    if test_set_file is not None and media_archive is not None:
        runtime_dir = Path(_runtime_dir_for_job(metadata.job_id))
        manifest_path = save_uploaded_file("test_set.json", test_set_file.file, runtime_dir)
        archive_path = save_uploaded_file("media.zip", media_archive.file, runtime_dir)
        predict_request.test_set_path = str(manifest_path)
        predict_request.media_archive_path = str(archive_path)
        job_store.update_request_payload(metadata.job_id, predict_request)
        logger.info(
            "Uploaded test set saved | job_id=%s manifest=%s archive=%s",
            metadata.job_id,
            manifest_path,
            archive_path,
        )

    if current_session is None:
        current_session = session_store.create_session(
            session_id=active_session_id,
            task_type=predict_request.task_type,
            model_id=registry_entry.model_id,
            input_source=predict_request.video_path or predict_request.media_url or predict_request.test_set_path or initial_input_source,
            video_path=predict_request.video_path,
            media_url=predict_request.media_url,
        )
        logger.info(
            "Session created | session_id=%s task_type=%s model_id=%s",
            current_session.session_id,
            current_session.task_type.value,
            current_session.model_id,
        )
    else:
        current_session = session_store.touch_session(current_session)
        logger.info(
            "Session follow-up accepted | session_id=%s task_type=%s requested_model_id=%s",
            current_session.session_id,
            current_session.task_type.value,
            registry_entry.model_id,
        )

    session_store.add_job(
        active_session_id,
        job_id=metadata.job_id,
        model_id=registry_entry.model_id,
        request_summary=_build_request_summary(predict_request, upload_file is not None),
    )
    if predict_request.task_type == TaskType.VQA:
        session_store.add_vqa_turn(
            active_session_id,
            job_id=metadata.job_id,
            model_id=registry_entry.model_id,
            question=str(predict_request.question or "").strip(),
        )

    try:
        queue = get_queue(settings)
        rq_job = enqueue_inference_job(
            queue,
            job_id=metadata.job_id,
            request_payload=predict_request.model_dump(mode="json"),
            model_id=metadata.model_id,
            model_generation=registry_record["generation"],
            job_timeout_seconds=settings.job_timeout_seconds,
        )
        metadata.queue_job_id = rq_job.id
        job_store.save_metadata(metadata)
        logger.info(
            "Job enqueued | job_id=%s session_id=%s queue_job_id=%s queue_name=%s job_timeout_seconds=%s",
            metadata.job_id,
            active_session_id,
            rq_job.id,
            settings.queue_name,
            settings.job_timeout_seconds,
        )
    except Exception as exc:
        job_store.mark_failed(metadata.job_id, f"Failed to enqueue job: {exc}")
        session_store.update_job(
            active_session_id,
            job_id=metadata.job_id,
            status=JobStatus.FAILED,
            error=f"Failed to enqueue job: {exc}",
        )
        if predict_request.task_type == TaskType.VQA:
            session_store.fail_vqa_turn(
                active_session_id,
                job_id=metadata.job_id,
                error=f"Failed to enqueue job: {exc}",
            )
        logger.exception("Failed to enqueue job | job_id=%s", metadata.job_id)
        raise HTTPException(status_code=500, detail="Failed to enqueue job.") from exc

    return PredictAccepted(
        job_id=metadata.job_id,
        session_id=active_session_id,
        status=metadata.status,
        task_type=metadata.task_type,
        model_id=metadata.model_id,
        created_at=metadata.created_at,
    )


async def _build_predict_request(
    request: Request,
    task_type: str | None,
    model_id: str | None,
    session_id: str | None,
    video_path: str | None,
    media_url: str | None,
    question: str | None,
    task_options: str | None,
    upload_file: UploadFile | None,
    test_set_file: UploadFile | None,
    media_archive: UploadFile | None,
) -> PredictRequest:
    content_type = request.headers.get("content-type", "").lower()

    if "application/json" in content_type:
        try:
            payload = await request.json()
            return PredictRequest.model_validate(payload)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Invalid JSON request body.") from exc

    if "multipart/form-data" in content_type:
        if upload_file is None and test_set_file is None and not video_path and not media_url and session_id is None:
            raise HTTPException(
                status_code=422,
                detail="Provide a video input or both `test_set_file` and `media_archive`.",
            )
        try:
            parsed_task_options = json.loads(task_options) if task_options else {}
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=422, detail="`task_options` must be valid JSON.") from exc
        if not isinstance(parsed_task_options, dict):
            raise HTTPException(status_code=422, detail="`task_options` must be a JSON object.")
        validation_video_path = video_path or (("uploaded://" + (upload_file.filename or "video")) if upload_file else None)
        validation_test_set_path = ("uploaded://" + (test_set_file.filename or "test_set.json")) if test_set_file else None
        payload = {
            "task_type": task_type,
            "model_id": model_id,
            "session_id": session_id,
            "video_path": validation_video_path,
            "media_url": media_url,
            "test_set_path": validation_test_set_path,
            "question": question,
            "task_options": parsed_task_options,
        }
        try:
            parsed = PredictRequest.model_validate(payload)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from exc
        if upload_file is not None:
            parsed.video_path = None
        if test_set_file is not None:
            parsed.test_set_path = None
        return parsed

    raise HTTPException(
        status_code=415,
        detail="Unsupported content type. Use application/json or multipart/form-data.",
    )


def _runtime_dir_for_job(job_id: str) -> str:
    return str(settings.tmp_dir / job_id)


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str) -> JobStatusResponse:
    metadata = job_store.load_metadata(job_id)
    if metadata is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    logger.info("Job status requested | job_id=%s status=%s", job_id, metadata.status.value)
    return JobStatusResponse(
        job_id=metadata.job_id,
        session_id=metadata.session_id,
        status=metadata.status,
        task_type=metadata.task_type,
        model_id=metadata.model_id,
        created_at=metadata.created_at,
        updated_at=metadata.updated_at,
        error=metadata.error,
    )


@app.get("/jobs/{job_id}/result")
def get_result(job_id: str) -> dict:
    metadata = job_store.load_metadata(job_id)
    if metadata is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if metadata.status.value != "succeeded":
        raise HTTPException(
            status_code=409,
            detail=f"Job is `{metadata.status.value}`. Result is not available yet.",
        )
    result = job_store.load_result(job_id)
    if result is None:
        raise HTTPException(status_code=500, detail="Result file is missing.")
    logger.info("Job result returned | job_id=%s model_id=%s", job_id, metadata.model_id)
    return {
        "job_id": metadata.job_id,
        "session_id": metadata.session_id,
        "status": metadata.status,
        "task_type": metadata.task_type,
        "model_id": metadata.model_id,
        "result": result,
        "raw_result_path": metadata.result_path,
        "reused_result": False,
    }


def _load_active_session_or_raise(session_id: str) -> SessionRecord:
    session = session_store.load_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session `{session_id}` was not found.")
    if session_store.is_expired(session):
        logger.info("Session expired | session_id=%s", session_id)
        runtime_cleaner.cleanup_session(session)
        raise HTTPException(status_code=410, detail=f"Session `{session_id}` has expired.")
    return session


def _has_new_input_source(
    predict_request: PredictRequest,
    upload_file: UploadFile | None,
    test_set_file: UploadFile | None,
) -> bool:
    return bool(upload_file is not None or test_set_file is not None or predict_request.video_path or predict_request.media_url)


def _initial_input_source(predict_request: PredictRequest, current_session: SessionRecord | None) -> str:
    if current_session is not None:
        return current_session.video_path or current_session.media_url or current_session.input_source
    return predict_request.video_path or predict_request.media_url or predict_request.test_set_path or ""


def _build_request_summary(predict_request: PredictRequest, has_upload: bool) -> dict[str, Any]:
    return {
        "task_type": predict_request.task_type.value,
        "question": predict_request.question,
        "has_upload": has_upload,
        "video_path": predict_request.video_path,
        "media_url": predict_request.media_url,
        "test_set_path": predict_request.test_set_path,
        "session_id": predict_request.session_id,
    }


def _reuse_latest_session_result_or_raise(session: SessionRecord) -> PredictionResult:
    latest_job = session_store.latest_successful_job(session.session_id)
    if latest_job is None:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Session `{session.session_id}` has no successful cached result yet. "
                "Submit a new input to create a job."
            ),
        )

    result = job_store.load_result(latest_job.job_id)
    if result is None:
        raise HTTPException(
            status_code=500,
            detail=f"Cached result for job `{latest_job.job_id}` is missing.",
        )

    return PredictionResult(
        job_id=latest_job.job_id,
        session_id=session.session_id,
        status=latest_job.status,
        task_type=session.task_type,
        model_id=latest_job.model_id,
        result=result,
        raw_result_path=latest_job.result_path,
        reused_result=True,
    )
