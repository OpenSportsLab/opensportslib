from __future__ import annotations

import json
from datetime import datetime, timezone

from redis import Redis
from rq import Queue

from config.settings import Settings


def get_redis_connection(settings: Settings) -> Redis:
    return Redis.from_url(settings.redis_url)


def get_queue(settings: Settings) -> Queue:
    return Queue(name=settings.queue_name, connection=get_redis_connection(settings))


def enqueue_inference_job(
    queue: Queue,
    *,
    job_id: str,
    request_payload: dict,
    model_id: str,
    model_generation: int,
    job_timeout_seconds: int,
):
    """Enqueue inference with the server-configured RQ execution limit."""
    return queue.enqueue(
        "worker.tasks.run_inference_job",
        job_id,
        request_payload,
        model_id,
        model_generation,
        job_id=job_id,
        job_timeout=job_timeout_seconds,
    )


def enqueue_model_registration(queue: Queue, operation_id: str, model_id: str, timeout: int):
    return queue.enqueue(
        "worker.tasks.register_model_job", operation_id, model_id,
        job_timeout=timeout,
    )


def enqueue_model_unregistration(queue: Queue, operation_id: str, model_id: str, generation: int, timeout: int):
    return queue.enqueue(
        "worker.tasks.unregister_model_job", operation_id, model_id, generation,
        job_timeout=timeout,
    )


def worker_heartbeat_key(settings: Settings) -> str:
    return f"osl:worker:{settings.worker_name}:heartbeat"


def write_worker_heartbeat(settings: Settings, loaded_models: list[str]) -> None:
    payload = {
        "worker_name": settings.worker_name,
        "loaded_models": loaded_models,
        "last_seen": datetime.now(timezone.utc).isoformat(),
    }
    redis_conn = get_redis_connection(settings)
    redis_conn.setex(
        worker_heartbeat_key(settings),
        settings.worker_heartbeat_ttl_seconds,
        json.dumps(payload),
    )


def get_worker_heartbeat(settings: Settings) -> dict:
    raw = get_redis_connection(settings).get(worker_heartbeat_key(settings))
    if not raw:
        return {"alive": False}
    data = json.loads(raw)
    last_seen = data.get("last_seen")
    return {
        "alive": True,
        "worker_name": data.get("worker_name"),
        "loaded_models": data.get("loaded_models", []),
        "last_seen": datetime.fromisoformat(last_seen) if last_seen else None,
    }
