from __future__ import annotations

import logging
import threading

from rq import SimpleWorker, Worker

from config.settings import get_settings
from storage.jobs import JobStore
from storage.runtime_cleanup import RuntimeCleaner
from storage.sessions import SessionStore
from worker.queueing import get_queue, write_worker_heartbeat
from worker.tasks import get_active_model_ids, maybe_unload_idle_model, preload_enabled_models, touch_worker_activity


def _heartbeat_loop(stop_event: threading.Event) -> None:
    settings = get_settings()
    runtime_cleaner = RuntimeCleaner(settings, SessionStore(settings), JobStore(settings))
    while not stop_event.is_set():
        try:
            runtime_cleaner.prune_expired_sessions()
            write_worker_heartbeat(settings, get_active_model_ids())
        except Exception:
            logging.exception("Failed to write worker heartbeat.")
        stop_event.wait(settings.worker_heartbeat_interval_seconds)


def _idle_unload_loop(stop_event: threading.Event) -> None:
    settings = get_settings()
    if settings.worker_idle_unload_seconds <= 0:
        return

    sleep_seconds = max(5, min(settings.worker_heartbeat_interval_seconds, settings.worker_idle_unload_seconds // 2 or 1))
    while not stop_event.is_set():
        try:
            maybe_unload_idle_model(settings.worker_idle_unload_seconds)
        except Exception:
            logging.exception("Failed during idle model unload check.")
        stop_event.wait(sleep_seconds)


def main() -> None:
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    preload_enabled_models()
    touch_worker_activity()
    worker_cls = _resolve_worker_class(settings.worker_execution_mode)
    logging.info(
        "Worker startup complete. Models will load lazily per job. execution_mode=%s worker_class=%s idle_unload_seconds=%s job_timeout_seconds=%s",
        settings.worker_execution_mode,
        worker_cls.__name__,
        settings.worker_idle_unload_seconds,
        settings.job_timeout_seconds,
    )

    stop_event = threading.Event()
    heartbeat_thread = threading.Thread(
        target=_heartbeat_loop,
        args=(stop_event,),
        daemon=True,
    )
    heartbeat_thread.start()

    idle_thread = threading.Thread(
        target=_idle_unload_loop,
        args=(stop_event,),
        daemon=True,
    )
    idle_thread.start()

    queue = get_queue(settings)
    worker = worker_cls([queue], name=settings.worker_name, connection=queue.connection)
    try:
        worker.work(with_scheduler=False)
    finally:
        stop_event.set()


def _resolve_worker_class(execution_mode: str) -> type[Worker]:
    if execution_mode == "simple":
        return SimpleWorker
    if execution_mode == "forking":
        return Worker
    raise ValueError(
        "Unsupported OSL_WORKER_EXECUTION_MODE. Use `simple` to keep models in-process "
        "or `forking` for default RQ worker behavior."
    )


if __name__ == "__main__":
    main()
