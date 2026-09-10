"""Shared task-level wrapper base for OpenSportsLib APIs."""

from __future__ import annotations

import json
import logging
import os
import copy
import tempfile
import time
import uuid
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

from opensportslib.core.config.accessors import get_component_name_by_kind
from opensportslib.core.utils.config import (
    expand,
    load_config_omega,
    fetch_and_merge_config_from_HF,
    resolve_config_path,
    resolve_inference_class_metadata,
)


def _manifest_media_references(payload: dict[str, Any]):
    """Yield mutable manifest media references as (value, setter) pairs."""

    for sample in payload.get("data", []):
        if not isinstance(sample, dict):
            continue
        if isinstance(sample.get("video_path"), str):
            yield sample["video_path"], lambda value, sample=sample: sample.__setitem__("video_path", value)
        for key in ("frame_paths", "video_frames"):
            values = sample.get(key)
            if isinstance(values, list):
                for index, value in enumerate(values):
                    if isinstance(value, str):
                        yield value, lambda replacement, values=values, index=index: values.__setitem__(index, replacement)
        inputs = sample.get("inputs")
        if not isinstance(inputs, list):
            continue
        for input_obj in inputs:
            if not isinstance(input_obj, dict):
                continue
            for key in ("path", "ball_path"):
                if isinstance(input_obj.get(key), str):
                    yield input_obj[key], lambda value, input_obj=input_obj, key=key: input_obj.__setitem__(key, value)
            values = input_obj.get("paths")
            if isinstance(values, list):
                for index, value in enumerate(values):
                    if isinstance(value, str):
                        yield value, lambda replacement, values=values, index=index: values.__setitem__(index, replacement)


class BaseTaskModel(ABC):
    """Thin shared contract for task-level OpenSportsLib wrappers."""

    def __init__(
        self,
        config=None,
        weights=None,
        remote: str | None = None,
        remote_timeout: float = 30.0,
        remote_poll_interval: float = 1.0,
        remote_result_timeout: float = 7200.0,
        remote_model_id: str | None = None,
    ):
        self._configure_logging()
        self.remote = remote.rstrip("/") if remote else None
        self.remote_timeout = float(remote_timeout)
        self.remote_poll_interval = float(remote_poll_interval)
        self.remote_result_timeout = float(remote_result_timeout)
        self.remote_model_id = remote_model_id
        if self.remote_timeout <= 0 or self.remote_poll_interval <= 0 or self.remote_result_timeout <= 0:
            raise ValueError("Remote timeout values must be positive.")

        if config is None:
            from huggingface_hub.utils import HFValidationError, validate_repo_id

            if (not isinstance(weights, str) or os.path.exists(expand(weights))
                    or weights.endswith((".pt", ".pth", ".tar"))):
                raise ValueError("config path is required unless weights is a Hugging Face model ID")
            try:
                validate_repo_id(weights)
            except HFValidationError as exc:
                raise ValueError("config path is required unless weights is a Hugging Face model ID") from exc
            try:
                config = resolve_config_path(weights)
            except Exception as exc:
                raise ValueError(
                    f"Could not load OpenSportsLib config.yaml from {weights!r}; "
                    "provide config explicitly or publish a compatible config.yaml."
                ) from exc

        self.config_path = resolve_config_path(config)
        self.config = load_config_omega(self.config_path)
        self.last_loaded_weights = None
        self.best_checkpoint = None

        if weights is not None and not self.is_remote:
            self.config = fetch_and_merge_config_from_HF(self.config, weights, merge_policy="compatibility")
            self.config = resolve_inference_class_metadata(self.config)
            self.last_loaded_weights = weights
            self.best_checkpoint = weights
        elif weights is not None:
            # The remote worker owns model loading; retain this only as caller metadata.
            self.last_loaded_weights = weights
            self.best_checkpoint = weights

        self.train_flag = False  # Flag to indicate whether we're in training mode (affects checkpoint loading behavior)

        data_cfg = getattr(self.config, "DATA", None)
        if data_cfg is not None and hasattr(data_cfg, "data_dir"):
            data_cfg.data_dir = expand(data_cfg.data_dir)
            logging.info(f"Data directory: {data_cfg.data_dir}")

        self.run_id = os.environ.get("RUN_ID") or str(uuid.uuid4())[:8]
        os.environ["RUN_ID"] = self.run_id

        system_cfg = getattr(self.config, "SYSTEM", None)
        if system_cfg is not None:
            system_paths = getattr(system_cfg, "paths", None)
            base_save_dir = expand(
                getattr(system_paths, "save_dir", None)
                or getattr(system_cfg, "save_dir", None)
                or "./checkpoints"
            )
            model_name = get_component_name_by_kind(self.config, "encoder") or "model"
            run_save_dir = os.path.join(base_save_dir, model_name, self.run_id)
            self.save_dir = run_save_dir
            if system_paths is not None:
                system_paths.save_dir = run_save_dir
                if hasattr(system_paths, "work_dir"):
                    system_paths.work_dir = run_save_dir
            else:
                system_cfg.save_dir = run_save_dir
                if hasattr(system_cfg, "work_dir"):
                    system_cfg.work_dir = run_save_dir
            os.makedirs(run_save_dir, exist_ok=True)
        else:
            self.save_dir = expand("./checkpoints")
            os.makedirs(self.save_dir, exist_ok=True)

        logging.info(f"Save directory: {self.save_dir}")

        self.model = None
        self.processor = None
        self.trainer = None
        self.last_remote_failures: list[dict[str, Any]] = []

        if weights is not None and not self.is_remote:
            self.load_weights(weights=weights)

    @property
    def is_remote(self) -> bool:
        """Whether inference requests are sent to an OpenSportsLib server."""

        return self.remote is not None

    def submit_inference(
        self,
        *,
        task_type: str,
        test_set: str,
        model_id: str | None = None,
        task_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Upload a complete JSON manifest and its referenced local media as one job."""

        if not self.remote:
            raise RuntimeError("Remote inference is not configured. Pass `remote=` to the model constructor.")
        manifest_path = Path(expand(test_set)).resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Test manifest not found: {manifest_path}")
        if manifest_path.suffix.lower() != ".json":
            raise ValueError("Remote test-set uploads currently require a JSON manifest.")

        with tempfile.TemporaryDirectory(prefix="opensportslib-remote-") as temp_dir:
            staged_manifest, archive_path = self._stage_manifest_upload(manifest_path, Path(temp_dir))
            logging.info(
                "Submitting remote test-set inference | task=%s manifest=%s manifest_bytes=%d archive=%s archive_bytes=%d",
                task_type,
                manifest_path,
                staged_manifest.stat().st_size,
                archive_path,
                archive_path.stat().st_size,
            )
            return self._post_multipart(
                "/predict",
                fields={
                    "task_type": task_type,
                    "model_id": model_id or self.remote_model_id or "",
                    "task_options": json.dumps(task_options or {}),
                },
                files={
                    "test_set_file": staged_manifest,
                    "media_archive": archive_path,
                },
            )

    def submit_per_sample_inference(
        self,
        *,
        task_type: str,
        test_set: str,
        model_id: str | None = None,
        task_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Submit one asynchronous remote job for every sample in an OSL manifest."""

        if not self.remote:
            raise RuntimeError("Remote inference is not configured. Pass `remote=` to the model constructor.")
        manifest_path, payload = self._load_remote_manifest(test_set)
        template = copy.deepcopy(payload)
        samples = template.pop("data")
        jobs: list[dict[str, Any]] = []
        submission_errors: list[dict[str, Any]] = []

        for sample_index, sample in enumerate(samples):
            sample_id = str(sample.get("id") or sample_index) if isinstance(sample, dict) else str(sample_index)
            record: dict[str, Any] = {"sample_index": sample_index, "sample_id": sample_id}
            try:
                with tempfile.TemporaryDirectory(prefix="opensportslib-remote-sample-") as temp_dir:
                    sample_payload = copy.deepcopy(template)
                    sample_payload["data"] = [copy.deepcopy(sample)]
                    staged_manifest, archive_path = self._stage_manifest_payload(
                        sample_payload,
                        manifest_path.parent,
                        Path(temp_dir),
                    )
                    response = self._post_multipart(
                        "/predict",
                        fields={
                            "task_type": task_type,
                            "model_id": model_id or self.remote_model_id or "",
                            "task_options": json.dumps(task_options or {}),
                        },
                        files={"test_set_file": staged_manifest, "media_archive": archive_path},
                    )
                record.update({"job_id": response.get("job_id"), "status": response.get("status", "queued")})
                logging.info("Submitted remote sample job | sample_id=%s job_id=%s", sample_id, record["job_id"])
            except Exception as exc:
                record.update({"status": "submission_failed", "error": str(exc)})
                submission_errors.append(dict(record))
                logging.error("Remote sample submission failed | sample_id=%s error=%s", sample_id, exc)
            jobs.append(record)

        return {
            "task_type": task_type,
            "source_manifest": str(manifest_path),
            "remote_mode": "per_sample",
            "manifest_template": template,
            "jobs": jobs,
            "submission_errors": submission_errors,
        }

    def submit_video_inference(
        self,
        *,
        task_type: str,
        video_path: str,
        question: str | None = None,
        session_id: str | None = None,
        model_id: str | None = None,
        task_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Submit a single uploaded video, primarily for direct VQA inference."""

        if not self.remote:
            raise RuntimeError("Remote inference is not configured. Pass `remote=` to the model constructor.")
        source = Path(expand(video_path)).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Video file not found: {source}")
        logging.info(
            "Submitting remote video inference | task=%s video=%s video_bytes=%d",
            task_type,
            source,
            source.stat().st_size,
        )
        fields = {
            "task_type": task_type,
            "model_id": model_id or self.remote_model_id or "",
            "task_options": json.dumps(task_options or {}),
        }
        if question is not None:
            fields["question"] = question
        if session_id is not None:
            fields["session_id"] = session_id
        return self._post_multipart("/predict", fields=fields, files={"upload_file": source})

    def get_remote_job(self, job_id: str) -> dict[str, Any]:
        return self._request_json(f"/jobs/{job_id}")

    def get_remote_result(self, job_id: str) -> dict[str, Any]:
        return self._request_json(f"/jobs/{job_id}/result")

    def wait_for_remote_result(self, job_id: str, timeout: float | None = None) -> dict[str, Any]:
        """Poll a submitted remote job until it succeeds, fails, or times out."""

        deadline = time.monotonic() + (self.remote_result_timeout if timeout is None else float(timeout))
        while True:
            status = self.get_remote_job(job_id)
            state = str(status.get("status", "")).lower()
            logging.info("Remote job status | job_id=%s status=%s", job_id, state or "unknown")
            if state == "succeeded":
                return self.get_remote_result(job_id)
            if state == "failed":
                raise RuntimeError(f"Remote inference job `{job_id}` failed: {status.get('error', 'unknown error')}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for remote inference job `{job_id}`.")
            time.sleep(self.remote_poll_interval)

    def wait_for_remote_batch(self, batch: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        """Collect a per-sample batch without discarding successful sample predictions."""

        if batch.get("remote_mode") != "per_sample":
            raise ValueError("`wait_for_remote_batch()` requires a `remote_mode=per_sample` submission payload.")
        deadline = time.monotonic() + (self.remote_result_timeout if timeout is None else float(timeout))
        pending = [record for record in batch.get("jobs", []) if record.get("job_id")]
        completed: dict[str, dict[str, Any]] = {}
        failures = list(batch.get("submission_errors") or [])

        while pending:
            next_pending = []
            for record in pending:
                job_id = str(record["job_id"])
                try:
                    status = self.get_remote_job(job_id)
                    state = str(status.get("status", "")).lower()
                    logging.info("Remote batch job status | sample_id=%s job_id=%s status=%s", record.get("sample_id"), job_id, state)
                    if state == "succeeded":
                        completed[job_id] = self.get_remote_result(job_id)
                    elif state == "failed":
                        failures.append({**record, "error": status.get("error", "Remote job failed.")})
                    else:
                        next_pending.append(record)
                except Exception as exc:
                    failures.append({**record, "error": str(exc)})
            pending = next_pending
            if pending:
                if time.monotonic() >= deadline:
                    failures.extend({**record, "error": "Timed out waiting for remote job."} for record in pending)
                    break
                time.sleep(self.remote_poll_interval)

        predictions = copy.deepcopy(batch.get("manifest_template") or {})
        predictions["data"] = []
        for record in batch.get("jobs", []):
            result = completed.get(str(record.get("job_id")))
            if result is None:
                continue
            result_payload = (result.get("result") or {}).get("predictions") or {}
            predictions["data"].extend(result_payload.get("data") or [])
        return {
            "task_type": batch.get("task_type"),
            "source_manifest": batch.get("source_manifest"),
            "predictions": predictions,
            "failures": failures,
            "jobs": batch.get("jobs") or [],
        }

    def _stage_manifest_upload(self, manifest_path: Path, temp_dir: Path) -> tuple[Path, Path]:
        manifest_path, payload = self._load_remote_manifest(manifest_path)
        return self._stage_manifest_payload(payload, manifest_path.parent, temp_dir)

    def _load_remote_manifest(self, test_set: str | Path) -> tuple[Path, dict[str, Any]]:
        manifest_path = Path(expand(str(test_set))).resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Test manifest not found: {manifest_path}")
        if manifest_path.suffix.lower() != ".json":
            raise ValueError("Remote test-set uploads currently require a JSON manifest.")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise ValueError("Test manifest must be an OSL JSON object with a `data` list.")
        return manifest_path, payload

    def _stage_manifest_payload(
        self,
        payload: dict[str, Any],
        source_root: Path,
        temp_dir: Path,
    ) -> tuple[Path, Path]:
        temp_dir.mkdir(parents=True, exist_ok=True)
        payload = copy.deepcopy(payload)

        files: dict[Path, str] = {}
        for value_ref in _manifest_media_references(payload):
            raw_path = str(value_ref[0])
            source = Path(raw_path)
            if not source.is_absolute():
                source = source_root / source
            source = source.resolve()
            if not source.is_file():
                raise FileNotFoundError(f"Manifest media file not found: {source}")
            archive_name = files.setdefault(source, f"media/{len(files):05d}_{source.name}")
            value_ref[1](archive_name)

        staged_manifest = temp_dir / "test_set.json"
        staged_manifest.write_text(json.dumps(payload), encoding="utf-8")
        archive_path = temp_dir / "media.zip"
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for source, archive_name in files.items():
                archive.write(source, archive_name)
        logging.info(
            "Prepared remote test-set upload | manifest=%s media_files=%d archive_bytes=%d",
            staged_manifest,
            len(files),
            archive_path.stat().st_size,
        )
        return staged_manifest, archive_path

    def _request_json(self, endpoint: str) -> dict[str, Any]:
        if not self.remote:
            raise RuntimeError("Remote inference is not configured. Pass `remote=` to the model constructor.")
        request = urlrequest.Request(f"{self.remote}{endpoint}", method="GET")
        return self._open_request(request)

    def _post_multipart(self, endpoint: str, *, fields: dict[str, str], files: dict[str, Path]) -> dict[str, Any]:
        boundary = f"----OpenSportsLib{uuid.uuid4().hex}"
        chunks: list[bytes] = []
        for name, value in fields.items():
            if value == "":
                continue
            chunks.extend((
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                value.encode(),
                b"\r\n",
            ))
        for name, path in files.items():
            chunks.extend((
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"; filename="{path.name}"\r\n'.encode(),
                b"Content-Type: application/octet-stream\r\n\r\n",
                path.read_bytes(),
                b"\r\n",
            ))
        chunks.append(f"--{boundary}--\r\n".encode())
        request = urlrequest.Request(
            f"{self.remote}{endpoint}",
            data=b"".join(chunks),
            method="POST",
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(sum(len(chunk) for chunk in chunks)),
            },
        )
        return self._open_request(request)

    def _open_request(self, request: urlrequest.Request) -> dict[str, Any]:
        started_at = time.monotonic()
        content_length = request.headers.get("Content-length", "0")
        logging.info(
            "Remote server request | method=%s url=%s content_bytes=%s",
            request.get_method(),
            request.full_url,
            content_length,
        )
        try:
            with urlrequest.urlopen(request, timeout=self.remote_timeout) as response:
                body = response.read()
                logging.info(
                    "Remote server response | method=%s url=%s status=%s response_bytes=%d elapsed_s=%.2f",
                    request.get_method(),
                    request.full_url,
                    response.status,
                    len(body),
                    time.monotonic() - started_at,
                )
                return json.loads(body.decode("utf-8"))
        except urlerror.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            logging.error(
                "Remote server error | method=%s url=%s status=%s elapsed_s=%.2f detail=%s",
                request.get_method(),
                request.full_url,
                exc.code,
                time.monotonic() - started_at,
                detail,
            )
            raise RuntimeError(f"Remote server returned HTTP {exc.code}: {detail}") from exc
        except urlerror.URLError as exc:
            logging.error(
                "Remote server connection error | method=%s url=%s elapsed_s=%.2f reason=%s",
                request.get_method(),
                request.full_url,
                time.monotonic() - started_at,
                exc.reason,
            )
            raise ConnectionError(f"Could not reach remote server `{self.remote}`: {exc.reason}") from exc

    @staticmethod
    def _configure_logging() -> None:
        root_logger = logging.getLogger()
        if not root_logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s | %(levelname)s | %(message)s",
            )
        elif root_logger.level > logging.INFO:
            root_logger.setLevel(logging.INFO)

    @abstractmethod
    def load_weights(
        self,
        weights: str | None = None,
        **kwargs,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def train(
        self,
        train_set: str | None = None,
        valid_set: str | None = None,
        weights: str | None = None,
        use_wandb: bool = True,
        **kwargs,
    ) -> str | None:
        raise NotImplementedError

    @abstractmethod
    def infer(
        self,
        test_set: str | None = None,
        weights: str | None = None,
        use_wandb: bool = True,
        **kwargs,
    ) -> dict:
        raise NotImplementedError

    @abstractmethod
    def evaluate(
        self,
        test_set: str | None = None,
        weights: str | None = None,
        predictions: str | dict[str, Any] | None = None,
        use_wandb: bool = True,
        **kwargs,
    ) -> dict | str | None:
        raise NotImplementedError

    def save_predictions(
        self,
        output_path: str,
        predictions: dict,
    ) -> str:
        """Persist in-memory prediction JSON payload to a target file path."""

        dst = expand(output_path)
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)

        if not isinstance(predictions, dict):
            raise TypeError(
                f"Unsupported predictions type: {type(predictions).__name__}. "
                "Expected dict."
            )

        with open(dst, "w", encoding="utf-8") as f:
            json.dump(predictions, f)
        return dst
