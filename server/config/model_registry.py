from __future__ import annotations

import hashlib
import hmac
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from redis import Redis
    from config.settings import ModelSettings, Settings

MODEL_KEY = "osl:model-registry:models"
DEFAULT_KEY = "osl:model-registry:defaults"
OPERATION_KEY = "osl:model-registry:operations"
SECRET_KEY_PREFIX = "osl:model-registry:secret:"
LOCAL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
READY, REGISTERING, FAILED, UNREGISTERING = "ready", "registering", "failed", "unregistering"
CHECKPOINT, CONFIG_ONLY = "checkpoint", "config_only"
CHECKPOINT_SUFFIXES = (".pt", ".pth", ".bin", ".safetensors", ".ckpt", ".pkl", ".pickle")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ModelRegistry:
    def __init__(self, settings: Settings, connection: Redis | None = None):
        self.settings = settings
        if connection is None:
            from redis import Redis

            connection = Redis.from_url(settings.redis_url)
        self.redis = connection

    def get(self, model_id: str) -> dict[str, Any] | None:
        raw = self.redis.hget(MODEL_KEY, model_id)
        return json.loads(raw) if raw else None

    def list(self) -> list[dict[str, Any]]:
        return sorted((json.loads(v) for v in self.redis.hvals(MODEL_KEY)), key=lambda x: x["model_id"])

    def put(self, record: dict[str, Any]) -> None:
        record["updated_at"] = utc_now()
        self.redis.hset(MODEL_KEY, record["model_id"], json.dumps(record))

    def reserve(self, record: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        if self.redis.hsetnx(MODEL_KEY, record["model_id"], json.dumps(record)):
            return record, True
        existing = self.get(record["model_id"])
        assert existing is not None
        # source_mode is resolved asynchronously for Hugging Face sources;
        # it must not turn an idempotent re-registration into a conflict.
        fields = ("model_id", "task_type", "source_type", "source", "config_path")
        if all(existing.get(k) == record.get(k) for k in fields):
            return existing, False
        raise FileExistsError(f"Model ID `{record['model_id']}` is already registered to another source.")

    def delete(self, model_id: str) -> None:
        self.redis.hdel(MODEL_KEY, model_id)
        for task, default_id in self.redis.hgetall(DEFAULT_KEY).items():
            if _decode(default_id) == model_id:
                self.redis.hdel(DEFAULT_KEY, _decode(task))

    def resolve(self, task_type: str, model_id: str | None) -> dict[str, Any] | None:
        if model_id is None:
            raw = self.redis.hget(DEFAULT_KEY, task_type)
            model_id = _decode(raw) if raw else None
        record = self.get(model_id) if model_id else None
        if record and record["task_type"] != task_type:
            raise TypeError(f"Model `{record['model_id']}` is registered for `{record['task_type']}`, not `{task_type}`.")
        return record

    def set_default(self, task_type: str, model_id: str) -> None:
        record = self.get(model_id)
        if not record:
            raise KeyError(model_id)
        if record["task_type"] != task_type or record["status"] != READY:
            raise ValueError("A default must be a READY model registered for the requested task.")
        self.redis.hset(DEFAULT_KEY, task_type, model_id)

    def create_operation(self, kind: str, model_id: str, status: str = "queued") -> dict[str, Any]:
        now = utc_now()
        operation = {"operation_id": str(uuid.uuid4()), "kind": kind, "model_id": model_id,
                     "status": status, "created_at": now, "updated_at": now, "error": None}
        self.save_operation(operation)
        return operation

    def get_operation(self, operation_id: str) -> dict[str, Any] | None:
        raw = self.redis.hget(OPERATION_KEY, operation_id)
        return json.loads(raw) if raw else None

    def save_operation(self, operation: dict[str, Any]) -> None:
        operation["updated_at"] = utc_now()
        self.redis.hset(OPERATION_KEY, operation["operation_id"], json.dumps(operation))

    def model_settings(self, record: dict[str, Any]) -> ModelSettings:
        from config.settings import ModelSettings

        source_mode = record.get("source_mode") or infer_source_mode(record.get("config_path"), record.get("source"))
        return ModelSettings(task_type=record["task_type"], model_id=record["model_id"],
                             enabled=record["status"] in {READY, UNREGISTERING},
                             config_path=record.get("config_path"),
                             weights=None if source_mode == CONFIG_ONLY else record["source"],
                             source_mode=source_mode)

    def store_operation_secret(self, operation_id: str, token: str) -> None:
        self.redis.setex(
            f"{SECRET_KEY_PREFIX}{operation_id}",
            self.settings.model_operation_timeout_seconds,
            token,
        )

    def consume_operation_secret(self, operation_id: str) -> str | None:
        key = f"{SECRET_KEY_PREFIX}{operation_id}"
        raw = self.redis.getdel(key)
        return _decode(raw) if raw else None

    def delete_operation_secret(self, operation_id: str) -> None:
        self.redis.delete(f"{SECRET_KEY_PREFIX}{operation_id}")


def validate_local_model_id(model_id: str) -> str:
    if not LOCAL_ID_RE.fullmatch(model_id):
        raise ValueError("model_id must be 1-128 characters using letters, numbers, '.', '_', '-', or ':'.")
    return model_id


def generated_local_model_id(weights_path: Path) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", weights_path.stem).strip("-._") or "model"
    digest = hashlib.sha256(str(weights_path).encode()).hexdigest()[:8]
    return f"local:{slug[:100]}-{digest}"


def resolve_local_source(settings: Settings, weights: str, config: str | None) -> tuple[Path, Path]:
    root = settings.model_root.resolve()
    weights_path = Path(weights).expanduser().resolve()
    _require_within_root(weights_path, root)
    if not weights_path.exists():
        raise ValueError(f"Local weights do not exist: {weights_path}")
    if weights_path.is_dir():
        config_path = Path(config).expanduser().resolve() if config else weights_path / "config.yaml"
    else:
        if not config:
            raise ValueError("config_path is required when weights_path is a file.")
        config_path = Path(config).expanduser().resolve()
    _require_within_root(config_path, root)
    if not config_path.is_file():
        raise ValueError(f"Model config does not exist: {config_path}")
    return weights_path, config_path


def infer_source_mode(config_path: str | None, source: str | None) -> str:
    """Return config_only for generic rule/config-driven runners."""
    if not config_path:
        return CHECKPOINT
    try:
        import yaml

        document = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
        runner = (((document.get("TRAIN") or {}).get("runner") or {}).get("type") or "")
        if isinstance(runner, str) and "rule" in runner.lower():
            return CONFIG_ONLY
    except (OSError, TypeError, ValueError):
        pass
    return CHECKPOINT


def resolve_huggingface_source(model_id: str, token: str | None = None) -> tuple[str, str, str | None]:
    """Resolve config and normalize checkpoint versus config-only HF sources."""
    from opensportslib.core.utils.config import resolve_config_path

    config_path = str(resolve_config_path(model_id, hf_token=token))
    try:
        from huggingface_hub import list_repo_files

        files = list_repo_files(model_id, token=token)
    except Exception as exc:
        raise ValueError(f"Could not access Hugging Face repository `{model_id}`.") from exc
    has_checkpoint = any(str(path).lower().endswith(CHECKPOINT_SUFFIXES) for path in files)
    if has_checkpoint:
        return config_path, CHECKPOINT, model_id
    if infer_source_mode(config_path, model_id) == CONFIG_ONLY:
        return config_path, CONFIG_ONLY, None
    raise ValueError(
        f"Hugging Face repository `{model_id}` has config.yaml but no supported checkpoint "
        "and its configuration does not declare a supported config-only runner."
    )


def require_api_key(settings: Settings, authorization: str | None) -> None:
    if not settings.api_key:
        raise RuntimeError("This operation is disabled because OSL_API_KEY is empty.")
    supplied = authorization.removeprefix("Bearer ") if authorization else ""
    if not hmac.compare_digest(supplied, settings.api_key):
        raise PermissionError("Invalid API key.")


def public_record(record: dict[str, Any], include_source: bool = False) -> dict[str, Any]:
    result = dict(record)
    if record.get("source_type") == "local" and not include_source:
        result.pop("source", None)
        result.pop("config_path", None)
        if result.get("error"):
            result["error"] = "Registration failed; inspect the authenticated operation for details."
    return result


def _require_within_root(path: Path, root: Path) -> None:
    if path != root and root not in path.parents:
        raise ValueError(f"Local model paths must be inside OSL_MODEL_ROOT ({root}).")


def _decode(value: bytes | str) -> str:
    return value.decode() if isinstance(value, bytes) else value
