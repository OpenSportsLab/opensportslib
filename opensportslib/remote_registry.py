from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Any
from urllib import error, parse, request


class RemoteRegistryError(RuntimeError):
    def __init__(self, status_code: int, detail: Any):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"Remote model registry returned HTTP {status_code}: {detail}")


class RemoteModelRegistry:
    """Administrative client for an OpenSportsLib server model registry."""

    def __init__(
        self,
        remote: str,
        api_key: str | None = None,
        timeout: float = 30.0,
        *,
        admin_token: str | None = None,
    ):
        if not remote:
            raise ValueError("remote is required.")
        if admin_token is not None:
            warnings.warn("admin_token is deprecated; use api_key.", DeprecationWarning, stacklevel=2)
            if api_key is not None and api_key != admin_token:
                raise ValueError("api_key and admin_token must match when both are provided.")
            api_key = admin_token
        self.remote = remote.rstrip("/")
        self.api_key = api_key
        self.timeout = float(timeout)
        if self.timeout <= 0:
            raise ValueError("timeout must be positive.")

    def register_model(
        self,
        *,
        task_type: str,
        huggingface_model_id: str | None = None,
        weights_path: str | Path | None = None,
        config_path: str | Path | None = None,
        model_id: str | None = None,
        hf_token: str | None = None,
    ) -> dict[str, Any]:
        """Register exactly one Hugging Face or server-local model source."""
        if (huggingface_model_id is None) == (weights_path is None):
            raise ValueError("Provide exactly one of huggingface_model_id or weights_path.")
        if huggingface_model_id is not None:
            if config_path is not None:
                raise ValueError("config_path applies only to local models.")
            if model_id is not None and model_id != huggingface_model_id:
                raise ValueError("A Hugging Face model uses its repository ID as model_id.")
            source = {"type": "huggingface", "model_id": huggingface_model_id}
            if hf_token is not None:
                source["hf_token"] = hf_token
            payload = {
                "task_type": task_type,
                "source": source,
            }
        else:
            payload = {
                "task_type": task_type,
                "model_id": model_id,
                "source": {
                    "type": "local",
                    "weights_path": str(weights_path),
                    "config_path": str(config_path) if config_path is not None else None,
                },
            }
        return self._request("POST", "/models", payload, authenticated=weights_path is not None)

    def list_models(self) -> list[dict[str, Any]]:
        return self._request("GET", "/models")["models"]

    def get_model(self, model_id: str) -> dict[str, Any]:
        query = parse.urlencode({"model_id": model_id})
        return self._request("GET", f"/models/status?{query}")

    def get_operation(self, operation_id: str) -> dict[str, Any]:
        return self._request("GET", f"/model-operations/{parse.quote(operation_id, safe='')}")

    def wait_for_operation(
        self,
        operation_id: str,
        *,
        timeout: float = 7200.0,
        poll_interval: float = 1.0,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while True:
            operation = self.get_operation(operation_id)
            if operation["status"] == "succeeded":
                return operation
            if operation["status"] == "failed":
                raise RuntimeError(operation.get("error") or "Model operation failed.")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for model operation `{operation_id}`.")
            time.sleep(poll_interval)

    def set_default(self, task_type: str, model_id: str) -> dict[str, Any]:
        task = parse.quote(task_type, safe="")
        return self._request("PUT", f"/models/defaults/{task}", {"model_id": model_id}, authenticated=True)

    def unregister_model(self, model_id: str, hf_token: str | None = None) -> dict[str, Any]:
        encoded = parse.quote(model_id, safe="")
        return self._request("DELETE", f"/models/{encoded}", authenticated=True, hf_token=hf_token)

    def reconcile_runtime(
        self,
        *,
        dry_run: bool = True,
        include_active: bool = False,
    ) -> dict[str, Any]:
        """Inspect or recover stale server jobs; dry-run is the safe default."""
        return self._request(
            "POST",
            "/admin/runtime/reconcile",
            {"dry_run": bool(dry_run), "include_active": bool(include_active)},
            authenticated=True,
        )

    def _request(
        self,
        method: str,
        endpoint: str,
        payload: dict[str, Any] | None = None,
        authenticated: bool = False,
        hf_token: str | None = None,
    ) -> dict[str, Any]:
        body = json.dumps(payload).encode() if payload is not None else None
        headers = {"Content-Type": "application/json"}
        if authenticated and self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if hf_token:
            headers["X-HF-Token"] = hf_token
        outgoing = request.Request(self.remote + endpoint, data=body, method=method, headers=headers)
        try:
            with request.urlopen(outgoing, timeout=self.timeout) as response:
                return json.loads(response.read().decode())
        except error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            try:
                parsed = json.loads(detail)
                detail = parsed.get("detail", parsed)
            except json.JSONDecodeError:
                pass
            raise RemoteRegistryError(exc.code, detail) from exc
        except error.URLError as exc:
            raise ConnectionError(f"Could not reach remote server `{self.remote}`: {exc.reason}") from exc
