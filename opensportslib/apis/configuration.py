"""Configuration lifecycle shared by task wrappers."""

from copy import deepcopy
from functools import wraps
import inspect
import os
import json
from threading import RLock
from urllib.parse import urlencode
from urllib.request import Request

from opensportslib.core.config.editable import Config, _get, _set
from opensportslib.core.config.runtime_adapter import dict_to_namespace, namespace_to_plain_dict


def config_operation(method):
    """Serialize operations and prevent edits while an operation is active."""
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        lock = self.__dict__.setdefault("_config_lock", RLock())
        with lock:
            depth = getattr(self, "_operation_depth", 0)
            self._operation_depth = depth + 1
            before = namespace_to_plain_dict(self.config)
            previous_inputs = getattr(self, "_call_config_inputs", {})
            bound = inspect.signature(method).bind(self, *args, **kwargs)
            self._call_config_inputs = {**previous_inputs, **{
                name.removesuffix("_set"): os.path.abspath(os.path.expanduser(str(value)))
                for name, value in bound.arguments.items()
                if name in {"train_set", "valid_set", "test_set"} and value is not None
            }}
            try:
                if method.__name__ == "train" and self.is_remote:
                    raise ValueError("Remote training is not supported")
                return method(self, *args, **kwargs)
            finally:
                self._operation_depth = depth
                self._call_config_inputs = previous_inputs
                if depth == 0:
                    # Local localization helpers write temporary input paths into
                    # config. Restore only those defaults, preserving model state.
                    current = namespace_to_plain_dict(self.config)
                    for split, cfg in _get(before, "DATA.common.splits", {}).items():
                        for key in ("annotation_path", "source_path"):
                            if key in cfg:
                                _set(current, f"DATA.common.splits.{split}.{key}", cfg[key])
                    self.config = dict_to_namespace(current)
                    self._refresh_config_references()
    return wrapped


class ConfigurationMixin:
    def _effective_config(self, config):
        editor = getattr(self, "_config_editor", None)
        result = dict_to_namespace(editor.apply_to(config)) if editor is not None else config
        if getattr(self, "_call_config_inputs", None):
            doc = namespace_to_plain_dict(result)
            for split, path in self._call_config_inputs.items():
                _set(doc, f"DATA.common.splits.{split}.annotation_path", path)
                if split == "valid" and _get(doc, "DATA.common.splits.valid_data_frames"):
                    _set(doc, "DATA.common.splits.valid_data_frames.annotation_path", path)
            result = dict_to_namespace(doc)
        return result

    def _refresh_config_references(self):
        for owner in (getattr(self, "model", None), getattr(self, "trainer", None)):
            if owner is None:
                continue
            for key in ("config", "cfg", "cfg_model"):
                if hasattr(owner, key):
                    setattr(owner, key, self.config)

    def get_config(self):
        return deepcopy(namespace_to_plain_dict(self.config))

    def config_options(self):
        editor = deepcopy(getattr(self, "_config_editor", None)) or Config(self.get_config(), source=self.config_path)
        # Inspection reflects normalization and runtime-selected checkpoint data.
        editor._document = self.get_config()
        return editor.options()

    def update_config(self, **options):
        lock = self.__dict__.setdefault("_config_lock", RLock())
        if not lock.acquire(blocking=False):
            raise RuntimeError("Cannot update configuration during an active operation")
        try:
            if getattr(self, "_operation_depth", 0):
                raise RuntimeError("Cannot update configuration during an active operation")
            original = getattr(self, "_config_editor", None)
            editor = deepcopy(original) if original is not None else Config(self.get_config(), source=self.config_path)
            # Carry current runtime values while retaining expressions and prior
            # explicit updates from the editable source.
            current = self.get_config()
            editor._document = editor.apply_to(current)
            if original is not None:
                from opensportslib.core.config.editable import _leaves
                for path, value in _leaves(original._document):
                    if isinstance(value, str) and "${" in value:
                        _set(editor._document, path, value)
            previous = deepcopy(editor._updates)
            editor.update(**options)
            safe = []
            for name, spec in editor._registry().items():
                if not spec.requires_initialization and name.split(".")[0] in {"data", "training", "inference", "runtime", "scheduler", "training_sampling", "sft", "prompt"}:
                    safe.extend(spec.paths)
            changed = [p for p, value in editor._updates.items() if p not in previous or previous[p] != value]
            for path in changed:
                if not any(path == p for p in safe):
                    raise ValueError(f"{path} requires a new model; update Config before initialization")
            if self.is_remote:
                editor.remote_overrides()
            candidate = dict_to_namespace(editor.apply_to(current))
            self.config = candidate
            self._config_editor = editor
            self._refresh_config_references()
            return self
        finally:
            lock.release()

    def _config_request_fields(self, fields):
        fields = dict(fields)
        editor = getattr(self, "_config_editor", None)
        overrides = editor.remote_overrides() if editor is not None else {}
        task_options = json.loads(fields.get("task_options") or "{}")
        if "config_overrides" in task_options:
            raise ValueError("Use Config.update(inference=...) for remote configuration overrides")
        if not overrides:
            return fields
        query = urlencode({"model_id": fields.get("model_id", ""), "task_type": fields.get("task_type", "")})
        try:
            capabilities = self._open_request(Request(f"{self.remote}/config-capabilities?{query}"))
        except Exception as exc:
            raise ValueError("Server does not expose configuration capabilities; upgrade the server or omit overrides") from exc
        if capabilities.get("version") != 1:
            raise ValueError("Unsupported server configuration protocol version")
        allowed = capabilities.get("options", {})
        for name, value in overrides.items():
            if name not in allowed:
                raise ValueError(f"Server does not support inference.{name}")
            spec = allowed[name]
            if spec.get("minimum") is not None and value < spec["minimum"] or spec.get("maximum") is not None and value > spec["maximum"]:
                raise ValueError(f"inference.{name} exceeds server limits")
        task_options["config_overrides"] = {"version": 1, "inference": overrides}
        fields["task_options"] = json.dumps(task_options)
        return fields
