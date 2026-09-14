"""Versioned request configuration and cached-model isolation."""

from copy import deepcopy
from functools import wraps
from threading import RLock


def model_config(model_settings):
    from opensportslib.apis import Config

    # Registry configs define the server's supported runtime. This endpoint
    # must remain available when the model hub is offline.
    return Config.from_file(model_settings.config_path)


def capabilities(config):
    allowed = {}
    for name, option in config.options().items():
        if not name.startswith("inference.") or name == "inference.num_workers":
            continue
        key = name.split(".", 1)[1]
        allowed[key] = dict(option)
        # Bound resource-heavy request options independently of client values.
        if key == "batch_size":
            allowed[key]["maximum"] = 64
        elif key == "max_new_tokens":
            from opensportslib.core.config.editable import _get
            cap = _get(config.get_config(), "TRAIN.execution.production.max_new_tokens_cap", 4096)
            allowed[key]["maximum"] = min(int(cap or 4096), 4096)
    return {"version": 1, "options": allowed}


def validate_overrides(config, envelope):
    if not isinstance(envelope, dict) or set(envelope) != {"version", "inference"} or type(envelope.get("version")) is not int or envelope["version"] != 1:
        raise ValueError("config_overrides must contain version=1 and an inference dictionary")
    inference = envelope["inference"]
    if not isinstance(inference, dict):
        raise ValueError("config_overrides.inference must be a dictionary")
    allowed = capabilities(config)["options"]
    for name, value in inference.items():
        if name not in allowed:
            raise ValueError(f"Unsupported remote inference option {name!r}")
        spec = allowed[name]
        config._check_value(f"inference.{name}", value, spec["value"])
        if spec.get("minimum") is not None and value < spec["minimum"] or spec.get("maximum") is not None and value > spec["maximum"]:
            raise ValueError(f"inference.{name} exceeds server limits")
    candidate = deepcopy(config)
    candidate.update(inference=inference)
    return deepcopy(inference)


def configured_prediction(method):
    @wraps(method)
    def wrapped(self, request_payload, working_dir):
        from opensportslib.apis import Config

        lock = self.__dict__.setdefault("_configuration_lock", RLock())
        with lock:
            if self.model is None:
                self.preload()
            model = self.model
            envelope = (request_payload.get("task_options") or {}).get("config_overrides")
            original_config = model.config
            original_editor = getattr(model, "_config_editor", None)
            original_refs = {key: getattr(model, key, None) for key in ("model", "trainer", "processor")}
            cached_configs = []
            for owner in original_refs.values():
                if owner is not None:
                    for key in ("config", "cfg", "cfg_model"):
                        if hasattr(owner, key):
                            cached_configs.append((owner, key, getattr(owner, key)))
            try:
                model.config = deepcopy(original_config)
                overrides = {}
                if envelope is not None:
                    overrides = validate_overrides(Config(model.get_config()), envelope)
                    model.update_config(inference=overrides)
                result = method(self, request_payload, working_dir)
                if envelope is not None:
                    result["config_overrides"] = {"version": 1, "inference": overrides}
                return result
            finally:
                model.config = original_config
                model._config_editor = original_editor
                for key, value in original_refs.items():
                    setattr(model, key, value)
                for owner, key, value in cached_configs:
                    setattr(owner, key, value)
    return wrapped
