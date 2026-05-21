"""Minimal runtime config adapter helpers.

Strict canonical runtime is enforced. This module now only provides
namespace/plain-dict conversion utilities plus a deprecated passthrough.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from typing import Any

from .schemas.schema_canonical import is_canonical_schema


def dict_to_namespace(d: Any, skip_keys: tuple[str, ...] = ("classes",)) -> Any:
    if isinstance(d, dict):
        out = {}
        for key, value in d.items():
            out[key] = value if key in skip_keys else dict_to_namespace(value, skip_keys)
        return SimpleNamespace(**out)
    if isinstance(d, list):
        return [dict_to_namespace(value, skip_keys) for value in d]
    return d


def namespace_to_plain_dict(ns: Any) -> Any:
    if ns is None or isinstance(ns, (str, int, float, bool)):
        return ns

    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf

        if isinstance(ns, (DictConfig, ListConfig)):
            ns = OmegaConf.to_container(ns, resolve=True)
    except ImportError:
        pass

    if isinstance(ns, dict):
        return {str(key): namespace_to_plain_dict(value) for key, value in ns.items()}
    if isinstance(ns, (list, tuple, set)):
        return [namespace_to_plain_dict(value) for value in ns]
    if hasattr(ns, "__dict__"):
        return {str(key): namespace_to_plain_dict(value) for key, value in vars(ns).items()}
    return ns


def maybe_namespace(payload: Any, *, as_namespace: bool) -> Any:
    return dict_to_namespace(payload) if as_namespace else payload


def adapt_config_to_runtime(config: Any, *, as_namespace: bool = True) -> Any:
    payload = namespace_to_plain_dict(config)
    if is_canonical_schema(payload):
        warnings.warn(
            "adapt_config_to_runtime is deprecated and returns canonical config unchanged.",
            DeprecationWarning,
            stacklevel=2,
        )
    return maybe_namespace(payload, as_namespace=as_namespace)
