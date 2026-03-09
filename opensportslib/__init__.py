import importlib
# from . import apis as model
# from . import metrics
# from . import datasets
# from . import core

def __getattr__(name):
    if name == "model":
        return importlib.import_module("opensportslib.apis")
    if name == "metrics":
        return importlib.import_module("opensportslib.metrics")
    if name == "datasets":
        return importlib.import_module("opensportslib.datasets")
    if name == "core":
        return importlib.import_module("opensportslib.core")
    raise AttributeError(f"module 'opensportslib' has no attribute '{name}'")

__all__ = ["model", "metrics", "datasets", "core"]