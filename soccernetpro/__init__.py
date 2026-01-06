import importlib
# from . import apis as model
# from . import metrics
# from . import datasets
# from . import core

def __getattr__(name):
    if name == "model":
        return importlib.import_module("soccernetpro.apis")
    if name == "metrics":
        return importlib.import_module("soccernetpro.metrics")
    if name == "datasets":
        return importlib.import_module("soccernetpro.datasets")
    if name == "core":
        return importlib.import_module("soccernetpro.core")
    raise AttributeError(f"module 'soccernetpro' has no attribute '{name}'")

__all__ = ["model", "metrics", "datasets", "core"]