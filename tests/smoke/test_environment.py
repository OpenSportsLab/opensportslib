"""Fast, actionable checks for the supported development environment."""

from __future__ import annotations

import importlib
import sys

import pytest


REQUIRED_IMPORTS = (
    "opensportslib",
    "torch",
    "torchvision",
    "omegaconf",
    "numpy",
    "pandas",
    "h5py",
)


def test_supported_python_version():
    assert sys.version_info >= (3, 12), (
        f"OpenSportsLib requires Python >=3.12; active interpreter is {sys.version.split()[0]} "
        f"at {sys.executable}"
    )


@pytest.mark.parametrize("module_name", REQUIRED_IMPORTS)
def test_required_runtime_dependency_imports(module_name):
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        pytest.fail(
            f"Required runtime dependency {module_name!r} could not be imported: "
            f"{type(exc).__name__}: {exc}",
            pytrace=True,
        )


def test_torch_cpu_device_is_accessible():
    import torch

    value = torch.tensor([1.0], device="cpu")
    assert value.item() == 1.0
