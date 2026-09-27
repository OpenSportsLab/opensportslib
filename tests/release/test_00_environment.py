"""Fail-fast contract for the explicitly enabled release environment."""

from __future__ import annotations

import importlib.util

import pytest

from tests.release._release_common import require_release_enabled


pytestmark = [pytest.mark.release, pytest.mark.gpu]


def test_release_environment_has_cuda_and_required_training_packages():
    require_release_enabled()
    import torch

    assert torch.cuda.is_available(), "Release verification requires an accessible CUDA GPU."
    assert torch.cuda.device_count() >= 1, "CUDA reported available but exposed no devices."

    required = {
        "nvidia.dali": "run `opensportslib setup --dali`",
        "peft": "run `opensportslib setup --vqa_qwen`",
        "trl": "run `opensportslib setup --vqa_qwen`",
    }
    missing = [f"{name} ({hint})" for name, hint in required.items() if importlib.util.find_spec(name) is None]
    assert not missing, "Release environment is missing required packages: " + ", ".join(missing)
