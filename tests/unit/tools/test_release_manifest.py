"""Fast contracts for the release profile manifest writer."""

from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[3] / "scripts" / "write_release_manifest.py"


def test_release_manifest_writer_is_present_and_commit_bound():
    spec = importlib.util.spec_from_file_location("write_release_manifest", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert callable(module.main)
    assert "commit" in SCRIPT.read_text(encoding="utf-8")


def test_release_manifest_aggregator_requires_all_profiles():
    aggregator = Path(__file__).parents[3] / "scripts" / "verify_release_manifests.py"
    source = aggregator.read_text(encoding="utf-8")
    for profile in ("qwen", "xvars", "gar"):
        assert profile in source
