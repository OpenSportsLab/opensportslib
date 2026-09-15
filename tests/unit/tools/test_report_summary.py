"""Contracts for diagnostic classification and secret redaction."""

from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[3] / "scripts" / "summarize_test_report.py"
SPEC = importlib.util.spec_from_file_location("summarize_test_report", SCRIPT)
reporter = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(reporter)


def test_failure_classifier_identifies_common_pipeline_phases():
    assert reporter.classify("CUDA GPU is unavailable") == "environment"
    assert reporter.classify("checkpoint state_dict missing backbone") == "checkpoint"
    assert reporter.classify("annotation manifest JSON malformed") == "data/annotation"
    assert reporter.classify("operation timed out") == "timeout"


def test_summary_redaction_removes_secret_values():
    message = "HF_TOKEN=hf_example password: hunter2 Authorization=Bearer-secret"
    redacted = reporter.redact(message)
    assert "hf_example" not in redacted
    assert "hunter2" not in redacted
    assert "Bearer-secret" not in redacted
    assert redacted.count("<redacted>") == 3


def test_phase_detection_distinguishes_fixture_and_test_failures():
    assert reporter.phase_for({"setup": {"outcome": "failed"}}) == "setup"
    assert reporter.phase_for({"setup": {"outcome": "passed"}, "call": {"outcome": "failed"}}) == "call"
    assert reporter.phase_for({"teardown": {"outcome": "failed"}}) == "teardown"

