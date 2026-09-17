"""Static contract for the repository's single test entry point."""

from __future__ import annotations

from pathlib import Path

from tests.helpers.configs import REPOSITORY_ROOT


def test_single_runner_is_executable_and_references_all_tiers():
    runner = REPOSITORY_ROOT / "scripts" / "run_tests.sh"
    source = runner.read_text(encoding="utf-8")
    assert runner.stat().st_mode & 0o111, "scripts/run_tests.sh must be executable"
    for path in ("tests/unit", "tests/smoke", "tests/integration", "tests/release"):
        assert path in source


def test_no_collected_tests_are_loose_at_tests_root():
    loose = sorted(path.name for path in (REPOSITORY_ROOT / "tests").glob("test_*.py"))
    assert not loose, f"Place tests in a tier/subsystem directory, not tests/: {loose}"


def test_unit_and_integration_tests_have_subsystem_owners():
    for tier in ("unit", "integration"):
        loose = sorted(path.name for path in (REPOSITORY_ROOT / "tests" / tier).glob("test_*.py"))
        assert not loose, f"Place {tier} tests in a subsystem directory: {loose}"


def test_agent_and_ci_contracts_use_single_runner():
    expected = "bash scripts/run_tests.sh"
    sources = [
        REPOSITORY_ROOT / "AGENTS.md",
        REPOSITORY_ROOT / "tests" / "AGENTS.md",
        REPOSITORY_ROOT / ".github" / "workflows" / "ci-tests.yml",
    ]
    for path in sources:
        assert expected in path.read_text(encoding="utf-8"), f"{path} does not use the single test runner"
