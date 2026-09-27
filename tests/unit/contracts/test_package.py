"""Distribution metadata, bundled assets, and public namespace contracts."""

from __future__ import annotations

import re
import tomllib

import pytest

import opensportslib
from opensportslib import apis
from tests.helpers.configs import CONFIG_ROOT, REPOSITORY_ROOT, experiment_config_paths


pytestmark = pytest.mark.unit


def _projects():
    with (REPOSITORY_ROOT / "pyproject.toml").open("rb") as stream:
        root = tomllib.load(stream)
    with (REPOSITORY_ROOT / "server" / "pyproject.toml").open("rb") as stream:
        server = tomllib.load(stream)
    return root, server


def test_project_metadata_and_console_entry_point():
    root, _ = _projects()
    project = root["project"]
    missing = sorted({"name", "version", "requires-python", "dependencies"} - set(project))
    assert not missing, f"project metadata is missing required keys: {missing}"
    assert project["name"] == "opensportslib"
    assert re.fullmatch(r"\d+\.\d+\.\d+(?:\.dev\d+)?", project["version"])
    assert project["requires-python"] == ">=3.12"
    assert project["scripts"]["opensportslib"] == "opensportslib.cli:main"


def test_server_dependency_pin_matches_library_version():
    root, server = _projects()
    expected = f"opensportslib=={root['project']['version']}"
    assert expected in server["project"]["dependencies"]


def test_packaging_includes_all_yaml_presets():
    root, _ = _projects()
    patterns = root["tool"]["setuptools"]["package-data"]["opensportslib"]
    assert "configs/**/*.yaml" in patterns
    assert experiment_config_paths(), f"No task presets found under {CONFIG_ROOT}"


def test_coverage_baseline_is_a_valid_committed_percentage():
    raw = (REPOSITORY_ROOT / "scripts" / "coverage-baseline.txt").read_text(encoding="utf-8").strip()
    assert raw.isdigit(), "coverage-baseline.txt must contain one integer percentage"
    assert 1 <= int(raw) <= 100


def test_public_namespaces_expose_required_contracts():
    assert {"model", "metrics", "datasets", "core", "tools"} <= set(opensportslib.__all__)
    required_api = {
        "Config", "BaseTaskModel", "ClassificationModel", "LocalizationModel", "VQAModel",
        "RemoteModelRegistry", "RemoteRegistryError",
    }
    assert required_api <= set(apis.__all__)
    for name in required_api:
        assert getattr(apis, name) is not None
