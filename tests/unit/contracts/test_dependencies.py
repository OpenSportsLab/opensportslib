"""Keep source imports aligned with packaging and optional-install contracts."""

from __future__ import annotations

import ast
from pathlib import Path
import sys
import tomllib

import pytest

from tests.helpers.configs import PACKAGE_ROOT, REPOSITORY_ROOT


pytestmark = pytest.mark.unit

IMPORT_ALIASES = {
    "opencv-python": "cv2",
    "pytorch-lightning": "pytorch_lightning",
    "scikit-learn": "sklearn",
    "soccernet": "SoccerNet",
}
OPTIONAL_IMPORTS = {
    "cupy", "fsspec", "matplotlib", "nvidia", "peft", "PIL", "safetensors",
    "torch_geometric", "trl", "yaml", "yaml_compat",
}
# These imports are currently supplied transitively by declared runtime
# dependencies. Keep the list explicit so a new undeclared import still fails;
# remove entries as their ownership is made direct in project metadata.
TRANSITIVE_LEGACY_IMPORTS = {"numpy", "packaging", "tqdm"}


def _dependency_name(requirement: str) -> str:
    return requirement.split(";", 1)[0].split("[", 1)[0].split("=", 1)[0].strip().lower()


def _third_party_imports() -> set[str]:
    imports = set()
    stdlib = set(sys.stdlib_module_names)
    for path in PACKAGE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name.split(".", 1)[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module.split(".", 1)[0]]
            else:
                continue
            imports.update(name for name in names if name not in stdlib and name != "opensportslib")
    return imports


def test_third_party_imports_are_declared_or_explicitly_optional():
    with (REPOSITORY_ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]
    declared = {_dependency_name(item) for item in project["dependencies"]}
    declared_imports = {IMPORT_ALIASES.get(name, name.replace("-", "_")) for name in declared}
    undeclared = sorted(
        _third_party_imports() - declared_imports - OPTIONAL_IMPORTS - TRANSITIVE_LEGACY_IMPORTS
    )
    assert not undeclared, (
        "Source imports undeclared third-party packages. Add a runtime dependency or "
        f"an intentional optional-install contract: {undeclared}"
    )


def test_test_dependencies_are_declared():
    with (REPOSITORY_ROOT / "pyproject.toml").open("rb") as stream:
        extras = tomllib.load(stream)["project"]["optional-dependencies"]
    names = {_dependency_name(item) for item in extras["test"]}
    assert {"pytest", "pytest-cov", "pytest-json-report", "pytest-timeout"} <= names
