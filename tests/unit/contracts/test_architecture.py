"""Semantic guardrails for OpenSportsLib's task-oriented architecture."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from opensportslib.apis import BaseTaskModel, ClassificationModel, LocalizationModel, VQAModel
from opensportslib.core.config.validate import validate_config
from tests.helpers.configs import PACKAGE_ROOT, experiment_config_paths


pytestmark = pytest.mark.unit

TASK_WRAPPERS = (ClassificationModel, LocalizationModel, VQAModel)


def _absolute_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module)
    return imported


def test_public_task_wrappers_implement_shared_contract():
    required = {"load_weights", "train", "infer", "evaluate", "save_predictions"}
    for wrapper in TASK_WRAPPERS:
        assert issubclass(wrapper, BaseTaskModel)
        missing = sorted(name for name in required if not callable(getattr(wrapper, name, None)))
        assert not missing, f"{wrapper.__name__} is missing task operations: {missing}"


def test_task_api_modules_do_not_import_other_task_wrappers():
    api_dir = PACKAGE_ROOT / "apis"
    task_modules = {
        "classification": "ClassificationModel",
        "localization": "LocalizationModel",
        "vqa": "VQAModel",
    }
    for task, own_wrapper in task_modules.items():
        path = api_dir / f"{task}.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported_names = {
            alias.asname or alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        foreign = sorted(name for name in task_modules.values() if name != own_wrapper and name in imported_names)
        assert not foreign, f"{task} API references unrelated task wrappers: {foreign}"


def test_shared_layers_do_not_depend_on_public_task_apis():
    violations = []
    for layer in ("core", "datasets", "metrics", "models"):
        for path in (PACKAGE_ROOT / layer).rglob("*.py"):
            imports = _absolute_imports(path)
            forbidden = sorted(name for name in imports if name.startswith("opensportslib.apis."))
            if forbidden:
                violations.append(f"{path.relative_to(PACKAGE_ROOT)}: {forbidden}")
    assert not violations, "Shared layers must not depend on task API wrappers:\n" + "\n".join(violations)


@pytest.mark.parametrize("config_path", experiment_config_paths(), ids=lambda path: str(path.relative_to(PACKAGE_ROOT)))
def test_every_packaged_experiment_config_composes_and_validates(config_path):
    validated = validate_config(str(config_path))
    assert validated["TASK"] == config_path.parent.name
