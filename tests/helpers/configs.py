"""Shared discovery helpers for packaged configuration contract tests."""

from __future__ import annotations

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPOSITORY_ROOT / "opensportslib"
CONFIG_ROOT = PACKAGE_ROOT / "configs"


def experiment_config_paths() -> list[Path]:
    """Return concrete task presets, excluding incomplete layering defaults."""

    return sorted(path for path in CONFIG_ROOT.glob("*/*.yaml") if path.name != "default.yaml")
