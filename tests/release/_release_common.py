"""Shared helpers for the release-verification test suite (tests/release/).

These tests are NOT part of the regular `pytest tests/test_*.py` contract.
They download real datasets from the OpenSportsLab Hugging Face org
(some of them large) and run real training/inference/evaluation on GPU.
They exist to be run manually after a big release to confirm that training
still works end-to-end for every model family the library ships.

See tests/release/README.md for the full contract, prerequisites, and
invocation examples.
"""

from __future__ import annotations

import importlib.util
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

RELEASE_ENV_FLAG = "RUN_OSL_RELEASE_TESTS"

# Where materialized configs / run outputs (checkpoints, logs, predictions)
# are cached. Override with OSL_RELEASE_CACHE_DIR to point at a disk with
# more room.
CACHE_ROOT = Path(
    os.environ.get("OSL_RELEASE_CACHE_DIR", str(REPO_ROOT / ".release_test_cache"))
).expanduser()
CONFIG_DIR = CACHE_ROOT / "configs"
OUTPUT_DIR = CACHE_ROOT / "outputs"

# Where datasets are downloaded to / read from. Deliberately independent of
# CACHE_ROOT: point OSL_RELEASE_DATA_DIR at a folder that already hosts these
# datasets (a shared drive, a previous run's download, ...) and nothing gets
# re-downloaded or overwritten -- snapshot_dataset()/download_files() below
# pass this straight through as `local_dir` to huggingface_hub, which only
# fetches files that are missing or whose content has changed (standard HF
# Hub behavior; see https://huggingface.co/docs/huggingface_hub/guides/download).
# Defaults to a subdirectory of CACHE_ROOT when unset.
DATA_DIR = Path(
    os.environ.get("OSL_RELEASE_DATA_DIR", str(CACHE_ROOT / "data"))
).expanduser()

for _d in (DATA_DIR, CONFIG_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Opt-in gate
# --------------------------------------------------------------------------


def release_tests_enabled() -> bool:
    return os.environ.get(RELEASE_ENV_FLAG, "") == "1"


def require_release_enabled() -> None:
    """Call at the top of every release test / fixture.

    Keeps these tests from ever running by accident (plain `pytest tests/`,
    an IDE "run all tests" button, a CI job someone forgot to scope) even
    though pytest can discover them. The flat `pytest tests/test_*.py`
    command from AGENTS.md never reaches this directory in the first place
    since it's a shell glob, not a recursive pattern — this is the second,
    explicit line of defense for anyone running `pytest tests/` directly.
    """
    if not release_tests_enabled():
        pytest.skip(
            f"Release verification tests are opt-in. "
            f"Set {RELEASE_ENV_FLAG}=1 to run them (see tests/release/README.md). "
            f"They download real datasets and run real training; do not enable "
            f"them in routine CI."
        )


# --------------------------------------------------------------------------
# Tunables (env-overridable so a maintainer can scale a run up or down)
# --------------------------------------------------------------------------


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def epochs_for(default: int) -> int:
    """TRAIN.epochs override. OSL_RELEASE_EPOCHS=0 means 'keep config default'."""
    value = _env_int("OSL_RELEASE_EPOCHS", default)
    return default if value == 0 else value


def max_items_for(env_name: str, default: int | None) -> int | None:
    """Generic dataset-subset-size override. 0 (or 'all') means 'download everything'."""
    raw = os.environ.get(env_name)
    if raw is None or raw == "":
        return default
    if raw.strip().lower() == "all":
        return None
    value = int(raw)
    return None if value == 0 else value


def hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")


def optional_module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


# --------------------------------------------------------------------------
# Hugging Face dataset access helpers
# --------------------------------------------------------------------------


def _hf_api():
    from huggingface_hub import HfApi

    return HfApi(token=hf_token())


def repo_accessible(repo_id: str, repo_type: str = "dataset") -> bool:
    """True if we can read repo metadata (i.e. it's public or we're authorized).

    Distinguishes "gated/private, no access" (returns False -> caller should
    skip) from genuine infrastructure problems (network down, HF outage),
    which are re-raised so the test fails loudly instead of silently skipping.
    """
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
    from requests.exceptions import ConnectionError as RequestsConnectionError

    try:
        _hf_api().repo_info(repo_id, repo_type=repo_type)
        return True
    except (GatedRepoError, RepositoryNotFoundError):
        return False
    except RequestsConnectionError:
        raise


def require_repo_access(repo_id: str, repo_type: str = "dataset") -> None:
    if not repo_accessible(repo_id, repo_type=repo_type):
        pytest.skip(
            f"No access to {repo_id!r} on Hugging Face. It may be gated — "
            f"request access at https://huggingface.co/datasets/{repo_id} and "
            f"export HF_TOKEN (or HUGGINGFACE_TOKEN) for an account that has "
            f"been granted access, then re-run."
        )


def list_repo_files(repo_id: str, repo_type: str = "dataset") -> list[str]:
    """List a repo's files with a single cheap API call -- use this to find
    what you actually need (e.g. small annotation JSONs) before downloading
    anything, rather than snapshot-downloading an entire many-file repo.
    """
    return list(_hf_api().list_repo_files(repo_id, repo_type=repo_type))


def repo_is_populated(repo_id: str, repo_type: str = "dataset", min_files: int = 2, revision: str = "main") -> bool:
    """Some OpenSportsLab dataset repos have nothing but a placeholder (just
    a .gitattributes / README) on their default branch while the real data
    lives on a named branch (see prefer_osl_ready_dataset() below) -- always
    pass the branch you actually intend to read from. Treat a repo/revision
    with no real files as "not yet available" rather than crashing on an
    empty split.
    """
    files = _hf_api().list_repo_files(repo_id, repo_type=repo_type, revision=revision)
    real_files = [f for f in files if f not in (".gitattributes", "README.md")]
    return len(real_files) >= min_files


def require_repo_populated(repo_id: str, repo_type: str = "dataset", min_files: int = 2, revision: str = "main") -> None:
    if not repo_is_populated(repo_id, repo_type=repo_type, min_files=min_files, revision=revision):
        pytest.skip(
            f"{repo_id!r}@{revision} does not have data uploaded yet. This "
            f"test is ready to run as soon as the dataset/branch is "
            f"published — re-run once it is."
        )


# --------------------------------------------------------------------------
# OSL-ready (sharded) datasets
#
# https://huggingface.co/collections/OpenSportsLab/osl-ready-datasets pins
# the datasets the org publishes as parquet + webdataset shards (a handful
# of large files per split) rather than one file per clip/game -- always
# prefer these when they're populated. Use prefer_osl_ready_dataset() to try
# a collection dataset first and fall back to a known-good loose-file
# dataset only while the sharded one isn't published yet.
#
# Important: for every OSL-ready repo observed so far, the *default* ("main")
# branch is an empty placeholder -- the actual shards live on named branches
# (e.g. "224p", "720p", "ResNET_PCA512", "224p-2024"). Always pass the
# specific revision you want; there is no sensible repo-wide default.
# --------------------------------------------------------------------------


def download_shard_split(repo_id: str, split: str, output_dir: Path, *, revision: str) -> Path:
    """Download one split of a parquet/webdataset-shard OSL dataset and
    convert it to a local OSL v2 JSON with media extracted alongside it.

    Thin wrapper around opensportslib.tools.hf_transfer.
    download_dataset_split_from_hf(..., download_format="parquet"), which
    itself does snapshot_download(allow_patterns=[f"{split}/*"]) -- for a
    sharded dataset that's a handful of `metadata.parquet` /
    `shard_manifest.parquet` / `shards/shard-*.tar` files (see
    opensportslib/tools/parquet_to_osl_json.py for the exact expected
    layout), not one request per sample, regardless of how many samples the
    split has.
    """
    from opensportslib.tools.hf_transfer import download_dataset_split_from_hf

    output_dir.mkdir(parents=True, exist_ok=True)
    result = download_dataset_split_from_hf(
        repo_id,
        revision,
        split,
        str(output_dir),
        download_format="parquet",
        token=hf_token(),
        progress_cb=lambda msg: report_step(f"[{repo_id}@{revision}:{split}] {msg}"),
    )
    return Path(result["json_path"])


def prefer_osl_ready_dataset(
    primary: str, fallback: str | None, *, primary_revision: str
) -> tuple[str, str, bool]:
    """Prefer `primary`@`primary_revision` (an OSL-ready/sharded dataset
    branch) if it's populated; otherwise fall back to `fallback` (a
    known-good, currently-populated but non-sharded dataset on its default
    branch) and say why. Returns (repo_id, revision, is_sharded).

    If `fallback` is None and `primary` isn't populated, skips the test --
    use this when there's no non-sharded alternative worth falling back to.
    """
    if repo_is_populated(primary, revision=primary_revision):
        return primary, primary_revision, True
    if fallback is None:
        require_repo_populated(primary, revision=primary_revision)  # raises pytest.skip
    report_step(
        f"{primary!r}@{primary_revision} (OSL-ready/sharded) is not populated "
        f"yet -- falling back to {fallback!r}. Re-run once {primary!r}@"
        f"{primary_revision} is published to automatically switch to the "
        f"sharded version."
    )
    return fallback, "main", False


def snapshot_dataset(
    repo_id: str,
    local_dir: Path,
    *,
    allow_patterns: list[str] | None = None,
) -> Path:
    """Download (or update) a full dataset repo, or a pattern-restricted
    subset of it, into local_dir. Safe to call repeatedly (resumable)."""
    from huggingface_hub import snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        token=hf_token(),
        allow_patterns=allow_patterns,
    )
    return local_dir


def download_files(repo_id: str, filenames: list[str], local_dir: Path) -> list[Path]:
    from huggingface_hub import hf_hub_download

    local_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for filename in filenames:
        path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
            local_dir=str(local_dir),
            token=hf_token(),
        )
        out.append(Path(path))
    return out


# --------------------------------------------------------------------------
# Canonical config materialization
#
# opensportslib/configs/<task>/<name>.yaml files are NOT self-contained: the
# library layers them on top of opensportslib/configs/default.yaml and
# opensportslib/configs/<task>/default.yaml at load time (see
# opensportslib/core/config/loader.py::_compose_yaml_layers), but only when
# the path is physically inside opensportslib/configs/<task>/. We reuse the
# library's own composer to get the real, maintained defaults, apply our
# dataset/run overrides on top in Python, then write the fully-resolved
# result out to CONFIG_DIR (outside opensportslib/configs/) so it loads as a
# plain standalone config.
# --------------------------------------------------------------------------


def _deep_merge(base: Any, override: Any) -> Any:
    if not isinstance(base, dict) or not isinstance(override, dict):
        return deepcopy(override)
    merged = deepcopy(base)
    for key, value in override.items():
        merged[key] = _deep_merge(merged.get(key), value) if key in merged else deepcopy(value)
    return merged


def materialize_config(task: str, name: str, overrides: dict, *, out_name: str | None = None) -> str:
    """Load opensportslib/configs/<task>/<name>.yaml with its real defaults
    applied, deep-merge `overrides` on top, write the result under
    CONFIG_DIR, and return the path to the materialized file.
    """
    from opensportslib.core.config import load_config, save_config

    canonical_path = REPO_ROOT / "opensportslib" / "configs" / task / f"{name}.yaml"
    if not canonical_path.is_file():
        raise FileNotFoundError(f"No such canonical config: {canonical_path}")

    base = load_config(str(canonical_path), as_namespace=False, validate=False)
    merged = _deep_merge(base, overrides)

    out_path = CONFIG_DIR / (out_name or f"{task}_{name}.yaml")
    save_config(merged, out_path)
    return str(out_path)


def system_block(run_name: str, *, gpu_count: int = 1) -> dict:
    """Standard SYSTEM override pointing checkpoints/logs at OUTPUT_DIR."""
    run_dir = OUTPUT_DIR / run_name
    return {
        "SYSTEM": {
            "paths": {
                "save_dir": str(run_dir / "checkpoints"),
                "work_dir": str(run_dir),
                "log_dir": str(run_dir / "logs"),
            },
            "device": "auto",
            "gpu": {"count": gpu_count, "id": 0},
            "reproducibility": {"use_seed": True, "seed": 0},
        }
    }


def report_step(label: str) -> None:
    print(f"\n=== {label} ===", flush=True)


def classes_from_osl_json(payload: dict, *, head: str | None = None) -> list[str]:
    """Extract a single_label head's class list from an OSL v2 JSON payload.
    Defaults to the first label head found (fine for datasets with exactly
    one classification/event head, which is the common case here); pass
    `head` explicitly for multi-head datasets (e.g. OSL-XFoul's `action` +
    `offence`).
    """
    labels = payload.get("labels") or {}
    if not labels:
        raise ValueError("No 'labels' block in this OSL v2 payload.")
    key = head or next(iter(labels))
    return list(labels[key]["labels"])
