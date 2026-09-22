"""Metadata and TAR staging for Hugging Face tracking datasets."""

from __future__ import annotations

import fcntl
import io
import json
import logging
import os
import re
import tarfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from opensportslib.core.config.accessors import get_input_cfg


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedTrackingSplit:
    annotations_path: Path
    index_path: Path
    repo_id: str
    revision: str
    split: str
    cache_dir: Path


def hf_tracking_source(config) -> dict:
    source = get_input_cfg(config).get("source") or {}
    return source if str(source.get("format", "")).lower() == "hf_webdataset" else {}


def _set_resolved_revision(config, revision: str) -> None:
    data = config.DATA
    inputs = data["inputs"] if isinstance(data, dict) else data.inputs
    values = inputs.values() if isinstance(inputs, dict) else vars(inputs).values()
    for input_cfg in values:
        source = input_cfg.get("source") if isinstance(input_cfg, dict) else input_cfg.source
        source_format = source.get("format") if isinstance(source, dict) else source.format
        if str(source_format).lower() == "hf_webdataset":
            if isinstance(source, dict):
                source["resolved_revision"] = revision
            else:
                source.resolved_revision = revision
            return
    raise ValueError("No hf_webdataset input was found in the configuration")


def _write_json_atomic(path: Path, payload) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _stage_shards(prepared: PreparedTrackingSplit) -> None:
    """Cache every TAR referenced by the split before a DataLoader can start."""
    from huggingface_hub import hf_hub_download

    with prepared.index_path.open(encoding="utf-8") as handle:
        index = json.load(handle)
    shards = sorted({shard for shard, _ in index["members"].values()})
    cached = index.get("shard_paths", {})
    paths = {}
    for shard in shards:
        path = cached.get(shard)
        if not path or not Path(path).is_file():
            try:
                path = hf_hub_download(
                    repo_id=prepared.repo_id,
                    repo_type="dataset",
                    revision=prepared.revision,
                    filename=f"{prepared.split}/shards/{shard}",
                    cache_dir=str(prepared.cache_dir / "hub"),
                    token=True,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Cannot download tracking TAR shard {prepared.split}/shards/{shard} "
                    f"from {prepared.repo_id}@{prepared.revision}"
                ) from exc
        if not Path(path).is_file():
            raise FileNotFoundError(f"Downloaded tracking TAR shard is missing: {path}")
        paths[shard] = str(path)
    if cached != paths:
        _write_json_atomic(prepared.index_path, {**index, "shard_paths": paths})
    logger.info("Cached all %d %s tracking TAR shards", len(paths), prepared.split)


def prepare_hf_tracking_split(config, split: str) -> PreparedTrackingSplit:
    """Index metadata and cache all required TAR shards for one split."""
    source = hf_tracking_source(config)
    if not source:
        raise ValueError("Tracking source format must be 'hf_webdataset'.")
    if split not in {"train", "valid", "test"}:
        raise ValueError(f"Unsupported Hugging Face tracking split: {split!r}")

    repo_id = str(source.get("repo_id") or "").strip()
    revision = str(source.get("revision") or "tracking").strip()
    cache_dir = Path(str(source.get("cache_dir") or "~/.cache/opensportslib/hf_tracking")).expanduser()
    if not repo_id or not revision:
        raise ValueError("Hugging Face tracking source requires repo_id and revision.")

    try:
        from huggingface_hub import HfApi

        resolved = str(source.get("resolved_revision") or "").strip()
        if not resolved:
            resolved = HfApi(token=True).repo_info(
                repo_id=repo_id, revision=revision, repo_type="dataset"
            ).sha
        if not resolved:
            raise ValueError("The Hub did not return a commit SHA.")
    except Exception as exc:
        raise RuntimeError(
            f"Cannot access gated dataset {repo_id!r} at {revision!r}. "
            "Request access and run `hf auth login`."
        ) from exc

    _set_resolved_revision(config, resolved)
    safe_repo = re.sub(r"[^A-Za-z0-9_.-]+", "--", repo_id)
    directory = cache_dir / safe_repo / resolved / split
    directory.mkdir(parents=True, exist_ok=True)
    annotations_path = directory / "annotations.json"
    index_path = directory / "index.json"
    prepared = PreparedTrackingSplit(
        annotations_path, index_path, repo_id, resolved, split, cache_dir
    )

    with (directory / ".prepare.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if annotations_path.is_file() and index_path.is_file():
            _stage_shards(prepared)
            return prepared
        logger.info("Staging %s tracking metadata from %s@%s", split, repo_id, resolved)

        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                    "Hugging Face tracking requires `datasets`; install with "
                    "`python -m pip install -e .`."
            ) from exc

        def rows(filename: str):
            uri = f"hf://datasets/{repo_id}@{resolved}/{split}/{filename}"
            return load_dataset(
                "parquet",
                data_files={split: uri},
                split=split,
                streaming=True,
                token=True,
                cache_dir=str(cache_dir / "datasets"),
            )

        try:
            samples = []
            locations = {}
            header = None
            for row in rows("metadata.parquet"):
                if header is None:
                    header = json.loads(row["header"])
                sample = json.loads(row["sample_payload"])
                if not isinstance(sample, dict):
                    raise ValueError("metadata.parquet contains a non-object sample_payload")
                sample_id = str(row["sample_id"])
                if sample_id in locations:
                    raise ValueError(f"Duplicate sample ID in metadata: {sample_id}")
                locations[sample_id] = (str(row["shard_name"]), sample)
                samples.append(sample)

            if not samples or not isinstance(header, dict):
                raise ValueError(f"{split}/metadata.parquet has no samples or header")

            manifest = {}
            for row in rows("shard_manifest.parquet"):
                if row.get("file_role") not in (None, "", "primary") or row.get("input_type") != "tracking_parquet":
                    continue
                key = (str(row["sample_id"]), str(row["relative_path"]))
                if row.get("status") == "ok" and row.get("wds_member"):
                    manifest[key] = (str(row["shard_name"]), str(row["wds_member"]))

            members = {}
            for sample_id, (shard_name, sample) in locations.items():
                for input_item in sample.get("inputs", []):
                    if input_item.get("type") != "tracking_parquet":
                        continue
                    path = str(input_item.get("path") or "")
                    location = manifest.get((sample_id, path))
                    if not path or location is None or location[0] != shard_name:
                        raise ValueError(f"Missing tracking TAR member for {sample_id!r}: {path!r}")
                    if path in members and members[path] != location:
                        raise ValueError(f"Tracking path maps to multiple TAR members: {path!r}")
                    members[path] = location
            if not members:
                raise ValueError(f"{split} contains no tracking_parquet inputs")

            _write_json_atomic(annotations_path, {**header, "data": samples})
            _write_json_atomic(index_path, {"members": members})
            logger.info(
                "Indexed %d %s samples across %d TAR shards",
                len(samples), split, len({shard for shard, _ in members.values()}),
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not stage {split} tracking metadata from {repo_id}@{resolved}: {exc}"
            ) from exc
        _stage_shards(prepared)

    return prepared


class HFTarTrackingReader:
    """Read one Parquet member without extracting it to the filesystem."""

    def __init__(self, prepared: PreparedTrackingSplit):
        self.prepared = prepared
        with prepared.index_path.open(encoding="utf-8") as handle:
            index = json.load(handle)
        self.members = index["members"]
        # SN-GAR training samples randomly across 13 shards. An eight-handle
        # cache repeatedly evicts and reindexes TARs (about 0.18 s per reopen).
        # Keep all of its shards open, while bounding descriptors for larger sets.
        self._max_open_tars = min(32, max(8, len({item[0] for item in self.members.values()})))
        self._pid = os.getpid()
        self._shard_paths = index.get("shard_paths", {}).copy()
        self._open_tars = OrderedDict()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_pid"] = None
        state["_open_tars"] = OrderedDict()
        return state

    def __del__(self):
        for tar in getattr(self, "_open_tars", {}).values():
            tar.close()

    def _reset_after_fork(self):
        if self._pid != os.getpid():
            for tar in self._open_tars.values():
                tar.close()
            self._pid = os.getpid()
            self._open_tars = OrderedDict()

    def _tar(self, shard_name: str):
        self._reset_after_fork()
        if shard_name in self._open_tars:
            self._open_tars.move_to_end(shard_name)
            return self._open_tars[shard_name]
        if shard_name not in self._shard_paths:
            from huggingface_hub import hf_hub_download

            try:
                logger.info("Caching tracking TAR shard %s/%s", self.prepared.split, shard_name)
                self._shard_paths[shard_name] = hf_hub_download(
                    repo_id=self.prepared.repo_id,
                    repo_type="dataset",
                    revision=self.prepared.revision,
                    filename=f"{self.prepared.split}/shards/{shard_name}",
                    cache_dir=str(self.prepared.cache_dir / "hub"),
                    token=True,
                )
            except Exception as exc:
                raise RuntimeError(f"Cannot download tracking TAR shard {shard_name!r}") from exc
        tar = tarfile.open(self._shard_paths[shard_name], mode="r:")
        self._open_tars[shard_name] = tar
        if len(self._open_tars) > self._max_open_tars:
            self._open_tars.popitem(last=False)[1].close()
        return tar

    def read_parquet(self, path: str):
        import pandas as pd

        if path not in self.members:
            raise FileNotFoundError(f"Tracking clip is absent from Hub manifest: {path}")
        shard_name, member_name = self.members[path]
        tar = self._tar(shard_name)
        try:
            member = tar.getmember(member_name)
            stream = tar.extractfile(member)
            if stream is None:
                raise KeyError(member_name)
            with stream:
                contents = stream.read()
        except KeyError as exc:
            raise FileNotFoundError(
                f"Tracking TAR member {member_name!r} is missing from {shard_name!r}"
            ) from exc
        return pd.read_parquet(io.BytesIO(contents))
