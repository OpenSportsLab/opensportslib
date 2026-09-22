"""Stage OSL JSON manifests and their selected media from a Hugging Face dataset."""

from __future__ import annotations

import fcntl
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from opensportslib.core.config.accessors import get_input_cfg


@dataclass(frozen=True)
class PreparedHFJsonSplit:
    annotation_path: Path
    source_path: Path
    revision: str


def hf_json_source(config) -> dict:
    source = get_input_cfg(config).get("source") or {}
    return source if str(source.get("format", "")).lower() == "hf_json" else {}


def _safe_path(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError(f"Unsafe path in Hugging Face JSON manifest: {value!r}")
    return path.as_posix()


def _set_resolved_revision(config, revision: str) -> None:
    inputs = config.DATA.inputs
    values = inputs.values() if isinstance(inputs, dict) else vars(inputs).values()
    for input_cfg in values:
        source = input_cfg.get("source") if isinstance(input_cfg, dict) else input_cfg.source
        if str(source.get("format") if isinstance(source, dict) else source.format).lower() == "hf_json":
            if isinstance(source, dict):
                source["resolved_revision"] = revision
            else:
                source.resolved_revision = revision
            return


def prepare_hf_json_split(config, split: str) -> PreparedHFJsonSplit:
    """Download one split's selected media, then publish its local manifest.

    The existing video loader probes each file during construction. Staging all
    videos in the requested split first therefore preserves its sampling and
    evaluation behavior. Hub blobs are linked into a small split directory.
    """
    source = hf_json_source(config)
    if not source:
        raise ValueError("Source format must be 'hf_json'.")
    if split not in {"train", "valid", "test"}:
        raise ValueError(f"Unsupported Hugging Face JSON split: {split!r}")

    repo_id = str(source.get("repo_id") or "").strip()
    branch = str(source.get("revision") or "main").strip()
    cache_dir = Path(str(source.get("cache_dir") or "~/.cache/opensportslib/hf_json")).expanduser()
    media_type = str(source.get("input_type") or "video_mp4")
    if not repo_id or not branch:
        raise ValueError("hf_json requires source.repo_id and source.revision.")

    try:
        from huggingface_hub import HfApi, hf_hub_download

        revision = str(source.get("resolved_revision") or "").strip()
        if not revision:
            revision = HfApi(token=True).repo_info(
                repo_id=repo_id, repo_type="dataset", revision=branch
            ).sha
        if not revision:
            raise ValueError("The Hub did not return a commit SHA.")
    except Exception as exc:
        raise RuntimeError(
            f"Cannot access Hugging Face dataset {repo_id!r} at {branch!r}. "
            "Request access and run `hf auth login`."
        ) from exc

    _set_resolved_revision(config, revision)
    safe_repo = re.sub(r"[^A-Za-z0-9_.-]+", "--", repo_id)
    root = cache_dir / safe_repo / revision
    root.mkdir(parents=True, exist_ok=True)
    manifest_name = _safe_path(str(source.get("annotation_pattern") or "annotations_{split}.json").format(split=split))
    annotation_path = root / f"selected_{media_type}_{split}.json"
    prepared = PreparedHFJsonSplit(annotation_path, root, revision)
    hub_cache = cache_dir / "hub"

    with (root / f".{split}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            cached_manifest = annotation_path.is_file()
            if cached_manifest:
                payload = json.loads(annotation_path.read_text(encoding="utf-8"))
            else:
                remote_manifest = hf_hub_download(
                    repo_id=repo_id, filename=manifest_name, repo_type="dataset",
                    revision=revision, token=True, cache_dir=str(hub_cache),
                )
                payload = json.loads(Path(remote_manifest).read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
                raise ValueError(f"{manifest_name} is not an OSL JSON manifest")
            if not payload["data"]:
                raise ValueError(f"{manifest_name} contains no samples")

            filenames = set()
            for sample in payload["data"]:
                expected_type = "video" if cached_manifest else media_type
                selected = [item for item in sample.get("inputs", []) if item.get("type") == expected_type]
                if len(selected) != 1:
                    raise ValueError(
                        f"Expected one {media_type!r} input for sample {sample.get('game_id', sample.get('id'))!r}"
                    )
                item = dict(selected[0])
                filename = _safe_path(str(item.get("path") or ""))
                if not filename.startswith(f"{split}/"):
                    raise ValueError(f"Media path {filename!r} is outside split {split!r}")
                filenames.add(filename)
                # The E2E video adapter expects the generic OSL video type.
                item["type"] = "video"
                sample["inputs"] = [item]

            def stage_media(filename: str) -> None:
                destination = root / filename
                if destination.is_file():
                    return
                downloaded = Path(hf_hub_download(
                    repo_id=repo_id, filename=filename, repo_type="dataset",
                    revision=revision, token=True, cache_dir=str(hub_cache),
                ))
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
                try:
                    temporary.symlink_to(downloaded)
                    os.replace(temporary, destination)
                finally:
                    temporary.unlink(missing_ok=True)

            with ThreadPoolExecutor(max_workers=min(4, len(filenames))) as executor:
                list(executor.map(stage_media, sorted(filenames)))

            if not cached_manifest:
                temporary = annotation_path.with_name(f".{annotation_path.name}.{os.getpid()}.tmp")
                try:
                    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                    os.replace(temporary, annotation_path)
                finally:
                    temporary.unlink(missing_ok=True)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to stage {split!r} from {repo_id}@{revision}: {exc}"
            ) from exc
    return prepared
