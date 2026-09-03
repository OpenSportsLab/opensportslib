import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

ProgressCallback = Callable[[str], None]
CancelCheck = Callable[[], bool]

HF_REPO_ID_KEY = "hf_repo_id"
HF_BRANCH_KEY = "hf_branch"
HF_SPLIT_KEY = "hf_split"
DEFAULT_SHARD_SIZE = 1_000_000_000


class HfTransferCancelled(RuntimeError):
    pass


def _import_osl_json_to_parquet():
    try:
        from .osl_json_to_parquet import DEFAULT_SHARD_SIZE as module_default_shard_size
        from .osl_json_to_parquet import convert_json_to_parquet as module_convert_json_to_parquet
        from .osl_json_to_parquet import parse_shard_size as module_parse_shard_size
    except ImportError as exc:
        raise RuntimeError(
            "Missing conversion dependencies for OSL JSON -> Parquet tools. "
            "Install the package with its data-conversion dependencies, including 'pandas' and 'pyarrow'."
        ) from exc
    return module_default_shard_size, module_convert_json_to_parquet, module_parse_shard_size


def _import_parquet_to_osl_json():
    try:
        from .parquet_to_osl_json import convert_parquet_to_json as module_convert_parquet_to_json
    except ImportError as exc:
        raise RuntimeError(
            "Missing conversion dependencies for Parquet -> OSL JSON tools. "
            "Install the package with its data-conversion dependencies, including 'pandas' and 'pyarrow'."
        ) from exc
    return module_convert_parquet_to_json


def parse_shard_size(value: int | str) -> int:
    _, _, module_parse_shard_size = _import_osl_json_to_parquet()
    return module_parse_shard_size(value)


def convert_json_to_parquet(*args, **kwargs):
    _, module_convert_json_to_parquet, _ = _import_osl_json_to_parquet()
    return module_convert_json_to_parquet(*args, **kwargs)


def convert_parquet_to_json(*args, **kwargs):
    module_convert_parquet_to_json = _import_parquet_to_osl_json()
    return module_convert_parquet_to_json(*args, **kwargs)


def _emit_progress(progress_cb: ProgressCallback | None, message: str) -> None:
    if progress_cb:
        progress_cb(message)


def _ensure_not_cancelled(is_cancelled: CancelCheck | None) -> None:
    if is_cancelled and is_cancelled():
        raise HfTransferCancelled("Transfer cancelled by user.")


def _import_hf_hub():
    try:
        from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'huggingface_hub'. Install it with: pip install huggingface_hub"
        ) from exc
    return HfApi, hf_hub_download, snapshot_download


def _import_hf_commit_operation_add():
    try:
        from huggingface_hub import CommitOperationAdd
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'huggingface_hub'. Install it with: pip install huggingface_hub"
        ) from exc
    return CommitOperationAdd


def human_size(num: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num < 1024.0:
            return f"{num:3.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} PB"


def get_json_repo_folder(path_in_repo: str) -> str:
    folder = os.path.dirname(path_in_repo)
    return folder if folder and folder != "." else ""


def extract_repo_paths_from_json(
    osl_json: dict[str, Any],
) -> list[str]:
    repo_paths: list[str] = []

    if "videos" in osl_json and isinstance(osl_json.get("videos"), list):
        for item in osl_json.get("videos", []):
            if isinstance(item, dict) and item.get("path"):
                repo_paths.append(str(item["path"]).lstrip("/"))

    if "data" in osl_json and isinstance(osl_json.get("data"), list):
        for sample in osl_json.get("data", []):
            inputs = sample.get("inputs", []) if isinstance(sample, dict) else []
            for inp in inputs:
                if not isinstance(inp, dict):
                    continue
                path = inp.get("path")
                if path:
                    repo_paths.append(str(path).lstrip("/"))
                # player_joints_h5 / player_centroids_h5 inputs may carry a
                # sidecar ball_path; download it alongside the primary input.
                ball_path = inp.get("ball_path")
                if ball_path:
                    repo_paths.append(str(ball_path).lstrip("/"))

    if not repo_paths:
        raise ValueError("No file paths found in the provided JSON (no inputs with 'path').")

    return repo_paths


def _build_allow_patterns(repo_paths: list[str], repo_json_folder: str) -> list[str]:
    def _full_repo_path(rel_path: str) -> str:
        rel_path = rel_path.lstrip("/")
        if repo_json_folder:
            prefix = repo_json_folder.rstrip("/") + "/"
            if not rel_path.startswith(prefix):
                return prefix + rel_path
        return rel_path

    return sorted(set(_full_repo_path(path) for path in repo_paths))


def write_hf_source_metadata_to_dataset_json(
    dataset_json_path: str,
    *,
    repo_id: str,
    branch: str,
    split: str = "",
) -> dict[str, str]:
    cleaned_path = os.path.abspath(str(dataset_json_path or "").strip())
    if not cleaned_path:
        raise ValueError("dataset_json_path is required.")
    if not os.path.isfile(cleaned_path):
        raise ValueError(f"JSON file does not exist: {cleaned_path}")

    with open(cleaned_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Invalid dataset JSON: expected root object.")

    metadata = {
        "repo_id": str(repo_id or "").strip(),
        "branch": str(branch or "").strip(),
        "split": str(split or "").strip(),
    }
    payload[HF_REPO_ID_KEY] = metadata["repo_id"]
    payload[HF_BRANCH_KEY] = metadata["branch"]
    payload[HF_SPLIT_KEY] = metadata["split"]

    with open(cleaned_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    return metadata


def read_hf_source_metadata_from_dataset(dataset_json: dict[str, Any] | None) -> dict[str, str]:
    payload = dataset_json if isinstance(dataset_json, dict) else {}

    return {
        "repo_id": str(payload.get(HF_REPO_ID_KEY) or "").strip(),
        "branch": str(payload.get(HF_BRANCH_KEY) or "").strip(),
        "split": str(payload.get(HF_SPLIT_KEY) or "").strip(),
    }


def _clean_hf_split(split: str) -> str:
    cleaned_split = _normalize_repo_path(split)
    if not cleaned_split:
        raise ValueError("split is required.")
    if cleaned_split.endswith(".json"):
        cleaned_split = cleaned_split[:-5]
    return cleaned_split


def _build_split_output_dir(output_dir: str, revision: str, split: str) -> str:
    cleaned_output_dir = str(output_dir or "").strip()
    if not cleaned_output_dir:
        raise ValueError("output_dir is required.")
    cleaned_revision = _normalize_repo_path(revision) or "main"
    cleaned_split = _clean_hf_split(split)
    return str(Path(cleaned_output_dir) / cleaned_revision / cleaned_split)


def _download_parquet_split_and_convert(
    repo_id: str,
    revision: str,
    split: str,
    output_dir: str,
    *,
    token: str | None = None,
    progress_cb: ProgressCallback | None = None,
    is_cancelled: CancelCheck | None = None,
) -> dict[str, Any]:
    cleaned_repo_id = str(repo_id or "").strip()
    cleaned_revision = str(revision or "").strip() or "main"
    cleaned_split = _clean_hf_split(split)
    if not cleaned_repo_id:
        raise ValueError("repo_id is required.")

    os.makedirs(output_dir, exist_ok=True)
    output_json_path = Path(output_dir) / f"{cleaned_split}.json"
    if output_json_path.is_file():
        _emit_progress(
            progress_cb,
            f"JSON already exists at {output_json_path}; skipping Parquet/WebDataset download and conversion.",
        )
        return {
            "repo_id": cleaned_repo_id,
            "revision": cleaned_revision,
            "split": cleaned_split,
            "folder_path": cleaned_split,
            "output_dir": output_dir,
            "json_path": str(output_json_path),
            "source": "parquet_split",
            "download_kind": "parquet",
            "downloaded_file_count": 0,
            "download_skipped": True,
            "extracted_media": True,
            "extracted_media_count": 0,
            "hf_source_metadata": {
                "repo_id": cleaned_repo_id,
                "branch": cleaned_revision,
                "split": cleaned_split,
            },
        }

    _ensure_not_cancelled(is_cancelled)
    _emit_progress(progress_cb, f"Downloading Parquet split '{cleaned_split}' from {cleaned_repo_id}@{cleaned_revision}...")

    _, _, snapshot_download = _import_hf_hub()
    tmp_dir = tempfile.mkdtemp(prefix="hf_parquet_dl_", dir=output_dir)
    try:
        snapshot_download(
            repo_id=cleaned_repo_id,
            repo_type="dataset",
            revision=cleaned_revision,
            allow_patterns=[f"{cleaned_split}/*"],
            local_dir=tmp_dir,
            token=token or None,
        )
        _ensure_not_cancelled(is_cancelled)

        parquet_dataset_dir = Path(tmp_dir) / cleaned_split
        _emit_progress(progress_cb, f"Converting Parquet split to JSON and extracting media into {output_dir}...")
        conversion_result = convert_parquet_to_json(
            dataset_dir=parquet_dataset_dir,
            output_json_path=output_json_path,
            extract_media=True,
            output_media_root=output_dir,
        )
        write_hf_source_metadata_to_dataset_json(
            str(output_json_path),
            repo_id=cleaned_repo_id,
            branch=cleaned_revision,
            split=cleaned_split,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    _emit_progress(
        progress_cb,
        (
            f"Conversion complete. JSON saved to {output_json_path}. "
            f"Extracted {conversion_result.get('extracted_media_files', 0)} media files."
        ),
    )
    return {
        "repo_id": cleaned_repo_id,
        "revision": cleaned_revision,
        "split": cleaned_split,
        "folder_path": cleaned_split,
        "output_dir": output_dir,
        "json_path": str(output_json_path),
        "source": "parquet_split",
        "download_kind": "parquet",
        "download_skipped": False,
        "num_samples": int(conversion_result.get("num_samples") or 0),
        "extracted_media": True,
        "extracted_media_count": int(conversion_result.get("extracted_media_files") or 0),
        "hf_source_metadata": {
            "repo_id": cleaned_repo_id,
            "branch": cleaned_revision,
            "split": cleaned_split,
        },
    }


def _download_json_path_from_hf(
    repo_id: str,
    revision: str,
    split: str,
    output_dir: str,
    *,
    dry_run: bool = False,
    token: str | None = None,
    progress_cb: ProgressCallback | None = None,
    is_cancelled: CancelCheck | None = None,
) -> dict[str, Any]:
    HfApi, hf_hub_download, _ = _import_hf_hub()
    api = HfApi(token=token or None)
    cleaned_split = _clean_hf_split(split)
    path_in_repo = f"{cleaned_split}.json"
    repo_json_folder = get_json_repo_folder(path_in_repo)

    os.makedirs(output_dir, exist_ok=True)
    _ensure_not_cancelled(is_cancelled)
    _emit_progress(progress_cb, f"Downloading JSON from {repo_id}@{revision}: {path_in_repo}")

    json_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=path_in_repo,
        revision=revision,
        local_dir=output_dir,
        local_dir_use_symlinks=False,
        token=token or None,
    )

    _ensure_not_cancelled(is_cancelled)
    with open(json_path, "r", encoding="utf-8") as handle:
        osl_json = json.load(handle)

    repo_paths = extract_repo_paths_from_json(osl_json)
    allow_patterns = _build_allow_patterns(repo_paths, repo_json_folder)

    result: dict[str, Any] = {
        "repo_id": repo_id,
        "revision": revision,
        "split": cleaned_split,
        "path_in_repo": path_in_repo,
        "json_path": json_path,
        "output_dir": output_dir,
        "dry_run": bool(dry_run),
        "referenced_file_count": len(allow_patterns),
    }

    if dry_run:
        _emit_progress(progress_cb, "Collecting repository file metadata for dry-run.")

        size_lookup: dict[str, int] = {}
        try:
            info_obj = api.repo_info(
                repo_id=repo_id,
                revision=revision,
                repo_type="dataset",
                files_metadata=True,
            )
            size_lookup = {
                sibling.rfilename: sibling.size
                for sibling in getattr(info_obj, "siblings", [])
                if getattr(sibling, "rfilename", None)
            }
        except Exception:
            size_lookup = {}

        files = []
        missing_files = []
        total_size = 0
        for full_repo_path in allow_patterns:
            _ensure_not_cancelled(is_cancelled)
            local_path = os.path.join(output_dir, full_repo_path)
            size = size_lookup.get(full_repo_path)
            if isinstance(size, int):
                total_size += size
            else:
                missing_files.append(full_repo_path)
            files.append(
                {
                    "repo_path": full_repo_path,
                    "local_path": local_path,
                    "size_bytes": size,
                    "size_human": human_size(size) if isinstance(size, int) else "Not found",
                }
            )

        result.update(
            {
                "files": files,
                "missing_files": missing_files,
                "estimated_total_size_bytes": total_size,
                "estimated_total_size_human": human_size(total_size),
            }
        )
        _emit_progress(progress_cb, f"Dry-run complete. Matched {len(allow_patterns)} files.")
        return result

    _emit_progress(progress_cb, f"Downloading {len(allow_patterns)} referenced files.")
    downloaded_count = 0
    for idx, full_repo_path in enumerate(allow_patterns, start=1):
        _ensure_not_cancelled(is_cancelled)
        _emit_progress(progress_cb, f"[{idx}/{len(allow_patterns)}] Downloading {full_repo_path}")
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=full_repo_path,
            revision=revision,
            local_dir=output_dir,
            local_dir_use_symlinks=False,
            token=token or None,
        )
        downloaded_count += 1

    _emit_progress(progress_cb, "Persisting Hugging Face source metadata into downloaded JSON.")
    hf_source_metadata = write_hf_source_metadata_to_dataset_json(
        json_path,
        repo_id=repo_id,
        branch=revision,
        split=cleaned_split,
    )

    result["download_kind"] = "json"
    result["downloaded_file_count"] = downloaded_count
    result["hf_source_metadata"] = hf_source_metadata
    _emit_progress(progress_cb, "Download completed.")
    return result


_PREFERRED_SPLIT_ORDER = ["train", "valid", "val", "validation", "test", "challenge"]
_NON_SPLIT_JSON_FILES = {"dataset_infos.json", "dataset_dict.json"}


def _sort_splits(splits: set[str]) -> list[str]:
    def _sort_key(name: str) -> tuple[int, str]:
        try:
            rank = _PREFERRED_SPLIT_ORDER.index(name.lower())
        except ValueError:
            rank = len(_PREFERRED_SPLIT_ORDER)
        return (rank, name.lower())

    return sorted(splits, key=_sort_key)


def list_dataset_branches_on_hf(
    repo_id: str,
    *,
    token: str | None = None,
) -> list[str]:
    cleaned_repo_id = str(repo_id or "").strip()
    if not cleaned_repo_id:
        raise ValueError("repo_id is required.")

    HfApi, _, _ = _import_hf_hub()
    api = HfApi(token=token or None)
    refs = api.list_repo_refs(cleaned_repo_id, repo_type="dataset")
    branch_names = [str(branch.name) for branch in getattr(refs, "branches", [])]

    unique_names = sorted(set(branch_names))
    if "main" in unique_names:
        unique_names.remove("main")
        return ["main"] + unique_names
    return unique_names


def list_dataset_splits_on_hf(
    repo_id: str,
    revision: str,
    *,
    token: str | None = None,
) -> dict[str, Any]:
    cleaned_repo_id = str(repo_id or "").strip()
    cleaned_revision = str(revision or "").strip() or "main"
    if not cleaned_repo_id:
        raise ValueError("repo_id is required.")

    HfApi, _, _ = _import_hf_hub()
    api = HfApi(token=token or None)
    repo_files = api.list_repo_files(
        cleaned_repo_id,
        revision=cleaned_revision,
        repo_type="dataset",
    )

    parquet_splits: set[str] = set()
    json_splits: set[str] = set()
    for path in repo_files:
        normalized = _normalize_repo_path(path)
        if "/" in normalized:
            folder, filename = normalized.split("/", 1)
            # Only the canonical Parquet+WebDataset export layout counts as a
            # parquet split (produced by convert_json_to_parquet / expected by
            # convert_parquet_to_json): `{split}/metadata.parquet` plus TAR
            # shards under `{split}/shards/`. A JSON-format dataset can also
            # reference arbitrary `.parquet` media files (e.g. tensor-encoded
            # videos) under a folder that happens to share the split's name,
            # so a loose "any .parquet/.tar anywhere under this folder" check
            # would misclassify those as Parquet+WebDataset splits.
            if folder and (
                filename == "metadata.parquet"
                or (filename.startswith("shards/") and filename.lower().endswith(".tar"))
            ):
                parquet_splits.add(folder)
        elif normalized.lower().endswith(".json") and normalized not in _NON_SPLIT_JSON_FILES:
            json_splits.add(normalized[: -len(".json")])

    if parquet_splits:
        return {"format": "parquet", "splits": _sort_splits(parquet_splits)}
    if json_splits:
        return {"format": "json", "splits": _sort_splits(json_splits)}
    return {"format": None, "splits": []}


def download_dataset_splits_from_hf(
    repo_id: str,
    revision: str,
    splits: list[str],
    output_dir: str,
    *,
    download_format: str = "parquet",
    dry_run: bool = False,
    token: str | None = None,
    progress_cb: ProgressCallback | None = None,
    is_cancelled: CancelCheck | None = None,
) -> list[dict[str, Any]]:
    cleaned_splits = [str(split or "").strip() for split in (splits or [])]
    cleaned_splits = [split for split in cleaned_splits if split]
    if not cleaned_splits:
        raise ValueError("At least one split is required.")

    total = len(cleaned_splits)
    results: list[dict[str, Any]] = []
    for idx, split in enumerate(cleaned_splits, start=1):
        _ensure_not_cancelled(is_cancelled)

        def _scoped_progress(message: str, _idx: int = idx, _split: str = split) -> None:
            _emit_progress(progress_cb, f"[{_idx}/{total}] {_split}: {message}")

        result = download_dataset_split_from_hf(
            repo_id,
            revision,
            split,
            output_dir,
            download_format=download_format,
            dry_run=dry_run,
            token=token,
            progress_cb=_scoped_progress,
            is_cancelled=is_cancelled,
        )
        results.append(result)

    return results


def download_dataset_split_from_hf(
    repo_id: str,
    revision: str,
    split: str,
    output_dir: str,
    *,
    download_format: str = "parquet",
    dry_run: bool = False,
    token: str | None = None,
    progress_cb: ProgressCallback | None = None,
    is_cancelled: CancelCheck | None = None,
) -> dict[str, Any]:
    cleaned_repo_id = str(repo_id or "").strip()
    cleaned_revision = str(revision or "").strip() or "main"
    cleaned_split = _clean_hf_split(split)
    cleaned_format = str(download_format or "parquet").strip().lower()
    if cleaned_format not in {"json", "parquet"}:
        raise ValueError("download_format must be 'json' or 'parquet'.")
    if not cleaned_repo_id:
        raise ValueError("repo_id is required.")
    split_output_dir = _build_split_output_dir(output_dir, cleaned_revision, cleaned_split)

    if cleaned_format == "parquet":
        if dry_run:
            raise ValueError("dry_run is only supported for JSON split downloads.")
        return _download_parquet_split_and_convert(
            cleaned_repo_id,
            cleaned_revision,
            cleaned_split,
            split_output_dir,
            token=token,
            progress_cb=progress_cb,
            is_cancelled=is_cancelled,
        )

    return _download_json_path_from_hf(
        cleaned_repo_id,
        cleaned_revision,
        cleaned_split,
        split_output_dir,
        dry_run=dry_run,
        token=token,
        progress_cb=progress_cb,
        is_cancelled=is_cancelled,
    )


def _normalize_repo_path(path: str) -> str:
    return str(path or "").strip().replace("\\", "/").lstrip("/")


def extract_local_input_upload_entries_from_json(dataset_json_path: str) -> list[dict[str, str]]:
    cleaned_json_path = os.path.abspath(str(dataset_json_path or "").strip())
    if not cleaned_json_path:
        raise ValueError("json_path is required.")
    if not os.path.isfile(cleaned_json_path):
        raise ValueError(f"JSON file does not exist: {cleaned_json_path}")

    with open(cleaned_json_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    data_items = payload.get("data", []) if isinstance(payload, dict) else []
    if not isinstance(data_items, list):
        raise ValueError("Invalid dataset JSON: expected top-level 'data' list.")

    base_dir = os.path.dirname(cleaned_json_path)
    entries: list[dict[str, str]] = []
    for sample in data_items:
        if not isinstance(sample, dict):
            continue
        inputs = sample.get("inputs", [])
        if not isinstance(inputs, list):
            continue

        for inp in inputs:
            if not isinstance(inp, dict):
                continue

            raw_paths = []
            path = str(inp.get("path") or "").strip()
            if path:
                raw_paths.append(path)
            # player_joints_h5 / player_centroids_h5 inputs may carry a sidecar
            # ball_path pointing at a separate ball-tracking h5 file; include it
            # alongside the primary input when present.
            ball_path = str(inp.get("ball_path") or "").strip()
            if ball_path:
                raw_paths.append(ball_path)

            for raw_path in raw_paths:
                local_path = raw_path if os.path.isabs(raw_path) else os.path.join(base_dir, raw_path)
                local_path = os.path.abspath(local_path)
                if not os.path.isfile(local_path):
                    raise FileNotFoundError(
                        f"Input file from dataset JSON not found on disk: {raw_path} (resolved: {local_path})"
                    )

                path_in_repo = _normalize_repo_path(raw_path)
                if not path_in_repo:
                    raise ValueError(f"Invalid input path in dataset JSON: {raw_path}")

                entries.append(
                    {
                        "local_path": local_path,
                        "path_in_repo": path_in_repo,
                    }
                )

    if not entries:
        raise ValueError("No valid data[].inputs[].path entries found in the provided dataset JSON.")
    return entries


def upload_dataset_inputs_from_json_to_hf(
    repo_id: str,
    json_path: str,
    *,
    revision: str | None = "main",
    split: str | None = None,
    commit_message: str | None = None,
    token: str | None = None,
    progress_cb: ProgressCallback | None = None,
    is_cancelled: CancelCheck | None = None,
) -> dict[str, Any]:
    HfApi, _, _ = _import_hf_hub()
    CommitOperationAdd = _import_hf_commit_operation_add()

    cleaned_repo_id = str(repo_id or "").strip()
    cleaned_json_path = os.path.abspath(str(json_path or "").strip())
    if not cleaned_repo_id:
        raise ValueError("repo_id is required.")
    if not cleaned_json_path:
        raise ValueError("json_path is required.")
    if not os.path.isfile(cleaned_json_path):
        raise ValueError(f"JSON file does not exist: {cleaned_json_path}")
    cleaned_revision = str(revision or "").strip() or "main"

    effective_commit_message = (commit_message or "").strip() or "Upload dataset inputs from JSON"
    input_upload_entries = extract_local_input_upload_entries_from_json(cleaned_json_path)
    unique_input_entries: list[dict[str, str]] = []
    input_entry_by_repo_path: dict[str, dict[str, str]] = {}
    duplicate_input_refs = 0
    for entry in input_upload_entries:
        path_in_repo = entry["path_in_repo"]
        existing = input_entry_by_repo_path.get(path_in_repo)
        if existing is None:
            input_entry_by_repo_path[path_in_repo] = entry
            unique_input_entries.append(entry)
            continue

        duplicate_input_refs += 1
        if os.path.abspath(existing["local_path"]) != os.path.abspath(entry["local_path"]):
            raise ValueError(
                "Conflicting local files mapped to the same repo path "
                f"'{path_in_repo}': '{existing['local_path']}' vs '{entry['local_path']}'."
            )

    if duplicate_input_refs:
        _emit_progress(
            progress_cb,
            (
                f"Deduplicated {duplicate_input_refs} repeated input references "
                f"into {len(unique_input_entries)} unique repo paths."
            ),
        )

    cleaned_split = _clean_hf_split(split) if split else ""
    json_path_in_repo = f"{cleaned_split}.json" if cleaned_split else _normalize_repo_path(os.path.basename(cleaned_json_path))
    if not json_path_in_repo:
        json_path_in_repo = "dataset.json"

    existing_repo_paths = {entry["path_in_repo"] for entry in unique_input_entries}
    json_already_listed_in_inputs = json_path_in_repo in existing_repo_paths
    upload_entries = [
        {
            "local_path": cleaned_json_path,
            "path_in_repo": json_path_in_repo,
        }
    ]
    # Always place dataset JSON first in the commit operations list.
    upload_entries.extend(
        entry for entry in unique_input_entries if entry["path_in_repo"] != json_path_in_repo
    )

    _ensure_not_cancelled(is_cancelled)
    api = HfApi(token=token or None)
    _emit_progress(
        progress_cb,
        (
            f"Preparing batched upload of {len(upload_entries)} files to {cleaned_repo_id}@{cleaned_revision} "
            f"(dataset JSON + {len(unique_input_entries)} unique inputs) from {cleaned_json_path}"
        ),
    )

    operations = []
    for idx, entry in enumerate(upload_entries, start=1):
        _ensure_not_cancelled(is_cancelled)
        _emit_progress(progress_cb, f"[{idx}/{len(upload_entries)}] Queueing {entry['path_in_repo']}")
        operations.append(
            CommitOperationAdd(
                path_in_repo=entry["path_in_repo"],
                path_or_fileobj=entry["local_path"],
            )
        )

    _ensure_not_cancelled(is_cancelled)
    _emit_progress(progress_cb, f"Submitting one Hugging Face commit with {len(operations)} files...")
    commit_info = api.create_commit(
        repo_id=cleaned_repo_id,
        repo_type="dataset",
        revision=cleaned_revision,
        operations=operations,
        commit_message=effective_commit_message,
    )
    commit_ref = (
        str(getattr(commit_info, "oid", "") or "").strip()
        or str(getattr(commit_info, "commit_id", "") or "").strip()
        or str(getattr(commit_info, "commit_url", "") or "").strip()
        or str(commit_info)
    )

    _emit_progress(progress_cb, f"Upload completed in one commit. Uploaded {len(upload_entries)} files.")
    return {
        "repo_id": cleaned_repo_id,
        "repo_type": "dataset",
        "upload_kind": "json",
        "json_path": cleaned_json_path,
        "revision": cleaned_revision,
        "split": cleaned_split,
        "json_path_in_repo": json_path_in_repo,
        "input_file_count": len(input_upload_entries),
        "unique_input_file_count": len(unique_input_entries),
        "uploaded_file_count": len(upload_entries),
        "uploaded_json_separately": not json_already_listed_in_inputs,
        "commit_message": effective_commit_message,
        "commit_ref": commit_ref,
    }


def is_hf_repo_not_found_error(error_message: str) -> bool:
    text = str(error_message or "").strip().lower()
    if not text:
        return False

    if "repository not found" in text:
        return True
    if "reponotfounderror" in text:
        return True
    return (
        "404 client error" in text
        and "/api/datasets/" in text
        and "preupload" in text
    )


def is_hf_revision_not_found_error(error_message: str) -> bool:
    text = str(error_message or "").strip().lower()
    if not text:
        return False
    if "revision not found" in text:
        return True
    if "revisionnotfounderror" in text:
        return True
    return (
        "404 client error" in text
        and "preupload" in text
        and "/api/datasets/" in text
        and "repository not found" not in text
    )


def is_hf_download_url_not_found_error(error_message: str) -> bool:
    text = str(error_message or "").strip().lower()
    if not text:
        return False
    if "404 client error" not in text:
        return False
    if "entry not found" in text:
        return True
    if "repository not found" in text:
        return True
    if "revision not found" in text:
        return True
    return "not found for url" in text and "huggingface.co" in text


def upload_dataset_as_parquet_to_hf(
    repo_id: str,
    json_path: str,
    *,
    revision: str | None = "main",
    split: str | None = None,
    commit_message: str | None = None,
    shard_mode: str = "size",
    shard_size: int | str = DEFAULT_SHARD_SIZE,
    samples_per_shard: int = 100,
    token: str | None = None,
    progress_cb: ProgressCallback | None = None,
    is_cancelled: CancelCheck | None = None,
) -> dict[str, Any]:
    """
    Convert an OSL JSON dataset to Parquet + WebDataset format and upload the result
    to a HuggingFace dataset repository under a folder named after the JSON file stem.

    For example, ``annotations_test.json`` is converted and uploaded to the
    ``annotations_test/`` folder on the repository.

    A temporary directory is used for the conversion output and removed when done.
    """
    HfApi, _, _ = _import_hf_hub()
    CommitOperationAdd = _import_hf_commit_operation_add()

    cleaned_repo_id = str(repo_id or "").strip()
    cleaned_json_path = os.path.abspath(str(json_path or "").strip())
    if not cleaned_repo_id:
        raise ValueError("repo_id is required.")
    if not os.path.isfile(cleaned_json_path):
        raise ValueError(f"JSON file does not exist: {cleaned_json_path}")

    cleaned_revision = str(revision or "").strip() or "main"
    effective_commit_message = (commit_message or "").strip() or "Upload dataset as Parquet + WebDataset"
    cleaned_shard_mode = str(shard_mode or "size").strip().lower()
    if cleaned_shard_mode not in {"size", "samples"}:
        raise ValueError("shard_mode must be either 'size' or 'samples'.")
    cleaned_shard_size = parse_shard_size(shard_size)
    cleaned_samples_per_shard = int(samples_per_shard or 100)
    if cleaned_samples_per_shard < 1:
        raise ValueError("samples_per_shard must be >= 1.")
    cleaned_split = _clean_hf_split(split) if split else ""
    folder_name = cleaned_split or Path(cleaned_json_path).stem
    media_root = Path(cleaned_json_path).parent

    _ensure_not_cancelled(is_cancelled)
    shard_desc = (
        f"shard_size={cleaned_shard_size}"
        if cleaned_shard_mode == "size"
        else f"samples_per_shard={cleaned_samples_per_shard}"
    )
    _emit_progress(
        progress_cb,
        f"Converting {cleaned_json_path} to Parquet + WebDataset ({shard_desc})...",
    )

    conversion_result: dict[str, Any] = {}
    total = 0
    commit_ref = ""
    tmp_dir = tempfile.mkdtemp(prefix="hf_parquet_ul_")
    try:
        parquet_output = Path(tmp_dir) / folder_name
        conversion_result = convert_json_to_parquet(
            json_path=cleaned_json_path,
            media_root=media_root,
            output_dir=parquet_output,
            shard_mode=cleaned_shard_mode,
            shard_size=cleaned_shard_size,
            samples_per_shard=cleaned_samples_per_shard,
            missing_policy="skip",
            overwrite=True,
        )

        _ensure_not_cancelled(is_cancelled)

        # Collect all files to upload, preserving sub-paths under folder_name/
        upload_entries: list[dict[str, str]] = []
        for local_file in sorted(parquet_output.rglob("*")):
            if not local_file.is_file():
                continue
            rel = local_file.relative_to(tmp_dir).as_posix()
            upload_entries.append({"local_path": str(local_file), "path_in_repo": rel})

        api = HfApi(token=token or None)
        total = len(upload_entries)
        _emit_progress(
            progress_cb,
            f"Preparing batched parquet upload of {total} files to {cleaned_repo_id}@{cleaned_revision} under '{folder_name}/'..."
        )

        operations = []
        for idx, entry in enumerate(upload_entries, start=1):
            _ensure_not_cancelled(is_cancelled)
            _emit_progress(progress_cb, f"[{idx}/{total}] Queueing {entry['path_in_repo']}")
            operations.append(
                CommitOperationAdd(
                    path_in_repo=entry["path_in_repo"],
                    path_or_fileobj=entry["local_path"],
                )
            )

        _ensure_not_cancelled(is_cancelled)
        _emit_progress(progress_cb, f"Submitting one Hugging Face commit with {len(operations)} parquet files...")
        commit_info = api.create_commit(
            repo_id=cleaned_repo_id,
            repo_type="dataset",
            revision=cleaned_revision,
            operations=operations,
            commit_message=effective_commit_message,
        )
        commit_ref = (
            str(getattr(commit_info, "oid", "") or "").strip()
            or str(getattr(commit_info, "commit_id", "") or "").strip()
            or str(getattr(commit_info, "commit_url", "") or "").strip()
            or str(commit_info)
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    _emit_progress(progress_cb, f"Parquet upload completed. Uploaded {total} files.")
    input_file_count = int(conversion_result.get("input_files_added") or 0)
    return {
        "repo_id": cleaned_repo_id,
        "revision": cleaned_revision,
        "split": cleaned_split,
        "upload_kind": "parquet",
        "json_path": cleaned_json_path,
        "folder_name": folder_name,
        "shard_mode": cleaned_shard_mode,
        "shard_size": cleaned_shard_size,
        "samples_per_shard": cleaned_samples_per_shard,
        "num_shards": int(conversion_result.get("num_shards") or 0),
        "num_samples": int(conversion_result.get("num_samples") or 0),
        "input_file_count": input_file_count,
        "uploaded_file_count": total,
        "commit_message": effective_commit_message,
        "commit_ref": commit_ref,
    }


def create_dataset_repo_on_hf(
    repo_id: str,
    *,
    token: str | None = None,
    private: bool = False,
    progress_cb: ProgressCallback | None = None,
) -> dict[str, Any]:
    cleaned_repo_id = str(repo_id or "").strip()
    if not cleaned_repo_id:
        raise ValueError("repo_id is required.")

    HfApi, _, _ = _import_hf_hub()
    api = HfApi(token=token or None)

    _emit_progress(progress_cb, f"Creating Hugging Face dataset repository: {cleaned_repo_id}")
    repo_url = api.create_repo(
        repo_id=cleaned_repo_id,
        repo_type="dataset",
        private=bool(private),
        exist_ok=True,
    )
    _emit_progress(progress_cb, f"Repository is ready: {cleaned_repo_id}")

    return {
        "repo_id": cleaned_repo_id,
        "repo_type": "dataset",
        "repo_url": str(repo_url),
    }


def dataset_repo_exists_on_hf(
    repo_id: str,
    *,
    token: str | None = None,
) -> bool:
    cleaned_repo_id = str(repo_id or "").strip()
    if not cleaned_repo_id:
        raise ValueError("repo_id is required.")

    HfApi, _, _ = _import_hf_hub()
    api = HfApi(token=token or None)
    try:
        api.repo_info(repo_id=cleaned_repo_id, repo_type="dataset")
        return True
    except Exception as exc:
        if is_hf_repo_not_found_error(str(exc)):
            return False
        raise


def create_dataset_branch_on_hf(
    repo_id: str,
    branch: str,
    *,
    source_revision: str = "main",
    token: str | None = None,
    progress_cb: ProgressCallback | None = None,
) -> dict[str, Any]:
    cleaned_repo_id = str(repo_id or "").strip()
    cleaned_branch = str(branch or "").strip()
    cleaned_source_revision = str(source_revision or "").strip() or "main"
    if not cleaned_repo_id:
        raise ValueError("repo_id is required.")
    if not cleaned_branch:
        raise ValueError("branch is required.")

    HfApi, _, _ = _import_hf_hub()
    api = HfApi(token=token or None)

    # Resolve the actual commit to branch from.
    # Always use the oldest (initial) commit so the new branch is never empty —
    # branching from HEAD would create an empty ref on a repo with no files yet.
    _emit_progress(
        progress_cb,
        f"Resolving initial commit for {cleaned_repo_id} to use as branch base...",
    )
    commits = api.list_repo_commits(cleaned_repo_id, repo_type="dataset")
    initial_commit_id = commits[-1].commit_id if commits else cleaned_source_revision

    _emit_progress(
        progress_cb,
        (
            f"Creating Hugging Face dataset branch '{cleaned_branch}' "
            f"from initial commit {initial_commit_id!r} in {cleaned_repo_id}"
        ),
    )
    api.create_branch(
        repo_id=cleaned_repo_id,
        repo_type="dataset",
        branch=cleaned_branch,
        revision=initial_commit_id,
        exist_ok=True,
    )
    _emit_progress(progress_cb, f"Branch is ready: {cleaned_repo_id}@{cleaned_branch}")

    return {
        "repo_id": cleaned_repo_id,
        "repo_type": "dataset",
        "branch": cleaned_branch,
        "source_revision": initial_commit_id,
    }
