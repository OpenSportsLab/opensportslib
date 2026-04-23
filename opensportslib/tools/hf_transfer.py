import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from .osl_json_to_parquet import convert_json_to_parquet
from .parquet_to_osl_json import convert_parquet_to_json


ProgressCallback = Callable[[str], None]
CancelCheck = Callable[[], bool]

HF_SOURCE_URL_KEY = "hf_source_url"
HF_REPO_ID_KEY = "hf_repo_id"
HF_BRANCH_KEY = "hf_branch"


class HfTransferCancelled(RuntimeError):
    pass


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


def fix_hf_url(hf_url: str) -> str:
    return str(hf_url or "").replace("/blob/", "/resolve/")


def is_hf_folder_url(hf_url: str) -> bool:
    """Return True when the URL points to a folder (contains ``/tree/``)."""
    return "/tree/" in str(hf_url or "")


def parse_hf_folder_url(hf_url: str) -> tuple[str, str, str]:
    """
    Parse a HuggingFace folder URL of the form::

        https://huggingface.co/datasets/<owner>/<repo>/tree/<revision>[/<folder_path>]

    Returns ``(repo_id, revision, folder_path)``.
    ``folder_path`` may be an empty string for the repo root.
    """
    parsed = urlparse(str(hf_url or ""))
    parts = parsed.path.strip("/").split("/")

    if "datasets" in parts:
        datasets_idx = parts.index("datasets")
        parts = parts[datasets_idx + 1:]

    if len(parts) < 3 or parts[2] != "tree":
        raise ValueError(f"URL does not look like a valid HuggingFace dataset folder URL: {hf_url}")

    repo_id = f"{parts[0]}/{parts[1]}"
    revision = parts[3] if len(parts) > 3 else "main"
    folder_path = "/".join(parts[4:]) if len(parts) > 4 else ""
    return repo_id, revision, folder_path


def parse_hf_url(hf_url: str) -> tuple[str, str, str]:
    url = fix_hf_url(hf_url)
    parsed = urlparse(url)
    parts = parsed.path.strip("/").split("/")

    if "datasets" in parts:
        datasets_idx = parts.index("datasets")
        parts = parts[datasets_idx + 1 :]

    if len(parts) < 5 or parts[2] != "resolve":
        raise ValueError(f"URL does not look like a valid HuggingFace dataset file URL: {url}")

    repo_id = f"{parts[0]}/{parts[1]}"
    revision = parts[3]
    path_in_repo = "/".join(parts[4:])
    return repo_id, revision, path_in_repo


def get_json_repo_folder(path_in_repo: str) -> str:
    folder = os.path.dirname(path_in_repo)
    return folder if folder and folder != "." else ""


def parse_types_arg(types_arg: str) -> str | set[str]:
    normalized = (types_arg or "video").strip().lower()
    if normalized in ("all", "*"):
        return "all"
    return {t.strip() for t in normalized.split(",") if t.strip()}


def extract_repo_paths_from_json(osl_json: dict[str, Any], want_types: str | set[str]) -> list[str]:
    repo_paths: list[str] = []

    if "videos" in osl_json and isinstance(osl_json.get("videos"), list):
        if want_types == "all" or ("video" in want_types):
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
                if not path:
                    continue
                inp_type = str(inp.get("type", "")).strip().lower()
                if want_types == "all" or inp_type in want_types:
                    repo_paths.append(str(path).lstrip("/"))

    if not repo_paths:
        if want_types == "all":
            raise ValueError("No file paths found in the provided JSON (no inputs with 'path').")
        raise ValueError(
            f"No matching file paths found for requested types={sorted(list(want_types))}. "
            "Check your JSON schema and --types."
        )

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
    source_url: str,
    repo_id: str,
    branch: str,
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
        "source_url": str(source_url or "").strip(),
        "repo_id": str(repo_id or "").strip(),
        "branch": str(branch or "").strip(),
    }
    payload[HF_SOURCE_URL_KEY] = metadata["source_url"]
    payload[HF_REPO_ID_KEY] = metadata["repo_id"]
    payload[HF_BRANCH_KEY] = metadata["branch"]

    with open(cleaned_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    return metadata


def read_hf_source_metadata_from_dataset(dataset_json: dict[str, Any] | None) -> dict[str, str]:
    payload = dataset_json if isinstance(dataset_json, dict) else {}

    source_url = str(payload.get(HF_SOURCE_URL_KEY) or "").strip()
    repo_id = str(payload.get(HF_REPO_ID_KEY) or "").strip()
    branch = str(payload.get(HF_BRANCH_KEY) or "").strip()

    if source_url and (not repo_id or not branch):
        try:
            parsed_repo_id, parsed_branch, _ = parse_hf_url(source_url)
            if not repo_id:
                repo_id = parsed_repo_id
            if not branch:
                branch = parsed_branch
        except ValueError:
            pass

    return {
        "source_url": source_url,
        "repo_id": repo_id,
        "branch": branch,
    }


def _download_parquet_folder_and_convert(
    folder_url: str,
    output_dir: str,
    *,
    token: str | None = None,
    progress_cb: ProgressCallback | None = None,
    is_cancelled: CancelCheck | None = None,
) -> dict[str, Any]:
    """
    Download a Parquet + WebDataset folder from HF and convert it to an OSL JSON file.

    The folder is expected to have been created by
    :func:`opensportslib.tools.convert_json_to_parquet`.
    A temporary directory is used for the raw download and removed when done.
    """
    _, _, snapshot_download = _import_hf_hub()

    repo_id, revision, folder_path = parse_hf_folder_url(folder_url)
    folder_name = folder_path.rstrip("/").split("/")[-1] if folder_path else repo_id.split("/")[-1]

    os.makedirs(output_dir, exist_ok=True)
    _ensure_not_cancelled(is_cancelled)
    _emit_progress(progress_cb, f"Downloading Parquet folder '{folder_path}' from {repo_id}@{revision}...")

    allow_patterns = [f"{folder_path}/*"] if folder_path else ["*"]

    tmp_dir = tempfile.mkdtemp(prefix="hf_parquet_dl_", dir=output_dir)
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            allow_patterns=allow_patterns,
            local_dir=tmp_dir,
            token=token or None,
        )
        _ensure_not_cancelled(is_cancelled)

        parquet_dataset_dir = Path(tmp_dir) / folder_path if folder_path else Path(tmp_dir)
        output_json_path = Path(output_dir) / f"{folder_name}.json"

        _emit_progress(progress_cb, f"Converting Parquet dataset to JSON and extracting media into {output_dir}...")
        conversion_result = convert_parquet_to_json(
            dataset_dir=parquet_dataset_dir,
            output_json_path=output_json_path,
            extract_media=True,
            output_media_root=output_dir,
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
        "repo_id": repo_id,
        "revision": revision,
        "folder_path": folder_path,
        "output_dir": output_dir,
        "json_path": str(output_json_path),
        "source": "parquet_folder",
        "download_kind": "parquet",
        "num_samples": int(conversion_result.get("num_samples") or 0),
        "extracted_media": True,
        "extracted_media_count": int(conversion_result.get("extracted_media_files") or 0),
    }


def download_dataset_from_hf(
    osl_json_url: str,
    output_dir: str,
    *,
    dry_run: bool = False,
    types_arg: str = "video",
    token: str | None = None,
    progress_cb: ProgressCallback | None = None,
    is_cancelled: CancelCheck | None = None,
) -> dict[str, Any]:
    if is_hf_folder_url(osl_json_url):
        return _download_parquet_folder_and_convert(
            osl_json_url,
            output_dir,
            token=token,
            progress_cb=progress_cb,
            is_cancelled=is_cancelled,
        )

    HfApi, hf_hub_download, _ = _import_hf_hub()
    api = HfApi(token=token or None)
    want_types = parse_types_arg(types_arg)

    repo_id, revision, path_in_repo = parse_hf_url(osl_json_url)
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

    repo_paths = extract_repo_paths_from_json(osl_json, want_types)
    allow_patterns = _build_allow_patterns(repo_paths, repo_json_folder)

    result: dict[str, Any] = {
        "repo_id": repo_id,
        "revision": revision,
        "path_in_repo": path_in_repo,
        "json_path": json_path,
        "output_dir": output_dir,
        "types": types_arg,
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
        source_url=osl_json_url,
        repo_id=repo_id,
        branch=revision,
    )

    result["download_kind"] = "json"
    result["downloaded_file_count"] = downloaded_count
    result["hf_source_metadata"] = hf_source_metadata
    _emit_progress(progress_cb, "Download completed.")
    return result


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
            raw_path = str(inp.get("path") or "").strip()
            if not raw_path:
                continue

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

    json_path_in_repo = _normalize_repo_path(os.path.basename(cleaned_json_path))
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
    commit_message: str | None = None,
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
    cleaned_samples_per_shard = int(samples_per_shard or 100)
    if cleaned_samples_per_shard < 1:
        raise ValueError("samples_per_shard must be >= 1.")
    folder_name = Path(cleaned_json_path).stem
    media_root = Path(cleaned_json_path).parent

    _ensure_not_cancelled(is_cancelled)
    _emit_progress(
        progress_cb,
        f"Converting {cleaned_json_path} to Parquet + WebDataset (samples_per_shard={cleaned_samples_per_shard})...",
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
        "upload_kind": "parquet",
        "json_path": cleaned_json_path,
        "folder_name": folder_name,
        "samples_per_shard": cleaned_samples_per_shard,
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
