"""OpenSportsLib dataset conversion and Hugging Face transfer tools."""

from __future__ import annotations

import importlib


_EXPORTS = {
    "convert_json_to_parquet": ("opensportslib.tools.osl_json_to_parquet", "convert_json_to_parquet"),
    "DEFAULT_SHARD_SIZE": ("opensportslib.tools.osl_json_to_parquet", "DEFAULT_SHARD_SIZE"),
    "parse_shard_size": ("opensportslib.tools.osl_json_to_parquet", "parse_shard_size"),
    "convert_parquet_to_json": ("opensportslib.tools.parquet_to_osl_json", "convert_parquet_to_json"),
    "convert_sn_vqa_2026_to_osl": ("opensportslib.tools.sn_vqa_2026_qwen_native", "convert_sn_vqa_2026_to_osl"),
    "evaluate_sn_vqa_predictions": ("opensportslib.tools.sn_vqa_2026_qwen_native", "evaluate_sn_vqa_predictions"),
    "HfTransferCancelled": ("opensportslib.tools.hf_transfer", "HfTransferCancelled"),
    "HF_REPO_ID_KEY": ("opensportslib.tools.hf_transfer", "HF_REPO_ID_KEY"),
    "HF_BRANCH_KEY": ("opensportslib.tools.hf_transfer", "HF_BRANCH_KEY"),
    "HF_SPLIT_KEY": ("opensportslib.tools.hf_transfer", "HF_SPLIT_KEY"),
    "download_dataset_split_from_hf": ("opensportslib.tools.hf_transfer", "download_dataset_split_from_hf"),
    "upload_dataset_inputs_from_json_to_hf": ("opensportslib.tools.hf_transfer", "upload_dataset_inputs_from_json_to_hf"),
    "upload_dataset_as_parquet_to_hf": ("opensportslib.tools.hf_transfer", "upload_dataset_as_parquet_to_hf"),
    "create_dataset_repo_on_hf": ("opensportslib.tools.hf_transfer", "create_dataset_repo_on_hf"),
    "dataset_repo_exists_on_hf": ("opensportslib.tools.hf_transfer", "dataset_repo_exists_on_hf"),
    "create_dataset_branch_on_hf": ("opensportslib.tools.hf_transfer", "create_dataset_branch_on_hf"),
    "read_hf_source_metadata_from_dataset": ("opensportslib.tools.hf_transfer", "read_hf_source_metadata_from_dataset"),
    "write_hf_source_metadata_to_dataset_json": ("opensportslib.tools.hf_transfer", "write_hf_source_metadata_to_dataset_json"),
    "is_hf_repo_not_found_error": ("opensportslib.tools.hf_transfer", "is_hf_repo_not_found_error"),
    "is_hf_revision_not_found_error": ("opensportslib.tools.hf_transfer", "is_hf_revision_not_found_error"),
    "is_hf_download_url_not_found_error": ("opensportslib.tools.hf_transfer", "is_hf_download_url_not_found_error"),
    "get_json_repo_folder": ("opensportslib.tools.hf_transfer", "get_json_repo_folder"),
    "extract_repo_paths_from_json": ("opensportslib.tools.hf_transfer", "extract_repo_paths_from_json"),
    "extract_local_input_upload_entries_from_json": ("opensportslib.tools.hf_transfer", "extract_local_input_upload_entries_from_json"),
}

__all__ = [
    "convert_json_to_parquet",
    "DEFAULT_SHARD_SIZE",
    "parse_shard_size",
    "convert_parquet_to_json",
    "convert_sn_vqa_2026_to_osl",
    "evaluate_sn_vqa_predictions",
    "HfTransferCancelled",
    "HF_REPO_ID_KEY",
    "HF_BRANCH_KEY",
    "HF_SPLIT_KEY",
    "download_dataset_split_from_hf",
    "upload_dataset_inputs_from_json_to_hf",
    "upload_dataset_as_parquet_to_hf",
    "create_dataset_repo_on_hf",
    "dataset_repo_exists_on_hf",
    "create_dataset_branch_on_hf",
    "read_hf_source_metadata_from_dataset",
    "write_hf_source_metadata_to_dataset_json",
    "is_hf_repo_not_found_error",
    "is_hf_revision_not_found_error",
    "is_hf_download_url_not_found_error",
    "get_json_repo_folder",
    "extract_repo_paths_from_json",
    "extract_local_input_upload_entries_from_json",
]


def __getattr__(name: str):
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
