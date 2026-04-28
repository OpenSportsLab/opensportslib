"""OpenSportsLib dataset conversion and Hugging Face transfer tools."""

from .hf_transfer import (
    HF_BRANCH_KEY,
    HF_REPO_ID_KEY,
    HF_SPLIT_KEY,
    HfTransferCancelled,
    create_dataset_branch_on_hf,
    create_dataset_repo_on_hf,
    dataset_repo_exists_on_hf,
    download_dataset_split_from_hf,
    extract_local_input_upload_entries_from_json,
    extract_repo_paths_from_json,
    get_json_repo_folder,
    is_hf_download_url_not_found_error,
    is_hf_repo_not_found_error,
    is_hf_revision_not_found_error,
    read_hf_source_metadata_from_dataset,
    upload_dataset_as_parquet_to_hf,
    upload_dataset_inputs_from_json_to_hf,
    write_hf_source_metadata_to_dataset_json,
)
from .osl_json_to_parquet import DEFAULT_SHARD_SIZE, convert_json_to_parquet, parse_shard_size
from .parquet_to_osl_json import convert_parquet_to_json

__all__ = [
    "convert_json_to_parquet",
    "DEFAULT_SHARD_SIZE",
    "parse_shard_size",
    "convert_parquet_to_json",
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
