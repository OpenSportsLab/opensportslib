# Tool and Utility Reference

> **Public / stable:** names exported from `opensportslib.tools` are supported.
> Implementation modules and helpers that are not exported are **internal**.

## Side effects and requirements

| Tool group | Main exports | Side effect / requirement |
| --- | --- | --- |
| OSL JSON → Parquet/WebDataset | `convert_json_to_parquet`, `parse_shard_size` | Reads an OSL manifest and media; writes metadata, tar shards, and Parquet output. |
| Parquet/WebDataset → OSL JSON | `convert_parquet_to_json`, `convert_parquet_metadata_to_json` | Reads local Parquet/TAR data and writes a reconstructed JSON document. |
| Hugging Face transfer | `download_dataset_*`, `upload_dataset_*`, repository helpers | Uses network and Hub credentials as needed; downloads media or mutates Hub repository contents. |
| SN VQA helpers | `convert_sn_vqa_2026_to_osl`, `evaluate_sn_vqa_predictions` | Converts/evaluates the supported SoccerNet VQA workflow. |

The remaining stable helpers work with source-history metadata and repositories:
`read_hf_source_metadata_from_dataset`,
`write_hf_source_metadata_to_dataset_json`, `get_json_repo_folder`,
`extract_repo_paths_from_json`, `extract_local_input_upload_entries_from_json`,
`is_hf_repo_not_found_error`, `is_hf_revision_not_found_error`, and
`is_hf_download_url_not_found_error`. The `HF_*_KEY` constants identify the
stored source provenance fields. They do not perform I/O on their own.

## Complete stable export catalog

| Export group | Stable names |
| --- | --- |
| Conversion | `DEFAULT_SHARD_SIZE`, `parse_shard_size`, `convert_json_to_parquet`, `convert_parquet_to_json`, `convert_parquet_metadata_to_json` |
| SoccerNet VQA | `convert_sn_vqa_2026_to_osl`, `evaluate_sn_vqa_predictions` |
| Transfer errors | `HfTransferCancelled`, `MissingDatasetInputsError` |
| Download/inspection | `download_dataset_split_from_hf`, `download_dataset_sample_inputs_from_hf`, `download_dataset_missing_inputs_from_hf`, `find_missing_dataset_inputs` |
| Upload/repository management | `upload_dataset_inputs_from_json_to_hf`, `upload_dataset_as_parquet_to_hf`, `create_dataset_repo_on_hf`, `dataset_repo_exists_on_hf`, `create_dataset_branch_on_hf` |
| Provenance and path helpers | `HF_REPO_ID_KEY`, `HF_BRANCH_KEY`, `HF_SPLIT_KEY`, `HF_FORMAT_KEY`, `HF_COMMIT_KEY`, `read_hf_source_metadata_from_dataset`, `write_hf_source_metadata_to_dataset_json`, `get_json_repo_folder`, `extract_repo_paths_from_json`, `extract_local_input_upload_entries_from_json`, and the three `is_hf_*_not_found_error` helpers |

Download helpers read pinned source metadata from an OSL JSON file and write
into their output directory. Upload and repository-management helpers can
create or mutate remote Hub repositories; require explicit credentials and
should never be invoked as an implicit side effect of data loading.

The helpers exposed by `opensportslib.tools` are lazy imports, so inspect the
generated signature below rather than assuming importing the package has
already loaded optional transfer dependencies.

::: opensportslib.tools
    options:
      members: false
      show_source: true

For commands and end-to-end examples, use [dataset conversion](../tools/dataset-conversion.md) and [Hugging Face transfer](../tools/hf-dataset-transfer.md).
