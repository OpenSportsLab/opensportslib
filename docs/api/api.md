# Public API Reference

> **Public / stable.** Import the names on this page from `opensportslib.apis`
> or `opensportslib.tools`. Other package modules may change unless a developer
> page explicitly describes them as an extension point.

## Task wrappers

Each task API accepts a supported configuration path or `Config`, optional local
or Hugging Face `weights`, and optional remote-server connection settings. A
task wrapper is the Python object you use to work with one task. `train()` uses
the configured or supplied manifests, `infer()` returns predictions in memory,
`evaluate()` returns metric values, and `save_predictions()` writes predictions
to disk when you ask it to.

| Class | Use | Important inputs and side effects |
| --- | --- | --- |
| `ClassificationModel` | Clip, frame-array, and supported tracking classification. | Creates run artifacts under the configured save directory; may load local/Hub weights. |
| `LocalizationModel` | Action spotting/localization for configured video, feature, tracking, and HDF5 paths. | Model family and loader backend determine the execution route. |
| `VQAModel` | X-VARS/Qwen visual question answering. | Requires the matching VQA dependency profile; direct inference accepts `video_path` and `question`. |
| `BaseTaskModel` | Shared abstract wrapper contract. | Provides local/remote request lifecycle; subclass methods implement task execution. |

```python
from opensportslib.apis import ClassificationModel

model = ClassificationModel(config="/path/to/classification.yaml")
predictions = model.infer(test_set="/path/to/test.json")
model.save_predictions("/path/to/predictions.json", predictions)
```

### Exact wrapper signatures

::: opensportslib.apis.base_task_model.BaseTaskModel
    options:
      members:
        - is_remote
        - submit_inference
        - submit_per_sample_inference
        - submit_video_inference
        - get_remote_job
        - get_remote_result
        - wait_for_remote_result
        - wait_for_remote_batch
        - clear_remote_session
      show_source: true

::: opensportslib.apis.classification.ClassificationModel
    options:
      members:
        - load_weights
        - train
        - infer
        - evaluate
      show_source: true

::: opensportslib.apis.localization.LocalizationModel
    options:
      members:
        - load_weights
        - train
        - infer
        - evaluate
      show_source: true

::: opensportslib.apis.vqa.VQAModel
    options:
      members:
        - load_weights
        - train
        - infer
        - evaluate
        - save_predictions
      show_source: true

## Configuration API

`Config` loads, composes, migrates, validates, and edits a configuration before
model allocation. `Config.from_pretrained()` downloads `config.yaml` from a Hub
model repository, so it needs network access and any required Hugging Face
authentication. Use `options()` before `update()` to discover supported,
validated settings.

::: opensportslib.core.config.editable.Config
    options:
      members:
        - from_file
        - from_pretrained
        - get_config
        - options
        - update
        - apply_to
        - remote_overrides
      show_source: true

## Remote model registry

`RemoteModelRegistry` is an administrative client for the separately deployed
server. `register_model`, `set_default`, `unregister_model`, and
`reconcile_runtime` mutate server state; use `reconcile_runtime(dry_run=True)`
to inspect safely. Local registration requires a server API key; Hub operations
may require an HF token.

::: opensportslib.remote_registry.RemoteRegistryError
    options:
      show_source: true

::: opensportslib.remote_registry.RemoteModelRegistry
    options:
      members:
        - register_model
        - list_models
        - get_model
        - get_operation
        - wait_for_operation
        - set_default
        - unregister_model
        - reconcile_runtime
      show_source: true

## Conversion and Hugging Face tools

These exports are stable, but many perform disk and network work. Conversion
functions create Parquet/WebDataset or JSON output. Hugging Face download and
upload functions access remote repositories and can write local media or remote
repository content. Inspect their signatures before use and use the
[dataset transfer guide](../tools/hf-dataset-transfer.md) for complete flows.

::: opensportslib.tools.osl_json_to_parquet
    options:
      members:
        - parse_shard_size
        - convert_json_to_parquet
      show_source: true

::: opensportslib.tools.parquet_to_osl_json
    options:
      members:
        - convert_parquet_metadata_to_json
        - convert_parquet_to_json
      show_source: true

::: opensportslib.tools.hf_transfer
    options:
      members:
        - HfTransferCancelled
        - MissingDatasetInputsError
        - download_dataset_split_from_hf
        - download_dataset_sample_inputs_from_hf
        - download_dataset_missing_inputs_from_hf
        - find_missing_dataset_inputs
        - upload_dataset_inputs_from_json_to_hf
        - upload_dataset_as_parquet_to_hf
        - create_dataset_repo_on_hf
        - dataset_repo_exists_on_hf
        - create_dataset_branch_on_hf
      show_source: true

See [extension APIs](../developer/config-reference.md) for builders and runtime
modules, and [internal catalog](../developer/internal-catalog.md) for concrete
implementation classes.
