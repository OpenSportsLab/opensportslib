# Configuration Extension API

> **Extension point.** These modules define canonical configuration lifecycle.
> Use them when adding config fields or routing behavior; do not make them a
> dependency of an external application without accepting internal change risk.

## Module map

| Module | Non-private entry points | Responsibility |
| --- | --- | --- |
| `core.config.editable` | `Config`, option TypedDicts | Pre-allocation composition, inspection, validated updates, and remote overrides. |
| `core.config.loader` | `load_raw_config`, `load_config`, `load_config_omega`, `resolve_config`, `save_config` | YAML/JSON loading, canonical layering, runtime backend normalization, and config persistence. |
| `core.config.migrate` / `validate` | `migrate_config`, `validate_config` | Legacy ingestion and canonical schema/topology checks. |
| `core.config.accessors` | `get_*`, `set_*` helpers | Runtime-safe access to system, split, data, component, train, and VQA settings. |
| `core.config.runtime_adapter` | namespace conversion/adaptation helpers | Convert canonical mappings to runtime attribute access. |

`save_config()` writes a config file. `Config.from_pretrained()` and any
Hugging Face config source use network/authentication. All other loaders read
from the local config path supplied by the caller.

::: opensportslib.core.config.loader
    options:
      members:
        - load_raw_config
        - load_config
        - load_config_omega
        - resolve_config
        - save_config
      show_source: true

::: opensportslib.core.config.migrate
    options:
      members:
        - migrate_config
      show_source: true

::: opensportslib.core.config.validate
    options:
      members:
        - validate_config
      show_source: true

## Accessor families

Accessors are intentionally grouped instead of duplicating every leaf setting:
`get_system_*` / `set_system_*`, `get_split_*` / `set_split_*`, `get_data_*`,
`get_component_*`, `get_train_*`, and `get_vqa_*`. Add a new accessor only
when a setting is shared across task modules; task-only config consumption
belongs in that task’s implementation.

::: opensportslib.core.config.accessors
    options:
      members:
        - get_loader_backend
        - set_loader_backend
        - get_split_cfg
        - get_split_dataloader_cfg
        - get_split_annotation_path
        - set_split_annotation_path
        - get_data_classes
        - get_data_modality
        - get_component_by_kind
        - get_component_name_by_kind
        - get_component_params_by_kind
        - get_model_family
        - get_runner_type
      show_source: true
