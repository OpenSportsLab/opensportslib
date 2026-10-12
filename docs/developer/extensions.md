# Supported Extension Paths

These patterns are for contributors changing the repository. They do not make
every concrete class a stable public API. Begin with the closest existing task,
and add tests for every routing or configuration change.

## Add or change a configuration

Place new canonical configs under `opensportslib/configs/<task>/`. Preserve required top-level sections, component graph, and task defaults. Use the [configuration developer guide](../config/developer-guide.md), then load the config through `Config.from_file()` as a validation check.

Use a task default plus a focused experiment YAML rather than copying a fully materialized config. Canonical task YAML files compose with root and task defaults. Keep legacy aliases out of new configs: legacy input is a compatibility ingestion path, not an authoring interface.

## Add a dataset or modality

Implement task-specific dataset behavior in the corresponding dataset module and wire its route through `opensportslib/datasets/builder.py`. Add an OSL JSON example and tests for parsing, split paths, and the returned sample/collation contract.

For an existing task, add the smallest modality branch that preserves its output contract:

1. Define the needed `DATA.inputs` and split configuration.
2. Add parsing, sampling, and collating to the task dataset.
3. Update the model/trainer only if the new batch shape requires it.
4. Update OSL JSON, config, and fast tests.

Adding a top-level task is broader: it requires a public wrapper, dataset dispatch, model dispatch, task execution, metrics, canonical configs, and contract/integration tests. It is not enabled merely by accepting another `TASK` value in config validation.

## Add a model component or route

Implement reusable components in the appropriate `models/` area and extend `opensportslib/models/builder.py` only for supported canonical names/families. Classification routes are selected by encoder name; localization routes are selected first by model family; VQA routes are selected by backend.

Configure a component using its role (`kind`), provider/registry metadata, name, and parameters. Do not document a component as available until its branch is reachable from the canonical builder and a config demonstrates it. Keep task orchestration in the wrapper/trainer rather than a shared component.

## Add execution or metrics behavior

Keep task-specific execution in `core/trainer/*_trainer.py` and metric logic in `metrics/`. Preserve the public contract: `infer()` returns predictions in memory and `save_predictions()` persists them explicitly.

Metrics consume the matching inferer’s prediction representation and return a serializable dictionary. Classification uses accuracy, balanced accuracy, macro F1, precision, recall, and optional top-k accuracy. Localization implements temporal spotting evaluation; VQA implements normalized answer matching and optional referee-semantic scoring. Add successful and malformed/missing input coverage for new metric behavior.

## Testing and documentation contract

Put deterministic config/data/builder/metric/API contracts under `tests/unit/`, lightweight package checks under `tests/smoke/`, and bounded offline workflows under `tests/integration/<task>/`. Real-data, GPU, network, and full-model coverage belongs in opt-in `tests/release/`. Reuse fixtures before adding helpers.

When a change affects an API, configuration, OSL format, dependency, packaged asset, or task boundary, update its corresponding contract test and published documentation. See [developer testing](testing.md) for the test runner and review checklist.

```bash
bash scripts/run_tests.sh
mkdocs build --strict
```
