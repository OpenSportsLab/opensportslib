# Project Structure

Public users normally start in `opensportslib/apis/` and `opensportslib/configs/`. Other package modules are implementation details unless explicitly documented as extension points.

```text
opensportslib/
├── apis/                 public wrappers: Config, classification, localization, VQA
├── configs/              canonical YAML root, task defaults, and experiment configs
├── core/config/          composition, migration, validation, editable config support
├── core/trainer/         task training, inference, and evaluation routes
├── datasets/             task dispatch plus video, tracking, HDF5, and Hub data handling
├── metrics/              classification, localization, and VQA metrics
├── models/               canonical model dispatch and implementation components
├── setup/                `opensportslib setup` implementation
├── tools/                package conversion and Hugging Face transfer APIs
├── examples/             config mirrors and minimal Python examples
├── tools/                command-line training, conversion, download, and upload scripts
├── tests/                smoke, unit, integration, and opt-in release suites
├── server/               separately installed FastAPI/RQ inference service
└── docs/                 this MkDocs site
```

## Config locations

Canonical configurations are under `opensportslib/configs/classification/`, `localization/`, and `vqa/`. Loading YAML from one of these task directories composes the root `configs/default.yaml`, the task `default.yaml`, and the selected config. `opensportslib/legacy_config/` holds compatibility fixtures, not new experiment configs.

See [architecture](../developer/architecture.md) for runtime flow and [configuration](../config/configuration-guide.md) for the canonical contract.
