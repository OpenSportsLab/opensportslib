# Project Structure

Most users only need `opensportslib/apis/` and `opensportslib/configs/`. The other package modules implement the library internally unless a developer page explicitly identifies them as supported extension points.

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

The supported configuration files are in `opensportslib/configs/classification/`, `localization/`, and `vqa/`. When you load a YAML file from one of these directories, OpenSportsLib combines the root `configs/default.yaml`, the task-level `default.yaml`, and the file you selected. `opensportslib/legacy_config/` exists for backward-compatibility tests, not for new experiments.

See [architecture](../developer/architecture.md) to understand how the library runs an experiment, or the [configuration guide](../config/configuration-guide.md) for the supported configuration format.
