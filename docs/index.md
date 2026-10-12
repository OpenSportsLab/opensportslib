# OpenSportsLib

OpenSportsLib is a Python library for building sports-video machine-learning workflows. You describe an experiment in a YAML configuration file, then use the library to train, run, and evaluate a model. It currently supports classification, localization (action spotting), and visual question answering (VQA).

![OpenSportsLib interface](assets/osl.jpg)

```text
YAML configuration file -> task API -> data and model setup
-> training, prediction, or evaluation -> metrics or OSL JSON predictions
```

Retrieval and captioning are planned for the future. They do not yet have runnable APIs, datasets, or model workflows in this package.

## Start here

1. [Install OpenSportsLib](getting-started/installation.md).
2. Read the [first workflow](getting-started/first-workflow.md). OpenSportsLib does not include a demo dataset, so you will provide video files and JSON manifest files that list them.
3. Choose a [configuration file](config/configuration-guide.md) and prepare an [OSL JSON dataset split](data/osl-json-format.md).
4. Use the [workflow guide](tni/tni.md) to train, make predictions, evaluate them, and save the results.

## Guides by audience

- New users: [installation](getting-started/installation.md), [project structure](getting-started/project_structure.md), and [first workflow](getting-started/first-workflow.md).
- Researchers: [configuration](config/configuration-guide.md), [data format](data/osl-json-format.md), [model zoo](model-zoo.md), and [training and prediction workflows](tni/tni.md).
- Developers: [architecture](developer/architecture.md), [supported extensions](developer/extensions.md), and the [configuration developer guide](config/developer-guide.md).
- Contributors: [contributing](contributing.md).
- Remote serving: [inference server](server/inference-server.md).

## License

OpenSportsLib is available under AGPL-3.0 and commercial licensing terms. See the repository license files for authoritative terms.

## Acknowledgments

OpenSportsLib is developed within the broader OpenSportsLab effort for sports
video understanding. Core contributors affiliated with KAUST include:

- [Jeet Vora](https://jeetv.github.io/) — Remote Research Engineer
- [Dr. Merey Ramazanova](https://meryusha.github.io/) — Post-Doc
- [Dr. Silvio Giancola](https://www.silviogiancola.com/) — Research Scientist
