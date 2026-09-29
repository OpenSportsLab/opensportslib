# OpenSportsLib

OpenSportsLib is a configuration-driven Python library for sports-video experiments. The currently runnable task APIs are classification, localization (action spotting), and visual question answering (VQA).

![OpenSportsLib interface](assets/osl.jpg)

```text
Canonical YAML config -> task wrapper -> dataset and model builders
-> trainer, inferer, or evaluator -> task metric / OSL JSON prediction payload
```

Retrieval and captioning are roadmap areas. They do not currently have runnable task wrappers, dataset dispatch, or model routes in this package.

## Start here

1. [Install OpenSportsLib](getting-started/installation.md).
2. Read the [first workflow](getting-started/first-workflow.md). The package does not ship a demo dataset, so you must supply media and manifests.
3. Choose a [canonical configuration](config/configuration-guide.md) and prepare an [OSL JSON](data/osl-json-format.md) split.
4. Train, infer, evaluate, and explicitly save results using the [workflow guide](tni/tni.md).

## Guides by audience

- New users: [installation](getting-started/installation.md), [project structure](getting-started/project_structure.md), and [first workflow](getting-started/first-workflow.md).
- Researchers: [configuration](config/configuration-guide.md), [data format](data/osl-json-format.md), [model zoo](model-zoo.md), and [workflows](tni/tni.md).
- Developers: [architecture](developer/architecture.md), [supported extensions](developer/extensions.md), and the [configuration developer guide](config/developer-guide.md).
- Contributors: [contributing](contributing.md).
- Remote serving: [inference server](server/inference-server.md).

## License

OpenSportsLib is available under AGPL-3.0 and commercial licensing terms. See the repository license files for authoritative terms.

## Acknowledgments

OpenSportsLib is developed within the broader OpenSportsLab effort for sports
video understanding. Core contributors include:

- [Jeet Vora](https://jeetv.github.io/) — Remote Research Engineer
- [Dr. Merey Ramazanova](https://meryusha.github.io/) — Post-Doc
- [Dr. Silvio Giancola](https://www.silviogiancola.com/) — Research Scientist
