# OpenSportsLib

OpenSportsLib is a modular Python library for sports video understanding.

It provides a unified framework to **train, evaluate, and run inference** for key temporal understanding tasks in sports video, including:

- **Action classification**
- **Action localization / spotting**
- **Action retrieval**
- **Action description / captioning**

OpenSportsLib is designed for **researchers, ML engineers, and sports analytics teams** who want reproducible and extensible workflows for sports video AI.

## Why OpenSportsLib?

- Unified workflow for training and inference
- Modular design for adding new tasks, datasets, and models
- Config driven experiments for reproducibility
- Support for multiple modalities and sports workflows
- Research friendly while still usable in applied settings

## Quick links

- **Documentation:** https://opensportslab.github.io/opensportslib/
- **PyPI:** https://pypi.org/project/opensportslib/
- **Issues:** https://github.com/OpenSportsLab/opensportslib/issues

---

## Installation

> Requires **Python 3.12+**.

### Stable release

```bash
pip install opensportslib
```

### Pre release

```bash
pip install --pre opensportslib
```

---

## Data and pretrained models

OpenSportsLib uses external annotation files, datasets, and pretrained checkpoints.

Public assets are hosted under the **OpenSportsLab Hugging Face organization**:

**https://huggingface.co/OpenSportsLab**

Use it as the main entry point to find:
- datasets
- annotation files
- extracted features
- pretrained models and checkpoints

--

## Quickstart

### Import the library

```python
import opensportslib
print("OpenSportsLib imported successfully")
```

### Train a classification model

```python
from opensportslib import model

myModel = model.classification(
    config="/path/to/classification.yaml"
)

myModel.train(
    train_set="/path/to/train_annotations.json",
    valid_set="/path/to/valid_annotations.json",
    pretrained="/path/to/pretrained.pt",  # optional
)
```

### Run inference

```python
from opensportslib import model

myModel = model.classification(
    config="/path/to/classification.yaml"
)

metrics = myModel.infer(
    test_set="/path/to/test_annotations.json",
    pretrained="/path/to/checkpoints/final_model",
    predictions="/path/to/predictions.json"
)

print(metrics)
```

### Localization example

```python
from opensportslib import model

myModel = model.localization(
    config="/path/to/localization.yaml"
)
```

---

## What you can do with OpenSportsLib

### Action Classification
Classify clips or event centered samples into predefined categories.

### Action Localization / Spotting
Predict when key events happen in long untrimmed sports videos.

### Action Retrieval
Search and retrieve relevant clips or moments from a collection of sports videos.

### Action Description / Captioning
Generate text descriptions for sports events and temporal segments.

---

## Typical workflow

1. Prepare your dataset in the expected format
2. Select or create a YAML config
3. Initialize the task specific model
4. Train on your annotations
5. Run inference on new data
6. Extend the pipeline with your own datasets or models

---

## Examples and documentation

Use the README for the fast start, then go deeper through:

- Full documentation: https://opensportslab.github.io/opensportslib/
- Example configs: [examples/configs/](examples/configs/)
- Quickstart scripts: [examples/quickstart/](examples/quickstart/)
- Contribution guide: [CONTRIBUTING.md](CONTRIBUTING.md)
- Developer guide: [DEVELOPERS.md](DEVELOPERS.md)

---

## Development setup

For contributors who want to work from source:

```bash
git clone https://github.com/OpenSportsLab/opensportslib.git
cd opensportslib
pip install -e .
```

### With extras

```bash
pip install -e ".[localization]"
pip install -e ".[py-geometric]" -f https://pytorch-geometric.com/whl/torch-2.10.0+cu128.html
```

### Conda option

If you prefer conda:

```bash
conda create -n osl python=3.12 pip
conda activate osl
pip install -e .
```

### Git workflow

1. Make sure you are branching from `dev`
2. Create your feature or fix branch from `dev`
3. Open a pull request back into `dev`

---

## Contributing

We welcome contributions to OpenSportsLib.

Please check:

- [CONTRIBUTING.md](CONTRIBUTING.md)
- [DEVELOPERS.md](DEVELOPERS.md)

These documents describe:

- how to add models and datasets
- coding standards
- training pipeline structure
- how to run and test the framework

---

## License

OpenSportsLib is available under dual licensing.

### Open source license
[AGPL 3.0](LICENSE) for research, academic, and community use.

### Commercial license
For proprietary or commercial deployment, please refer to [LICENSE-COMMERCIAL](LICENSE-COMMERCIAL).

---

## Citation

If you use OpenSportsLib in your research, please cite the project.

```bibtex
@misc{opensportslib,
  title={OpenSportsLib},
  author={OpenSportsLab},
  year={2026},
  howpublished={\url{https://github.com/OpenSportsLab/opensportslib}}
}
```

---

## Acknowledgments

OpenSportsLib is developed within the broader OpenSportsLab effort for sports video understanding.
