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
> Supports CUDA 12.6 / 12.8 / 13.0 (with CPU fallback).  
> PyTorch Geometric is supported up to PyTorch 2.10.*.

### Stable release

```bash
pip install opensportslib
```

### Pre release

```bash
pip install --pre opensportslib
```

### Setup Environment (PyTorch, CUDA aware & Optional Dependencies)
```bash
# Install PyTorch (CPU/GPU auto-detected)
opensportslib setup

# Optional: install PyTorch Geometric support
opensportslib setup --pyg

# Optional: install for DALI support
opensportslib setup --dali
``` 
---

**Note:**  
Run `opensportslib setup` to automatically configure dependencies.  
If issues occur, manually install compatible versions of `torch`, `torchvision`, and related libraries according to your CUDA version or system compatibility.

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
from opensportslib.apis import ClassificationModel

my_model = ClassificationModel(
    config="/path/to/classification.yaml",
    weights="/path/to/weights.pt",  # optional
)

my_model.train(
    train_set="/path/to/train_annotations.json",
    valid_set="/path/to/valid_annotations.json",
)
```

### Run inference

```python
from opensportslib.apis import ClassificationModel

my_model = ClassificationModel(
    config="/path/to/classification.yaml",
    weights="/path/to/weights.pt",  # optional
)

predictions = my_model.infer(
    test_set="/path/to/test_annotations.json",
)

saved_predictions = my_model.save_predictions(
    output_path="/path/to/predictions.json",
    predictions=predictions,
)

metrics = my_model.evaluate(
    test_set="/path/to/test_annotations.json",
)

metrics_from_file = my_model.evaluate(
    test_set="/path/to/test_annotations.json",
    predictions=saved_predictions,
)

print(metrics)
```

### Localization example

```python
from opensportslib.apis import LocalizationModel

my_model = LocalizationModel(
    config="/path/to/localization.yaml",
    weights="/path/to/weights.pt",  # optional
)

predictions = my_model.infer(
    test_set="/path/to/test_annotations.json",
)

saved_predictions = my_model.save_predictions(
    output_path="/path/to/predictions.json",
    predictions=predictions,
)

metrics = my_model.evaluate(
    test_set="/path/to/test_annotations.json",
)

metrics_from_file = my_model.evaluate(
    test_set="/path/to/test_annotations.json",
    predictions=saved_predictions,
)
```


---

## Hugging Face Dataset Transfer

OpenSportsLib provides APIs and scripts for downloading and uploading OSL datasets with Hugging Face.

### Python API

```python
from opensportslib.tools import (
    download_dataset_split_from_hf,
    upload_dataset_inputs_from_json_to_hf,
    upload_dataset_as_parquet_to_hf,
)
```

### Scripts

```bash
python tools/download_osl_hf.py --repo-id <org/repo> --revision main --split test --format parquet --output-dir downloaded_data
python tools/upload_osl_hf.py --repo-id <org/repo> --json-path <local_dataset.json> --split test --revision main
```

Downloads are placed under `<output-dir>/<revision>/<split>`.

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
- High-level API guide: [opensportslib/apis/README.md](opensportslib/apis/README.md)
- Configuration guide: https://opensportslab.github.io/opensportslib/tni/config-guide/
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

### Conda option

If you prefer conda:

```bash
conda create -n osl python=3.12 pip
conda activate osl
pip install -e .
```

### Setup Environment (PyTorch, CUDA aware & Optional Dependencies)
```bash
# Install PyTorch (CPU/GPU auto-detected)
opensportslib setup

# Optional: install PyTorch Geometric support
opensportslib setup --pyg

# Optional: install for DALI support
opensportslib setup --dali
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
