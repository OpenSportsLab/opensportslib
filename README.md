# OpenSportsLib

<img src="docs/assets/osl.jpg" height="400">

OpenSportsLib is a Python library for building machine-learning workflows with
sports video.

It gives you one consistent way to **train models, make predictions, and
evaluate results** for sports-video tasks, including:

- **Action classification**
- **Action localization / spotting**
- **Visual Question Answering (VQA)**

Retrieval and action description/captioning are planned for the future. They do
not yet have dedicated task APIs or training workflows in this package.

OpenSportsLib is for **researchers, ML engineers, and sports analytics teams**
who need reproducible, adaptable sports-video AI workflows.

## Why OpenSportsLib?

- One workflow for training, prediction, and evaluation
- A modular design for adding tasks, datasets, and models
- YAML configuration files that make experiments reproducible
- Optional SpoTTA test-time adaptation for E2ESpot prediction
- Support for several input types and sports workflows
- Suitable for research and applied projects

## Quick links

- **Documentation:** https://opensportslab.github.io/opensportslib/
- **OSL JSON format:** https://opensportslab.github.io/opensportslib/data/osl-json-format/
- **Inference server:** [server/README.md](server/README.md)
- **PyPI:** https://pypi.org/project/opensportslib/
- **Issues:** https://github.com/OpenSportsLab/opensportslib/issues

---

## Installation

> Requires **Python 3.12+**.  
> Supports CUDA 12.6 / 12.8 / 13.0 (with CPU fallback).  
> PyTorch Geometric uses a dedicated PyTorch 2.12.1 compatibility profile.

### Create conda env

```bash
conda create -n osl python=3.12 pip -y
conda activate osl
```

### Stable release

```bash
pip install opensportslib
```

### Pre release

```bash
pip install --pre opensportslib
```

### Source development version

```bash
pip install -e .
```

### Setup Environment (PyTorch, CUDA aware & Optional Dependencies)
```bash
# Install PyTorch (CPU/GPU auto-detected)
opensportslib setup

# Optional: install PyTorch Geometric support. This replaces the installed
# Torch stack with the PyG-compatible PyTorch 2.12.1 profile.
opensportslib setup --pyg

# Optional: install for DALI support
opensportslib setup --dali

# Optional: install the X-VARS-compatible VQA dependency profile
opensportslib setup --vqa_xvars

# Optional: install the Qwen-compatible VQA dependency profile
opensportslib setup --vqa_qwen
```
---

**Note:**  
Run `opensportslib setup` to install the matching dependencies automatically.
If setup fails, install versions of `torch`, `torchvision`, and related
libraries that match your CUDA version or system.

For VQA, install exactly one backend-specific dependency profile:

- `--vqa_xvars` installs the X-VARS-compatible Hugging Face stack from `XVARS_DEPENDENCY_PINS`
- `--vqa_qwen` installs the Qwen-compatible Hugging Face stack from `QWEN_DEPENDENCY_PINS`

The `vqa_qwen` config supports `Qwen/Qwen2.5-7B-Instruct` and `Qwen/Qwen3.5-9B-Base`.

---

## Data and pretrained models

OpenSportsLib uses annotation files, datasets, and pretrained checkpoints that
are stored outside the package.

Public assets are hosted under the **OpenSportsLab Hugging Face organization**:

**https://huggingface.co/OpenSportsLab**

Use it to find:
- datasets
- annotation files
- extracted features
- pretrained models and checkpoints

See the [Model Zoo](docs/model-zoo.md) for available pretrained models, their
reported scores and datasets, and loading examples.

---

## Dataset format

OpenSportsLib uses the **OSL JSON v2.0** format for datasets and predictions. A
dataset manifest is an OSL JSON file that lists samples, their input files, and
their annotations. See the [OSL JSON format guide](docs/data/osl-json-format.md)
for the full structure, examples, and conversion notes.

---

## Quickstart

### Classification

```python
from opensportslib.apis import ClassificationModel

my_model = ClassificationModel(
    config="/path/to/classification.yaml",
    weights=None,  # optional: path or Hugging Face model ID
)

my_model.train(
    train_set="/path/to/train_annotations.json",
    valid_set="/path/to/valid_annotations.json",
)

predictions = my_model.infer(test_set="/path/to/test_annotations.json")
my_model.save_predictions(
    output_path="/path/to/predictions.json",
    predictions=predictions,
)
```

For localization, VQA, and complete examples, see the
[API guide](opensportslib/apis/README.md), [quickstart scripts](examples/quickstart/),
and [VQA guide](docs/tools/vqa.md).


---

## Hugging Face Dataset Transfer

OpenSportsLib provides Python APIs and scripts for downloading and uploading
OSL datasets through Hugging Face.

For SN-GAR classification and action-spotting configurations, setup, caching,
and training commands, see the [SN-GAR examples](examples/sngar/README.md).

### Python API

```python
from opensportslib.tools import (
    download_dataset_split_from_hf,
    download_dataset_sample_inputs_from_hf,
    upload_dataset_inputs_from_json_to_hf,
    upload_dataset_as_parquet_to_hf,
)
```

### Scripts

```bash
python tools/download/download_osl_hf.py --repo-id <org/repo> --revision main --split test --format parquet --output-dir downloaded_data --annotations-only
python tools/download/upload_osl_hf.py --repo-id <org/repo> --json-path <local_dataset.json> --split test --revision main
```

---

## Examples and documentation

Use this README to get started, then use these guides when you need more detail:

- Full documentation: https://opensportslab.github.io/opensportslib/
- OSL JSON format: [docs/data/osl-json-format.md](docs/data/osl-json-format.md)
- High-level API guide: [opensportslib/apis/README.md](opensportslib/apis/README.md)
- Configuration guide: https://opensportslab.github.io/opensportslib/config/configuration-guide/
- Example configs: [examples/configs/](examples/configs/)
- Quickstart scripts: [examples/quickstart/](examples/quickstart/)
- Contribution guide: [CONTRIBUTING.md](CONTRIBUTING.md)
- Developer guide: [DEVELOPERS.md](DEVELOPERS.md)

---

## Development setup

Use these steps if you want to contribute or run OpenSportsLib directly from
the source repository:

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

# Optional: install PyTorch Geometric support. This replaces the installed
# Torch stack with the PyG-compatible PyTorch 2.12.1 profile.
opensportslib setup --pyg

# Optional: install for DALI support
opensportslib setup --dali

# Optional: install the X-VARS-compatible VQA dependency profile
opensportslib setup --vqa_xvars

# Optional: install the Qwen-compatible VQA dependency profile
opensportslib setup --vqa_qwen
```

---

## Contributing

We welcome contributions. Pull requests must target `dev`, and each
GitHub-linked commit author must accept the [Individual Contributor License
Agreement](.github/CLA.md) when prompted by the `CLA check`. Each listed author
must post this exact comment on the pull request:

> I have read the OpenSportsLab Individual Contributor License Agreement and I hereby sign it.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the contribution workflow and
[DEVELOPERS.md](DEVELOPERS.md) for architecture and extension guidance.

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

OpenSportsLib is part of the broader OpenSportsLab effort in sports-video
understanding. Core contributors affiliated with KAUST include:

- [Jeet Vora](https://jeetv.github.io/) — Remote Research Engineer
- [Dr. Merey Ramazanova](https://meryusha.github.io/) — Post-Doc
- [Dr. Silvio Giancola](https://www.silviogiancola.com/) — Research Scientist
