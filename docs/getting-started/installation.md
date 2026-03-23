# Installation

OpenSportsLib can be installed either from **PyPI** or from source in **editable mode** (recommended for development).

---

## Option 1 — Install from PyPI

#### Create a Virtual Environment
Use Conda to manage dependencies and ensure Python 3.12 compatibility.
```bash
conda create -n osl python=3.12 pip
conda activate osl
```

#### PyPI install
Stable version
```bash
pip install opensportslib
```

Pre-release version
```bash
pip install --pre opensportslib
```
!!! note
    The `--pre` flag installs the latest pre-release version from PyPI.

## Verify installation

```python
import opensportslib
print("OpenSportsLib installed successfully")
```

## Option 2 — Install from Source (Editable Mode) ⭐ Recommended

- Use this method if you:
- want the latest development version
- plan to modify the code
- are contributing to the project

#### Step 1: Clone the Repository
```bash
git clone https://github.com/OpenSportsLab/opensportslib.git 
cd opensportslib
```
#### Step 2: Create a Virtual Environment
Use Conda to manage dependencies and ensure Python 3.12 compatibility.
```bash
conda create -n osl python=3.12 pip
conda activate osl
```
#### Step 3: Install in Editable Mode
Install the base package or include optional dependencies for specific tasks like localization:
```bash
# Install core package in editable mode
pip install -e .

# OR for localization support
pip install -e .[localization]
 
# OR want to use "torch-geometric","torch-scatter", "torch-sparse", "torch-cluster", "torch-spline-conv"
pip install -e ".[py-geometric]" -f https://pytorch-geometric.com/whl/torch-2.10.0+cu128.html
```