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

> Requires **Python 3.12+**  
> Supports CUDA 12.6 / 12.8 / 13.0 (with CPU fallback).  
> PyTorch Geometric is supported up to PyTorch 2.10.*.


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

#### Setup Environment (PyTorch, CUDA aware & Optional Dependencies)
```bash
# Install PyTorch (CPU/GPU auto-detected)
opensportslib setup

# Optional: install PyTorch Geometric support
opensportslib setup --pyg

# Optional: install for DALI support
opensportslib setup --dali
```

---

!!! note   
    Run `opensportslib setup` to automatically configure dependencies.
    If issues occur, manually install compatible versions of `torch`, `torchvision`, and related libraries according to your CUDA version or system compatibility.


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
```

#### Step 4: Setup Environment (PyTorch, CUDA aware & Optional Dependencies)
```bash
# Install PyTorch (CPU/GPU auto-detected)
opensportslib setup

# Optional: install PyTorch Geometric support
opensportslib setup --pyg

# Optional: install for DALI support
opensportslib setup --dali
```