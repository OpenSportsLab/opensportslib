# Installation

SoccerNetPro can be installed either from **PyPI** or from source in **editable mode** (recommended for development).

---

## Option 1 — Install from PyPI

#### Create a Virtual Environment
Use Conda to manage dependencies and ensure Python 3.12 compatibility.
```bash
conda create -n SoccerNet python=3.12 pip
conda activate SoccerNet
```

#### PyPI install
```bash
pip install --pre soccernetpro
```
!!! note
    The `--pre` flag installs the latest pre-release version from PyPI.

## Verify installation

```python
import soccernetpro
print("SoccerNetPro installed successfully")
```

## Option 2 — Install from Source (Editable Mode) ⭐ Recommended

- Use this method if you:
- want the latest development version
- plan to modify the code
- are contributing to the project

#### Step 1: Clone the Repository
```bash
git clone https://github.com/OpenSportsLab/soccernetpro.git 
cd soccernetpro
```
#### Step 2: Create a Virtual Environment
Use Conda to manage dependencies and ensure Python 3.12 compatibility.
```bash
conda create -n SoccerNet python=3.12 pip
conda activate SoccerNet
```
#### Step 3: Install in Editable Mode
Install the base package or include optional dependencies for specific tasks like localization:
```bash
# Install core package in editable mode
pip install -e .

# OR for localization support
pip install -e .[localization]
 
# OR for tracking support
pip install -e .[tracking]
```