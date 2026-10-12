# Install PyTorch, DALI, CuPy, and PyG

## Supported path

Run `opensportslib setup` after installing OpenSportsLib. It checks the visible
GPU and driver, chooses a supported Torch package profile, and installs optional
profiles when requested. It replaces the installed Torch packages, so use a
dedicated Python 3.12+ environment.

```bash
conda create -n opensportslib python=3.12 pip
conda activate opensportslib
python -m pip install --upgrade pip setuptools wheel packaging
pip install -e .
opensportslib setup
```

The manual `pip` commands below are for diagnosis or recovery. They do not
replace setup’s compute-capability validation or guarantee a supported profile.

| Command | Purpose | Behavior |
| --- | --- | --- |
| `opensportslib setup --dali` | NVIDIA DALI video loading | On a CUDA host installs `nvidia-dali-cuda120` plus `cupy-cuda12x` for CUDA 12.x, or `nvidia-dali-cuda130` plus `cupy-cuda13x` for CUDA 13. |
| `opensportslib setup --pyg` | Graph/tracking workloads | Replaces Torch with the PyG-compatible Torch `2.12.1` / torchvision `0.27.1` profile and installs `torch-geometric`. |
| `opensportslib setup --pyg --pyg_extensions` | Optional compiled PyG extensions | Verifies matching binary wheels before installing `pyg-lib`, `torch-scatter`, and `torch-sparse`. |
| `opensportslib setup --vqa_xvars` | X-VARS VQA | Installs the X-VARS Hugging Face dependency pins. |
| `opensportslib setup --vqa_qwen` | Qwen VQA | Installs the Qwen Hugging Face dependency pins. |

## Supported install targets

| Target | PyTorch wheel tag | DALI package | CuPy package |
| --- | --- | --- | --- |
| CPU only | `cpu` | not needed | not needed |
| CUDA 12.6 | `cu126` | `nvidia-dali-cuda120` | `cupy-cuda12x` |
| CUDA 12.8 | `cu128` | `nvidia-dali-cuda120` | `cupy-cuda12x` |
| CUDA 13.0 | `cu130` | `nvidia-dali-cuda130` | `cupy-cuda13x` |

`nvidia-smi` reports a driver-supported CUDA version such as `12.8`; PyTorch
wheel tags use values such as `cu128`. The setup command selects the highest
compatible supported tag using both driver version and visible GPU compute
capabilities. It installs CPU wheels when no CUDA driver is available.

Notes:

- Choose one Torch wheel tag: `cpu`, `cu126`, `cu128`, or `cu130`.
- CUDA 12.x uses `nvidia-dali-cuda120` and `cupy-cuda12x`; CUDA 13.x uses
  `nvidia-dali-cuda130` and `cupy-cuda13x`.
- PyTorch wheels include their CUDA runtime. The driver must support the
  selected wheel tag; a CUDA toolkit version shown elsewhere on the host is
  not itself a wheel-selection rule.

GPUs below compute capability 5.0 are rejected. A visible legacy GPU (5.0–7.4)
uses the `cu126` route when compatible; on Linux ARM64 that route supports
Ampere and newer only. A mixed legacy/newer visible set can require selecting
one group with `CUDA_VISIBLE_DEVICES`. GPUs requiring CUDA 13 need a driver
reporting CUDA 13.0 or newer.

## 1. Clean an existing environment

If an earlier experiment left incompatible packages installed, remove them
before running the supported setup command again:

```bash
python -m pip uninstall -y \
  torch torchvision torchaudio torch-geometric \
  pyg-lib torch-scatter torch-sparse \
  nvidia-dali-cuda120 nvidia-dali-cuda130 \
  cupy cupy-cuda12x cupy-cuda13x
opensportslib setup
```

## 2. Manual Torch recovery commands

Pick exactly one command only when recovering a broken environment. The normal
workflow is still `opensportslib setup`.

### CPU only

```bash
python -m pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cpu
```

### CUDA 12.6

```bash
python -m pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu126
```

### CUDA 12.8

```bash
python -m pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128
```

### CUDA 13.0

```bash
python -m pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu130
```

OpenSportsLib does not provide a general fixed-version Torch profile. Do not
copy historical Torch pins: use the current setup command or inspect its
selected wheel profile.

## 3. Optional DALI and CuPy support

DALI is for CUDA video workloads; it is not installed for CPU-only setup.

```bash
# CUDA 12.6 or 12.8
python -m pip install nvidia-dali-cuda120 cupy-cuda12x

# CUDA 13.0
python -m pip install nvidia-dali-cuda130 cupy-cuda13x
```

The supported equivalent is:

```bash
opensportslib setup --dali
```

## 4. PyTorch Geometric

Graph/tracking workloads use the dedicated PyG compatibility profile:

```bash
opensportslib setup --pyg
```

This replaces the Torch stack with Torch `2.12.1` and torchvision `0.27.1`,
then installs `torch-geometric`. Optional compiled extension wheels are not
installed by default. Install them only if the workload requires them:

```bash
opensportslib setup --pyg --pyg_extensions
```

That command verifies binary wheel availability before installing `pyg-lib`,
`torch-scatter`, and `torch-sparse`; it does not fall back to source builds.
For manual recovery, first install the PyG-compatible Torch profile and
`torch-geometric`, then select an extension-wheel URL matching the installed
CUDA tag. The supported PyG Torch base version is `2.12.1`.

```bash
python -m pip install torch==2.12.1 torchvision==0.27.1 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install torch-geometric
```

Use the matching index for the first command: `cpu`, `cu126`, `cu128`, or
`cu130`. Check the result before installing extensions:

```bash
python - <<'PY'
import torch

version = torch.__version__.split("+")[0]
tag = "cpu" if torch.version.cuda is None else "cu" + torch.version.cuda.replace(".", "")
print(version, tag)
PY
```

### CPU extensions

```bash
TORCH_VERSION=2.12.1
CUDA_TAG=cpu
python -m pip install pyg-lib torch-scatter torch-sparse --only-binary=:all: \
  -f "https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_TAG}.html"
```

### CUDA 12.6 extensions

```bash
TORCH_VERSION=2.12.1
CUDA_TAG=cu126
python -m pip install pyg-lib torch-scatter torch-sparse --only-binary=:all: \
  -f "https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_TAG}.html"
```

### CUDA 12.8 extensions

```bash
TORCH_VERSION=2.12.1
CUDA_TAG=cu128
python -m pip install pyg-lib torch-scatter torch-sparse --only-binary=:all: \
  -f "https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_TAG}.html"
```

### CUDA 13.0 extensions

```bash
TORCH_VERSION=2.12.1
CUDA_TAG=cu130
python -m pip install pyg-lib torch-scatter torch-sparse --only-binary=:all: \
  -f "https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_TAG}.html"
```

If the extensions do not publish a matching binary wheel for your platform,
keep the base `torch-geometric` installation; compiled extensions are optional.

## 5. Verify installation

### PyTorch

```bash
python - <<'PY'
import torch

print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

### DALI

```bash
python - <<'PY'
import nvidia.dali as dali

print("DALI:", dali.__version__)
PY
```

### CuPy

```bash
python - <<'PY'
import cupy as cp

print("CuPy:", cp.__version__)
print("CUDA devices:", cp.cuda.runtime.getDeviceCount())
PY
```

### PyTorch Geometric

```bash
python - <<'PY'
import torch_geometric

print("PyG:", torch_geometric.__version__)
PY
```

## 6. VQA dependency profiles

Install one VQA profile after the base Torch setup. X-VARS and Qwen pin
incompatible Hugging Face dependency sets, so use separate environments when
both are required.

### X-VARS

```bash
opensportslib setup --vqa_xvars
```

This installs `transformers==4.38.2`, `peft==0.9.0`,
`tokenizers==0.15.2`, `accelerate==0.27.2`, and `trl==0.10.1`.

### Qwen

```bash
opensportslib setup --vqa_qwen
```

This installs `transformers==5.13.0`, `peft==0.19.0`,
`tokenizers==0.22.1`, `accelerate==1.14.0`, and `trl==1.7.1`.

## 7. Troubleshooting

### `torch.cuda.is_available()` is `False`

Confirm that a CPU wheel was not installed and that the host exposes the GPU:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
nvidia-smi
```

Re-run `opensportslib setup` after resolving driver visibility or selecting the
intended GPU group with `CUDA_VISIBLE_DEVICES`.

### The driver cannot support the selected wheel

The setup command requires a CUDA 12.6+ compatible driver for its CUDA wheel
profiles, and CUDA 13.0+ for GPUs that require `cu130`. Upgrade the NVIDIA
driver rather than forcing a wheel tag unsupported by the driver.

### `nvidia-smi` reports CUDA 12.8 but Torch uses `cu126`

That can be valid. `nvidia-smi` reports the maximum CUDA version supported by
the driver, whereas a Torch wheel supplies its own CUDA runtime. Let
`opensportslib setup` select the highest compatible profile, or use one of the
manual wheel tags above only for recovery.

### DALI import fails

Use the DALI package that matches the selected CUDA profile. DALI dynamically
links CUDA libraries; on managed systems ensure the appropriate CUDA runtime is
available before retrying `opensportslib setup --dali`. A typical diagnostic is
to confirm the host's CUDA library path is visible to the environment, for
example `CUDA_PATH=/usr/local/cuda` and `$CUDA_PATH/lib64` on Linux.

### PyG extensions fail to install

Run `opensportslib setup --pyg --pyg_extensions` so wheel availability is
validated first. On platforms without matching binary extensions, retain
`torch-geometric` from `opensportslib setup --pyg` and install extensions only
when the selected model genuinely needs them.
