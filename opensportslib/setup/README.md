# Install PyTorch, DALI, CuPy, and PyG

This README replaces the automatic installation script with explicit `pip` commands.  
Choose the command that matches the CUDA wheel you want to use.

## Supported install targets

| Target | PyTorch wheel tag | DALI package | CuPy package |
| --- | --- | --- | --- |
| CPU only | `cpu` | not needed | not needed |
| CUDA 12.6 | `cu126` | `nvidia-dali-cuda120` | `cupy-cuda12x` |
| CUDA 12.8 | `cu128` | `nvidia-dali-cuda120` | `cupy-cuda12x` |
| CUDA 13.0 | `cu130` | `nvidia-dali-cuda130` | `cupy-cuda13x` |

Notes:

- `nvidia-smi` reports versions such as `12.8`, while PyTorch wheel tags use values such as `cu128`.
- For PyTorch, select one wheel tag: `cpu`, `cu126`, `cu128`, or `cu130`.
- For DALI, CUDA 12.x uses the `nvidia-dali-cuda120` package, while CUDA 13.x uses `nvidia-dali-cuda130`.
- For CuPy, CUDA 12.x uses `cupy-cuda12x`, while CUDA 13.x uses `cupy-cuda13x`.
- `opensportslib setup` selects the wheel using both the driver CUDA version and visible GPU compute capabilities.
- GPUs below `SM 7.5` use `cu126` even when the installed driver reports CUDA 13.0, because newer CUDA 13 PyTorch wheels do not include kernels for those architectures.
- GPUs at `SM 7.5` or newer, including DGX Spark (`SM 12.1`), use the highest wheel supported by the installed driver.

## 1. Create a clean environment

```bash
conda create -n opensportslib python=3.12
conda activate opensportslib

python -m pip install --upgrade pip setuptools wheel packaging
```

Optional cleanup if you already tried another Torch install:

```bash
python -m pip uninstall -y \
  torch torchvision torchaudio \
  torch-geometric pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
  nvidia-dali-cuda120 nvidia-dali-cuda130 \
  cupy cupy-cuda12x cupy-cuda13x
```

## 2. Install PyTorch

Pick exactly one option.

### Option A: CPU only

```bash
python -m pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cpu
```

### Option B: CUDA 12.6

```bash
python -m pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu126
```

### Option C: CUDA 12.8

```bash
python -m pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128
```

### Option D: CUDA 13.0

```bash
python -m pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu130
```

## 3. Reproducible pinned PyTorch install

If you want a pinned version instead of the latest available compatible wheel, use one of the following commands.

### CPU only

```bash
python -m pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cpu
```

### CUDA 12.6

```bash
python -m pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu126
```

### CUDA 12.8

```bash
python -m pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

### CUDA 13.0

```bash
python -m pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu130
```

## 4. Install optional DALI and CuPy support

Skip this section for a CPU only install.

### For CUDA 12.6 or CUDA 12.8

```bash
python -m pip install --extra-index-url https://pypi.nvidia.com --upgrade nvidia-dali-cuda120
python -m pip install cupy-cuda12x
```

### For CUDA 13.0

```bash
python -m pip install --extra-index-url https://pypi.nvidia.com --upgrade nvidia-dali-cuda130
python -m pip install cupy-cuda13x
```

## 5. Install PyTorch Geometric

For basic PyTorch Geometric usage, this is usually enough:

```bash
python -m pip install torch-geometric
```

For the optional compiled extensions, first check your installed Torch and CUDA versions:

```bash
python - <<'PY'
import torch

torch_version = torch.__version__.split("+")[0]
cuda_tag = "cpu" if torch.version.cuda is None else "cu" + torch.version.cuda.replace(".", "")

print("Torch version:", torch_version)
print("CUDA tag:", cuda_tag)
PY
```

Then choose the matching command.

### CPU only

```bash
TORCH=2.10.0
CUDA=cpu

python -m pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html

python -m pip install torch-geometric
```

### CUDA 12.6

```bash
TORCH=2.10.0
CUDA=cu126

python -m pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html

python -m pip install torch-geometric
```

### CUDA 12.8

```bash
TORCH=2.10.0
CUDA=cu128

python -m pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html

python -m pip install torch-geometric
```

### CUDA 13.0

```bash
TORCH=2.10.0
CUDA=cu130

python -m pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html

python -m pip install torch-geometric
```

If you installed a Torch version different from `2.10.0`, replace `TORCH=2.10.0` with the exact base version printed by:

```bash
python -c "import torch; print(torch.__version__.split('+')[0])"
```

## 6. Verify the installation

### Verify PyTorch

```bash
python - <<'PY'
import torch

print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("Running on CPU")
PY
```

### Verify DALI

```bash
python - <<'PY'
import nvidia.dali as dali

print("DALI:", dali.__version__)
PY
```

### Verify CuPy

```bash
python - <<'PY'
import cupy as cp

print("CuPy:", cp.__version__)
print("CUDA devices:", cp.cuda.runtime.getDeviceCount())
PY
```

## 7. Optional VQA dependency profiles

If you want to use OpenSportsLib VQA, install one backend-specific Hugging Face
dependency profile after the base Torch setup:

### X-VARS-compatible VQA

Use:

```bash
opensportslib setup --vqa_xvars
```

This installs the pinned versions from `XVARS_DEPENDENCY_PINS`:

- `transformers==4.38.2`
- `peft==0.9.0`
- `tokenizers==0.15.2`
- `accelerate==0.27.2`
- `trl==0.10.1`

### Qwen-compatible VQA

Use:

```bash
opensportslib setup --vqa_qwen
```

This installs the pinned versions from `QWEN_DEPENDENCY_PINS`:

- `transformers==5.13.0`
- `peft==0.19.0`
- `tokenizers==0.22.1`
- `accelerate==1.14.0`
- `trl==1.7.1`

The `vqa_qwen` config supports `Qwen/Qwen2.5-7B-Instruct` and
`Qwen/Qwen3.5-9B-Base`.

### Verify PyTorch Geometric

```bash
python - <<'PY'
import torch_geometric

print("PyG:", torch_geometric.__version__)
PY
```

## 7. Common issues

### `torch.cuda.is_available()` is `False`

Check that you did not accidentally install the CPU wheel:

```bash
python - <<'PY'
import torch

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
PY
```

If `torch.version.cuda` prints `None`, uninstall Torch and reinstall using one of the CUDA commands above.

### `nvidia-smi` shows CUDA 12.8, but you installed `cu126`

This can still be valid. `nvidia-smi` reports the maximum CUDA version supported by the driver. PyTorch wheels include their own CUDA runtime. Use a PyTorch wheel supported by your driver and by your project dependencies.

### DALI import fails

DALI dynamically links against CUDA libraries. Make sure your system has a compatible CUDA Toolkit installed and that the CUDA libraries are visible in your environment.

For example:

```bash
export CUDA_PATH=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
```

### PyG extensions fail to install

Use the exact Torch version and CUDA tag in the PyG wheel URL.

Example for Torch 2.10.0 with CUDA 12.8:

```bash
python -m pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-2.10.0+cu128.html
```

On some platforms, especially Linux arm64, prebuilt wheels may not exist for every optional PyG extension. In that case, start with:

```bash
python -m pip install torch-geometric
```

and only install compiled extensions when your code requires them.
