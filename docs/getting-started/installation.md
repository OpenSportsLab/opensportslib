# Installation

OpenSportsLib requires Python 3.12 or newer. Install the package first, then run its setup command to install a PyTorch profile for the current machine. The setup command replaces the installed Torch stack; use a dedicated environment.

## Base installation

```bash
conda create -n osl python=3.12 pip
conda activate osl
python -m pip install --upgrade pip
pip install opensportslib
opensportslib setup
```

For the latest prerelease, use `pip install --pre opensportslib`. For source development, clone the repository and replace the install command with `pip install -e .`.

`opensportslib setup` detects `nvidia-smi` when available and installs a CPU wheel when it is not. It selects among CUDA 12.6, 12.8, and 13.0 wheel profiles from the reported driver and visible GPU compute capabilities; it is not a general CUDA toolkit installer.

## Optional profiles

| Command | Use when | Important behavior |
| --- | --- | --- |
| `opensportslib setup --dali` | Using DALI video loading on an NVIDIA GPU | Installs DALI/CuPy for the selected CUDA profile; DALI is not a CPU loader. |
| `opensportslib setup --pyg` | Using graph/tracking models | Replaces Torch with the pinned PyG-compatible Torch 2.12.1 profile and installs `torch-geometric`. |
| `opensportslib setup --pyg --pyg_extensions` | Optional compiled PyG extensions are required | Verifies binary wheels before installing extension packages. |
| `opensportslib setup --vqa_xvars` | Running X-VARS VQA | Replaces the Hugging Face dependency set with X-VARS pins. |
| `opensportslib setup --vqa_qwen` | Running Qwen VQA | Replaces the Hugging Face dependency set with Qwen pins. |

The X-VARS and Qwen profiles have incompatible dependency pins; use separate environments when both are needed. On CPU, compatible DALI-video configs are normalized to the OpenCV loader.

## Verify and authenticate

```bash
python -c "from opensportslib.apis import ClassificationModel, LocalizationModel, VQAModel, Config; print('OpenSportsLib ready')"
opensportslib setup --help
```

Hugging Face access is required only for gated datasets or model repositories. Authenticate with `hf auth login` before such a workflow. The separately packaged inference server is not installed by `pip install opensportslib`; see the [server guide](../server/inference-server.md).
