# Installation

OpenSportsLib requires Python 3.12 or newer. First install the package, then run its setup command to install the PyTorch version that matches your machine. Because setup replaces the installed PyTorch packages, use a separate environment for OpenSportsLib.

## Base installation

```bash
conda create -n osl python=3.12 pip
conda activate osl
python -m pip install --upgrade pip
pip install opensportslib
opensportslib setup
```

Use `pip install --pre opensportslib` for the latest prerelease. If you are developing from source, clone the repository and use `pip install -e .` instead of the normal install command.

When available, `opensportslib setup` checks `nvidia-smi` to identify your NVIDIA GPU. Without it, the command installs CPU packages. For supported GPUs, it chooses a CUDA 12.6, 12.8, or 13.0 package profile from the reported driver and GPU capabilities. It does not install or manage a general CUDA toolkit.

## Optional profiles

| Command | Use when | Important behavior |
| --- | --- | --- |
| `opensportslib setup --dali` | Using DALI video loading on an NVIDIA GPU | Installs DALI/CuPy for the selected CUDA profile; DALI is not a CPU loader. |
| `opensportslib setup --pyg` | Using graph/tracking models | Replaces Torch with the pinned PyG-compatible Torch 2.12.1 profile and installs `torch-geometric`. |
| `opensportslib setup --pyg --pyg_extensions` | Optional compiled PyG extensions are required | Verifies binary wheels before installing extension packages. |
| `opensportslib setup --vqa_xvars` | Running X-VARS VQA | Replaces the Hugging Face dependency set with X-VARS pins. |
| `opensportslib setup --vqa_qwen` | Running Qwen VQA | Replaces the Hugging Face dependency set with Qwen pins. |

The X-VARS and Qwen profiles require incompatible package versions, so use separate environments if you need both. On a CPU-only machine, compatible DALI video configurations automatically use the OpenCV video loader instead.

## Verify and authenticate

```bash
python -c "from opensportslib.apis import ClassificationModel, LocalizationModel, VQAModel, Config; print('OpenSportsLib ready')"
opensportslib setup --help
```

You only need Hugging Face access for gated datasets or model repositories. Before using one, sign in with `hf auth login`. The inference server is a separate package and is not installed by `pip install opensportslib`; see the [server guide](../server/inference-server.md) when you need remote inference.
