import platform
import subprocess
import sys


CUDA_WHEEL_VERSIONS = {
    "cu126": (12, 6),
    "cu128": (12, 8),
    "cu130": (13, 0),
}
MIN_SUPPORTED_COMPUTE_CAPABILITY = (5, 0)
LEGACY_GPU_MAX_COMPUTE_CAPABILITY = (7, 4)
LEGACY_GPU_CUDA_WHEEL = "cu126"
LEGACY_GPU_CUDA_WHEEL_MAX_COMPUTE_CAPABILITY = (9, 0)
CUDA13_REQUIRED_MIN_COMPUTE_CAPABILITY = (10, 0)

XVARS_DEPENDENCY_PINS = {
    "transformers": "4.38.2",
    "peft": "0.9.0",
    "tokenizers": "0.15.2",
    "accelerate": "0.27.2",
    "trl": "0.10.1",
}

QWEN_DEPENDENCY_PINS = {
    "transformers": "5.13.0",
    "peft": "0.19.0",
    "tokenizers": "0.22.1",
    "accelerate": "1.14.0",
    "trl": "1.7.1",
}

def get_cuda_version():
    try:
        output = subprocess.check_output(["nvidia-smi"]).decode()

        for line in output.split("\n"):
            if "CUDA Version" in line:
                ver  = line.split("CUDA Version:")[1].strip().split()[0]
                print(f"CUDA Version found : {ver}")
                cuda_tag = f"cu{ver.replace('.', '')}"
                return ver, cuda_tag
    except Exception:
        return None, None
    return None, None


def get_gpu_compute_capabilities():
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True,
        )
    except Exception:
        return []

    capabilities = []
    for line in output.splitlines():
        value = line.strip()
        if not value:
            continue
        try:
            major, minor = value.split(".", 1)
            capabilities.append((int(major), int(minor)))
        except ValueError:
            print(f"Ignoring unrecognized GPU compute capability: {value}")
    return capabilities


def select_cuda_wheel(cuda_version, compute_capabilities):
    if not cuda_version:
        return "cpu"

    try:
        driver_version = tuple(int(part) for part in cuda_version.split(".", 1))
    except ValueError as exc:
        raise RuntimeError(f"Unable to parse CUDA version reported by nvidia-smi: {cuda_version}") from exc

    if any(capability < MIN_SUPPORTED_COMPUTE_CAPABILITY for capability in compute_capabilities):
        raise RuntimeError(
            "OpenSportsLib PyTorch wheels require GPUs with compute capability "
            f"{MIN_SUPPORTED_COMPUTE_CAPABILITY[0]}.{MIN_SUPPORTED_COMPUTE_CAPABILITY[1]} or newer. "
            f"Detected: {compute_capabilities}."
        )

    compatible_tags = [
        tag for tag, wheel_version in CUDA_WHEEL_VERSIONS.items() if wheel_version <= driver_version
    ]
    has_legacy_gpu = compute_capabilities and any(
        capability <= LEGACY_GPU_MAX_COMPUTE_CAPABILITY for capability in compute_capabilities
    )
    if has_legacy_gpu:
        if platform.machine().lower() in {"aarch64", "arm64"}:
            raise RuntimeError(
                "Official Linux ARM64 CUDA 12.6 PyTorch wheels support Ampere and newer GPUs only. "
                "Use an x86_64 host or build PyTorch from source for the visible legacy GPU architecture."
            )
        if any(
            capability > LEGACY_GPU_CUDA_WHEEL_MAX_COMPUTE_CAPABILITY
            for capability in compute_capabilities
        ):
            raise RuntimeError(
                "Visible GPUs require incompatible prebuilt PyTorch wheels. Set CUDA_VISIBLE_DEVICES to "
                "either the legacy GPUs or the newer GPUs, then run setup again."
            )
        compatible_tags = [tag for tag in compatible_tags if tag == LEGACY_GPU_CUDA_WHEEL]

    if any(
        capability >= CUDA13_REQUIRED_MIN_COMPUTE_CAPABILITY
        for capability in compute_capabilities
    ):
        compatible_tags = [tag for tag in compatible_tags if tag == "cu130"]

    if not compatible_tags:
        if any(
            capability >= CUDA13_REQUIRED_MIN_COMPUTE_CAPABILITY
            for capability in compute_capabilities
        ):
            raise RuntimeError(
                "The visible GPU architecture requires the CUDA 13.0 PyTorch wheel. "
                "Update the NVIDIA driver so nvidia-smi reports CUDA 13.0 or newer, "
                "then run setup again."
            )
        raise RuntimeError(
            f"CUDA {cuda_version} is too old for the supported PyTorch wheels: "
            f"{', '.join(CUDA_WHEEL_VERSIONS)}."
        )

    selected = max(compatible_tags, key=lambda tag: CUDA_WHEEL_VERSIONS[tag])
    print(
        "Selected PyTorch wheel "
        f"{selected} for CUDA {cuda_version} and GPU compute capabilities {compute_capabilities or 'unknown'}"
    )
    return selected


def select_torch_packages(compute_capabilities):
    return ("torch", "torchvision", "torchaudio")


CUDA_VERSION, _DETECTED_CUDA_TAG = get_cuda_version()
GPU_COMPUTE_CAPABILITIES = get_gpu_compute_capabilities()
CUDA_TAG = select_cuda_wheel(CUDA_VERSION, GPU_COMPUTE_CAPABILITIES)


def install_xvars_dependencies(DEPENDENCY_PINS):
    python = sys.executable
    packages = list(DEPENDENCY_PINS)
    pinned_packages = [f"{name}=={version}" for name, version in DEPENDENCY_PINS.items()]

    print(f"\nInstalling {list(DEPENDENCY_PINS.keys())} dependency overrides...\n")
    print("This overrides the default Hugging Face dependency set with XVars-compatible versions.")
    subprocess.call([python, "-m", "pip", "uninstall", "-y", *packages])
    subprocess.check_call([python, "-m", "pip", "install", *pinned_packages])
    print("Dependencies installed successfully.")

def install_torch():
    python = sys.executable
    subprocess.call([python, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"])
    packages = select_torch_packages(GPU_COMPUTE_CAPABILITIES)

    subprocess.check_call([
        python, "-m", "pip", "install",
        *packages,
        "--index-url",
        f"https://download.pytorch.org/whl/{CUDA_TAG}",
    ])
    print(f"\nSuccess with {CUDA_TAG}: {', '.join(packages)}")
    return CUDA_TAG

def install_dali():

    python = sys.executable

    print("\nInstalling dali extras...\n")

    # DALI (only if GPU)
    if CUDA_VERSION:
        
        if CUDA_TAG == "cu130":
            subprocess.check_call([
                python, "-m", "pip", "install",
                "nvidia-dali-cuda130"
            ])

            # CuPy (CUDA-aware but auto-resolves internally)
            subprocess.check_call([
                python, "-m", "pip", "install",
                "cupy-cuda130"
            ])
        else:
            subprocess.check_call([
                python, "-m", "pip", "install",
                "nvidia-dali-cuda120"
            ])

            # CuPy (CUDA-aware but auto-resolves internally)
            subprocess.check_call([
                python, "-m", "pip", "install",
                "cupy-cuda12x"
            ])

def install_pyg():
    import torch
    from packaging import version

    python = sys.executable
    torch_version = "2.10.0" if version.parse(torch.__version__.split("+")[0]) > version.parse("2.10.0") else torch.__version__.split("+")[0]
    cuda_tag = CUDA_TAG
    print("\nInstalling Py-Geometric ecosystem...\n")
    if cuda_tag == "cpu":
        url =  f"https://data.pyg.org/whl/torch-{torch_version}+cpu.html"
    else:
        url = f"https://data.pyg.org/whl/torch-{torch_version}+{cuda_tag}.html"

    subprocess.check_call([
        python, "-m", "pip", "install",
        "torch-geometric", "-f", url
    ])
    subprocess.check_call([
        python, "-m", "pip", "install",
        "torch-scatter", "-f", url
    ])
    subprocess.check_call([
        python, "-m", "pip", "install",
        "torch-sparse", "-f", url
    ])
    subprocess.check_call([
        python, "-m", "pip", "install",
        "torch-cluster", "-f", url
    ])
    subprocess.check_call([
        python, "-m", "pip", "install",
        "torch-spline-conv", "-f", url
    ])

def install_extras(dali=False, pyg=False):
    if dali:
        install_dali()
        print("NVIDIA DALI installed successfully.")
    if pyg:
        install_pyg()
        print("PyTorch Geometric installed successfully.")


def verify():
    import torch

    print("\n Verifying installation...\n")
    print("Torch:", torch.__version__)

    if torch.cuda.is_available():
        print("CUDA available")
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("Running on CPU")

def setup(dali=False, pyg=False, vqa_xvars=False, vqa_qwen=False):
    install_torch()
    install_extras(dali=dali, pyg=pyg)
    if vqa_xvars:
        install_xvars_dependencies(XVARS_DEPENDENCY_PINS)
    if vqa_qwen:
        install_xvars_dependencies(QWEN_DEPENDENCY_PINS)
    verify()


# ----------------------------
# CLI entry
# ----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dali", action="store_true")
    parser.add_argument("--pyg", action="store_true")
    parser.add_argument("--vqa_xvars", action="store_true")
    parser.add_argument("--vqa_qwen", action="store_true")

    args = parser.parse_args()

    setup(dali=args.dali, pyg=args.pyg, vqa_xvars=args.vqa_xvars, vqa_qwen=args.vqa_qwen)
