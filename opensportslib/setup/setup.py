import subprocess
import sys


CUDA_SUPPORT = [
    "cu126",
    "cu128",
    "cu130",
    "cpu"
]

def get_cuda_version():
    try:
        output = subprocess.check_output(["nvidia-smi"]).decode()

        for line in output.split("\n"):
            if "CUDA Version" in line:
                ver  = line.split("CUDA Version:")[1].strip().split()[0]
                print(f"CUDA Version found : {ver}")
                return ver
    except Exception:
        return None


CUDA_VERSION = get_cuda_version()

def get_cpu_tag():
    if not CUDA_VERSION:
        return "cpu"

def install_torch():
    python = sys.executable
    for cuda in CUDA_SUPPORT:

        print(f"\n Trying installation: {cuda}\n")
        try:
            if get_cpu_tag() == "cpu":
                subprocess.check_call([
                    python, "-m", "pip", "install",
                    "torch", "torchvision", "torchaudio"
                ])
            else:

                subprocess.check_call([
                    python, "-m", "pip", "install",
                    "torch", "torchvision", "torchaudio",
                    "--index-url",
                    f"https://download.pytorch.org/whl/{cuda}"
                ])
            print(f"\nSuccess with {cuda}")
            return cuda

        except Exception as e:
            print(f"Failed with {cuda}: {e}")

    raise RuntimeError("All CUDA installs failed")

def install_dali():

    python = sys.executable

    print("\nInstalling dali extras...\n")

    # DALI (only if GPU)
    if CUDA_VERSION:
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
    cuda_tag = next((f"cu{CUDA_VERSION.replace('.', '')}" for _ in [0] if CUDA_VERSION), CUDA_SUPPORT[0])
    cuda_tag = cuda_tag if cuda_tag in CUDA_SUPPORT else CUDA_SUPPORT[0]
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

def setup(dali=False, pyg=False):
    install_torch()
    install_extras(dali=dali, pyg=pyg)
    verify()


# ----------------------------
# CLI entry
# ----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dali", action="store_true")
    parser.add_argument("--pyg", action="store_true")

    args = parser.parse_args()

    setup(dali=args.dali, pyg=args.pyg)