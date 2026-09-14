from __future__ import annotations

import os

from huggingface_hub import snapshot_download


REPOS = [
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "OpenSportsLab/OSL-VQA-XFOUL-qwen2.5-7B-VL-lora",
    "Qwen/Qwen3-VL-8B-Instruct",
    "OpenSportsLab/OSL-VQA-XFOUL-qwen3-8B-VL-lora",
    "OpenSportsLab/OSL-VQA-XFOUL-XVARS-lora",
    "OpenSportsLab/OSL-cls-action-mvitv2",
    "OpenSportsLab/trained-clip-vit-large-patch14",
    "OpenSportsLab/base_model_videoChatGPT",
    "OpenSportsLab/OSL-loc-snbas-2025-e2e",
    #"OpenSportsLab/OSL-loc-snbas-2023-e2e",
]


def main() -> None:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    cache_dir = os.getenv("OSL_HF_CACHE_DIR") or os.getenv("HF_HOME")

    print("Predownloading Hugging Face assets...")
    if cache_dir:
        print(f"Using cache dir: {cache_dir}")

    for repo_id in REPOS:
        print(f"Downloading {repo_id}")
        snapshot_download(
            repo_id=repo_id,
            token=token,
            cache_dir=cache_dir,
            resume_download=True,
        )

    print("All listed Hugging Face assets are available in cache.")


if __name__ == "__main__":
    main()
