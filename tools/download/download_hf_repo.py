"""Download a full HuggingFace repository (dataset or model) using snapshot_download.

Usage:
    python tools/download/download_hf_repo.py \
        --repo-id OpenSportsLab/OSL-XFoul \
        --revision main-parquet \
        --output-dir /ibex/project/c2134/opensportslab/datasets/OSL-XFoul/main-parquet
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("ERROR: huggingface_hub is not installed. Run: pip install huggingface-hub")
    sys.exit(1)


def main(
    repo_id: str,
    revision: str = "main",
    output_dir: str | None = None,
    repo_type: str = "dataset",
    token: str | None = None,
    ignore_patterns: list[str] | None = None,
) -> None:
    if output_dir is None:
        # Default: <repo_name>/<revision> in cwd
        repo_name = repo_id.split("/")[-1]
        output_dir = os.path.join(repo_name, revision)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Repo   : {repo_id}")
    print(f"Version: {revision}")
    print(f"Type   : {repo_type}")
    print(f"Dest   : {os.path.abspath(output_dir)}")
    if ignore_patterns:
        print(f"Ignoring patterns: {ignore_patterns}")
    print("-" * 60)

    local_dir = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        repo_type=repo_type,
        local_dir=output_dir,
        token=token,
        ignore_patterns=ignore_patterns,
    )

    print(f"Done. Repository downloaded to: {os.path.abspath(local_dir)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download a full HuggingFace repository via snapshot_download."
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="HuggingFace repo ID, e.g. OpenSportsLab/OSL-XFoul",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Branch, tag, or commit SHA to download (default: main).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Local directory to download into. "
            "Defaults to <repo_name>/<revision> in the current directory."
        ),
    )
    parser.add_argument(
        "--repo-type",
        default="dataset",
        choices=["dataset", "model", "space"],
        help="Type of HuggingFace repository (default: dataset).",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HuggingFace access token for gated repos. Falls back to HF_TOKEN env var.",
    )
    parser.add_argument(
        "--ignore",
        nargs="*",
        default=None,
        metavar="PATTERN",
        help="Glob patterns to exclude, e.g. --ignore '*.parquet' '*.arrow'.",
    )

    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")

    main(
        repo_id=args.repo_id,
        revision=args.revision,
        output_dir=args.output_dir,
        repo_type=args.repo_type,
        token=token,
        ignore_patterns=args.ignore,
    )
