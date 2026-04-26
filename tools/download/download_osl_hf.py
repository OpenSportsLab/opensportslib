import argparse
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from opensportslib.tools.hf_transfer import download_dataset_from_hf


def main(
    osl_json_url: str,
    output_dir: str = "downloaded_data",
    dry_run: bool = False,
    types_arg: str = "video",
    token: str | None = None,
) -> None:
    result = download_dataset_from_hf(
        osl_json_url,
        output_dir,
        dry_run=dry_run,
        types_arg=types_arg,
        token=token,
        progress_cb=lambda msg: print(f"[HF] {msg}"),
    )

    print(f"Repo: {result['repo_id']} @ {result['revision']}")
    print(f"JSON path: {result['json_path']}")
    download_kind = result.get("download_kind", "json")
    if "referenced_file_count" in result:
        print(f"Matched files: {result['referenced_file_count']}")
    elif download_kind == "parquet":
        print(
            "Folder download mode: converted parquet folder to OSL JSON and "
            f"extracted {result.get('extracted_media_count', 0)} media files."
        )

    if dry_run:
        print("Running in DRY-RUN mode (no files downloaded).")
        files = result.get("files", [])
        for file_info in files:
            print(f"[DRY RUN] Repo file : {file_info['repo_path']} ({file_info['size_human']})")
            print(f"[DRY RUN] Local path: {file_info['local_path']}")

        print("-" * 48)
        print(f"Total estimated storage needed: {result.get('estimated_total_size_human', '0.0 B')}")
        missing = result.get("missing_files", [])
        if missing:
            print(f"WARNING: {len(missing)} files not found in repo metadata.")
            for path in missing[:50]:
                print(f"  - {path}")
            if len(missing) > 50:
                print(f"  ... and {len(missing) - 50} more")
        return

    print(
        f"Done. Downloaded {result.get('downloaded_file_count', 0)} files to: "
        f"{os.path.abspath(result['output_dir'])}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Download OSL data from HuggingFace using either a JSON file URL "
            "or a dataset folder URL (tree/...)."
        )
    )
    parser.add_argument(
        "--url",
        required=True,
        help=(
            "URL of an OSL JSON file (blob/resolve) or a dataset folder URL "
            "(tree/<revision>)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="downloaded_data",
        help="Directory to store downloaded files.",
    )
    parser.add_argument(
        "--types",
        default="video",
        help=(
            "Comma-separated input types (e.g. 'video', 'video,captions,features') "
            "or 'all'. Default: video."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files without downloading them and estimate storage size.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face token override. If omitted, local HF login is used.",
    )

    args = parser.parse_args()
    main(
        osl_json_url=args.url,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        types_arg=args.types,
        token=args.token,
    )
