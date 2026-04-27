import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from opensportslib.tools.hf_transfer import (
    create_dataset_branch_on_hf,
    create_dataset_repo_on_hf,
    upload_dataset_as_parquet_to_hf,
    upload_dataset_inputs_from_json_to_hf,
)
from opensportslib.tools.osl_json_to_parquet import DEFAULT_SHARD_SIZE, parse_shard_size


def main(
    repo_id: str,
    json_path: str,
    revision: str = "main",
    commit_message: str | None = None,
    token: str | None = None,
    upload_format: str = "json",
    shard_mode: str = "size",
    shard_size: int | str = DEFAULT_SHARD_SIZE,
    samples_per_shard: int = 100,
) -> None:
    # Ensure target repo exists (idempotent via exist_ok=True in helper).
    print(f"[HF] Ensuring dataset repo exists: {repo_id}")
    create_dataset_repo_on_hf(
        repo_id=repo_id,
        token=token,
        progress_cb=lambda msg: print(f"[HF] {msg}"),
    )

    # Ensure target revision exists when uploading to a non-main branch.
    cleaned_revision = str(revision or "").strip() or "main"
    if cleaned_revision != "main":
        print(f"[HF] Ensuring branch exists: {repo_id}@{cleaned_revision}")
        create_dataset_branch_on_hf(
            repo_id=repo_id,
            branch=cleaned_revision,
            source_revision="main",
            token=token,
            progress_cb=lambda msg: print(f"[HF] {msg}"),
        )

    if upload_format == "parquet":
        result = upload_dataset_as_parquet_to_hf(
            repo_id=repo_id,
            json_path=json_path,
            revision=cleaned_revision,
            commit_message=commit_message,
            shard_mode=shard_mode,
            shard_size=shard_size,
            samples_per_shard=samples_per_shard,
            token=token,
            progress_cb=lambda msg: print(f"[HF] {msg}"),
        )
    else:
        result = upload_dataset_inputs_from_json_to_hf(
            repo_id=repo_id,
            json_path=json_path,
            revision=cleaned_revision,
            commit_message=commit_message,
            token=token,
            progress_cb=lambda msg: print(f"[HF] {msg}"),
        )

    print("Upload complete.")
    print(f"Format: {result.get('upload_kind', upload_format)}")
    print(f"Repo: {result['repo_id']}")
    print(f"Branch: {result['revision']}")
    print(f"Dataset JSON: {result['json_path']}")
    if "folder_name" in result:
        print(f"Repo folder: {result['folder_name']}")
    if "num_samples" in result:
        print(f"Samples: {result['num_samples']}")
    if "shard_size" in result:
        print(f"Shard mode: {result.get('shard_mode', 'size')}")
        print(f"Shard size: {result['shard_size']}")
    if result.get("shard_mode") == "samples" and "samples_per_shard" in result:
        print(f"Samples per shard: {result['samples_per_shard']}")
    if "unique_input_file_count" in result:
        print(f"Unique input files: {result['unique_input_file_count']}")
    print(f"Uploaded files: {result['uploaded_file_count']}")
    print(f"Commit message: {result['commit_message']}")
    print(f"Commit ref: {result['commit_ref']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Upload a local dataset JSON to a Hugging Face dataset repo as either "
            "raw JSON-linked inputs or Parquet + WebDataset shards."
        )
    )
    parser.add_argument("--repo-id", required=True, help="Dataset repo id, e.g. OpenSportsLab/OSL-loc-tennis-public")
    parser.add_argument("--json-path", required=True, help="Local dataset JSON path.")
    parser.add_argument(
        "--revision",
        default="main",
        help="Target branch/revision in the dataset repo (default: main).",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload dataset inputs from JSON",
        help="Optional commit message prefix.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face token override. If omitted, local HF login is used.",
    )
    parser.add_argument(
        "--format",
        default="json",
        choices=["json", "parquet"],
        help="Upload mode: json (raw inputs) or parquet (Parquet + WebDataset).",
    )
    parser.add_argument(
        "--shard-mode",
        default="size",
        choices=["size", "samples"],
        help="Shard grouping mode for parquet mode (default: size).",
    )
    parser.add_argument(
        "--shard-size",
        default="1GB",
        help="Target TAR shard size for parquet size mode, e.g. 500MB, 1GB, 1024MiB (default: 1GB).",
    )
    parser.add_argument(
        "--samples-per-shard",
        type=int,
        default=100,
        help="Samples per shard for parquet sample mode (default: 100).",
    )

    args = parser.parse_args()
    if args.shard_mode != "samples" and args.samples_per_shard != 100:
        parser.error("--samples-per-shard can only be used with --shard-mode samples")
    main(
        repo_id=args.repo_id,
        json_path=args.json_path,
        revision=args.revision,
        commit_message=args.commit_message,
        token=args.token,
        upload_format=args.format,
        shard_mode=args.shard_mode,
        shard_size=parse_shard_size(args.shard_size),
        samples_per_shard=args.samples_per_shard,
    )
