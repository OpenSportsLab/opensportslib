#!/usr/bin/env python3
"""Upload completed native-rate SNGAR tracking with OSL Parquet/WebDataset tools."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pyarrow.parquet as pq

from opensportslib.tools.hf_transfer import (
    create_dataset_branch_on_hf,
    dataset_repo_exists_on_hf,
    upload_dataset_as_parquet_to_hf,
)
from opensportslib.tools.osl_json_to_parquet import parse_shard_size

BRANCH = "tracking-full"
COUNTS = {"train": 62159, "valid": 12091, "test": 13689}
DEFAULT_ROOT = Path("/home/giancos/OSLdata/sngar_tracking_classification_full_fps/tracking")


def preflight(root: Path) -> dict:
    """Check the complete local build before any Hub calls, including on retries."""
    root = root.resolve()
    report = json.loads((root.parent / "reports/build.json").read_text())
    if report.get("status") != "passed" or report.get("completed") != sum(COUNTS.values()):
        raise ValueError("Finish the tracking conversion first: reports/build.json is not a complete passed build.")
    summary = {}
    all_ids = set()
    for split, expected in COUNTS.items():
        manifest = root / f"annotations_{split}.json"
        data = json.loads(manifest.read_text())
        samples = data["data"]
        if data.get("task") != "action_classification" or len(samples) != expected:
            raise ValueError(f"{manifest}: expected {expected} classification samples")
        size = 0
        for sample in samples:
            sample_id = sample["id"]
            if sample_id in all_ids:
                raise ValueError(f"Duplicate sample ID: {sample_id}")
            all_ids.add(sample_id)
            inputs = sample["inputs"]
            conversion = sample.get("metadata", {}).get("conversion", {})
            if len(inputs) != 1 or inputs[0].get("type") != "tracking_parquet":
                raise ValueError(f"{sample_id}: expected one tracking_parquet input")
            if (conversion.get("identity") != report.get("identity")
                    or conversion.get("frame_interval") != 1
                    or conversion.get("window_slots") != 144):
                raise ValueError(f"{sample_id}: missing or inconsistent full-FPS provenance")
            path = (root / inputs[0]["path"]).resolve()
            if not path.is_relative_to(root / split) or path.suffix != ".parquet":
                raise ValueError(f"{sample_id}: input must be a local Parquet file under {split}/")
            metadata = pq.read_metadata(path)
            rows = conversion.get("output_rows")
            if metadata.num_rows != rows or not 0 < rows <= 144:
                raise ValueError(f"{sample_id}: Parquet row count disagrees with native window")
            if conversion.get("gap_policy") == "pad" and rows != 144:
                raise ValueError(f"{sample_id}: padded native window must contain 144 rows")
            if conversion.get("gap_policy") not in ("pad", "observed"):
                raise ValueError(f"{sample_id}: unknown tracking gap policy")
            if "tracking_valid" not in metadata.schema.names:
                raise ValueError(f"{sample_id}: missing tracking_valid column")
            size += path.stat().st_size
        summary[split] = {"samples": len(samples), "source_bytes": size}
        print(f"Checked {split}: {len(samples):,} samples, {size / 2**30:.2f} GiB", flush=True)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--repo-id", default="OpenSportsLab/SoccerNet-GAR")
    parser.add_argument("--shard-size", default="1GB", help="OSL size target per TAR shard (default: 1GB)")
    parser.add_argument("--splits", nargs="+", choices=list(COUNTS), default=list(COUNTS),
                        help="Splits to upload; all three are always checked first")
    parser.add_argument("--check-only", action="store_true", help="Validate locally; no Hub calls or uploads")
    args = parser.parse_args(argv)
    try:
        shard_size = parse_shard_size(args.shard_size)
        preflight(args.data_root)
        print(f"Target: {args.repo_id}@{BRANCH}; splits: {', '.join(args.splits)}", flush=True)
        if args.check_only:
            print("Local checks passed. Nothing uploaded.")
            return 0
        if not dataset_repo_exists_on_hf(args.repo_id):
            raise ValueError(f"Dataset repository does not exist or is inaccessible: {args.repo_id}")
        create_dataset_branch_on_hf(args.repo_id, BRANCH, progress_cb=print)
        for split in dict.fromkeys(args.splits):
            result = upload_dataset_as_parquet_to_hf(
                args.repo_id, str(args.data_root.resolve() / f"annotations_{split}.json"),
                revision=BRANCH, split=split, shard_mode="size", shard_size=shard_size,
                commit_message=f"Upload SNGAR full-FPS tracking: {split}",
                progress_cb=lambda message: print(message, flush=True),
            )
            print(json.dumps(result, indent=2, default=str), flush=True)
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
