"""
CLI wrapper — convert an OSL JSON annotation file into Parquet + WebDataset TAR shards.

All logic lives in opensportslib.tools.osl_json_to_parquet.
Run with --help for full usage.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

# Allow running directly from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from opensportslib.tools import convert_json_to_parquet
from opensportslib.tools.osl_json_to_parquet import DEFAULT_SHARD_SIZE, parse_shard_size


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert an OSL-style JSON annotation file to Parquet + WebDataset TAR shards."
    )
    parser.add_argument("json_path", help="Path to the OSL JSON annotation file.")
    parser.add_argument("media_root", help="Root directory containing the input files referenced in the JSON.")
    parser.add_argument("output_dir", help="Destination directory for the converted dataset.")
    parser.add_argument(
        "--shard-mode",
        default="size",
        choices=["size", "samples"],
        help="Shard grouping mode (default: size).",
    )
    parser.add_argument(
        "--shard-size",
        default="1GB",
        metavar="SIZE",
        help="Target TAR shard size for size mode, e.g. 500MB, 1GB, 1024MiB (default: 1GB).",
    )
    parser.add_argument(
        "--samples-per-shard",
        type=int,
        default=100,
        metavar="N",
        help="Number of samples per TAR shard when --shard-mode samples is used (default: 100).",
    )
    parser.add_argument(
        "--compression",
        default="zstd",
        choices=["zstd", "snappy", "gzip", "brotli", "none"],
        help="Parquet compression codec (default: zstd). Use 'none' for no compression.",
    )
    parser.add_argument(
        "--shard-prefix",
        default="shard",
        help="Prefix for TAR shard file names (default: shard).",
    )
    parser.add_argument(
        "--missing-policy",
        default="raise",
        choices=["raise", "skip"],
        help="What to do when a referenced input file is missing (default: raise).",
    )
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        help="Store resolved absolute paths in Parquet instead of the original relative paths.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output directory if it already exists.",
    )

    args = parser.parse_args()
    if args.shard_mode != "samples" and args.samples_per_shard != 100:
        parser.error("--samples-per-shard can only be used with --shard-mode samples")
    compression_value: Optional[str] = None if args.compression == "none" else args.compression
    shard_size_value = parse_shard_size(args.shard_size or DEFAULT_SHARD_SIZE)

    result = convert_json_to_parquet(
        json_path=args.json_path,
        media_root=args.media_root,
        output_dir=args.output_dir,
        shard_mode=args.shard_mode,
        shard_size=shard_size_value,
        samples_per_shard=args.samples_per_shard,
        compression=compression_value,
        shard_prefix=args.shard_prefix,
        missing_policy=args.missing_policy,
        keep_relative_paths_in_parquet=not args.absolute_paths,
        overwrite=args.overwrite,
    )

    print(json.dumps(result, indent=2))
    missing_input_files = int(result.get("missing_input_files") or 0)
    if missing_input_files > 0:
        print(
            f"\nWARNING: {missing_input_files} input file(s) were missing.",
            file=sys.stderr,
        )
