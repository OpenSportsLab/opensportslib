"""
CLI wrapper — convert Parquet + WebDataset TAR shards back to an OSL JSON file.

All logic lives in opensportslib.tools.parquet_to_osl_json.
Run with --help for full usage.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running directly from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from opensportslib.tools import convert_parquet_to_json


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert Parquet + WebDataset TAR shards back to an OSL-style JSON annotation file."
    )
    parser.add_argument(
        "dataset_dir",
        help="Directory produced by osl_json_to_parquet_webdataset.py (must contain metadata.parquet and shards/).",
    )
    parser.add_argument("output_json_path", help="Destination path for the reconstructed OSL JSON file.")
    parser.add_argument(
        "--extract-media",
        action="store_true",
        help="Extract media files from TAR shards into --output-media-root, preserving their original relative paths.",
    )
    parser.add_argument(
        "--output-media-root",
        default=None,
        metavar="DIR",
        help="Directory to extract media files into. Required when --extract-media is set.",
    )
    parser.add_argument(
        "--overwrite-media",
        action="store_true",
        help="Overwrite already-extracted media files.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        metavar="N",
        help="JSON indentation level for the output file (default: 2).",
    )

    args = parser.parse_args()

    if args.extract_media and args.output_media_root is None:
        parser.error("--output-media-root is required when --extract-media is set.")

    result = convert_parquet_to_json(
        dataset_dir=args.dataset_dir,
        output_json_path=args.output_json_path,
        extract_media=args.extract_media,
        output_media_root=args.output_media_root,
        overwrite_media=args.overwrite_media,
        json_indent=args.indent,
    )

    print(json.dumps(result, indent=2))
