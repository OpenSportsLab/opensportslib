"""Write one non-secret, commit-bound release profile manifest."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--profile", required=True, choices=("qwen", "xvars", "gar"))
    parser.add_argument("--scale", required=True, choices=("full", "bounded"))
    parser.add_argument("--exit-code", required=True, type=int)
    args = parser.parse_args()

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except subprocess.CalledProcessError:
        commit = "unknown"
    report_dir = Path(args.report_dir)
    metadata = report_dir / "release-metadata.jsonl"
    records = []
    if metadata.exists():
        for line in metadata.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "commit": commit,
        "profile": args.profile,
        "scale": args.scale,
        "result": "passed" if args.exit_code == 0 else "failed",
        "exit_code": args.exit_code,
        "metadata_records": records,
    }
    (report_dir / "release-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
