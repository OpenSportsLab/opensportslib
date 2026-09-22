"""Verify the Qwen, X-VARS, and GAR release manifests describe one passing commit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_PROFILES = {"qwen", "xvars", "gar"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifests", nargs="+", help="release-manifest.json paths copied from release servers")
    args = parser.parse_args()
    records = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.manifests]
    profiles = {record.get("profile") for record in records}
    commits = {record.get("commit") for record in records}
    failed = [record.get("profile") for record in records if record.get("result") != "passed"]
    if profiles != REQUIRED_PROFILES:
        raise SystemExit(f"Expected manifests for {sorted(REQUIRED_PROFILES)}, got {sorted(profiles)}.")
    if len(commits) != 1 or "unknown" in commits:
        raise SystemExit(f"Release manifests must identify one known commit, got {sorted(commits)}.")
    if failed:
        raise SystemExit(f"Release profiles failed: {failed}.")
    print(f"Release manifests verified for commit {next(iter(commits))}: {', '.join(sorted(profiles))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
