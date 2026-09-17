#!/usr/bin/env python3
"""Inspect or recover stale OpenSportsLib server runtime jobs."""

from __future__ import annotations

import argparse
import json
import os
import sys

from opensportslib import RemoteModelRegistry, RemoteRegistryError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default=os.getenv("OSL_SERVER_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--token", default=os.getenv("OSL_MODEL_ADMIN_TOKEN"))
    parser.add_argument("--apply", action="store_true", help="Apply recovery and cleanup; default is dry-run.")
    parser.add_argument("--include-active", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args()
    if not args.token:
        parser.error("provide --token or set OSL_MODEL_ADMIN_TOKEN")
    try:
        report = RemoteModelRegistry(args.server, args.token).reconcile_runtime(
            dry_run=not args.apply,
            include_active=args.include_active,
        )
    except (RemoteRegistryError, ConnectionError, TimeoutError) as exc:
        print(f"runtime reconciliation failed: {exc}", file=sys.stderr)
        return 1
    if args.json_output:
        print(json.dumps(report, indent=2))
    else:
        mode = "applied" if args.apply else "dry-run"
        print(f"Runtime reconciliation ({mode})")
        for key in ("inspected", "active", "recovered", "cleaned"):
            print(f"{key}: {report.get(key, 0)}")
        for job in report.get("jobs", []):
            print(f"{job.get('job_id')}: {job.get('reason')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
