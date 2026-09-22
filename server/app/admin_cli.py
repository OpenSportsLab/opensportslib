from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from opensportslib import RemoteModelRegistry, RemoteRegistryError


load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect or recover stale server jobs.")
    parser.add_argument("--server", default=os.getenv("OSL_SERVER_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--api-key", default=os.getenv("OSL_API_KEY"))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--include-active", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args()
    if not args.api_key:
        parser.error("provide --api-key or set OSL_API_KEY")
    try:
        report = RemoteModelRegistry(args.server, api_key=args.api_key).reconcile_runtime(
            dry_run=not args.apply, include_active=args.include_active
        )
    except (RemoteRegistryError, ConnectionError, TimeoutError) as exc:
        print(f"runtime reconciliation failed: {exc}", file=sys.stderr)
        return 1
    if args.json_output:
        print(json.dumps(report, indent=2))
    else:
        print(f"Runtime reconciliation ({'applied' if args.apply else 'dry-run'})")
        for key in ("inspected", "active", "recovered", "cleaned"):
            print(f"{key}: {report.get(key, 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
