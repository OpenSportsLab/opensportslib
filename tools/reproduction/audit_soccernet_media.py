"""Count actual decoded frames in a local SoccerNet comparison plan.

This tool only reads local media. A plan's files list specifies official_local_path,
official_sha256, expected_frames, and the pinned source provenance. PyAV is
required. Container frame-count estimates are retained for comparison only.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time


def decode_file(row):
    import av

    path = Path(row["official_local_path"])
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != row["official_sha256"]:
        raise ValueError(f"SHA-256 differs from pinned official source: {path}")
    count, first, last, previous, pts_gaps = 0, None, None, None, 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        time_base = stream.time_base
        for frame in container.decode(stream):
            if count == 0:
                first = frame.pts
            if previous is not None and frame.pts is not None:
                if abs(float((frame.pts - previous) * time_base) - .04) > .002:
                    pts_gaps += 1
            previous = frame.pts
            count += 1
            last = frame.pts
    return {
        **row, "sha256_verified": True, "decoded_frames": count,
        "decoded_shortfall": max(0, row["expected_frames"] - count),
        "first_pts": first, "last_pts": last, "time_base": str(time_base),
        "pts_gaps_over_2ms": pts_gaps, "decode_error": None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    plan = json.loads(args.plan.read_text())
    started = time.monotonic()
    report = {
        "repo_id": plan["repo_id"], "files": [], "complete": False,
        "scope": "SHA-256 and full sequential decoding of every planned file",
        "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.output.with_name(args.output.name + ".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.output)

    save()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        tasks = {pool.submit(decode_file, row): row for row in plan["files"]}
        for future in as_completed(tasks):
            row = tasks[future]
            try:
                result = future.result()
            except Exception as error:
                result = {**row, "decode_error": f"{type(error).__name__}: {error}"}
            report["files"].append(result)
            save()
            print(json.dumps({
                key: result.get(key) for key in (
                    "path", "decoded_frames", "expected_frames", "decoded_shortfall",
                    "decode_error",
                )
            }), flush=True)
    report.update(
        complete=True, elapsed_seconds=time.monotonic() - started,
        ended_utc=datetime.now(timezone.utc).isoformat(),
    )
    save()
    raise SystemExit(2 if any(
        row.get("decode_error") or row.get("decoded_shortfall", 0) > 0
        for row in report["files"]
    ) else 0)


if __name__ == "__main__":
    main()
