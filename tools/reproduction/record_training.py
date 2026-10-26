"""Record a training command and mirror its small reproducibility artifacts.

Run this recorder under nohup (or a service) for training across terminal exits.
The child exit code is preserved; signals are forwarded to the training group.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = (
    "run.json", "history.json", "progress.json", "preflight.json",
    "data-identity.json", "results.json", "status.json", "failure.json",
)


def write_changed(path, content):
    path = Path(path)
    if path.is_file() and path.read_bytes() == content:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def snapshot(source, archive):
    """Copy only explicitly allowed, valid JSON; never weights or predictions."""
    source, archive = Path(source), Path(archive)
    paths = [source / "ledger.json"]
    for run in sorted(source.glob("*/seed-*")):
        if run.is_dir():
            paths.extend(run / name for name in ARTIFACTS)
    for path in paths:
        if not path.is_file():
            continue
        content = path.read_bytes()
        try:
            json.loads(content)
        except (ValueError, UnicodeError):
            # A writer may be replacing an artifact; retry on the next poll.
            continue
        write_changed(archive / "runs" / path.relative_to(source), content)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.poll_seconds <= 0:
        parser.error("A command and positive poll interval are required")
    args.source.mkdir(parents=True, exist_ok=True)
    # The lock belongs to the output campaign, so two archives cannot train it.
    lock = (args.source / ".training-recorder.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error("A training recorder is already using this campaign")
    session = args.archive / "sessions" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{os.getpid()}"
    )
    session.mkdir(parents=True)
    metadata = {
        "started_utc": utc_now(), "command": command, "cwd": str(ROOT),
        "recorder_pid": os.getpid(), "status": "starting",
        "environment": {key: os.environ.get(key) for key in (
            "CUBLAS_WORKSPACE_CONFIG", "OPENCV_FFMPEG_THREADS", "OMP_NUM_THREADS",
            "MPLCONFIGDIR", "CUDA_VISIBLE_DEVICES", "PYTHONHASHSEED",
        )},
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--short", "--untracked-files=normal"],
            cwd=ROOT, text=True,
        ),
    }
    write_changed(session / "working-tree.patch", subprocess.check_output(
        ["git", "diff", "HEAD", "--", ".", ":(exclude)reproduction/training-runs"],
        cwd=ROOT,
    ))

    def save_metadata():
        write_changed(session / "session.json", (
            json.dumps(metadata, indent=2) + "\n"
        ).encode())

    save_metadata()
    snapshot(args.source, args.archive)
    stopped = threading.Event()

    def mirror():
        while not stopped.wait(args.poll_seconds):
            try:
                snapshot(args.source, args.archive)
            except OSError as error:
                print(f"Artifact snapshot failed: {error}", file=sys.stderr, flush=True)

    worker = threading.Thread(target=mirror, daemon=True)
    child = None

    def forward(signum, frame):
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, forward)
    try:
        # Direct file output cannot block training if a terminal/pipe disappears.
        with (session / "console.log").open("ab", buffering=0) as log:
            child = subprocess.Popen(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            metadata.update(child_pid=child.pid, status="running")
            save_metadata()
            worker.start()
            returncode = child.wait()
        metadata.update(
            status="completed" if returncode == 0 else "failed",
            returncode=returncode,
        )
    except Exception as error:
        metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        stopped.set()
        if worker.is_alive():
            worker.join()
        snapshot(args.source, args.archive)
        metadata["ended_utc"] = utc_now()
        save_metadata()
    # A successful campaign can contain blocked/failed jobs; inspect its ledger.
    raise SystemExit(returncode if returncode >= 0 else 128 - returncode)


if __name__ == "__main__":
    main()
