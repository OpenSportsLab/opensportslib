"""Training evidence must survive child failures without archiving large assets."""

import json
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("returncode", [0, 7])
def test_recorder_preserves_logs_artifacts_and_exit_status(tmp_path, returncode):
    root = Path(__file__).resolve().parents[3]
    source, archive = tmp_path / "outputs", tmp_path / "archive"
    run = source / "example" / "seed-1"
    run.mkdir(parents=True)
    (run / "history.json").write_text("[]")
    (run / "best.pt").write_bytes(b"weights")
    (run / "predictions.json").write_text('{"large": true}')
    child = (
        "import json, pathlib, sys; "
        "run = pathlib.Path(sys.argv[1]); "
        "(run / 'history.json').write_text(json.dumps([{'epoch': 0, 'loss': .25}])); "
        "print('training stdout', flush=True); "
        "print('training stderr', file=sys.stderr, flush=True); "
        "sys.exit(int(sys.argv[2]))"
    )
    result = subprocess.run([
        sys.executable, str(root / "tools/reproduction/record_training.py"),
        "--source", str(source), "--archive", str(archive), "--",
        sys.executable, "-c", child, str(run), str(returncode),
    ], capture_output=True, text=True)
    assert result.returncode == returncode, result.stderr
    sessions = list((archive / "sessions").iterdir())
    assert len(sessions) == 1
    metadata = json.loads((sessions[0] / "session.json").read_text())
    assert metadata["returncode"] == returncode
    assert metadata["status"] == ("completed" if returncode == 0 else "failed")
    assert metadata["git_head"]
    assert metadata["command"][-1] == str(returncode)
    console = (sessions[0] / "console.log").read_text()
    assert "training stdout" in console and "training stderr" in console
    archived_run = archive / "runs/example/seed-1"
    assert json.loads((archived_run / "history.json").read_text()) == [
        {"epoch": 0, "loss": .25}
    ]
    assert not (archived_run / "best.pt").exists()
    assert not (archived_run / "predictions.json").exists()
