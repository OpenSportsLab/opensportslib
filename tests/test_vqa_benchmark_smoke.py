import json
import subprocess
import sys
from pathlib import Path


def test_benchmark_smoke_writes_report(vqa_config_path, tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = root / "tools" / "benchmark_vqa.py"
    output = tmp_path / "benchmark_report.json"
    output_2 = tmp_path / "benchmark_report_second.json"

    cmd = [
        sys.executable,
        str(script),
        "--config",
        vqa_config_path,
        "--output",
        str(output),
        "--seed",
        "42",
    ]
    subprocess.run(cmd, check=True)
    cmd[cmd.index(str(output))] = str(output_2)
    subprocess.run(cmd, check=True)

    payload = json.loads(output.read_text(encoding="utf-8"))
    payload_2 = json.loads(output_2.read_text(encoding="utf-8"))
    assert payload["seed"] == 42
    assert "profiles" in payload
    assert "baseline" in payload["profiles"]
    assert "xvars_hf" in payload["profiles"]
    assert "metrics" in payload["profiles"]["baseline"]
    assert "metrics" in payload["profiles"]["xvars_hf"]
    assert payload["profiles"]["baseline"]["metrics"] == payload_2["profiles"]["baseline"]["metrics"]
    assert payload["profiles"]["xvars_hf"]["metrics"] == payload_2["profiles"]["xvars_hf"]["metrics"]
