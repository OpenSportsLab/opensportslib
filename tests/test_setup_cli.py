from __future__ import annotations

from opensportslib import cli
from opensportslib.setup import setup as setup_lib


def test_cli_setup_forwards_xvars_flag(monkeypatch):
    captured: dict[str, object] = {}

    def fake_setup(*, dali: bool, pyg: bool, vqa_xvars: bool, vqa_qwen: bool):
        captured["dali"] = dali
        captured["pyg"] = pyg
        captured["vqa_xvars"] = vqa_xvars
        captured["vqa_qwen"] = vqa_qwen

    monkeypatch.setattr(cli, "setup", fake_setup)

    rc = cli.main(["setup", "--vqa_xvars", "--dali"])

    assert rc == 0
    assert captured == {"dali": True, "pyg": False, "vqa_xvars": True, "vqa_qwen": False}


def test_cli_setup_forwards_qwen_flag(monkeypatch):
    captured: dict[str, object] = {}

    def fake_setup(*, dali: bool, pyg: bool, vqa_xvars: bool, vqa_qwen: bool):
        captured["dali"] = dali
        captured["pyg"] = pyg
        captured["vqa_xvars"] = vqa_xvars
        captured["vqa_qwen"] = vqa_qwen

    monkeypatch.setattr(cli, "setup", fake_setup)

    rc = cli.main(["setup", "--vqa_qwen", "--pyg"])

    assert rc == 0
    assert captured == {"dali": False, "pyg": True, "vqa_xvars": False, "vqa_qwen": True}


def test_install_xvars_dependencies_uninstalls_then_reinstalls(monkeypatch):
    calls: list[tuple[str, list[str]]] = []

    def fake_call(cmd: list[str]):
        calls.append(("call", cmd))
        return 0

    def fake_check_call(cmd: list[str]):
        calls.append(("check_call", cmd))
        return 0

    monkeypatch.setattr(setup_lib.subprocess, "call", fake_call)
    monkeypatch.setattr(setup_lib.subprocess, "check_call", fake_check_call)
    monkeypatch.setattr(setup_lib.sys, "executable", "/usr/bin/python3")

    setup_lib.install_xvars_dependencies(setup_lib.XVARS_DEPENDENCY_PINS)

    assert calls == [
        (
            "call",
            [
                "/usr/bin/python3",
                "-m",
                "pip",
                "uninstall",
                "-y",
                "transformers",
                "peft",
                "tokenizers",
                "accelerate",
                "trl",
            ],
        ),
        (
            "check_call",
            [
                "/usr/bin/python3",
                "-m",
                "pip",
                "install",
                "transformers==4.38.2",
                "peft==0.9.0",
                "tokenizers==0.15.2",
                "accelerate==0.27.2",
                "trl==0.10.1",
            ],
        ),
    ]


def test_setup_skips_vqa_dependency_install_when_flags_are_false(monkeypatch):
    calls: list[str] = []

    monkeypatch.setattr(setup_lib, "install_extras", lambda dali=False, pyg=False: calls.append(f"extras:{dali}:{pyg}"))
    monkeypatch.setattr(setup_lib, "install_xvars_dependencies", lambda pins: calls.append(f"deps:{sorted(pins)}"))
    monkeypatch.setattr(setup_lib, "verify", lambda: calls.append("verify"))
    monkeypatch.setattr(setup_lib, "install_torch", lambda: calls.append("torch"))

    setup_lib.setup(dali=True, pyg=False, vqa_xvars=False, vqa_qwen=False)

    assert calls == ["torch", "extras:True:False", "verify"]
