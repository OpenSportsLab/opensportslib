from __future__ import annotations

import pytest

from opensportslib import cli
from opensportslib.setup import setup as setup_lib


def test_select_cuda_wheel_uses_cu126_for_pre_sm75_gpu_with_cuda_13():
    assert setup_lib.select_cuda_wheel("13.0", [(7, 0)]) == "cu126"


def test_select_cuda_wheel_uses_cu126_for_pascal_with_cuda_13():
    assert setup_lib.select_cuda_wheel("13.0", [(6, 0)]) == "cu126"


def test_select_torch_packages_returns_unpinned_package_names():
    assert setup_lib.select_torch_packages([(7, 0)]) == (
        "torch",
        "torchvision",
        "torchaudio",
    )


def test_install_torch_uses_the_pinned_pyg_compatibility_profile(monkeypatch):
    calls: list[tuple[str, list[str]]] = []

    monkeypatch.setattr(setup_lib.sys, "executable", "/usr/bin/python3")
    monkeypatch.setattr(setup_lib, "CUDA_TAG", "cu130")
    monkeypatch.setattr(setup_lib.subprocess, "call", lambda cmd: calls.append(("call", cmd)) or 0)
    monkeypatch.setattr(
        setup_lib.subprocess,
        "check_call",
        lambda cmd: calls.append(("check_call", cmd)) or 0,
    )

    setup_lib.install_torch(pyg_compatible=True)

    assert calls[1] == (
        "check_call",
        [
            "/usr/bin/python3", "-m", "pip", "install",
            "torch==2.12.1", "torchvision==0.27.1", "torchaudio==2.12.1",
            "--index-url", "https://download.pytorch.org/whl/cu130",
        ],
    )


def test_validate_pyg_wheels_checks_all_binary_wheels_before_torch_reinstall(monkeypatch):
    calls: list[list[str]] = []

    monkeypatch.setattr(setup_lib.sys, "executable", "/usr/bin/python3")
    monkeypatch.setattr(setup_lib, "CUDA_TAG", "cu130")
    monkeypatch.setattr(setup_lib.subprocess, "check_call", lambda cmd: calls.append(cmd) or 0)

    setup_lib.validate_pyg_wheels()

    assert calls[0][:9] == [
        "/usr/bin/python3", "-m", "pip", "download", "--no-deps", "--only-binary=:all:",
        "--dest", calls[0][7], "torch-geometric",
    ]
    assert calls[0][8:14] == list(setup_lib.PYG_EXTENSION_PACKAGES)
    assert calls[0][-2:] == ["-f", "https://data.pyg.org/whl/torch-2.12.1+cu130.html"]


def test_pyg_extension_packages_match_the_supported_optional_libraries():
    assert setup_lib.PYG_EXTENSION_PACKAGES == (
        "pyg-lib",
        "torch-scatter",
        "torch-sparse",
    )


def test_setup_selects_pyg_torch_profile_before_installing_pyg(monkeypatch):
    calls: list[object] = []

    monkeypatch.setattr(
        setup_lib,
        "install_torch",
        lambda *, pyg_compatible=False: calls.append(("torch", pyg_compatible)),
    )
    monkeypatch.setattr(setup_lib, "validate_pyg_wheels", lambda: calls.append("validate"))
    monkeypatch.setattr(
        setup_lib,
        "install_extras",
        lambda dali=False, pyg=False: calls.append(("extras", dali, pyg)),
    )
    monkeypatch.setattr(setup_lib, "verify", lambda: calls.append("verify"))

    setup_lib.setup(pyg=True)

    assert calls == ["validate", ("torch", True), ("extras", False, True), "verify"]


def test_select_cuda_wheel_uses_cu130_for_dgx_spark():
    assert setup_lib.select_cuda_wheel("13.0", [(12, 1)]) == "cu130"


def test_select_cuda_wheel_rejects_new_architecture_without_cuda_13_driver():
    with pytest.raises(RuntimeError, match="CUDA 13.0"):
        setup_lib.select_cuda_wheel("12.8", [(10, 0)])


def test_select_cuda_wheel_uses_highest_driver_compatible_wheel():
    assert setup_lib.select_cuda_wheel("12.8", [(8, 0)]) == "cu128"


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
    monkeypatch.setattr(
        setup_lib,
        "install_torch",
        lambda *, pyg_compatible=False: calls.append(f"torch:{pyg_compatible}"),
    )

    setup_lib.setup(dali=True, pyg=False, vqa_xvars=False, vqa_qwen=False)

    assert calls == ["torch:False", "extras:True:False", "verify"]
