"""Fast command-line interface contracts."""

from __future__ import annotations

import pytest

from opensportslib import cli


pytestmark = [pytest.mark.unit, pytest.mark.smoke]


def test_cli_help_is_available(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    assert "opensportslib" in capsys.readouterr().out


def test_cli_rejects_unknown_command(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["unknown"])
    assert exc.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_cli_setup_forwards_optional_install_flags(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "setup", lambda **kwargs: calls.append(kwargs))
    assert cli.main(["setup", "--pyg", "--dali", "--vqa_xvars", "--vqa_qwen"]) == 0
    assert calls == [{"pyg": True, "dali": True, "vqa_xvars": True, "vqa_qwen": True}]
