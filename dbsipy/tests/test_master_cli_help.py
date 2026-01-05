from __future__ import annotations

import sys

import pytest

from dbsipy import master_cli


def test_master_cli_prints_help_when_no_args(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["DBSI"])
    master_cli.main()
    out = capsys.readouterr().out
    assert "DBSIpy command line interface" in out
    assert "run" in out
    assert "benchmark" in out


def test_master_cli_help_flag(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["DBSI", "--help"])
    with pytest.raises(SystemExit) as e:
        master_cli.main()
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert "DBSIpy command line interface" in out
