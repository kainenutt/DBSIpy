from __future__ import annotations

from pathlib import Path

import pytest

from dbsipy.benchmark_cli import BenchmarkCLI
from dbsipy.cli import CLI


def test_run_cli_validate_args_accepts_existing_cfg(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.ini"
    cfg.write_text("[GLOBAL]\nmodel_engine = DBSI\n", encoding="utf-8")

    cli = CLI(subparsers=None)  # subparsers unused for validate_args
    args = cli.validate_args({"cfg_path": str(cfg)})
    assert args["cfg_path"] == str(cfg)


def test_run_cli_validate_args_rejects_missing_cfg(tmp_path: Path) -> None:
    missing = tmp_path / "missing.ini"
    cli = CLI(subparsers=None)
    with pytest.raises(FileNotFoundError):
        cli.validate_args({"cfg_path": str(missing)})


def test_benchmark_cli_validate_args_happy_path(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.ini"
    dwi = tmp_path / "dwi.nii.gz"
    bval = tmp_path / "bvals"
    bvec = tmp_path / "bvecs"

    cfg.write_text("[GLOBAL]\nmodel_engine = DBSI\n", encoding="utf-8")
    dwi.write_bytes(b"dummy")
    bval.write_text("0 1000\n", encoding="utf-8")
    bvec.write_text("0 0\n", encoding="utf-8")

    bc = BenchmarkCLI(subparsers=None)
    args = bc.validate_args(
        {
            "cfg_path": str(cfg),
            "dwi_file": str(dwi),
            "bval_file": str(bval),
            "bvec_file": str(bvec),
            "mask_file": "auto",
            "output_root": str(tmp_path / "out"),
            "repeats": 1,
            "resource_sample_interval_s": 0.1,
            "no_resource_monitor": True,
        }
    )

    assert args["mask_file"] == "auto"
    assert Path(args["output_root"]).exists()
    assert args["repeats"] == 1
    assert args["resource_sample_interval_s"] == 0.1
    assert args["resource_monitor"] is False
