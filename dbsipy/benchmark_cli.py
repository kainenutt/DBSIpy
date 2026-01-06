"""Benchmark CLI for DBSIpy.

Adds `DBSI benchmark` to execute all supported engines on the same dataset and
emit a consolidated JSON summary.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dbsipy.core.benchmark import run_benchmark


class BenchmarkCLI:
    def __init__(self, subparsers) -> None:
        self.subparsers = subparsers

    def add_subparser_args(self) -> argparse:
        subparser = self.subparsers.add_parser(
            "benchmark",
            description="benchmark all DBSIpy engines on the same input dataset",
        )

        subparser.add_argument(
            "--cfg_path",
            type=str,
            required=True,
            help="Path to a configuration .ini to use as a template",
        )
        subparser.add_argument("--dwi_file", type=str, required=True, help="Path to DWI NIfTI (*.nii or *.nii.gz)")
        subparser.add_argument("--bval_file", type=str, required=True, help="Path to bvals file")
        subparser.add_argument("--bvec_file", type=str, required=True, help="Path to bvecs file")
        subparser.add_argument(
            "--mask_file",
            type=str,
            required=False,
            default=None,
            help="Optional mask NIfTI path, or 'auto' to auto-generate",
        )
        subparser.add_argument(
            "--output_root",
            type=str,
            required=False,
            default=None,
            help="Optional directory to place all benchmark run folders + summary JSON",
        )
        subparser.add_argument(
            "--repeats",
            type=int,
            required=False,
            default=1,
            help="Number of repeats per engine (default: 1)",
        )

        subparser.add_argument(
            "--output_mode",
            type=str,
            required=False,
            default=None,
            choices=["quiet", "standard", "verbose", "debug"],
            help="Terminal output mode (overrides config): quiet | standard | verbose | debug",
        )

        subparser.add_argument(
            "--resource_sample_interval_s",
            type=float,
            required=False,
            default=0.2,
            help="Sampling interval (seconds) for per-process peak RSS monitoring (default: 0.2)",
        )

        subparser.add_argument(
            "--no_resource_monitor",
            action="store_true",
            help="Disable per-process peak RSS sampling (still records start/end deltas when available)",
        )

        return self.subparsers

    def validate_args(self, args):
        # master_cli passes a dict; keep parity with dbsipy.cli.CLI
        if not isinstance(args, dict):
            args = vars(args)

        def _req_file(key: str) -> str:
            p = str(args.get(key, "") or "")
            if not p:
                raise ValueError(f"Missing required argument: {key}")
            if not os.path.exists(p):
                raise FileNotFoundError(f"File not found: {p}")
            return p

        args["cfg_path"] = _req_file("cfg_path")
        args["dwi_file"] = _req_file("dwi_file")
        args["bval_file"] = _req_file("bval_file")
        args["bvec_file"] = _req_file("bvec_file")

        mask = args.get("mask_file", None)
        if mask is not None:
            mask = str(mask).strip()
            if mask == "":
                mask = None
            elif mask.lower() not in {"auto", "n/a"} and not os.path.exists(mask):
                raise FileNotFoundError(f"Mask file not found: {mask}")
        args["mask_file"] = mask

        out_root = args.get("output_root", None)
        if out_root is not None:
            out_root = str(out_root).strip()
            if out_root == "":
                out_root = None
            else:
                Path(out_root).mkdir(parents=True, exist_ok=True)
        args["output_root"] = out_root

        repeats = int(args.get("repeats", 1) or 1)
        if repeats < 1:
            raise ValueError("--repeats must be >= 1")
        args["repeats"] = repeats

        # Output mode is applied to the effective config used per run.
        mode = args.get("output_mode", None)
        if mode is not None:
            mode = str(mode).strip().lower()
            if mode == "":
                mode = None
        args["output_mode"] = mode

        interval = float(args.get("resource_sample_interval_s", 0.2) or 0.2)
        if interval <= 0:
            raise ValueError("--resource_sample_interval_s must be > 0")
        args["resource_sample_interval_s"] = interval

        args["resource_monitor"] = not bool(args.get("no_resource_monitor", False))

        return args

    def run(self, args):
        summary = run_benchmark(
            cfg_path=args["cfg_path"],
            dwi_file=args["dwi_file"],
            bval_file=args["bval_file"],
            bvec_file=args["bvec_file"],
            mask_file=args.get("mask_file", None),
            output_root=args.get("output_root", None),
            repeats=int(args.get("repeats", 1)),
            run_tag="benchmark",
            resource_sample_interval_s=float(args.get("resource_sample_interval_s", 0.2) or 0.2),
            resource_monitor=bool(args.get("resource_monitor", True)),
            output_mode=args.get("output_mode", None),
        )

        # If any run failed, exit nonzero.
        failures = [r for r in summary.get("runs", []) if not r.get("ok", False)]
        if failures:
            raise SystemExit(
                f"Benchmark completed with {len(failures)} failure(s). Summary: {summary.get('summary_path')}"
            )

        print(f"Benchmark complete. Summary: {summary.get('summary_path')}")
