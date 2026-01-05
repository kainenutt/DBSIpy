from __future__ import annotations

import configparser
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


SUPPORTED_ENGINES: tuple[str, ...] = ("DBSI", "IA", "DTI", "NODDI")


def _ensure_section(cfg: configparser.ConfigParser, section: str) -> None:
    if not cfg.has_section(section):
        cfg.add_section(section)


def _load_base_config(cfg_path: str) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg.read(cfg_path)

    _ensure_section(cfg, "DEBUG")
    cfg.set("DEBUG", "cfg_source", str(cfg_path))

    # Ensure required top-level sections exist; the per-engine validator will
    # enforce additional requirements.
    _ensure_section(cfg, "INPUT")
    _ensure_section(cfg, "GLOBAL")

    return cfg


def _apply_input_overrides(
    cfg: configparser.ConfigParser,
    *,
    dwi_file: str,
    bval_file: str,
    bvec_file: str,
    mask_file: str | None,
) -> None:
    cfg.set("INPUT", "dwi_file", str(dwi_file))
    cfg.set("INPUT", "bval_file", str(bval_file))
    cfg.set("INPUT", "bvec_file", str(bvec_file))
    if mask_file is not None:
        cfg.set("INPUT", "mask_file", str(mask_file))


def _apply_output_overrides(
    cfg: configparser.ConfigParser,
    *,
    output_root: str | None,
    run_tag: str | None,
) -> None:
    if output_root is None and run_tag is None:
        return
    _ensure_section(cfg, "OUTPUT")
    if output_root is not None:
        cfg.set("OUTPUT", "save_dir", str(output_root))
    if run_tag is not None:
        cfg.set("OUTPUT", "run_tag", str(run_tag))


def run_benchmark(
    *,
    cfg_path: str,
    dwi_file: str,
    bval_file: str,
    bvec_file: str,
    mask_file: str | None = None,
    output_root: str | None = None,
    repeats: int = 1,
    run_tag: str = "benchmark",
) -> dict[str, Any]:
    """Run all supported DBSIpy engines as a benchmark.

    This is intended for real-world datasets (NIfTI + bval + bvec), and produces
    normal DBSIpy output folders for each run plus a single JSON summary.

    Notes:
    - Uses the provided cfg as a *template* and only overrides INPUT + engine.
    - Runs are executed sequentially in-process.
    - Failures are captured per-engine; a final nonzero exit should be handled
      by the caller if desired.
    """

    if repeats < 1:
        raise ValueError("repeats must be >= 1")

    cfg_path = str(Path(cfg_path))
    dwi_file = str(Path(dwi_file))
    bval_file = str(Path(bval_file))
    bvec_file = str(Path(bvec_file))
    if mask_file is not None:
        mask_file = str(mask_file)

    if output_root is not None:
        # Resolve to an absolute path so downstream config resolution does not
        # interpret relative paths as relative to the template cfg directory.
        # This is especially important for benchmarks, where callers typically
        # expect --output_root to be relative to CWD.
        output_root = str(Path(output_root).resolve())
        Path(output_root).mkdir(parents=True, exist_ok=True)

    # Import torch lazily so importing the module doesn't force a heavy import.
    try:
        import torch
    except Exception:  # pragma: no cover
        torch = None  # type: ignore

    started_utc = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    started_t = time.time()

    runs: list[dict[str, Any]] = []

    for engine in SUPPORTED_ENGINES:
        for rep in range(1, repeats + 1):
            cfg = _load_base_config(cfg_path)
            _apply_input_overrides(
                cfg,
                dwi_file=dwi_file,
                bval_file=bval_file,
                bvec_file=bvec_file,
                mask_file=mask_file,
            )
            cfg.set("GLOBAL", "model_engine", engine)
            _apply_output_overrides(cfg, output_root=output_root, run_tag=run_tag)

            run_rec: dict[str, Any] = {
                "engine": engine,
                "repeat": rep,
                "ok": False,
                "error": None,
                "save_dir": None,
                "total_runtime_s": None,
                "timings": None,
                "device": None,
                "host": None,
                "cuda": {
                    "available": bool(getattr(torch, "cuda", None) and torch.cuda.is_available()) if torch else False,
                    "max_memory_allocated": None,
                    "max_memory_reserved": None,
                },
            }

            try:
                if torch and torch.cuda.is_available():
                    try:
                        torch.cuda.reset_peak_memory_stats()
                    except Exception:
                        pass

                from dbsipy.core.fast_DBSI import DBSIpy

                pipeline = DBSIpy(cfg)
                pipeline.__call__()

                run_rec["ok"] = True
                run_rec["save_dir"] = str(getattr(pipeline, "save_dir", None))
                run_rec["total_runtime_s"] = float(getattr(pipeline, "_total_runtime_s", 0.0) or 0.0)
                run_rec["timings"] = dict(getattr(pipeline, "_timings", {}) or {})
                run_rec["device"] = str(getattr(pipeline.configuration, "DEVICE", ""))
                run_rec["host"] = str(getattr(pipeline.configuration, "HOST", ""))

                if torch and torch.cuda.is_available():
                    try:
                        run_rec["cuda"]["max_memory_allocated"] = int(torch.cuda.max_memory_allocated())
                    except Exception:
                        pass
                    try:
                        run_rec["cuda"]["max_memory_reserved"] = int(torch.cuda.max_memory_reserved())
                    except Exception:
                        pass

            except Exception as e:  # noqa: BLE001
                run_rec["error"] = f"{type(e).__name__}: {e}"
            finally:
                if torch and torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass

            runs.append(run_rec)

    finished_utc = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    total_wall_s = float(time.time() - started_t)

    summary: dict[str, Any] = {
        "schema_version": 1,
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "total_wall_s": total_wall_s,
        "inputs": {
            "cfg_path": cfg_path,
            "dwi_file": dwi_file,
            "bval_file": bval_file,
            "bvec_file": bvec_file,
            "mask_file": mask_file,
        },
        "repeats": repeats,
        "engines": list(SUPPORTED_ENGINES),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
        },
        "runs": runs,
    }

    # Write the summary next to the grouped benchmark runs if output_root is set,
    # otherwise next to the DWI input.
    out_base = Path(output_root) if output_root is not None else Path(dwi_file).resolve().parent
    out_base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = out_base / f"{stamp}_DBSIpy_benchmark_summary.json"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    summary["summary_path"] = str(summary_path)

    return summary
