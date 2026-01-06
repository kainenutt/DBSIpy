from __future__ import annotations

import configparser
import json
import os
import platform
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any


SUPPORTED_ENGINES: tuple[str, ...] = ("DBSI", "IA", "DTI", "NODDI")


def _utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _cfg_to_dict(cfg: configparser.ConfigParser) -> dict[str, dict[str, str]]:
    """Serialize an INI config to a JSON-friendly dict.

    Values are kept as strings as they appear in the config.
    """

    out: dict[str, dict[str, str]] = {}
    for section in cfg.sections():
        out[section] = {}
        for opt, val in cfg.items(section):
            out[section][opt] = str(val)
    return out


def get_process_resource_snapshot() -> dict[str, Any]:
    """Snapshot of *this process* resource counters.

    Designed for shared environments: only process-local resources are reported.
    Some fields may be unavailable and remain None.
    """

    snap: dict[str, Any] = {
        "pid": os.getpid(),
        "t_wall": time.time(),
        "t_process": time.process_time(),
        "rss_bytes": None,
        "vms_bytes": None,
        "cpu_user_s": None,
        "cpu_system_s": None,
        "io_read_bytes": None,
        "io_write_bytes": None,
        "num_threads": None,
    }

    try:
        import psutil  # type: ignore

        proc = psutil.Process(os.getpid())

        try:
            mi = proc.memory_info()
            snap["rss_bytes"] = int(getattr(mi, "rss", 0) or 0)
            snap["vms_bytes"] = int(getattr(mi, "vms", 0) or 0)
        except Exception:
            pass

        try:
            ct = proc.cpu_times()
            snap["cpu_user_s"] = float(getattr(ct, "user", 0.0) or 0.0)
            snap["cpu_system_s"] = float(getattr(ct, "system", 0.0) or 0.0)
        except Exception:
            pass

        try:
            io = proc.io_counters()
            snap["io_read_bytes"] = int(getattr(io, "read_bytes", 0) or 0)
            snap["io_write_bytes"] = int(getattr(io, "write_bytes", 0) or 0)
        except Exception:
            pass

        try:
            snap["num_threads"] = int(proc.num_threads())
        except Exception:
            pass
    except Exception:
        pass

    return snap


def diff_process_resource_snapshots(start: dict[str, Any], end: dict[str, Any]) -> dict[str, Any]:
    """Compute deltas between two snapshots."""

    def _delta_num(key: str) -> float | int | None:
        a = start.get(key)
        b = end.get(key)
        if a is None or b is None:
            return None
        try:
            return b - a
        except Exception:
            return None

    return {
        "wall_s": _delta_num("t_wall"),
        "process_cpu_s": _delta_num("t_process"),
        "cpu_user_s": _delta_num("cpu_user_s"),
        "cpu_system_s": _delta_num("cpu_system_s"),
        "io_read_bytes": _delta_num("io_read_bytes"),
        "io_write_bytes": _delta_num("io_write_bytes"),
        "rss_bytes_start": start.get("rss_bytes"),
        "rss_bytes_end": end.get("rss_bytes"),
        "vms_bytes_start": start.get("vms_bytes"),
        "vms_bytes_end": end.get("vms_bytes"),
        "num_threads_start": start.get("num_threads"),
        "num_threads_end": end.get("num_threads"),
    }


class _PeakRssMonitor:
    """Background sampler for peak RSS of the current process."""

    def __init__(self, *, interval_s: float = 0.2) -> None:
        self.interval_s = float(interval_s)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_rss_bytes: int | None = None

    def __enter__(self):
        try:
            import psutil  # type: ignore

            proc = psutil.Process(os.getpid())

            def _run():
                peak: int | None = None
                while not self._stop.is_set():
                    try:
                        rss = int(getattr(proc.memory_info(), "rss", 0) or 0)
                        peak = rss if peak is None else max(peak, rss)
                    except Exception:
                        pass
                    self._stop.wait(self.interval_s)
                self.peak_rss_bytes = peak

            self._thread = threading.Thread(target=_run, name="dbsipy-peak-rss", daemon=True)
            self._thread.start()
        except Exception:
            self._thread = None
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread is not None:
            try:
                self._thread.join(timeout=1.0)
            except Exception:
                pass
        return False


def _get_git_state() -> dict[str, Any] | None:
    """Return git metadata when running from a git checkout."""

    try:
        import subprocess

        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True)
            .strip()
        )
        dirty = subprocess.call(["git", "diff", "--quiet"], stderr=subprocess.DEVNULL) != 0
        return {"commit": commit, "dirty": bool(dirty)}
    except Exception:
        return None


def _get_torch_env(torch) -> dict[str, Any] | None:
    if torch is None:
        return None
    env: dict[str, Any] = {"version": getattr(torch, "__version__", None)}
    try:
        env["cuda_compiled_version"] = getattr(getattr(torch, "version", None), "cuda", None)
    except Exception:
        env["cuda_compiled_version"] = None
    try:
        env["cudnn_version"] = int(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else None
    except Exception:
        env["cudnn_version"] = None
    try:
        env["num_threads"] = int(torch.get_num_threads())
        env["num_interop_threads"] = int(torch.get_num_interop_threads())
    except Exception:
        pass
    return env


def _get_cuda_device_info(torch) -> dict[str, Any] | None:
    if torch is None or not getattr(torch, "cuda", None) or not torch.cuda.is_available():
        return None
    try:
        idx = int(torch.cuda.current_device())
    except Exception:
        idx = 0
    info: dict[str, Any] = {"device_index": idx}
    try:
        info["name"] = str(torch.cuda.get_device_name(idx))
    except Exception:
        info["name"] = None
    try:
        cap = torch.cuda.get_device_capability(idx)
        info["capability"] = [int(cap[0]), int(cap[1])]
    except Exception:
        info["capability"] = None
    try:
        props = torch.cuda.get_device_properties(idx)
        info["total_memory_bytes"] = int(getattr(props, "total_memory", 0) or 0)
        info["multi_processor_count"] = int(getattr(props, "multi_processor_count", 0) or 0)
    except Exception:
        info["total_memory_bytes"] = None
    return info


def _get_installed_package_versions() -> dict[str, str]:
    """Versions for key packages (without importing them).

    Uses importlib.metadata to avoid heavy imports.
    """

    try:
        from importlib.metadata import PackageNotFoundError, version
    except Exception:  # pragma: no cover
        return {}

    # Keep this curated and stable; avoid collecting the entire environment.
    pkg_names = [
        "DBSIpy",
        "numpy",
        "scipy",
        "pandas",
        "nibabel",
        "dipy",
        "torch",
        "psutil",
        "joblib",
        "tqdm",
    ]

    out: dict[str, str] = {}
    for name in pkg_names:
        try:
            out[name] = str(version(name))
        except PackageNotFoundError:
            out[name] = "unavailable"
        except Exception:
            out[name] = "unavailable"
    return out


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
    resource_sample_interval_s: float = 0.2,
    resource_monitor: bool = True,
    output_mode: str | None = None,
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

    if resource_sample_interval_s <= 0:
        raise ValueError("resource_sample_interval_s must be > 0")

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

    started_utc = _utc_now()
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

            if output_mode is not None:
                _ensure_section(cfg, "DEBUG")
                cfg.set("DEBUG", "output_mode", str(output_mode))

            run_started_utc = _utc_now()
            snap0 = get_process_resource_snapshot()

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
                "run_started_utc": run_started_utc,
                "run_finished_utc": None,
                "effective_config": _cfg_to_dict(cfg),
                "process": {
                    "pid": os.getpid(),
                },
                "resources": {
                    "delta": None,
                    "peak_rss_bytes": None,
                },
                "cuda": {
                    "available": bool(getattr(torch, "cuda", None) and torch.cuda.is_available()) if torch else False,
                    "device": _get_cuda_device_info(torch),
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

                if resource_monitor:
                    with _PeakRssMonitor(interval_s=resource_sample_interval_s) as mon:
                        pipeline = DBSIpy(cfg)
                        pipeline.__call__()
                        run_rec["resources"]["peak_rss_bytes"] = mon.peak_rss_bytes
                else:
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
                snap1 = get_process_resource_snapshot()
                run_rec["resources"]["delta"] = diff_process_resource_snapshots(snap0, snap1)
                run_rec["run_finished_utc"] = _utc_now()
                if torch and torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass

            runs.append(run_rec)

    finished_utc = _utc_now()
    total_wall_s = float(time.time() - started_t)

    summary: dict[str, Any] = {
        "schema_version": 2,
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "total_wall_s": total_wall_s,
        "benchmark_process": {
            "pid": os.getpid(),
        },
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
            "git": _get_git_state(),
            "torch": _get_torch_env(torch),
            "packages": _get_installed_package_versions(),
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
