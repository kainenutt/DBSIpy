"""Provenance and run-manifest utilities.

This module is intentionally designed to be imported without triggering GUI code.
It centralizes creation of the on-disk `run_manifest.json` written after a run.

Provenance should never abort a computation.
"""

from __future__ import annotations

import importlib
import json
import logging
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dbsipy.core import utils


def _safe_version(mod_name: str) -> str:
    try:
        mod = importlib.import_module(mod_name)
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return "unavailable"


def write_run_manifest(model: Any, *, dbsipy_version: str) -> None:
    """Write a provenance manifest alongside saved outputs.

    Parameters
    ----------
    model:
        A DBSIpy-like object (typically `DBSIpy`) with attributes set during a run.
    dbsipy_version:
        The version string to record. Passed in to avoid circular imports.

    Notes
    -----
    Failures are logged and ignored.
    """

    try:
        save_dir = getattr(model, "save_dir", None)
        if not isinstance(save_dir, str) or not save_dir:
            return

        cfg = getattr(getattr(model, "configuration", None), "cfg_file", None)

        def _cfg_get(section: str, option: str, fallback=None):
            try:
                if cfg is None:
                    return fallback
                if not cfg.has_section(section):
                    return fallback
                return cfg.get(section, option, fallback=fallback)
            except Exception:
                return fallback

        cfg_source = None
        if cfg is not None:
            try:
                cfg_source = cfg.get("DEBUG", "cfg_source", fallback=None)
            except Exception:
                cfg_source = None

        configuration = getattr(model, "configuration", None)
        engine = getattr(configuration, "ENGINE", None)

        # Curated subset of configuration values (scalars + key file paths).
        config_selected: dict[str, Any] = {
            "engine": engine,
            "output_map_set": getattr(configuration, "output_map_set", None),
            "dti_bval_cut": getattr(configuration, "dti_bval_cutoff", None),
            "signal_normalization": getattr(configuration, "signal_normalization", None),
            "learnable_s0": bool(getattr(configuration, "learnable_s0", False)),
            "output_mode": str(getattr(configuration, "output_mode", "standard")),
            "inputs": {
                "dwi_file": getattr(configuration, "dwi_path", None),
                "mask_file": getattr(configuration, "mask_path", None),
                "bval_file": getattr(configuration, "bval_path", None),
                "bvec_file": getattr(configuration, "bvec_path", None),
            },
            "basis_paths": {
                "angle_basis": _cfg_get("STEP_1", "angle_basis", None),
                "iso_basis": _cfg_get("STEP_1", "iso_basis", None),
                "step_2_axials": _cfg_get("STEP_2", "step_2_axials", None),
                "step_2_radials": _cfg_get("STEP_2", "step_2_radials", None),
            },
        }

        if engine in {"DBSI", "IA"}:
            config_selected["global"] = {
                "weight_threshold": getattr(configuration, "weight_threshold", None),
                "max_group_number": getattr(configuration, "max_group_number", None),
                "fiber_threshold": getattr(configuration, "fiber_threshold", None),
            }
            try:
                max_fibers = int(getattr(configuration, "max_group_number", 1) or 1)
            except Exception:
                max_fibers = 1
            config_selected["fiber_outputs"] = {
                "max_group_number": max_fibers,
                "multifiber_enabled": bool(max_fibers > 1),
                "aggregate_maps_enabled": bool(
                    (max_fibers > 1)
                    and (getattr(configuration, "output_map_set", None) in {"default", "expanded"})
                ),
            }
            if engine == "IA":
                config_selected["fiber_outputs"]["aggregate_ia_ea_maps_enabled"] = bool(
                    (max_fibers > 1)
                    and (getattr(configuration, "output_map_set", None) in {"default", "expanded"})
                )

            config_selected["optimizer"] = {
                "step_1_lr": getattr(configuration, "step_1_LR", None),
                "step_1_epochs": getattr(configuration, "step_1_epochs", None),
                "step_1_loss_fn": getattr(configuration, "step_1_loss", None),
                "step_2_lr": getattr(configuration, "step_2_LR", None),
                "step_2_epochs": getattr(configuration, "step_2_epochs", None),
                "step_2_loss_fn": getattr(configuration, "step_2_loss", None),
            }
            config_selected["step_1"] = {
                "angle_threshold": getattr(configuration, "angle_threshold", None),
                "step_1_axial": float(getattr(configuration, "step_1_axial", [np.nan])[0])
                if hasattr(configuration, "step_1_axial")
                else None,
                "step_1_radial": float(getattr(configuration, "step_1_radial", [np.nan])[0])
                if hasattr(configuration, "step_1_radial")
                else None,
            }
            config_selected["step_2"] = {
                "intra_threshold": getattr(configuration, "intra_threshold", None),
            }
            config_selected["isotropic"] = {
                "four_iso": bool(getattr(configuration, "four_iso", False)),
                "highly_restricted_threshold": getattr(configuration, "highly_restricted_threshold", None),
                "restricted_threshold": getattr(configuration, "restricted_threshold", None),
                "free_water_threshold": getattr(configuration, "water_threshold", None),
            }
        elif engine == "NODDI":
            config_selected["noddi"] = {
                "noddi_lr": getattr(configuration, "noddi_lr", None),
                "noddi_epochs": getattr(configuration, "noddi_epochs", None),
                "noddi_d_ic": getattr(configuration, "noddi_d_ic", None),
                "noddi_d_iso": getattr(configuration, "noddi_d_iso", None),
                "noddi_use_tortuosity": getattr(configuration, "noddi_use_tortuosity", None),
            }

        dwi_shape = None
        try:
            dwi_shape = getattr(model, "dwi_nifti_shape", None)
            if dwi_shape is not None:
                dwi_shape = tuple(int(x) for x in dwi_shape)
        except Exception:
            dwi_shape = None

        n_volumes = None
        try:
            if dwi_shape and len(dwi_shape) >= 4:
                n_volumes = int(dwi_shape[-1])
            elif hasattr(model, "bvals") and getattr(model.bvals, "shape", None) is not None:
                n_volumes = int(model.bvals.shape[0])
        except Exception:
            n_volumes = None

        spatial_dims = None
        try:
            if dwi_shape and len(dwi_shape) >= 3:
                spatial_dims = [int(dwi_shape[0]), int(dwi_shape[1]), int(dwi_shape[2])]
            else:
                sd = getattr(configuration, "spatial_dims", None)
                if sd is not None:
                    spatial_dims = [int(sd[0]), int(sd[1]), int(sd[2])]
        except Exception:
            spatial_dims = None

        masked_voxels = None
        try:
            cfg_mask = getattr(configuration, "mask", None)
            if cfg_mask is not None:
                masked_voxels = int(np.asarray(cfg_mask, dtype=bool).sum())
            elif hasattr(model, "dwi") and model.dwi is not None:
                masked_voxels = int(getattr(model.dwi, "shape", [None])[0])
        except Exception:
            masked_voxels = None

        outputs: list[str] = []
        try:
            outputs = sorted([p.name for p in Path(save_dir).glob("*.nii.gz")])
        except Exception:
            outputs = []

        output_map_specs = []
        try:
            from dbsipy.maps.map_metadata import get_map_spec_safe

            try:
                max_fibers = int(getattr(configuration, "max_group_number", 1) or 1)
            except Exception:
                max_fibers = 1

            on_disk = set(outputs)
            for pmap in getattr(model, "params", {}).values():
                original_name = getattr(pmap, "pmap_name", None)
                if not isinstance(original_name, str) or not original_name:
                    continue
                save_name = utils.legacy_fiber_save_name(original_name, max_fibers)
                file_name = save_name + ".nii.gz"
                if file_name not in on_disk:
                    continue
                spec = get_map_spec_safe(save_name)
                output_map_specs.append(
                    {
                        "file": file_name,
                        "map_name": save_name,
                        **spec,
                    }
                )
            output_map_specs.sort(key=lambda x: str(x.get("file", "")))
        except Exception:
            output_map_specs = []

        manifest = {
            "schema_version": 1,
            "created_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "run_started_utc": getattr(model, "_run_started_utc", None),
            "run_finished_utc": getattr(model, "_run_finished_utc", None),
            "total_runtime_s": getattr(model, "_total_runtime_s", None),
            "save_dir": save_dir,
            "engine": getattr(configuration, "ENGINE", None),
            "output_map_set": getattr(configuration, "output_map_set", None),
            "device": getattr(configuration, "DEVICE", None),
            "host": getattr(configuration, "HOST", None),
            "diagnostics_enabled": bool(getattr(configuration, "diagnostics_enabled", False)),
            "output_mode": str(getattr(configuration, "output_mode", "standard")),
            "dti_bval_cut": getattr(configuration, "dti_bval_cutoff", None),
            "signal_normalization": getattr(
                model, "signal_normalization_mode", getattr(configuration, "signal_normalization", None)
            ),
            "signal_scale_stats": getattr(model, "signal_scale_stats", None),
            "mask_source": getattr(model, "mask_source", None),
            "learnable_s0": bool(getattr(configuration, "learnable_s0", False)),
            "fiber_outputs": config_selected.get("fiber_outputs", None),
            "cfg_source": cfg_source,
            "config_selected": config_selected,
            "inputs": {
                "dwi_file": getattr(configuration, "dwi_path", None),
                "mask_file": getattr(configuration, "mask_path", None),
                "bval_file": getattr(configuration, "bval_path", None),
                "bvec_file": getattr(configuration, "bvec_path", None),
            },
            "data": {
                "dwi_shape": dwi_shape,
                "spatial_dims": spatial_dims,
                "n_volumes": n_volumes,
                "masked_voxels": masked_voxels,
            },
            "timings_s": dict(getattr(model, "_timings", {}) or {}),
            "status": dict(getattr(model, "_flags", {}) or {}),
            "runtime": {
                "dbsipy_version": str(dbsipy_version),
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "numpy": _safe_version("numpy"),
                "torch": _safe_version("torch"),
                "nibabel": _safe_version("nibabel"),
                "dipy": _safe_version("dipy"),
                "joblib": _safe_version("joblib"),
            },
            "rng": dict(getattr(model, "_rng", {}) or {}),
            "cuda": {
                "available": bool(torch.cuda.is_available()),
                "device_index": (int(torch.cuda.current_device()) if torch.cuda.is_available() else None),
                "device_name": (
                    torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else None
                ),
            },
            "provenance": {
                "argv": getattr(model, "_argv_str", None),
                "git_commit": getattr(model, "_git_commit", None),
            },
            "artifacts": {
                "config_final_ini": str(Path(save_dir) / "config_final.ini"),
                "analysis_config_ini": str(Path(save_dir) / "analysis_config.ini"),
                "log": str(Path(save_dir) / "log"),
                "outputs_nii_gz": outputs,
                "output_map_schema_version": 1,
                "output_map_specs": output_map_specs,
            },
        }

        out_path = Path(save_dir) / "run_manifest.json"
        out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        logging.info(f"Run manifest saved to: {out_path}")
    except Exception:
        logging.exception("Failed to write run_manifest.json")
