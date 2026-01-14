from __future__ import annotations

import configparser
from pathlib import Path

import numpy as np
import torch

from dbsipy.core.configuration import configuration
from dbsipy.core.io import mask_dwi
from dbsipy.core.utils import MIN_POSITIVE_SIGNAL


def test_empty_mask_file_in_ini_uses_non_trivial_signal_mask(tmp_path: Path) -> None:
    dwi = tmp_path / "dwi.nii.gz"
    bval = tmp_path / "bvals"
    bvec = tmp_path / "bvecs"
    dwi.write_bytes(b"dummy")
    bval.write_text("0 1000\n", encoding="utf-8")
    bvec.write_text("0 0\n", encoding="utf-8")

    cfg = configparser.ConfigParser()
    cfg["GLOBAL"] = {"model_engine": "DTI"}
    cfg["INPUT"] = {
        "dwi_file": str(dwi),
        "bval_file": str(bval),
        "bvec_file": str(bvec),
        "mask_file": "",  # explicit empty means: minimal signal-based masking
    }

    c = configuration(cfg)
    assert isinstance(c.mask_path, str)
    assert c.mask_path == ""


def test_none_like_mask_file_in_ini_uses_non_trivial_signal_mask(tmp_path: Path) -> None:
    dwi = tmp_path / "dwi.nii.gz"
    bval = tmp_path / "bvals"
    bvec = tmp_path / "bvecs"
    dwi.write_bytes(b"dummy")
    bval.write_text("0 1000\n", encoding="utf-8")
    bvec.write_text("0 0\n", encoding="utf-8")

    none_like_values = ["n/a", "NA", "none", r"n\a"]
    for v in none_like_values:
        cfg = configparser.ConfigParser()
        cfg["GLOBAL"] = {"model_engine": "DTI"}
        cfg["INPUT"] = {
            "dwi_file": str(dwi),
            "bval_file": str(bval),
            "bvec_file": str(bvec),
            "mask_file": v,
        }
        c = configuration(cfg)
        assert isinstance(c.mask_path, str)
        assert c.mask_path == ""


def test_non_trivial_signal_mask_drops_floor_only() -> None:
    # 2 voxels, 2 volumes
    # voxel 0: entirely at MIN_POSITIVE_SIGNAL (represents clipped background)
    # voxel 1: has real signal in at least one volume
    # Note: `mask_dwi` determines the volume axis by matching `bvals` length.
    # Avoid ambiguous shapes where a spatial dimension also equals n_volumes.
    dwi = np.zeros((3, 1, 1, 2), dtype=np.float64)
    dwi[0, 0, 0, :] = float(MIN_POSITIVE_SIGNAL)
    dwi[1, 0, 0, 0] = float(MIN_POSITIVE_SIGNAL)
    dwi[1, 0, 0, 1] = 0.1
    dwi[2, 0, 0, :] = float(MIN_POSITIVE_SIGNAL)

    bvals = torch.tensor([0.0, 1000.0], dtype=torch.float32)

    mask, dwi_masked, mask_source, spatial_dims, vol_idx, non_trivial = mask_dwi(
        dwi,
        bvals=bvals,
        mask_path="",  # triggers non_trivial_signal path
    )

    assert mask_source == "non_trivial_signal"
    assert spatial_dims == (3, 1, 1)
    assert vol_idx == 3
    assert non_trivial.shape == (3, 1, 1)

    # Only the second voxel should survive.
    assert mask.sum() == 1
    assert dwi_masked.shape == (1, 2)
    np.testing.assert_allclose(dwi_masked[0], dwi[1, 0, 0, :])
