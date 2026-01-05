from __future__ import annotations

from typing import Any
from pathlib import Path

import numpy as np
import nibabel as nb
import os
import torch
from dipy.segment.mask import median_otsu

from dbsipy.core.utils import MIN_POSITIVE_SIGNAL


def auto_mask_output_path(dwi_path: str) -> str:
    """Return the output path for an auto-generated mask.

    If the input DWI is called "dwi.nii.gz" or "dwi.nii", the mask is saved as
    "dwi_auto_mask.nii.gz" in the same directory as the DWI.
    """

    p = Path(dwi_path)
    name = p.name

    # Handle .nii.gz explicitly.
    if name.lower().endswith(".nii.gz"):
        stem = name[:-7]
    elif name.lower().endswith(".nii"):
        stem = name[:-4]
    else:
        stem = p.stem

    return str(p.with_name(f"{stem}_auto_mask.nii.gz"))


def save_auto_mask_nifti(mask: np.ndarray, *, dwi_path: str, affine: Any, header: Any) -> str:
    """Save an auto-generated 3D mask NIfTI next to the input DWI.

    Returns the written path.
    """

    out_path = auto_mask_output_path(dwi_path)

    # NIfTI expects spatial dims; store as uint8 (0/1).
    mask_u8 = (mask > 0).astype(np.uint8)
    hdr = header.copy() if header is not None else None
    if hdr is not None:
        try:
            hdr.set_data_dtype(np.uint8)
        except Exception:
            pass

    img = nb.Nifti1Image(mask_u8, affine, header=hdr)
    nb.save(img, out_path)
    return out_path


def load_dwi_nifti(dwi_path: str) -> tuple[np.ndarray, Any, Any]:
    """Load a DWI NIfTI as a numpy array and return (data, header, affine).

    Preserves the legacy behavior in `DBSIpy.load()`:
    - uses `get_fdata()` (float64 by default)
    - clips to `MIN_POSITIVE_SIGNAL`
    - expands 2D/3D to 4D by inserting a singleton z-dimension
    """

    dwi_nifti = nb.load(dwi_path)
    dwi = np.clip(dwi_nifti.get_fdata(), a_min=MIN_POSITIVE_SIGNAL, a_max=None)

    header = dwi_nifti.header
    affine = dwi_nifti.affine

    if dwi.ndim < 4:
        dwi = np.expand_dims(dwi, axis=2)

    return dwi, header, affine


def load_bvals_bvecs(bval_path: str, bvec_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Load bvals/bvecs from text files and return (bvals, bvecs).

    Behavior matches legacy `DBSIpy.load()`:
    - tries whitespace first, then comma-delimited
    - accepts bvecs as 3xN or Nx3 and returns Nx3
    - normalizes bvecs row-wise (with safe handling for all-zero rows)
    """

    try:
        bval_data = np.loadtxt(bval_path)
    except ValueError:
        bval_data = np.loadtxt(bval_path, delimiter=",")
    bvals = torch.from_numpy(bval_data).float()

    try:
        bvec_data = np.loadtxt(bvec_path)
    except ValueError:
        bvec_data = np.loadtxt(bvec_path, delimiter=",")
    bvec_tensor = torch.from_numpy(bvec_data).float()

    # Transpose if bvecs are in 3xN format.
    if bvec_tensor.ndim > 1 and bvec_tensor.shape[0] == 3:
        bvecs = bvec_tensor.T
    else:
        bvecs = bvec_tensor

    # B-vectors should already be normalized, but perform normalization to be safe.
    bvecs[(bvecs == 0).all(dim=1)] = 1e-3
    bvecs = bvecs / torch.linalg.norm(bvecs, axis=1)[:, None]

    return bvals, bvecs


def mask_dwi(
    dwi: np.ndarray,
    *,
    bvals: torch.Tensor,
    mask_path: str,
) -> tuple[np.ndarray, np.ndarray, str, tuple[int, ...], int, np.ndarray]:
    """Generate/apply a spatial mask and return flattened voxels.

    Returns
    -------
    mask:
        3D boolean mask.
    dwi_masked:
        Masked dwi as a 2D array (n_voxels, n_volumes) in numpy.
    mask_source:
        One of: auto | file | non_trivial_signal
    spatial_dims:
        3D spatial dimensions (x, y, z).
    vol_idx:
        Index of the volume dimension in the original `dwi` array.
    non_trivial_signal_mask:
        3D mask selecting voxels with any nonzero signal.
    """

    # Spatial dims are everything except the volume dimension (legacy heuristic).
    spatial_dims = tuple(int(dim) for dim in dwi.shape if dim != int(bvals.shape[0]))

    vol_candidates = [inx for (inx, dim) in enumerate(dwi.shape) if dim == int(bvals.shape[0])]
    if not vol_candidates:
        raise ValueError("Could not determine DWI volume dimension (no axis matches bvals length).")
    vol_idx = int(np.array(vol_candidates).item())

    non_trivial_signal_mask = np.logical_and(~(dwi == 0).all(axis=vol_idx), np.ones(spatial_dims, dtype=bool))

    # Determine mask.
    if mask_path != 'auto':
        if mask_path and isinstance(mask_path, str) and os.path.exists(mask_path):
            mask = np.logical_and(nb.load(mask_path).get_fdata() > 0, non_trivial_signal_mask)
            mask_source = 'file'
        else:
            mask = non_trivial_signal_mask
            mask_source = 'non_trivial_signal'
    else:
        _, mask = median_otsu(
            dwi,
            median_radius=4,
            numpass=4,
            vol_idx=np.arange(int(bvals.shape[0])),
            dilate=3,
        )
        mask = np.logical_and(mask, non_trivial_signal_mask)
        mask_source = 'auto'

    dwi_masked = dwi[mask]
    dwi_masked = dwi_masked.reshape(-1, int(bvals.shape[0]))

    return mask, dwi_masked, mask_source, spatial_dims, vol_idx, non_trivial_signal_mask
