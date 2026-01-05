from __future__ import annotations

from pathlib import Path

import numpy as np
import nibabel as nb

from dbsipy.core.io import auto_mask_output_path, save_auto_mask_nifti


def test_auto_mask_output_path_nii_gz() -> None:
    p = auto_mask_output_path("/tmp/dwi.nii.gz")
    out = Path(p)
    assert out.parent == Path("/tmp")
    assert out.name == "dwi_auto_mask.nii.gz"


def test_auto_mask_output_path_nii() -> None:
    p = auto_mask_output_path("/tmp/dwi.nii")
    out = Path(p)
    assert out.parent == Path("/tmp")
    assert out.name == "dwi_auto_mask.nii.gz"


def test_save_auto_mask_writes_uint8(tmp_path: Path) -> None:
    dwi_path = tmp_path / "dwi.nii.gz"

    # Minimal valid 4D DWI NIfTI.
    data = np.zeros((5, 6, 7, 2), dtype=np.float32)
    affine = np.eye(4)
    nb.save(nb.Nifti1Image(data, affine), str(dwi_path))

    mask = np.zeros((5, 6, 7), dtype=bool)
    mask[0, 0, 0] = True

    out = save_auto_mask_nifti(mask, dwi_path=str(dwi_path), affine=affine, header=nb.load(str(dwi_path)).header)
    assert out.endswith("dwi_auto_mask.nii.gz")

    saved = nb.load(out)
    assert saved.shape == (5, 6, 7)
    assert saved.get_fdata().max() == 1.0
    assert saved.get_data_dtype() == np.dtype("uint8")
