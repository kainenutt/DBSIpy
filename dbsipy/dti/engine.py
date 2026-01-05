"""DTI engine runner.

This module contains the extracted DTI branch from `DBSIpy.calc()`.
The public entrypoint is `run_dti(dbsi)`.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import torch

from dbsipy.core.dti_pipeline import apply_dti_maps, set_if_present
from dbsipy.core.logfmt import log_banner
from dbsipy.misc.models.Linear_Models import diffusion_tensor_model as DiffusionTensorModel


def run_dti(dbsi: Any) -> None:
    """Run the DTI engine branch for a DBSIpy instance.

    This is a behavior-preserving extraction of the `ENGINE == 'DTI'` path from
    `dbsipy/core/fast_DBSI.py::DBSIpy.calc()`.
    """
    start = time.time()

    log_banner('Starting DTI')

    dti_sel = dbsi.bvals < dbsi.configuration.dti_bval_cutoff
    bvals_dti = dbsi.bvals[dti_sel].detach().cpu()
    bvecs_dti = dbsi.bvecs[dti_sel].detach().cpu()

    dwi_dti = dbsi.dwi_raw[:, dti_sel]
    if isinstance(dwi_dti, np.ndarray):
        dwi_dti = torch.from_numpy(dwi_dti)
    dwi_dti = dwi_dti.detach().cpu()

    fit_method = str(getattr(dbsi.configuration, 'dti_fit_method', 'WLS')).upper()
    logging.info(
        f"DTI fit: voxels={int(dwi_dti.shape[0]):,}, volumes={int(bvals_dti.shape[0])}, fit_method={fit_method}"
    )

    tenmodel = DiffusionTensorModel(bvals_dti, bvecs_dti, device='cpu')
    opt_args = getattr(dbsi.configuration, 'DTI_OPTIMIZER_ARGS', None) if fit_method == 'ADAM' else None
    tenfit = tenmodel.fit(dwi_dti, fit_method=fit_method, optimizer_args=opt_args)

    apply_dti_maps(dbsi.params, tenfit)
    set_if_present(dbsi.params, 'b0_map', dbsi.dwi_raw[:, torch.argmin(dbsi.bvals)])

    dbsi._timings['dti_fit_s'] = float(time.time() - start)
    logging.info(f"DTI complete in {dbsi._timings['dti_fit_s']:.4f} seconds")
    return
