"""NODDI engine runner.

This module contains the extracted NODDI branch from `DBSIpy.calc()`.
The public entrypoint is `run_noddi(dbsi)`.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import torch

from dbsipy.core.dti_pipeline import apply_dti_maps, set_if_present
from dbsipy.core.logfmt import log_banner
from dbsipy.misc.models.Linear_Models import diffusion_tensor_model as DiffusionTensorModel
from dbsipy.misc.models.Linear_Models import NODDIModel


def run_noddi(dbsi: Any) -> None:
    """Run the NODDI engine branch for a DBSIpy instance.

    Behavior-preserving extraction of the `ENGINE == 'NODDI'` path from
    `dbsipy/core/fast_DBSI.py::DBSIpy.calc()`.
    """
    start = time.time()
    log_banner('Starting NODDI')
    logging.info(
        f"NODDI fit: voxels={int(dbsi.dwi.shape[0]):,}, volumes={int(dbsi.bvals.shape[0])}"
    )

    # NODDI fitting with Watson distribution for dispersed sticks
    noddimodel = NODDIModel(
        dbsi.bvals,
        dbsi.bvecs,
        device=dbsi.configuration.DEVICE,
    )
    noddifit = noddimodel.fit(
        dbsi.dwi_raw.to(dbsi.configuration.DEVICE),
        optimizer_args=dbsi.configuration.NODDI_OPTIMIZER_ARGS,
        signal_normalization=getattr(dbsi.configuration, 'signal_normalization', 'auto'),
    )

    # Assign NODDI parameter maps
    dbsi.params['noddi_ndi'].pmap = noddifit.ndi[:, None]
    dbsi.params['noddi_odi'].pmap = noddifit.odi[:, None]
    dbsi.params['noddi_fiso'].pmap = noddifit.fiso[:, None]
    dbsi.params['noddi_fec'].pmap = noddifit.fec[:, None]
    dbsi.params['noddi_kappa'].pmap = noddifit.kappa[:, None]
    dbsi.params['noddi_d_ic'].pmap = noddifit.d_ic[:, None]
    dbsi.params['noddi_d_ec_par'].pmap = noddifit.d_ec_par[:, None]
    dbsi.params['noddi_d_ec_perp'].pmap = noddifit.d_ec_perp[:, None]
    dbsi.params['noddi_fiber_direction'].pmap = noddifit.fiber_direction
    dbsi.params['noddi_fiber_direction_cfa'].pmap = noddifit.fiber_direction_cfa

    noddi_time = time.time()
    dbsi._timings['noddi_fit_s'] = float(noddi_time - start)
    logging.info(
        f"NODDI fit complete in {round(noddi_time-start, 4)} seconds; starting DTI comparison..."
    )

    dti_start = time.time()
    try:
        dti_sel = dbsi.bvals < dbsi.configuration.dti_bval_cutoff
        bvals_dti = dbsi.bvals[dti_sel].detach().cpu()
        bvecs_dti = dbsi.bvecs[dti_sel].detach().cpu()
        dwi_dti = dbsi.dwi[:, dti_sel].detach().cpu()

        tenmodel = DiffusionTensorModel(bvals_dti, bvecs_dti, device='cpu')
        tenfit = tenmodel.fit(dwi_dti, fit_method='OLS')

        apply_dti_maps(dbsi.params, tenfit)
        dbsi._timings['noddi_dti_compare_s'] = float(time.time() - dti_start)
        dbsi._flags['noddi_dti_compare_ok'] = True
    except Exception:
        dbsi._timings['noddi_dti_compare_s'] = float(time.time() - dti_start)
        dbsi._flags['noddi_dti_compare_ok'] = False
        logging.exception('DTI comparison failed; continuing without DTI maps')

    set_if_present(dbsi.params, 'b0_map', dbsi.dwi_raw[:, torch.argmin(dbsi.bvals)])
    if dbsi._flags.get('noddi_dti_compare_ok'):
        logging.info('DTI comparison complete')

    return
