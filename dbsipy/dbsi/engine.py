"""DBSI engine runner.

This module contains the extracted DBSI branch from `DBSIpy.calc()`.
The public entrypoint is `run_dbsi(dbsi)`.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import torch

from dbsipy.core.dti_pipeline import apply_dti_maps, set_if_present
from dbsipy.core.logfmt import log_banner
from dbsipy.dbsi.step1 import DiffusionBasisFunctionModel
from dbsipy.misc.models.Linear_Models import diffusion_tensor_model as DiffusionTensorModel
from dbsipy.nn import leastsquares


def run_dbsi(dbsi: Any) -> None:
    """Run the DBSI engine branch for a DBSIpy instance.

    Behavior-preserving extraction of the `ENGINE in {'DBSI', 'IA'}` path from
    `dbsipy/core/fast_DBSI.py::DBSIpy.calc()`.

    Note: The extracted logic still uses the historical timing/flag keys
    (e.g., `dbsi_ia_total_fit_s`) to preserve downstream expectations.
    """
    start = time.time()
    log_banner('Starting DBSI')
    logging.info('Fitting Step 1 (orientation NNLS)...')
    orientation_model = DiffusionBasisFunctionModel(dbsi.configuration).fit(
        dbsi.bvals, dbsi.bvecs, dbsi.dwi
    )
    fit_first = time.time()
    dbsi._timings['step1_s'] = float(fit_first - start)

    logging.info(
        f"Step 1 complete in {round(fit_first-start, 4)} seconds; fitting Step 2 (multi-fiber diffusivity)..."
    )

    leastsquares.MultiFiberModel(device=dbsi.configuration.DEVICE, engine=dbsi.configuration.ENGINE)(
        dbsi, ModelPriors=orientation_model.priors
    )

    second_opt = time.time()
    dbsi._timings['step2_s'] = float(second_opt - fit_first)
    dbsi._timings['dbsi_ia_total_fit_s'] = float(second_opt - start)
    logging.info(f'Step 2 complete in {round(second_opt-fit_first, 4)} seconds.')
    logging.info('DBSI fitting complete; starting DTI comparison...')

    if dbsi.configuration.diagnostics_enabled:
        try:
            logging.info(
                f"Diagnostics: Step1_time={round(fit_first-start, 4)}s, Step2_time={round(second_opt-fit_first, 4)}s"
            )
        except Exception:
            pass

    dti_start = time.time()
    try:
        dti_sel = dbsi.bvals < dbsi.configuration.dti_bval_cutoff
        bvals_dti = dbsi.bvals[dti_sel].detach().cpu()
        bvecs_dti = dbsi.bvecs[dti_sel].detach().cpu()
        dwi_dti = dbsi.dwi[:, dti_sel].detach().cpu()

        tenmodel = DiffusionTensorModel(bvals_dti, bvecs_dti, device='cpu')
        tenfit = tenmodel.fit(dwi_dti, fit_method='OLS')

        apply_dti_maps(dbsi.params, tenfit)
        set_if_present(dbsi.params, 'b0_map', dbsi.dwi_raw[:, torch.argmin(dbsi.bvals)])
        dbsi._timings['dbsi_ia_dti_compare_s'] = float(time.time() - dti_start)
        dbsi._flags['dbsi_ia_dti_compare_ok'] = True
        logging.info('DTI comparison complete')
    except Exception:
        dbsi._timings['dbsi_ia_dti_compare_s'] = float(time.time() - dti_start)
        dbsi._flags['dbsi_ia_dti_compare_ok'] = False
        logging.exception('DTI comparison failed; continuing without DTI maps')

    return
