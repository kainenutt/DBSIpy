from __future__ import annotations

import torch


def set_if_present(params: dict, map_name: str, value) -> None:
    """Set a parameter-map value only if it was allocated.

    `params` is expected to be `DBSIpy.params` (a dict-like of parameter_map objects).
    """
    try:
        if map_name in params:
            params[map_name].pmap = value
    except Exception:
        # A missing map or incompatible shape should not crash unrelated flows.
        return


def apply_dti_maps(params: dict, tenfit) -> None:
    """Populate DTI maps from a `DiffusionTensorModel.fit(...)` result.

    This writes both the default and expanded DTI outputs, but uses `set_if_present`
    so it only populates maps that were actually allocated by `output_map_set`.
    """

    # Core DTI maps (expected in the default map set)
    set_if_present(params, 'dti_fa', tenfit.fa[:, None])
    set_if_present(params, 'dti_axial', tenfit.ad[:, None])
    set_if_present(params, 'dti_adc', tenfit.adc[:, None])
    set_if_present(params, 'dti_radial', tenfit.rd[:, None])
    set_if_present(params, 'dti_cfa', tenfit.cfa)

    # Expanded DTI maps (only if allocated)
    set_if_present(params, 'dti_lambda_1', tenfit.ad[:, None])
    set_if_present(params, 'dti_lambda_2', tenfit.eval_2[:, None])
    set_if_present(params, 'dti_lambda_3', tenfit.eval_3[:, None])

    set_if_present(params, 'dti_eigenvec_1', tenfit.eigen_frame[:, :, -1])
    set_if_present(params, 'dti_eigenvec_2', tenfit.eigen_frame[:, :, -2])
    set_if_present(params, 'dti_eigenvec_3', tenfit.eigen_frame[:, :, -3])

    # CFA-weighted eigenvectors (only if allocated)
    try:
        fa_clip = torch.clip(tenfit.fa, 0, 1)
        set_if_present(
            params,
            'dti_eigenvec_1_cfa',
            torch.einsum('ij, i -> ij', torch.abs(tenfit.eigen_frame[:, :, -1]), fa_clip),
        )
        set_if_present(
            params,
            'dti_eigenvec_2_cfa',
            torch.einsum('ij, i -> ij', torch.abs(tenfit.eigen_frame[:, :, -2]), fa_clip),
        )
        set_if_present(
            params,
            'dti_eigenvec_3_cfa',
            torch.einsum('ij, i -> ij', torch.abs(tenfit.eigen_frame[:, :, -3]), fa_clip),
        )
    except Exception:
        # If tenfit doesn't expose expected tensor shapes, skip these.
        return
