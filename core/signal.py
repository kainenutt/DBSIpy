from __future__ import annotations

from typing import Any

import torch

from dbsipy.core.utils import MIN_POSITIVE_SIGNAL
from dbsipy.core.validation import ConfigurationError


def normalize_signal(
    dwi: torch.Tensor,
    bvals: torch.Tensor,
    *,
    mode: str,
    engine: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float] | None, str]:
    """Normalize diffusion signal according to DBSIpy semantics.

    Parameters
    ----------
    dwi:
        (n_voxels, n_volumes) float tensor.
    bvals:
        (n_volumes,) float tensor.
    mode:
        One of: auto | max | b0 | minb | none
    engine:
        Used only when mode='auto'. For DBSI/IA/NODDI auto->max, otherwise auto->none.

    Returns
    -------
    dwi_raw:
        Copy of input signal.
    dwi_norm:
        Normalized signal.
    scale:
        Per-voxel scale estimate ("S0" estimate for attenuation normalization). Always shape (n_voxels,).
    stats:
        Dict with min/median/max of scale for nonzero voxels, or None when mode='none'.
    mode_used:
        Final resolved mode (auto resolved to max/none).
    """

    dwi_raw = dwi.clone()
    dwi_norm = dwi.clone()

    nonzero_vox = ~(dwi_raw == 0).all(dim=1)

    mode_norm = str(mode or 'auto').strip().lower()
    if mode_norm == 'auto':
        eng = (engine or '').strip().upper()
        mode_norm = 'max' if eng in {'DBSI', 'IA', 'NODDI'} else 'none'

    # Default scale is 1 (no normalization).
    scale = torch.ones((dwi_raw.shape[0],), dtype=dwi_raw.dtype, device=dwi_raw.device)
    stats: dict[str, float] | None = None

    if mode_norm != 'none':
        # Ensure b-value selection masks live on the same device as the signal.
        bvals_local = bvals.detach()
        if bvals_local.device != dwi_raw.device:
            bvals_local = bvals_local.to(dwi_raw.device)

        minb_val = float(bvals_local.min().item())
        b0_sel = (bvals_local == 0)
        minb_sel = (bvals_local == minb_val)

        if mode_norm == 'max':
            scale_nonzero = dwi_raw[nonzero_vox].max(dim=1).values
        elif mode_norm == 'b0':
            if bool(b0_sel.any()):
                scale_nonzero = dwi_raw[nonzero_vox][:, b0_sel].mean(dim=1)
            elif bool(minb_sel.any()):
                scale_nonzero = dwi_raw[nonzero_vox][:, minb_sel].mean(dim=1)
            else:
                scale_nonzero = dwi_raw[nonzero_vox].max(dim=1).values
        elif mode_norm == 'minb':
            if bool(minb_sel.any()):
                scale_nonzero = dwi_raw[nonzero_vox][:, minb_sel].mean(dim=1)
            else:
                scale_nonzero = dwi_raw[nonzero_vox].max(dim=1).values
        else:
            raise ConfigurationError(
                "Invalid signal_normalization mode.\n"
                "Valid options: max | b0 | minb | none\n"
                f"Current value: '{mode_norm}'"
            )

        scale_nonzero = torch.clamp(scale_nonzero, min=MIN_POSITIVE_SIGNAL)
        scale[nonzero_vox] = scale_nonzero
        dwi_norm[nonzero_vox] = dwi_raw[nonzero_vox] / scale_nonzero[:, None]

        try:
            s = scale_nonzero.detach().cpu()
            stats = {
                'min': float(torch.min(s).item()),
                'median': float(torch.median(s).item()),
                'max': float(torch.max(s).item()),
            }
        except Exception:
            stats = None

    return dwi_raw, dwi_norm, scale, stats, mode_norm


def _as_float_dict(x: Any) -> dict[str, float] | None:
    """Best-effort conversion helper (used for defensive copying)."""
    if x is None:
        return None
    if isinstance(x, dict):
        out: dict[str, float] = {}
        for k in ('min', 'median', 'max'):
            if k in x:
                try:
                    out[k] = float(x[k])
                except Exception:
                    pass
        return out or None
    return None
