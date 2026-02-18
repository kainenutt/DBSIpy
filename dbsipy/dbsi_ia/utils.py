import torch
from typing import List, Dict
from dbsipy.misc.models.Linear_Models import diffusion_tensor_model


def _infer_device_from_fit_result(fit_result: Dict, fallback_device: str) -> torch.device:
    for v in fit_result.values():
        if isinstance(v, torch.Tensor):
            return v.device
    return torch.device(fallback_device)


def _maybe_set_param(DBSI_cls, key: str, index, value: torch.Tensor) -> None:
    param = DBSI_cls.params.get(key)
    if param is None:
        return
    param.pmap[index] = value


def _write_tensor_maps(DBSI_cls, tenfit, inds_torch: torch.Tensor, fiber_inds: torch.Tensor, prefix: str, fractions_hat: torch.Tensor) -> None:
    fiber_sel = inds_torch[fiber_inds]

    _maybe_set_param(DBSI_cls, f'{prefix}_axial', fiber_sel, tenfit.ad)
    _maybe_set_param(DBSI_cls, f'{prefix}_radial', fiber_sel, tenfit.rd[:, None])
    _maybe_set_param(DBSI_cls, f'{prefix}_adc', fiber_sel, 1 / 3 * (tenfit.ad + tenfit.eval_2[:, None] + tenfit.eval_3[:, None]))
    _maybe_set_param(DBSI_cls, f'{prefix}_fa', fiber_sel, tenfit.fa[:, None])
    _maybe_set_param(DBSI_cls, f'{prefix}_fraction', inds_torch, fractions_hat[:, None])

    # CFA maps are optional; write if allocated.
    _maybe_set_param(DBSI_cls, f'{prefix}_cfa', fiber_sel, torch.squeeze(tenfit.cfa[:, None]))

    # Expanded maps
    _maybe_set_param(DBSI_cls, f'{prefix}_lambda_1', fiber_sel, tenfit.ad)
    _maybe_set_param(DBSI_cls, f'{prefix}_lambda_2', fiber_sel, tenfit.eval_2[:, None])
    _maybe_set_param(DBSI_cls, f'{prefix}_lambda_3', fiber_sel, tenfit.eval_3[:, None])
    _maybe_set_param(DBSI_cls, f'{prefix}_eigenvec_1', fiber_sel, tenfit.eigen_frame[:, :, -1])
    _maybe_set_param(DBSI_cls, f'{prefix}_eigenvec_2', fiber_sel, tenfit.eigen_frame[:, :, -2])
    _maybe_set_param(DBSI_cls, f'{prefix}_eigenvec_3', fiber_sel, tenfit.eigen_frame[:, :, -3])
    _maybe_set_param(DBSI_cls, f'{prefix}_eigenvec_1_cfa', fiber_sel, torch.squeeze(tenfit.cfa[:, None]))
    _maybe_set_param(DBSI_cls, f'{prefix}_eigenvec_2_cfa', fiber_sel, torch.einsum('ij, i -> ij', torch.abs(tenfit.eigen_frame[:, :, -2]), torch.clip(tenfit.fa, 0, 1)))
    _maybe_set_param(DBSI_cls, f'{prefix}_eigenvec_3_cfa', fiber_sel, torch.einsum('ij, i -> ij', torch.abs(tenfit.eigen_frame[:, :, -3]), torch.clip(tenfit.fa, 0, 1)))

def fit_results_to_parameter_maps_dbsi_ia(fit_results: List[Dict[str, torch.FloatTensor]], indicies: List[int], DBSI_cls):
    for ii, (fit_result, inds) in enumerate(zip(fit_results, indicies)):
        max_num_fibers = fit_result.get('max_group_number', 1)
        if isinstance(max_num_fibers, torch.Tensor):
            max_num_fibers = int(max_num_fibers.item())

        try:
            max_num_fibers = int(max_num_fibers)
        except Exception:
            max_num_fibers = 1

        try:
            cfg_max_fibers = int(getattr(DBSI_cls.configuration, 'max_group_number', max_num_fibers) or max_num_fibers)
        except Exception:
            cfg_max_fibers = max_num_fibers

        # A valid outcome is a "0-fiber" model (purely isotropic). In that case,
        # the fit result will not contain any fiber_* keys, so reconstruction
        # must not assume at least one fiber.
        n_fibers = max(0, min(max_num_fibers, cfg_max_fibers))

        current_device = _infer_device_from_fit_result(fit_result, fallback_device=DBSI_cls.configuration.DEVICE)
        inds_torch = torch.tensor(inds, dtype=int, device=current_device)

        # Learnable S0 output (optional)
        if 's0_map' in fit_result:
            s0 = fit_result['s0_map']
            if isinstance(s0, torch.Tensor):
                s0 = s0.to(current_device)
            _maybe_set_param(DBSI_cls, 's0_map', inds_torch, s0[:, None])

        # Hidden output: full isotropic spectrum coefficients (optional)
        if 'isotropic_spectrum' in fit_result and DBSI_cls.params.get('isotropic_spectrum') is not None:
            iso_spec = fit_result.get('isotropic_spectrum')
            if isinstance(iso_spec, torch.Tensor):
                iso_spec = iso_spec.to(current_device)
            if iso_spec is not None:
                _maybe_set_param(DBSI_cls, 'isotropic_spectrum', inds_torch, iso_spec)

        tenmodel = diffusion_tensor_model(DBSI_cls.bvals, DBSI_cls.bvecs, device=str(current_device))

        n_vox = len(inds)

        # Aggregates for overall fiber compartment
        agg_den = torch.zeros((n_vox, 1), device=current_device)
        agg_num: dict[str, torch.Tensor] = {
            'fa': torch.zeros((n_vox, 1), device=current_device),
            'axial': torch.zeros((n_vox, 1), device=current_device),
            'radial': torch.zeros((n_vox, 1), device=current_device),
            'adc': torch.zeros((n_vox, 1), device=current_device),
        }

        # IA/EA-specific aggregates (only written if corresponding maps are allocated)
        agg_den_ia = torch.zeros((n_vox, 1), device=current_device)
        agg_num_ia: dict[str, torch.Tensor] = {
            'fa': torch.zeros((n_vox, 1), device=current_device),
            'axial': torch.zeros((n_vox, 1), device=current_device),
            'radial': torch.zeros((n_vox, 1), device=current_device),
            'adc': torch.zeros((n_vox, 1), device=current_device),
        }
        agg_den_ea = torch.zeros((n_vox, 1), device=current_device)
        agg_num_ea: dict[str, torch.Tensor] = {
            'fa': torch.zeros((n_vox, 1), device=current_device),
            'axial': torch.zeros((n_vox, 1), device=current_device),
            'radial': torch.zeros((n_vox, 1), device=current_device),
            'adc': torch.zeros((n_vox, 1), device=current_device),
        }

        for compartment in DBSI_cls.configuration.DEFAULT_FIBER_CUTS.keys():
            for fiber_idx in range(1, n_fibers + 1):
                frac_key = f'fiber_{fiber_idx:02d}_local_{compartment}_fractions'
                sig_key = f'fiber_{fiber_idx:02d}_local_{compartment}_signal'

                fiber_c_fractions_hat = fit_result.get(frac_key)
                fiber_c_signal_hat = fit_result.get(sig_key)

                # Some Step 2 outcomes can be isotropic-only (0-fiber) even when
                # the config allows fibers; in that case the per-fiber keys may be absent.
                if fiber_c_fractions_hat is None or fiber_c_signal_hat is None:
                    continue

                if compartment == 'fiber':
                    prefix = f'fiber_{fiber_idx - 1}d'
                else:
                    prefix = f'fiber_{fiber_idx - 1}d_{compartment}'

                fiber_inds = fiber_c_fractions_hat >= DBSI_cls.configuration.fiber_threshold
                if torch.any(fiber_inds):
                    tenfit = tenmodel.fit(fiber_c_signal_hat[fiber_inds])
                    _write_tensor_maps(DBSI_cls, tenfit, inds_torch, fiber_inds, prefix, fiber_c_fractions_hat)

                    # Aggregates
                    if compartment == 'fiber' and DBSI_cls.params.get('fiber_total_fraction') is not None:
                        ad = torch.zeros((n_vox, 1), device=current_device)
                        rd = torch.zeros((n_vox, 1), device=current_device)
                        adc = torch.zeros((n_vox, 1), device=current_device)
                        fa = torch.zeros((n_vox, 1), device=current_device)

                        ad[fiber_inds] = tenfit.ad
                        rd[fiber_inds] = tenfit.rd[:, None]
                        adc[fiber_inds] = tenfit.adc[:, None]
                        fa[fiber_inds] = tenfit.fa[:, None]

                        w = fiber_c_fractions_hat[:, None]
                        agg_num['axial'] += w * ad
                        agg_num['radial'] += w * rd
                        agg_num['adc'] += w * adc
                        agg_num['fa'] += w * fa

                    if compartment == 'IA' and DBSI_cls.params.get('fiber_total_IA_fraction') is not None:
                        ad = torch.zeros((n_vox, 1), device=current_device)
                        rd = torch.zeros((n_vox, 1), device=current_device)
                        adc = torch.zeros((n_vox, 1), device=current_device)
                        fa = torch.zeros((n_vox, 1), device=current_device)

                        ad[fiber_inds] = tenfit.ad
                        rd[fiber_inds] = tenfit.rd[:, None]
                        adc[fiber_inds] = tenfit.adc[:, None]
                        fa[fiber_inds] = tenfit.fa[:, None]

                        w = fiber_c_fractions_hat[:, None]
                        agg_num_ia['axial'] += w * ad
                        agg_num_ia['radial'] += w * rd
                        agg_num_ia['adc'] += w * adc
                        agg_num_ia['fa'] += w * fa

                    if compartment == 'EA' and DBSI_cls.params.get('fiber_total_EA_fraction') is not None:
                        ad = torch.zeros((n_vox, 1), device=current_device)
                        rd = torch.zeros((n_vox, 1), device=current_device)
                        adc = torch.zeros((n_vox, 1), device=current_device)
                        fa = torch.zeros((n_vox, 1), device=current_device)

                        ad[fiber_inds] = tenfit.ad
                        rd[fiber_inds] = tenfit.rd[:, None]
                        adc[fiber_inds] = tenfit.adc[:, None]
                        fa[fiber_inds] = tenfit.fa[:, None]

                        w = fiber_c_fractions_hat[:, None]
                        agg_num_ea['axial'] += w * ad
                        agg_num_ea['radial'] += w * rd
                        agg_num_ea['adc'] += w * adc
                        agg_num_ea['fa'] += w * fa
                else:
                    if compartment == 'fiber':
                        _maybe_set_param(DBSI_cls, f'{prefix}_fraction', inds_torch, fiber_c_fractions_hat[:, None])
                    else:
                        _maybe_set_param(DBSI_cls, f'{prefix}_fraction', inds_torch, fiber_c_fractions_hat[:, None])

                if compartment == 'fiber' and DBSI_cls.params.get('fiber_total_fraction') is not None:
                    agg_den += fiber_c_fractions_hat[:, None]
                if compartment == 'IA' and DBSI_cls.params.get('fiber_total_IA_fraction') is not None:
                    agg_den_ia += fiber_c_fractions_hat[:, None]
                if compartment == 'EA' and DBSI_cls.params.get('fiber_total_EA_fraction') is not None:
                    agg_den_ea += fiber_c_fractions_hat[:, None]

        if DBSI_cls.params.get('fiber_total_fraction') is not None:
            eps = torch.tensor(1.0e-8, device=current_device)
            denom = torch.clamp(agg_den, min=eps)
            _maybe_set_param(DBSI_cls, 'fiber_total_fraction', inds_torch, agg_den)
            if DBSI_cls.params.get('fiber_agg_axial') is not None:
                _maybe_set_param(DBSI_cls, 'fiber_agg_axial', inds_torch, agg_num['axial'] / denom)
            if DBSI_cls.params.get('fiber_agg_radial') is not None:
                _maybe_set_param(DBSI_cls, 'fiber_agg_radial', inds_torch, agg_num['radial'] / denom)
            if DBSI_cls.params.get('fiber_agg_adc') is not None:
                _maybe_set_param(DBSI_cls, 'fiber_agg_adc', inds_torch, agg_num['adc'] / denom)
            if DBSI_cls.params.get('fiber_agg_fa') is not None:
                _maybe_set_param(DBSI_cls, 'fiber_agg_fa', inds_torch, agg_num['fa'] / denom)

        if DBSI_cls.params.get('fiber_total_IA_fraction') is not None:
            eps = torch.tensor(1.0e-8, device=current_device)
            denom = torch.clamp(agg_den_ia, min=eps)
            _maybe_set_param(DBSI_cls, 'fiber_total_IA_fraction', inds_torch, agg_den_ia)
            if DBSI_cls.params.get('fiber_agg_IA_axial') is not None:
                _maybe_set_param(DBSI_cls, 'fiber_agg_IA_axial', inds_torch, agg_num_ia['axial'] / denom)
            if DBSI_cls.params.get('fiber_agg_IA_radial') is not None:
                _maybe_set_param(DBSI_cls, 'fiber_agg_IA_radial', inds_torch, agg_num_ia['radial'] / denom)
            if DBSI_cls.params.get('fiber_agg_IA_adc') is not None:
                _maybe_set_param(DBSI_cls, 'fiber_agg_IA_adc', inds_torch, agg_num_ia['adc'] / denom)
            if DBSI_cls.params.get('fiber_agg_IA_fa') is not None:
                _maybe_set_param(DBSI_cls, 'fiber_agg_IA_fa', inds_torch, agg_num_ia['fa'] / denom)

        if DBSI_cls.params.get('fiber_total_EA_fraction') is not None:
            eps = torch.tensor(1.0e-8, device=current_device)
            denom = torch.clamp(agg_den_ea, min=eps)
            _maybe_set_param(DBSI_cls, 'fiber_total_EA_fraction', inds_torch, agg_den_ea)
            if DBSI_cls.params.get('fiber_agg_EA_axial') is not None:
                _maybe_set_param(DBSI_cls, 'fiber_agg_EA_axial', inds_torch, agg_num_ea['axial'] / denom)
            if DBSI_cls.params.get('fiber_agg_EA_radial') is not None:
                _maybe_set_param(DBSI_cls, 'fiber_agg_EA_radial', inds_torch, agg_num_ea['radial'] / denom)
            if DBSI_cls.params.get('fiber_agg_EA_adc') is not None:
                _maybe_set_param(DBSI_cls, 'fiber_agg_EA_adc', inds_torch, agg_num_ea['adc'] / denom)
            if DBSI_cls.params.get('fiber_agg_EA_fa') is not None:
                _maybe_set_param(DBSI_cls, 'fiber_agg_EA_fa', inds_torch, agg_num_ea['fa'] / denom)

        for compartment in DBSI_cls.configuration.DEFAULT_ISOTROPIC_CUTS.keys():
            _maybe_set_param(DBSI_cls, f'{compartment}_fraction', inds, fit_result[f'{compartment}_fraction'][:, None])
            _maybe_set_param(DBSI_cls, f'{compartment}_adc', inds, fit_result[f'{compartment}_adc'][:, None])

    return
