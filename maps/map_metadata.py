"""Unified map metadata schema.

This module provides a single canonical source of truth for map metadata across
engines (DTI/DBSI/IA/NODDI): units, descriptions, and expected output shape
kinds.

The project historically stored:
- allocation/shape info in per-engine modules (e.g., `src/maps/dti_maps.py`)
- richer metadata only for NODDI (`src/maps/noddi_maps.py`)

To reduce drift, use this module for metadata lookups and for reporting in
manifests.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


ShapeKind = str  # 'scalar' | '3-vector'


@dataclass(frozen=True)
class MapSpec:
    description: str
    units: str
    shape_kind: ShapeKind
    valid_range: Optional[Tuple[float, float]] = None
    notes: Optional[str] = None

    def as_dict(self) -> Dict:
        d = {
            'description': self.description,
            'units': self.units,
            'shape_kind': self.shape_kind,
        }
        if self.valid_range is not None:
            d['range'] = [float(self.valid_range[0]), float(self.valid_range[1])]
        if self.notes:
            d['notes'] = str(self.notes)
        return d


# ---- Canonical specs for non-fiber maps ----

# Convention for diffusivity-like outputs in this codebase: µm²/ms.
_DIFF_UNITS = 'µm²/ms'
_DIFF_RANGE = (0.0, 5.0)  # broad QC range for in-vivo diffusion metrics


DTI_MAP_SPECS: Dict[str, MapSpec] = {
    'dti_fa': MapSpec(
        'DTI fractional anisotropy (FA)',
        'dimensionless',
        'scalar',
        (0.0, 1.0),
        'Fractional anisotropy derived from the diffusion tensor eigenvalues.',
    ),
    'dti_cfa': MapSpec(
        'DTI color FA (CFA)',
        'RGB [0-1]',
        '3-vector',
        (0.0, 1.0),
        'RGB orientation encoding modulated by FA (|v| components scaled by FA).',
    ),
    'dti_axial': MapSpec('DTI axial diffusivity (AD)', _DIFF_UNITS, 'scalar', _DIFF_RANGE, 'AD = λ1.'),
    'dti_radial': MapSpec('DTI radial diffusivity (RD)', _DIFF_UNITS, 'scalar', _DIFF_RANGE, 'RD = (λ2+λ3)/2.'),
    'dti_adc': MapSpec('DTI mean diffusivity (MD/ADC)', _DIFF_UNITS, 'scalar', _DIFF_RANGE, 'MD = (λ1+λ2+λ3)/3.'),
    'dti_lambda_1': MapSpec('DTI eigenvalue λ1', _DIFF_UNITS, 'scalar', _DIFF_RANGE, 'Principal diffusion tensor eigenvalue.'),
    'dti_lambda_2': MapSpec('DTI eigenvalue λ2', _DIFF_UNITS, 'scalar', _DIFF_RANGE, 'Second diffusion tensor eigenvalue.'),
    'dti_lambda_3': MapSpec('DTI eigenvalue λ3', _DIFF_UNITS, 'scalar', _DIFF_RANGE, 'Third diffusion tensor eigenvalue.'),
    'dti_eigenvec_1': MapSpec('DTI principal eigenvector', 'unit vector (x, y, z)', '3-vector', None, 'Unit vector for λ1 direction.'),
    'dti_eigenvec_2': MapSpec('DTI second eigenvector', 'unit vector (x, y, z)', '3-vector', None, 'Unit vector for λ2 direction.'),
    'dti_eigenvec_3': MapSpec('DTI third eigenvector', 'unit vector (x, y, z)', '3-vector', None, 'Unit vector for λ3 direction.'),
    'dti_eigenvec_1_cfa': MapSpec('DTI principal eigenvector (CFA-weighted)', 'RGB [0-1]', '3-vector', (0.0, 1.0), 'abs(eigenvec_1) * clip(FA,0,1).'),
    'dti_eigenvec_2_cfa': MapSpec('DTI second eigenvector (CFA-weighted)', 'RGB [0-1]', '3-vector', (0.0, 1.0), 'abs(eigenvec_2) * clip(FA,0,1).'),
    'dti_eigenvec_3_cfa': MapSpec('DTI third eigenvector (CFA-weighted)', 'RGB [0-1]', '3-vector', (0.0, 1.0), 'abs(eigenvec_3) * clip(FA,0,1).'),
}


DBSI_BASE_MAP_SPECS: Dict[str, MapSpec] = {
    'isotropic_adc': MapSpec('Isotropic apparent diffusion coefficient', _DIFF_UNITS, 'scalar', _DIFF_RANGE, 'Composite isotropic diffusivity estimate.'),
    'isotropic_fraction': MapSpec('Isotropic signal fraction', 'dimensionless', 'scalar', (0.0, 1.0), 'Sum of isotropic compartment fractions.'),
    'restricted_adc': MapSpec('Restricted isotropic diffusivity', _DIFF_UNITS, 'scalar', _DIFF_RANGE, 'Restricted isotropic component diffusivity.'),
    'restricted_fraction': MapSpec('Restricted isotropic fraction', 'dimensionless', 'scalar', (0.0, 1.0), 'Fraction assigned to restricted isotropic compartment.'),
    'highly_restricted_adc': MapSpec('Highly-restricted isotropic diffusivity', _DIFF_UNITS, 'scalar', _DIFF_RANGE, 'Highly-restricted isotropic component diffusivity (optional).'),
    'highly_restricted_fraction': MapSpec('Highly-restricted isotropic fraction', 'dimensionless', 'scalar', (0.0, 1.0), 'Fraction assigned to highly-restricted isotropic compartment.'),
    'hindered_adc': MapSpec('Hindered isotropic diffusivity', _DIFF_UNITS, 'scalar', _DIFF_RANGE, 'Hindered isotropic component diffusivity.'),
    'hindered_fraction': MapSpec('Hindered isotropic fraction', 'dimensionless', 'scalar', (0.0, 1.0), 'Fraction assigned to hindered isotropic compartment.'),
    'water_adc': MapSpec('Free-water isotropic diffusivity', _DIFF_UNITS, 'scalar', _DIFF_RANGE, 'Free-water (fast) isotropic diffusivity.'),
    'water_fraction': MapSpec('Free-water fraction', 'dimensionless', 'scalar', (0.0, 1.0), 'Fraction assigned to free-water isotropic compartment.'),
}


COMMON_MAP_SPECS: Dict[str, MapSpec] = {
    'b0_map': MapSpec('Estimated b=0 signal (S0) map', 'a.u.', 'scalar', None, 'Signal intensity baseline (arbitrary units).'),
    's0_map': MapSpec('Estimated b=0 signal (S0) map (learned)', 'a.u.', 'scalar', None, 'Voxelwise S0 learned during Step 2 when learnable_s0 is enabled.'),
}


# ---- NODDI canonical metadata (moved from src/maps/noddi_maps.py) ----

NODDI_SCALAR_MAP_SPECS: Dict[str, MapSpec] = {
    'noddi_ndi': MapSpec('Neurite Density Index (NDI)', 'dimensionless', 'scalar', (0.0, 1.0),
                        'Intra-cellular volume fraction (ICVF). Defined as (1-FISO)*v_ic.'),
    'noddi_odi': MapSpec('Orientation Dispersion Index (ODI)', 'dimensionless', 'scalar', (0.0, 1.0),
                        'Fiber orientation dispersion. Low in coherent bundles, high in crossing regions.'),
    'noddi_fiso': MapSpec('Free Water Fraction (FISO)', 'dimensionless', 'scalar', (0.0, 1.0),
                         'CSF/free water fraction. Elevated near ventricles and in edema.'),
    'noddi_fec': MapSpec('Extra-cellular Fraction (FEC)', 'dimensionless', 'scalar', (0.0, 1.0),
                        'Extra-axonal space fraction. Computed as (1-FISO)-NDI.'),
    'noddi_kappa': MapSpec('Watson Concentration Parameter', 'dimensionless', 'scalar', (0.01, 64.0),
                          'Concentration of Watson distribution. High = aligned, low = dispersed.'),
    'noddi_d_ic': MapSpec('Intra-cellular Diffusivity', _DIFF_UNITS, 'scalar', (0.5, 3.0),
                         'Parallel diffusivity in neurites. Typically fixed at 1.7 µm²/ms.'),
    'noddi_d_ec_par': MapSpec('Extra-cellular Parallel Diffusivity', _DIFF_UNITS, 'scalar', (0.5, 3.0),
                             'Hindered diffusion parallel to fibers.'),
    'noddi_d_ec_perp': MapSpec('Extra-cellular Perpendicular Diffusivity', _DIFF_UNITS, 'scalar', (0.1, 1.5),
                              'Hindered diffusion perpendicular to fibers. Tortuosity constrained.'),
}

NODDI_VECTOR_MAP_SPECS: Dict[str, MapSpec] = {
    'noddi_fiber_direction': MapSpec('Principal Neurite Orientation', 'unit vector (x, y, z)', '3-vector',
                                   notes='Mean fiber direction. Can be used for tractography.'),
    'noddi_fiber_direction_cfa': MapSpec('Color-coded Fiber Direction (weighted by NDI)', 'RGB [0-1]', '3-vector',
                                       notes='Red=|x|, Green=|y|, Blue=|z|, brightness=NDI (float RGB, not 8-bit).'),
}

# Typical value ranges for NODDI (QC heuristics)
NODDI_TYPICAL_RANGES = {
    'brain_wm': {
        'noddi_ndi': (0.4, 0.8),
        'noddi_odi': (0.05, 0.4),
        'noddi_fiso': (0.0, 0.15),
        'noddi_kappa': (0.7, 13.0),
    },
    'brain_gm': {
        'noddi_ndi': (0.2, 0.5),
        'noddi_odi': (0.3, 0.8),
        'noddi_fiso': (0.0, 0.2),
        'noddi_kappa': (0.3, 2.0),
    },
    'brain_csf': {
        'noddi_ndi': (0.0, 0.1),
        'noddi_odi': (0.5, 1.0),
        'noddi_fiso': (0.7, 1.0),
        'noddi_kappa': (0.01, 1.0),
    },
}


# ---- Fiber and compartmented fiber specs (pattern based) ----

_FIBER_PREFIXES = ('IA_', 'EA_', '')

_FIBER_SCALAR_METRICS = {
    'fraction': MapSpec('Fiber signal fraction', 'dimensionless', 'scalar', (0.0, 1.0), 'Fraction of signal attributed to this fiber component.'),
    'fa': MapSpec('Fiber fractional anisotropy (FA)', 'dimensionless', 'scalar', (0.0, 1.0), 'FA from fiber-specific diffusion tensor.'),
    'axial': MapSpec('Fiber axial diffusivity (AD)', _DIFF_UNITS, 'scalar', _DIFF_RANGE, 'AD = λ1 from fiber-specific tensor.'),
    'radial': MapSpec('Fiber radial diffusivity (RD)', _DIFF_UNITS, 'scalar', _DIFF_RANGE, 'RD = (λ2+λ3)/2 from fiber-specific tensor.'),
    'adc': MapSpec('Fiber mean diffusivity (MD/ADC)', _DIFF_UNITS, 'scalar', _DIFF_RANGE, 'MD = (λ1+λ2+λ3)/3 from fiber-specific tensor.'),
    'lambda_1': MapSpec('Fiber eigenvalue λ1', _DIFF_UNITS, 'scalar', _DIFF_RANGE, 'Principal eigenvalue of fiber-specific tensor.'),
    'lambda_2': MapSpec('Fiber eigenvalue λ2', _DIFF_UNITS, 'scalar', _DIFF_RANGE, 'Second eigenvalue of fiber-specific tensor.'),
    'lambda_3': MapSpec('Fiber eigenvalue λ3', _DIFF_UNITS, 'scalar', _DIFF_RANGE, 'Third eigenvalue of fiber-specific tensor.'),
}

_FIBER_VECTOR_METRICS = {
    'cfa': MapSpec('Fiber color FA (CFA)', 'RGB [0-1]', '3-vector', (0.0, 1.0), 'RGB orientation encoding modulated by fiber FA.'),
    'eigenvec_1': MapSpec('Fiber principal eigenvector', 'unit vector (x, y, z)', '3-vector', None, 'Unit vector for fiber principal direction.'),
    'eigenvec_2': MapSpec('Fiber second eigenvector', 'unit vector (x, y, z)', '3-vector', None, 'Unit vector for fiber second direction.'),
    'eigenvec_3': MapSpec('Fiber third eigenvector', 'unit vector (x, y, z)', '3-vector', None, 'Unit vector for fiber third direction.'),
    'eigenvec_1_cfa': MapSpec('Fiber principal eigenvector (CFA-weighted)', 'RGB [0-1]', '3-vector', (0.0, 1.0), 'abs(eigenvec_1) * clip(FA,0,1).'),
    'eigenvec_2_cfa': MapSpec('Fiber second eigenvector (CFA-weighted)', 'RGB [0-1]', '3-vector', (0.0, 1.0), 'abs(eigenvec_2) * clip(FA,0,1).'),
    'eigenvec_3_cfa': MapSpec('Fiber third eigenvector (CFA-weighted)', 'RGB [0-1]', '3-vector', (0.0, 1.0), 'abs(eigenvec_3) * clip(FA,0,1).'),
}


def _fiber_metric_spec(metric: str, compartment_prefix: str) -> Optional[MapSpec]:
    base = metric
    if base in _FIBER_SCALAR_METRICS:
        spec = _FIBER_SCALAR_METRICS[base]
    elif base in _FIBER_VECTOR_METRICS:
        spec = _FIBER_VECTOR_METRICS[base]
    else:
        return None

    if not compartment_prefix:
        return spec

    # Adjust the description for compartmented outputs (IA/EA).
    label = 'Intra-axonal' if compartment_prefix == 'IA_' else 'Extra-axonal'
    return MapSpec(
        description=f"{label} {spec.description}",
        units=spec.units,
        shape_kind=spec.shape_kind,
        valid_range=spec.valid_range,
        notes=spec.notes,
    )


def get_map_spec(map_name: str) -> Optional[Dict]:
    """Return a normalized metadata dict for a map name.

    Supports both internal keys (e.g., `fiber_0d_adc`) and legacy on-disk names
    (e.g., `fiber_adc`, `fiber_01_adc`, `fiber_IA_adc`).

    Returns None if unknown.
    """
    if not isinstance(map_name, str) or not map_name:
        return None

    # Exact matches
    if map_name in COMMON_MAP_SPECS:
        return COMMON_MAP_SPECS[map_name].as_dict()
    if map_name in DTI_MAP_SPECS:
        return DTI_MAP_SPECS[map_name].as_dict()
    if map_name in DBSI_BASE_MAP_SPECS:
        return DBSI_BASE_MAP_SPECS[map_name].as_dict()
    if map_name in NODDI_SCALAR_MAP_SPECS:
        return NODDI_SCALAR_MAP_SPECS[map_name].as_dict()
    if map_name in NODDI_VECTOR_MAP_SPECS:
        return NODDI_VECTOR_MAP_SPECS[map_name].as_dict()

    # Fiber internal: fiber_0d_adc, fiber_2d_IA_adc, ...
    m = re.match(r'^fiber_(\d+)d_(.+)$', map_name)
    if m is not None:
        suffix = m.group(2)
        for pfx in _FIBER_PREFIXES:
            if suffix.startswith(pfx):
                metric = suffix[len(pfx):]
                spec = _fiber_metric_spec(metric, pfx)
                return spec.as_dict() if spec is not None else None

    # Fiber legacy on disk: fiber_adc | fiber_IA_adc | fiber_01_adc | fiber_01_IA_adc
    m = re.match(r'^fiber_(\d{2})_(.+)$', map_name)
    if m is not None:
        suffix = m.group(2)
        for pfx in _FIBER_PREFIXES:
            if suffix.startswith(pfx):
                metric = suffix[len(pfx):]
                spec = _fiber_metric_spec(metric, pfx)
                return spec.as_dict() if spec is not None else None

    m = re.match(r'^fiber_(.+)$', map_name)
    if m is not None:
        suffix = m.group(1)
        for pfx in _FIBER_PREFIXES:
            if suffix.startswith(pfx):
                metric = suffix[len(pfx):]
                spec = _fiber_metric_spec(metric, pfx)
                return spec.as_dict() if spec is not None else None

    return None


def get_map_spec_safe(map_name: str) -> Dict:
    """Like get_map_spec(), but never returns None."""
    spec = get_map_spec(map_name)
    if spec is None:
        return {
            'description': None,
            'units': None,
            'shape_kind': None,
        }
    return spec
