"""Backward-compatible NODDI map metadata.

This module historically duplicated definitions from `src/maps/noddi_maps.py`.
To prevent drift, it now re-exports the canonical definitions from there.
"""

from dbsipy.maps.noddi_maps import SCALAR_MAPS, VECTOR_MAPS, TYPICAL_RANGES, get_map_info


def validate_noddi_values(params, tissue_type='brain_wm', tolerance=0.2):
    """
    Check if NODDI parameters are within expected physiological ranges.
    
    Parameters
    ----------
    params : dict
        Dictionary with keys like 'noddi_ndi', 'noddi_odi', etc.
    tissue_type : str
        Expected tissue type: 'brain_wm', 'brain_gm', 'brain_csf'
    tolerance : float
        Fraction outside typical range to allow (default 0.2 = 20%)
    
    Returns
    -------
    valid : bool
        True if parameters are within typical ranges
    warnings : list
        List of warning messages for out-of-range values
    """
    if tissue_type not in TYPICAL_RANGES:
        raise ValueError(f"Unknown tissue type: {tissue_type}")
    
    typical = TYPICAL_RANGES[tissue_type]
    warnings = []
    
    for param_name, (min_val, max_val) in typical.items():
        if param_name not in params:
            continue
        
        value = params[param_name]
        
        # Allow some tolerance
        min_allowed = min_val * (1 - tolerance)
        max_allowed = max_val * (1 + tolerance)
        
        if value < min_allowed or value > max_allowed:
            warnings.append(
                f"{param_name}={value:.3f} outside typical {tissue_type} range "
                f"[{min_val:.3f}, {max_val:.3f}]"
            )
    
    return len(warnings) == 0, warnings


__all__ = [
    "SCALAR_MAPS",
    "VECTOR_MAPS",
    "TYPICAL_RANGES",
    "get_map_info",
    "validate_noddi_values",
]
