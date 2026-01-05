"""Backward-compatible NODDI map metadata.

This module historically held the canonical map metadata for NODDI. To reduce
drift with other engines, metadata now lives in `src/maps/map_metadata.py` and
this module re-exports the same public API.
"""

from dbsipy.maps.map_metadata import (
    NODDI_SCALAR_MAP_SPECS,
    NODDI_TYPICAL_RANGES,
    NODDI_VECTOR_MAP_SPECS,
)


SCALAR_MAPS = {k: v.as_dict() for k, v in NODDI_SCALAR_MAP_SPECS.items()}
VECTOR_MAPS = {k: v.as_dict() for k, v in NODDI_VECTOR_MAP_SPECS.items()}
TYPICAL_RANGES = dict(NODDI_TYPICAL_RANGES)


def get_map_info(map_name):
    """
    Get information about a NODDI parameter map.
    
    Parameters
    ----------
    map_name : str
        Name of the map (e.g., 'noddi_ndi')
    
    Returns
    -------
    info : dict
        Dictionary with description, units, range, notes
    """
    if map_name in SCALAR_MAPS:
        return SCALAR_MAPS[map_name]
    if map_name in VECTOR_MAPS:
        return VECTOR_MAPS[map_name]
    raise ValueError(f"Unknown NODDI map: {map_name}")
