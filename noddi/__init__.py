"""
NODDI (Neurite Orientation Dispersion and Density Imaging) Module

This module provides NODDI analysis as a standalone engine option in DBSIpy,
following an ADAM-based optimization pattern.

NODDI is a three-compartment biophysical model that estimates:
- **NDI** (Neurite Density Index): Intra-cellular volume fraction
- **ODI** (Orientation Dispersion Index): Fiber orientation dispersion  
- **FISO**: Free water/CSF volume fraction
- **Fiber orientation**: Principal neurite direction

The model combines:
1. Intra-cellular compartment: Sticks with Watson dispersion (restricted diffusion)
2. Extra-cellular compartment: Hindered diffusion with tortuosity constraint
3. Isotropic compartment: Free water (CSF)

References
----------
.. [1] Zhang, H., Schneider, T., Wheeler-Kingshott, C.A., Alexander, D.C., 2012.
       NODDI: practical in vivo neurite orientation dispersion and density imaging
       of the human brain. Neuroimage 61(4), 1000-1016.

Notes
-----
Units Convention
~~~~~~~~~~~~~~~~
- **Internal computation**: mm²/s (SI standard)
- **Output files**: µm²/ms (multiply by 1000)
- **Fractions (NDI, ODI, FISO)**: Dimensionless [0, 1]

Typical Brain White Matter Values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- NDI: 0.5-0.7 (neurite density)
- ODI: 0.1-0.3 (low dispersion in coherent fibers)
- FISO: 0.0-0.1 (minimal free water in healthy WM)

Data Requirements
~~~~~~~~~~~~~~~~~
- **Minimum**: Single-shell high b-value (b≥1000 s/mm²)
- **Recommended**: Multi-shell (2+ b-values) for better compartment separation
- **Typical**: b=1000 and b=2000 s/mm² with 30+ directions each

See Also
--------
src.misc.models.Linear_Models : NODDIModel implementation
src.noddi.watson_lut : Watson distribution lookup table
src.core.fast_DBSI : Main orchestrator with NODDI engine option
"""

# Placeholder for now - full implementation in Linear_Models.py
__version__ = '1.0.0'
__author__ = 'DBSIpy Development Team'
