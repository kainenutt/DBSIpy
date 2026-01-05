"""Parameter map generation for DBSI (Diffusion Basis Spectrum Imaging) engine.

This module converts raw DBSI optimization results into interpretable quantitative
parameter maps for diffusion MRI analysis. It handles both anisotropic (fiber) and
isotropic (restricted/hindered/free water) compartments.

Output Maps
-----------
Scalar Maps:
    - Fiber fractions (per fiber)
    - Fiber FA (fractional anisotropy, per fiber)
    - Fiber ADC (apparent diffusion coefficient, per fiber)
    - Fiber AD (axial diffusivity, per fiber)
    - Fiber RD (radial diffusivity, per fiber)
    - Restricted fraction (cellular compartment)
    - Hindered fraction (extracellular)
    - Free water fraction (CSF-like)
    - Total isotropic fraction (sum of above)

3-Vector Maps:
    - Fiber eigenvectors (fiber directions, per fiber)
    - Fiber RGB color maps (direction-encoded color)

Units Convention
----------------
All diffusivity outputs (AD, RD, ADC) are saved in **um^2/ms**:
- Internal computation: mm^2/s
- Output conversion: multiply by 1e3
- Typical brain WM: 0.6-0.9 um^2/ms

Isotropic Segmentation
----------------------
Spectrum partitioned by diffusivity thresholds (configurable in INI):
- Restricted: < 0.3 um^2/ms (cell bodies)
- Hindered: 0.3-3.0 um^2/ms (extracellular space)
- Free water: > 3.0 um^2/ms (CSF)

See Also
--------
src.dbsi_ia : IA engine with intra/extra-axonal separation
src.core.fast_DBSI : Main DBSI orchestrator
"""
