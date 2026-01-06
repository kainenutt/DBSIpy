"""Parameter map generation for IA (Intra-axonal/Extra-axonal) engine.

This module extends DBSI output maps to separate fiber compartments into intra-axonal
and extra-axonal components based on radial diffusivity thresholds. This enables
quantification of axonal vs. myelin contributions to the diffusion signal.

IA Segmentation Strategy
-------------------------
Each fiber is split into two sub-compartments:
- **Intra-axonal (IA)**: RD < threshold (restricted radial diffusion, axon interior)
- **Extra-axonal (EA)**: RD >= threshold (hindered radial diffusion, myelin sheath)

Default thresholds (from fast_DBSI.py:124-130):
- White matter: 0.4 um^2/ms
- Configurable per tissue type in INI file

Output Maps
-----------
All standard DBSI maps plus IA/EA variants:
    - Fiber_1_Fraction_IA / Fiber_1_Fraction_EA
    - Fiber_1_FA_IA / Fiber_1_FA_EA
    - Fiber_1_ADC_IA / Fiber_1_ADC_EA
    - Fiber_1_AD_IA / Fiber_1_AD_EA
    - Fiber_1_RD_IA / Fiber_1_RD_EA
    - Fiber_1_EigenVec_IA / Fiber_1_EigenVec_EA (3-vectors)
    - Fiber_1_RGB_IA / Fiber_1_RGB_EA (direction-encoded color)

Use Case
--------
Useful for distinguishing:
- Axonal loss (decreased IA fraction)
- Demyelination (increased EA fraction, decreased RD ratio)
- Inflammation (increased free water + hindered fractions)

Units Convention
----------------
Same as DBSI engine: all diffusivity outputs in **um^2/ms** (x1e3 from mm^2/s).

See Also
--------
dbsipy.dbsi : Standard DBSI engine without IA/EA separation
dbsipy.core.fast_DBSI : Contains DEFAULT_FIBER_CUTS thresholds
"""
