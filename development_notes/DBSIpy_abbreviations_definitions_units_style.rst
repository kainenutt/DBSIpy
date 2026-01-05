Abbreviations, Definitions, Unit Conventions, and Style for DBSIpy
==================================================================

Date: 2025-12-31

This document defines the preferred terms, units, and naming conventions used throughout DBSIpy (code + outputs + notes).

.. _0-do-not-use--prefer:

0) Do Not Use / Prefer
----------------------

- **Do not use:** **MD**

  - **Prefer:** **ADC** (apparent diffusion coefficient)
  - Rationale: outputs are consistently labeled ``*_adc`` in maps. Keeping "ADC" avoids ambiguity about which "MD" is intended.

.. _1-core-abbreviations-and-definitions:

1) Core Abbreviations and Definitions
-------------------------------------

.. _11-diffusion-tensor-imaging-dti:

1.1 Diffusion Tensor Imaging (DTI)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **ADC**: apparent diffusion coefficient (a.k.a. tensor mean diffusivity). Output map: ``dti_adc``.
- **AD**: axial diffusivity (principal eigenvalue). Output map: ``dti_axial``.
- **RD**: radial diffusivity (mean of the 2 minor eigenvalues). Output map: ``dti_radial``.
- **FA**: fractional anisotropy. Output map: ``dti_fa``.
- **CFA**: color fractional anisotropy (direction-weighted color map). Output map: ``dti_cfa``.
- **lambda1, lambda2, lambda3**: diffusion tensor eigenvalues (ordered). Output maps: ``dti_lambda_1``, ``dti_lambda_2``, ``dti_lambda_3``.

  - Convention: **lambda1 == AD**.

- **evec1/2/3**: diffusion tensor eigenvectors. Output maps: ``dti_eigenvec_1/2/3``.

.. _12-dbsi--dbsi-ia:

1.2 DBSI / DBSI-IA
~~~~~~~~~~~~~~~~~~

General:

- **DBSI**: Diffusion Basis Spectrum Imaging.
- **DBSI-IA**: DBSI with explicit intra-axonal (**IA**) and extra-axonal (**EA**) compartment separation.

Signal baseline:

- **S0**: estimated :math:`b=0` signal baseline (arbitrary units).

  - Map keys: ``b0_map`` (measured :math:`b=0` volume, when produced) and ``s0_map`` (learned voxelwise S0 when ``learnable_s0=True``).

Isotropic components:

- **isotropic_adc**: diffusivity of the isotropic compartment mixture (scalar). Output map: ``isotropic_adc``.
- **isotropic_fraction**: fraction of signal attributed to isotropic compartments. Output map: ``isotropic_fraction``.

Fiber (0-direction) components (naming reflects the current single-fiber index used in outputs):

- **fiber_0d_fraction / fa / axial / radial / adc / cfa**: standard DTI-like metrics for the fiber compartment.

  - Output maps: ``fiber_0d_fraction``, ``fiber_0d_fa``, ``fiber_0d_axial``, ``fiber_0d_radial``, ``fiber_0d_adc``, ``fiber_0d_cfa``.

- **fiber_0d_lambda_{1,2,3}** and eigenvectors: expanded outputs.

DBSI-IA additional compartment outputs:

- **IA**: intra-axonal compartment (maps prefixed ``fiber_0d_IA_*``).
- **EA**: extra-axonal compartment (maps prefixed ``fiber_0d_EA_*``).

Multi-segment isotropic models:

- **restricted / hindered / water** are isotropic sub-compartments used in the 3-segment / 4-segment outputs:

  - Example maps: ``restricted_adc``, ``restricted_fraction``, ``hindered_adc``, ``hindered_fraction``, ``water_adc``, ``water_fraction``.
  - 4-segment adds ``highly_restricted_adc/fraction``.

.. _14-noddi:

1.3 NODDI
~~~~~~~~~

- **NDI**: neurite density index (intra-cellular volume fraction). Output map: ``noddi_ndi``.
- **ODI**: orientation dispersion index. Output map: ``noddi_odi``.
- **FISO**: free-water / isotropic volume fraction. Output map: ``noddi_fiso``.
- **FEC**: extra-cellular fraction. Output map: ``noddi_fec``.
- **kappa**: Watson concentration parameter. Output map: ``noddi_kappa``.
- **d_ic**: intra-cellular diffusivity (often fixed). Output map: ``noddi_d_ic``.
- **d_ec_par / d_ec_perp**: extra-cellular parallel / perpendicular diffusivity. Output maps: ``noddi_d_ec_par``, ``noddi_d_ec_perp``.
- **fiber_direction**: principal neurite orientation. Output maps: ``noddi_fiber_direction``, ``noddi_fiber_direction_cfa``.

.. _2-unit-conventions:

2) Unit Conventions
-------------------

.. _21-acquisition--input:

2.1 Acquisition / input
~~~~~~~~~~~~~~~~~~~~~~~

- **b-values**: stored/assumed in **s/mm^2**.
- **b-vectors**: unit vectors in scanner coordinates (shape ``(n_volumes, 3)``), normalized in code.

.. _22-internal-model-units:

2.2 Internal model units
~~~~~~~~~~~~~~~~~~~~~~~~

- Diffusivities are commonly represented internally as **mm^2/s** (e.g., fixed values like :math:`1.7\times10^{-3}` mm^2/s).

.. _23-output-map-units:

2.3 Output map units
~~~~~~~~~~~~~~~~~~~~

- **Diffusivities in outputs (ADC/AD/RD/lambda_i, etc.)**: reported in **um^2/ms**.

  - Conversion: :math:`1\ \mathrm{mm}^2/\mathrm{s} = 10^3\ \mathrm{\mu m}^2/\mathrm{ms}`.

- **Fractions** (``*_fraction``, NDI/ODI/FISO/FEC, AWF): **dimensionless**, typically in :math:`[0,1]`.
- **Directions** (``*_orientation``, ``*_eigenvec_*``, ``*_fiber_direction``): **unit vectors** (3 components).
- **CFA maps** (``*_cfa``, ``*_fiber_direction_cfa``): **3-vectors** used as color encodings; treat as display vectors (not physical units).

.. _3-naming-conventions-code--outputs:

3) Naming Conventions (Code + Outputs)
--------------------------------------

.. _31-output-map-prefixes:

3.1 Output map prefixes
~~~~~~~~~~~~~~~~~~~~~~~

Use these prefixes consistently:

- ``dti_*`` for DTI outputs
- ``noddi_*`` for NODDI outputs
- ``fiber_0d_*``, ``fiber_0d_IA_*``, ``fiber_0d_EA_*`` for DBSI/DBSI-IA fiber compartment outputs
- ``isotropic_*``, ``restricted_*``, ``hindered_*``, ``water_*``, ``highly_restricted_*`` for isotropic compartment outputs
- ``b0_map`` / ``s0_map`` for b=0 baseline signal outputs

.. _32-prefer-these-short-names-in-notes:

3.2 Prefer these short names in notes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **ADC** (not MD)
- **AD**, **RD**, **FA**
- **NDI**, **ODI**, **FISO**, **FEC**, **kappa**

.. _33-eigenvalue-naming:

3.3 Eigenvalue naming
~~~~~~~~~~~~~~~~~~~~~

- Use **lambda1/lambda2/lambda3** in prose.
- In filenames and code variables, prefer ``lambda_1``, ``lambda_2``, ``lambda_3`` (matches ``dbsipy/maps/dti_maps.py``).

.. _4-style-guidelines-documentation--code:

4) Style Guidelines (Documentation + Code)
------------------------------------------

.. _41-documentation-style:

4.1 Documentation style
~~~~~~~~~~~~~~~~~~~~~~~

- Always include **units** when stating typical values or thresholds.
- Prefer **map names** that match filenames and code keys.
- When comparing methods/runs, prefer "old/new" tables with:

  - ROI name
  - voxel count ``n``
  - median (optionally add IQR later)

.. _42-logging-style:

4.2 Logging style
~~~~~~~~~~~~~~~~~

- Use ``logging.info()`` for high-level steps and timings.
- Use ``logging.warning()`` for recoverable fallbacks (e.g., initialization fallback), and include the exception message.
- When CUDA is available, log the selected device and GPU model (already done in runtime environment header).

.. _43-parameter-naming-style-in-code:

4.3 Parameter naming style in code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Prefer ``adc`` over ``md``.
- Use ``*_par`` / ``*_perp`` for parallel/perpendicular (e.g., ``d_ec_par``, ``d_ec_perp``).
- Use explicit compartment tags when ambiguous: ``v_ic``, ``v_iso``, ``v_ec``.
- Use ``*_clamped`` suffix only for clamped/validated tensors.

.. _5-quick-reference-canonical-map-lists:

5) Quick Reference: Canonical Map Lists
---------------------------------------

These lists are the "source of truth" for names:

- DTI: ``dbsipy/maps/dti_maps.py``
- DBSI: ``dbsipy/maps/dbsi_maps.py``
- DBSI-IA: ``dbsipy/maps/dbsi_ia_maps.py``
- NODDI: ``dbsipy/maps/noddi_maps.py``

Project structure note:

- The canonical import root is ``dbsipy/`` (the legacy ``src/`` tree is removed).
