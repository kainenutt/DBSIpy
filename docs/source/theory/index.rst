Theory
======

This section summarizes the *modeling* and *fitting philosophy* used throughout DBSIpy.
It is intentionally high-level: enough to interpret outputs and configure runs, without
reproducing full derivations.

Modeling philosophy
-------------------

DBSIpy treats diffusion MRI as a **forward model** problem: given acquisition settings
(:math:`b`-values and :math:`b`-vectors), each engine defines a parametric signal model
:math:`S(\theta)` that predicts the voxelwise diffusion signal.

Two practical principles show up across engines:

1) **Conservative assumptions, explicit units**

   - Acquisition b-values are assumed in **s/mm²**.
   - Diffusivities are represented internally in **mm²/s**, but are typically reported
     in outputs as **µm²/ms** (same numerical scale, different unit label).
   - Naming conventions favor **ADC** over “MD” (see the project unit/style notes).

2) **Optimization as an engineering tool**

   - Many fits are **nonconvex**. DBSIpy uses iterative optimizers and practical
     stopping rules (epochs, patience/min-delta) to find stable solutions.
   - Determinism and reproducibility matter: tests use tiny phantoms and CPU execution
     to reduce stochasticity.

Signal normalization and :math:`S_0`
--------------------------------------------------

DBSIpy supports multiple signal normalization modes (``signal_normalization``):

- ``max``: per-voxel divide by max signal (commonly used for attenuation-like fitting)
- ``b0`` / ``minb``: normalize by mean of b=0 or minimum-b volumes
- ``none``: fit raw signal
- ``auto``: engine-dependent default

In particular, ``auto`` resolves to ``max`` for DBSI/DBSI-IA/NODDI and ``none`` for DTI
to match common usage.

Separately, DBSI/DBSI-IA can optionally treat :math:`S_0` as a learnable parameter
(``learnable_s0``); otherwise :math:`S_0` is implicitly absorbed by normalization.

Contents
--------

.. toctree::
   :maxdepth: 2

   dbsi
   dbsi_ia
   dti
   noddi
