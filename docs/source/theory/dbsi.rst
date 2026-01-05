DBSI
====

Conceptual model
----------------

Diffusion Basis Spectrum Imaging (DBSI) models a voxel’s diffusion signal as a mixture
of:

- **Anisotropic (fiber) components**, intended to capture coherently oriented tissue
  structures (e.g., axonal bundles), and
- **Isotropic components**, intended to capture environments like restricted isotropic
  diffusion, hindered diffusion, and free water.

In DBSIpy, outputs for the primary fiber compartment are exposed as ``fiber_0d_*`` maps
(e.g., ``fiber_0d_fraction``, ``fiber_0d_fa``, ``fiber_0d_axial``).

Basis sets
----------

DBSI uses basis sets (CSV tables) to represent the candidate diffusion behaviors.
These are shipped with the package and referenced from configs using paths like:

- ``BasisSets/<dataset>/angle_basis.csv``
- ``BasisSets/<dataset>/axial_basis.csv``

DBSIpy resolves these paths at runtime so configs remain portable.

Fitting strategy (two-stage)
----------------------------

DBSIpy’s DBSI fitting is organized into two conceptual stages:

1) **Stage 1 (orientation / grouping):**
   identify dominant fiber orientations and assign signal to candidate anisotropic
   components subject to an angular threshold.

2) **Stage 2 (parameter refinement):**
   refine diffusivity/fraction parameters for the fiber and isotropic compartments.

Both stages are iterative; convergence is controlled by epoch counts and (optionally)
early stopping (patience/min-delta).

Practical notes
---------------

- The optimization is generally **nonconvex**; solutions are expected to be stable and
  physically plausible rather than exact.
- Signal normalization (often ``max``) is used to make fitting behave like attenuation
  modeling when :math:`S_0` is not explicitly learned.
