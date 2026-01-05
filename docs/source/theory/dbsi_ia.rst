DBSI-IA
=======

Conceptual model
----------------

DBSI-IA extends DBSI by decomposing the anisotropic (fiber) compartment into:

- **IA (intra-axonal)** sub-compartment, and
- **EA (extra-axonal)** sub-compartment.

In outputs, these appear as ``fiber_0d_IA_*`` and ``fiber_0d_EA_*`` maps, alongside the
aggregate fiber maps (``fiber_0d_*``).

IA/EA separation heuristic
--------------------------

DBSIpy uses a practical separation rule tied to the DBSI basis definitions. In many
configurations, a radial diffusivity threshold (``intra_threshold``) is used to
distinguish IA-like vs EA-like contributions.

Fitting strategy
----------------

DBSI-IA follows the same high-level two-stage structure as DBSI (orientation/grouping
followed by refinement), but produces additional parameter maps for the IA/EA split.

Practical notes
---------------

- IA/EA fits are still **nonconvex** and are best interpreted as a stable decomposition
  under the model assumptions, not a guaranteed unique biological ground truth.
- When using signal normalization defaults, ``auto`` resolves to ``max`` for IA.
