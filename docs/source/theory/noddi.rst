NODDI
=====

Conceptual model
----------------

NODDI (Neurite Orientation Dispersion and Density Imaging) is a multi-compartment model
intended to separate:

- an **intra-cellular / neurite** component (NDI-related),
- an **extra-cellular** component, and
- an **isotropic (free-water)** component (FISO).

DBSIpy exposes common summary maps such as ``noddi_ndi``, ``noddi_odi``, and
``noddi_fiso``.

Fitting strategy
----------------

NODDI fitting is a nonlinear optimization problem. In DBSIpy, the optimizer is run for
a configured number of epochs and learning rate, producing a stable solution for the
chosen model settings.

Practical notes
---------------

- NODDI is sensitive to acquisition design; multi-shell data improves identifiability
  of compartments.
- When ``signal_normalization`` is ``auto``, NODDI uses attenuation-like normalization
  (``max``) by default.
