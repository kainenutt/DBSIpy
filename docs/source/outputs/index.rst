Outputs
=======

DBSIpy writes NIfTI maps to the chosen output directory.

Benchmark summary
-----------------

The ``DBSIpy benchmark`` command also writes a consolidated JSON file named like:

- ``*_DBSIpy_benchmark_summary.json``

This includes per-run timings, process-scoped resource deltas, CUDA peak memory (when applicable),
and the effective configuration used to run each engine.

Map naming
----------

DBSI/DBSI-IA default outputs include:

- ``isotropic_fraction`` and related isotropic compartment maps
- ``fiber_0d_*`` maps for the primary fiber compartment

DBSI-IA additionally reports IA/EA sub-compartment maps:

- ``fiber_0d_IA_*``
- ``fiber_0d_EA_*``

For the authoritative list of maps and units, see the map metadata in the codebase.
