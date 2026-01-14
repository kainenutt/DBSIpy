Changelog
=========

All notable changes to this project will be documented in this file.

1.0.0 (2026-01-05)
------------------

- First 1.0 release of DBSIpy.
- Documentation/text cleanup: normalized units and math notation to plain ASCII for consistent rendering across terminals/editors.

1.0.1 (2026-01-05)
------------------

- CLI: rename installed console script from ``DBSI`` to ``DBSIpy``.

1.0.2 (2026-01-05)
------------------

- Masking: when ``mask_file = auto``, save the generated mask next to the input DWI as ``*_auto_mask.nii.gz``.
- Configs: reorganize shipped DBSI configs under ``dbsipy/configs/DBSI_Configs/`` and update runtime path resolution for configs/basis sets.
- Testing: add opt-in ``accuracy`` tests using simulated phantoms (skipped by default; run with ``python -m pytest -m accuracy``).
- Docs: add Sphinx + Read the Docs scaffolding under ``docs/`` and document the opt-in accuracy test suite.
- Maintenance: remove obsolete deployment config initialization script.

1.1.0 (2026-01-06)
------------------

- Output modes: standardize console behavior on ``[DEBUG] output_mode = quiet|standard|verbose|debug`` and add ``--output_mode`` overrides to CLI/benchmark.
- Configs/UI: fully retire writing ``[DEBUG] verbose`` (templates, GUI flow, and saved ``config_final.ini`` snapshots no longer emit it; deprecated parsing fallback remains for older configs).
- Provenance: record ``output_mode`` explicitly in ``run_manifest.json``.

1.1.1 (2026-01-06)
------------------

- Outputs: downgrade non-finite map sanitization messages from WARNING to DEBUG (still errors when ``DBSIPY_STRICT=1``).

1.2.0 (2026-01-06)
------------------

- Benchmark: add NVML snapshots (total GPU memory/utilization + process list when supported).
- Benchmark: add coarse H2D/D2H transfer timing accumulation (Step 1/Step 2 uploads + map-save downloads).

1.2.1 (2026-01-13)
------------------

- Masking: treat an explicitly empty ``mask_file`` value as a minimal signal mask (removes only voxels with no signal).
- Masking: treat ``mask_file = n/a|na|none|n\a`` (case-insensitive) the same as empty (minimal signal mask).
- DBSI/IA Step 1: allow passing ``DBSI_CONFIG`` through to ``nnlsq`` (fixes ``TypeError: invalid optimizer argument``).

1.2.2 (2026-01-14)
------------------

- Output modes: in non-quiet modes, progress bars now respect TTY/env defaults again (prevents per-update newline spam in batch/redirected logs).
