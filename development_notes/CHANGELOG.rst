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
