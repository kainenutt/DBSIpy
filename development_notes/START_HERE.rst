Developer onboarding (START HERE)
=================================

This folder contains developer-facing notes that are useful before making changes.

- This folder **is tracked** in git.

Quickstart (Windows)
--------------------

1) Install package deps

- Install (editable): ``python -m pip install -U pip``
- Install project: ``python -m pip install -e .``

2) Run tests

- Run the test suite: ``python -m pytest``

If you see odd import issues on Windows, prefer running tests via ``python -m pytest`` (not bare ``pytest``) to ensure you're using the same interpreter that has DBSIpy installed.

3) Run the CLI

After install, the console script is ``DBSI``.

- Help: ``DBSI --help``
- Example run (using included test data):

  - Example configs live under ``dbsipy/configs/*.ini``

The run will write NIfTI outputs and a provenance manifest (``run_manifest.json``) into the chosen output folder.

Repo orientation
----------------

- ``dbsipy/``: canonical package root.
- ``dbsipy/core/fast_DBSI.py``: main orchestrator.
- ``dbsipy/configs/``: shipping configs and basis sets.
- ``dbsipy/tests/``: pytest suite.

Notes worth reading next
------------------------

- ``DBSIpy_abbreviations_definitions_units_style.md``: naming, units, abbreviations.
- Any recent reports in this folder (e.g., server QC writeups) if you’re touching related code.

Release hygiene
---------------

Every code change should include:

- Version bump (see ``setup.py`` and ``dbsipy/_version.py``), and
- A new entry in ``development_notes/CHANGELOG.rst``.
