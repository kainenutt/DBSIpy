Versioning and Release Process
==============================

This repo follows a lightweight, semver-style approach: ``MAJOR.MINOR.PATCH``.

When to bump
------------

- **PATCH**: bug fixes, small behavior additions, output/manifest fixes, CLI/doc improvements.
- **MINOR**: new features, new outputs/maps, new configs/flags, behavioral changes that remain backward compatible.
- **MAJOR**: breaking changes (renamed maps, changed default behavior, removed APIs, incompatible config changes).

What to update on every version change
--------------------------------------

1. Update the version in:

   - ``setup.py`` (``version=...``)
   - ``dbsipy/_version.py`` (``__version__ = ...``) so ``dbsipy.__version__`` (and manifests/logs) report the correct version

2. Add a new entry to ``development_notes/CHANGELOG.rst`` describing:

   - user-visible changes
   - any output map changes (new/renamed maps)
   - any config/CLI changes

3. Run tests (recommended default):

   - Windows: ``python -m pytest``

   Optional (opt-in physical accuracy checks):

   - ``python -m pytest -m accuracy``

Tagging
-------

For releases, create an annotated tag matching the version:

- ``git tag -a vX.Y.Z -m "DBSIpy X.Y.Z"``
- ``git push origin vX.Y.Z``

Suggested release checklist
---------------------------

- [ ] Tests pass locally (``python -m pytest``)
- [ ] Optional: accuracy tests pass (``python -m pytest -m accuracy``)
- [ ] Version bumped in code + ``setup.py``
- [ ] ``development_notes/CHANGELOG.rst`` updated
- [ ] Commit merged to the release branch
- [ ] Tag pushed (``vX.Y.Z``)
