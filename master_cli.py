"""Backwards-compatible CLI shim.

Installed DBSIpy uses the package-scoped entrypoint `dbsipy.master_cli:main`.
This file remains for running from a git checkout (or older workflows).
"""

from __future__ import annotations

from dbsipy.master_cli import main


if __name__ == "__main__":
    main()
