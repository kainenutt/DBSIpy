"""Packaged configuration templates and basis sets.

This package ships:
- `*.ini` configuration templates
- `BasisSets/*/*.csv` basis tables

Use `get_configs_dir()` to locate these resources on disk.
"""

from __future__ import annotations

from pathlib import Path

from .paths import get_configs_dir, resolve_basis_path, resolve_config_path

__all__ = [
    "get_configs_dir",
    "resolve_basis_path",
    "resolve_config_path",
]
