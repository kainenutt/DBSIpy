from __future__ import annotations

from pathlib import Path
from typing import Optional


def get_configs_dir() -> Path:
    """Return the on-disk directory containing shipped config templates.

    Works for editable installs and installed wheels.
    """

    try:
        # Python 3.9+
        import importlib.resources as resources

        return Path(resources.files("dbsipy.configs"))
    except Exception:
        # Fallback: relative to this file
        return Path(__file__).resolve().parent


def _norm_path_str(p: str) -> str:
    return str(p).replace("\\", "/")


def resolve_config_path(raw: str) -> str:
    """Resolve a config path that may be given in legacy repo form.

    If `raw` exists on disk, it is returned unchanged.
    Otherwise, we try mapping legacy `DBSIpy_Configs/...` references into the
    packaged `dbsipy/configs/...` folder.
    """

    if not raw:
        return raw

    try:
        if Path(raw).exists():
            return raw
    except Exception:
        pass

    s = _norm_path_str(raw)

    # Legacy: DBSIpy_Configs/Configuration_*.ini
    if "DBSIpy_Configs/" in s:
        suffix = s.split("DBSIpy_Configs/", 1)[1]
        # New layout: configs are grouped under DBSI_Configs/
        candidate = get_configs_dir() / "DBSI_Configs" / suffix
        if candidate.exists():
            return str(candidate)

        # Older layout fallback: flat under configs/
        candidate = get_configs_dir() / suffix
        if candidate.exists():
            return str(candidate)

    # If user passed just a filename, try within shipped configs.
    name = Path(raw).name

    candidate = get_configs_dir() / "DBSI_Configs" / name
    if candidate.exists():
        return str(candidate)

    candidate = get_configs_dir() / name
    if candidate.exists():
        return str(candidate)

    return raw


def resolve_basis_path(raw: str, *, cfg_source: Optional[str] = None) -> str:
    """Resolve a basis CSV path for DBSI/IA.

    Handles:
    - absolute paths (returned if they exist)
    - paths relative to the config file location (`cfg_source`)
    - legacy paths containing a `BasisSets/...` suffix
    """

    if not raw:
        return raw

    # Existing absolute/relative path.
    try:
        if Path(raw).exists():
            return raw
    except Exception:
        pass

    # Relative-to-config resolution.
    if cfg_source:
        try:
            cfg_dir = Path(cfg_source).resolve().parent
            rel_candidate = (cfg_dir / raw).resolve()
            if rel_candidate.exists():
                return str(rel_candidate)
        except Exception:
            pass

    s = _norm_path_str(raw)

    # Re-root anything containing "BasisSets/" under the packaged configs.
    if "BasisSets/" in s:
        suffix = s.split("BasisSets/", 1)[1]
        # Current layout: configs/DBSI_Configs/BasisSets/...
        candidate = get_configs_dir() / "DBSI_Configs" / "BasisSets" / suffix
        if candidate.exists():
            return str(candidate)

        # Legacy layout: configs/BasisSets/...
        candidate = get_configs_dir() / "BasisSets" / suffix
        if candidate.exists():
            return str(candidate)

    return raw
