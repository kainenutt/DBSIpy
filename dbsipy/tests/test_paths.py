from __future__ import annotations

from pathlib import Path

from dbsipy.configs.paths import get_configs_dir, resolve_basis_path, resolve_config_path


def test_get_configs_dir_exists() -> None:
    p = get_configs_dir()
    assert isinstance(p, Path)
    assert p.exists()


def test_resolve_config_path_legacy_prefix() -> None:
    # Legacy-style path references are mapped into the packaged configs dir.
    raw = "DBSIpy_Configs/Template_All_Engines_Minimal.ini"
    resolved = resolve_config_path(raw)
    assert Path(resolved).exists()
    assert Path(resolved).name == "Template_All_Engines_Minimal.ini"


def test_resolve_config_path_filename_only() -> None:
    resolved = resolve_config_path("Template_All_Engines_Minimal.ini")
    assert Path(resolved).exists()
    assert Path(resolved).name == "Template_All_Engines_Minimal.ini"


def test_resolve_basis_path_from_suffix() -> None:
    raw = "BasisSets/Human_Brain_InVivo_Modern/angle_basis.csv"
    resolved = resolve_basis_path(raw)
    assert Path(resolved).exists()
    assert resolved.replace("\\", "/").endswith("BasisSets/Human_Brain_InVivo_Modern/angle_basis.csv")
