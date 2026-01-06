from __future__ import annotations

from dbsipy.core.benchmark import diff_process_resource_snapshots, get_process_resource_snapshot
from dbsipy.core.benchmark import _get_installed_package_versions  # type: ignore


def test_process_resource_snapshot_and_diff_smoke() -> None:
    s0 = get_process_resource_snapshot()
    s1 = get_process_resource_snapshot()
    d = diff_process_resource_snapshots(s0, s1)

    assert isinstance(s0, dict)
    assert isinstance(s1, dict)
    assert isinstance(d, dict)

    # Required keys for reproducibility reporting.
    assert "pid" in s0
    assert "t_wall" in s0
    assert "t_process" in s0

    # Deltas should exist (may be None if psutil counters unavailable).
    assert "wall_s" in d
    assert "process_cpu_s" in d


def test_installed_package_versions_smoke() -> None:
    versions = _get_installed_package_versions()
    assert isinstance(versions, dict)
    # Curated key should always be present in output (value may be unavailable).
    assert "DBSIpy" in versions
