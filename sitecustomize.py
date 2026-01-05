"""Local test-run hardening.

Pytest will auto-discover and auto-load any third-party plugins installed in the
active environment via setuptools entry points (group: ``pytest11``). That can
make this project's tests flaky or slow in developer environments that happen
to have unrelated plugins installed.

We defensively disable that behavior *only* when it looks like we're running
pytest, while still allowing explicit opt-in by setting
``PYTEST_DISABLE_PLUGIN_AUTOLOAD`` yourself.

This file is imported automatically by Python's ``site`` module when it is
present on ``sys.path`` (the repo root is on ``sys.path`` when running tests
from the workspace root).
"""

from __future__ import annotations

import os
import sys


def _looks_like_pytest_invocation(argv: list[str]) -> bool:
    joined = " ".join(argv).lower()
    return "pytest" in joined


if os.environ.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD") is None and _looks_like_pytest_invocation(sys.argv):
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
