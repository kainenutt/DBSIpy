from __future__ import annotations

import os
import sys
from typing import Any, Optional

from tqdm import tqdm


DEFAULT_BAR_FORMAT = "|{bar:100}|{percentage:3.0f}% ({n_fmt}/{total_fmt}) [{desc}: {elapsed} < {remaining}]"


def is_progress_enabled(explicit: Optional[bool] = None) -> bool:
    """Determine whether progress bars should be displayed.

    Priority:
    1) explicit argument (if provided)
    2) environment variable DBSIPY_PROGRESS (0/1)
    3) only enable when stderr is a TTY
    """
    if explicit is not None:
        return bool(explicit)

    env = os.environ.get("DBSIPY_PROGRESS", "").strip().lower()
    if env in {"1", "true", "yes", "on"}:
        return True
    if env in {"0", "false", "no", "off"}:
        return False

    try:
        return bool(sys.stderr.isatty())
    except Exception:
        return False


def make_progress_bar(
    *,
    total: int,
    desc: str,
    colour: Optional[str] = None,
    enabled: Optional[bool] = None,
    bar_format: str = DEFAULT_BAR_FORMAT,
    **kwargs: Any,
) -> tqdm:
    """Create a consistently-formatted tqdm progress bar."""
    disable = not is_progress_enabled(enabled)
    return tqdm(
        total=int(total),
        desc=str(desc),
        ascii=True,
        colour=colour,
        bar_format=bar_format,
        disable=disable,
        **kwargs,
    )
