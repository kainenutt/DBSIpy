from __future__ import annotations

import logging
from typing import Optional


def log_banner(title: str, *, logger: Optional[logging.Logger] = None) -> None:
    """Log a formatted banner matching the historic fast_DBSI style."""
    lg = logger or logging.getLogger()
    border = "# " + ("-" * 79) + " #"
    lg.info(border)
    lg.info(f"# {str(title).center(79)} #")
    lg.info(border)
