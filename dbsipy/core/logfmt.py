from __future__ import annotations

import logging
from typing import Optional


# Custom verbosity levels used to implement DBSIpy output modes.
# STATUS: minimal milestones (quiet mode)
# INFO: standard user-facing output
# DETAIL/VERBOSE: extra user-facing detail (also captured in the user log)
STATUS = 25
DETAIL = 15
VERBOSE = 12


_LEVELS_REGISTERED = False


def ensure_custom_levels_registered() -> None:
    global _LEVELS_REGISTERED
    if _LEVELS_REGISTERED:
        return

    try:
        logging.addLevelName(STATUS, "STATUS")
        logging.addLevelName(DETAIL, "DETAIL")
        logging.addLevelName(VERBOSE, "VERBOSE")

        def _status(self: logging.Logger, msg, *args, **kwargs):  # type: ignore[no-redef]
            if self.isEnabledFor(STATUS):
                self._log(STATUS, msg, args, **kwargs)

        def _detail(self: logging.Logger, msg, *args, **kwargs):  # type: ignore[no-redef]
            if self.isEnabledFor(DETAIL):
                self._log(DETAIL, msg, args, **kwargs)

        def _verbose(self: logging.Logger, msg, *args, **kwargs):  # type: ignore[no-redef]
            if self.isEnabledFor(VERBOSE):
                self._log(VERBOSE, msg, args, **kwargs)

        if not hasattr(logging.Logger, "status"):
            setattr(logging.Logger, "status", _status)
        if not hasattr(logging.Logger, "detail"):
            setattr(logging.Logger, "detail", _detail)
        if not hasattr(logging.Logger, "verbose"):
            setattr(logging.Logger, "verbose", _verbose)
    except Exception:
        # Logging still functions without custom helpers.
        pass

    _LEVELS_REGISTERED = True


def log_banner(
    title: str,
    *,
    logger: Optional[logging.Logger] = None,
    level: int = logging.INFO,
) -> None:
    """Log a formatted banner matching the historic fast_DBSI style."""
    ensure_custom_levels_registered()
    lg = logger or logging.getLogger()
    border = "# " + ("-" * 79) + " #"
    lg.log(level, border)
    lg.log(level, f"# {str(title).center(79)} #")
    lg.log(level, border)
