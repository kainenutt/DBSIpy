"""Tkinter dialogs.

Do not import tkinter at module import time. Import it inside functions so that
non-GUI workflows (headless servers, CI) can import DBSIpy modules safely.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tkinter import Tk


def ask_dropdown_choice(
    root: "Tk",
    title: str,
    prompt: str,
    options: list[str],
    initial: str | None = None,
) -> str | None:
    """Minimal modal dropdown dialog returning selected string or None.

    Notes
    -----
    This function imports tkinter lazily so the rest of the package remains
    importable without Tk support.
    """

    try:
        from tkinter import Toplevel, StringVar, Label, Button, Frame, OptionMenu
    except Exception as e:
        raise RuntimeError(
            "Tkinter is required for GUI mode but could not be imported. "
            "On headless Linux, install a Python build with Tk support (e.g., python3-tk)."
        ) from e

    result: dict[str, str | None] = {"value": None}

    win = Toplevel(root)
    win.title(title)
    win.resizable(False, False)

    # Make sure the dialog is visible and raised; some window managers can
    # otherwise create it behind the file dialogs, which looks like a hang.
    try:
        win.deiconify()
        win.lift()
        win.attributes("-topmost", True)
    except Exception:
        pass

    var = StringVar(value=(initial if initial in options else (options[0] if options else "")))

    Label(win, text=prompt, justify="left", anchor="w").pack(padx=12, pady=(12, 6), fill="x")

    # Use a classic Tk dropdown (OptionMenu) for maximum compatibility.
    opt = OptionMenu(win, var, *(options or [""]))
    opt.config(width=20)
    opt.pack(padx=12, pady=(0, 12), fill="x")
    opt.focus_set()

    def _ok() -> None:
        result["value"] = var.get().strip() if var.get() is not None else None
        win.destroy()

    def _cancel() -> None:
        result["value"] = None
        win.destroy()

    btn_frame = Frame(win)
    btn_frame.pack(padx=12, pady=(0, 12), fill="x")
    Button(btn_frame, text="OK", command=_ok).pack(side="right")
    Button(btn_frame, text="Cancel", command=_cancel).pack(side="right", padx=(0, 8))

    win.protocol("WM_DELETE_WINDOW", _cancel)
    win.bind("<Return>", lambda _e: _ok())
    win.bind("<Escape>", lambda _e: _cancel())

    try:
        win.focus_force()
        win.update_idletasks()
        # Center on screen so it's not hidden off-screen.
        w = win.winfo_reqwidth()
        h = win.winfo_reqheight()
        sw = win.winfo_screenwidth()
        sh = win.winfo_screenheight()
        x = max(0, (sw - w) // 2)
        y = max(0, (sh - h) // 2)
        win.geometry(f"+{x}+{y}")
        win.update()
    except Exception:
        pass

    root.wait_window(win)

    # Ensure topmost is unset after close.
    try:
        win.attributes("-topmost", False)
    except Exception:
        pass

    return result["value"]
