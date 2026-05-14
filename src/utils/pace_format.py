"""Pure conversion helpers for athlete-facing numbers.

The LLM mis-reads decimal-minute paces as mm:ss notation under load. The
robust fix is to pre-format every athlete-facing number BEFORE the LLM
sees it, so the only valid output is to quote the pretty string
verbatim.

These helpers are deliberately pure: no I/O, no logging, no agent or
framework imports. Every caller (system prompt, planner, memory, tool
result post-processors) can import them safely.

Conventions:

- ``decimal_min_to_mmss(value)``: decimal minutes -> "m:ss". e.g.
  ``4.50 -> "4:30"``, ``0.5 -> "0:30"``, ``None -> None``.
- ``decimal_min_to_hms(value)``: decimal minutes -> "h:mm:ss" when
  >= 60min, "m:ss" otherwise. e.g. ``75.5 -> "1:15:30"``,
  ``7.5 -> "7:30"``.
- ``watts_to_pretty(value)``: integer-like watts -> "<int>W".
- ``hr_to_pretty(value)``: integer-like bpm -> "<int> bpm".

All helpers return ``None`` on missing or non-numeric input so the
caller can chain them without conditional logic.
"""

from __future__ import annotations


def decimal_min_to_mmss(value: float | int | None) -> str | None:
    """Convert decimal-minutes pace to ``"m:ss"`` notation.

    ``4.50 -> "4:30"`` (because ``0.50 min = 30 sec``).
    ``0.5  -> "0:30"``.
    ``4.99 -> "5:00"`` (rounding bumps the minute).
    ``None -> None``. Non-numeric input also returns ``None``.

    Negative values are treated as invalid input and return ``None`` so
    callers can plumb the result through string formatting without
    leaking ``-1:-30``-style output.
    """
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v < 0:
        return None
    whole = int(v)
    seconds = round((v - whole) * 60)
    if seconds == 60:
        whole += 1
        seconds = 0
    return f"{whole}:{seconds:02d}"


def decimal_min_to_hms(value: float | int | None) -> str | None:
    """Convert decimal minutes to ``"h:mm:ss"`` (>= 1h) or ``"m:ss"``.

    ``75.5  -> "1:15:30"``
    ``7.5   -> "7:30"``
    ``59.99 -> "1:00:00"`` (rounding carries through the boundary).
    """
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v < 0:
        return None
    total_seconds = round(v * 60)
    h, rem = divmod(total_seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def watts_to_pretty(value: float | int | None) -> str | None:
    """Format an FTP / power value as ``"<int>W"``.

    Floats are rounded to the nearest int because watts are reported
    integer-valued in coaching contexts.
    """
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v < 0:
        return None
    return f"{int(round(v))}W"


def hr_to_pretty(value: float | int | None) -> str | None:
    """Format a heart rate value as ``"<int> bpm"``."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v < 0:
        return None
    return f"{int(round(v))} bpm"


__all__ = [
    "decimal_min_to_mmss",
    "decimal_min_to_hms",
    "watts_to_pretty",
    "hr_to_pretty",
]
