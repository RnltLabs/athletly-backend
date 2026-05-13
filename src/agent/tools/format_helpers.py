"""Shared formatters for agent-facing tool output.

LLM-side (especially smaller models like Haiku) frequently misreads
decimal-minute values as mm:ss notation. To eliminate that ambiguity
the agent tools attach _pretty string variants next to the raw
numbers. This module is the single source of truth for those
formatters.

Two conventions:

- ``pace_to_mmss(decimal)``: decimal minutes per km / per 100m to
  "m:ss" notation. e.g. 4.41 -> "4:25".
- ``minutes_to_hms(decimal)``: decimal minutes to "h:mm:ss" or "m:ss"
  notation. e.g. 93.5 -> "1:33:30", 16.0 -> "16:00".

Both return ``None`` on missing input so callers can safely include
the result in dicts without conditional logic.
"""

from __future__ import annotations


def pace_to_mmss(pace_decimal: float | None) -> str | None:
    """Convert decimal minutes (e.g. 4.41 min/km) to "m:ss" notation.

    4.41 -> "4:25" (because 0.41 min = 24.6 sec, rounded to 25)
    """
    if pace_decimal is None:
        return None
    try:
        value = float(pace_decimal)
        whole = int(value)
        seconds = round((value - whole) * 60)
        if seconds == 60:
            whole += 1
            seconds = 0
        return f"{whole}:{seconds:02d}"
    except (TypeError, ValueError):
        return None


def minutes_to_hms(minutes_decimal: float | None) -> str | None:
    """Convert decimal minutes to "h:mm:ss" or "m:ss" notation.

    93.5  -> "1:33:30"
    16.0  -> "16:00"
    7.5   -> "7:30"
    """
    if minutes_decimal is None:
        return None
    try:
        total_seconds = round(float(minutes_decimal) * 60)
        h, rem = divmod(total_seconds, 3600)
        m, s = divmod(rem, 60)
        if h:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"
    except (TypeError, ValueError):
        return None


def minutes_to_hm(minutes_decimal: float | None) -> str | None:
    """Convert decimal minutes to "Xh Ymin" notation for sleep-style output.

    467.0 -> "7h 47min"
    113.0 -> "1h 53min"
     16.0 -> "16min"
    """
    if minutes_decimal is None:
        return None
    try:
        total_minutes = int(round(float(minutes_decimal)))
        h, m = divmod(total_minutes, 60)
        if h and m:
            return f"{h}h {m}min"
        if h:
            return f"{h}h"
        return f"{m}min"
    except (TypeError, ValueError):
        return None
