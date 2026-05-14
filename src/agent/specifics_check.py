"""Sprint P: deterministic specifics-grounding check for budget-skip path.

When the critic chain runs out of budget we have a tempting but unsafe
option: ship the regenerated rewrite anyway, marked as ``degraded``. In
practice this has shipped fabricated stats to users (Elena/Marco/Lisa
Iter 3, 2026-05-14 personas), e.g. "Dein heutiger Morgen-HRV war 40 ms"
with no ``get_health_metrics`` tool call in the turn.

This module provides ``specifics_grounded_in_tools`` which extracts
numeric "specifics" from a piece of coach text (paces, HRV values,
distances, durations, heart rates) and verifies each one appears, in
some equivalent form, somewhere in the concatenated tool-result payload
of the current turn.

Design goals:

1. **Deterministic.** No LLM. Sub-millisecond. Pure functions.
2. **German-friendly.** Decimal comma and decimal point are treated as
   interchangeable when looking up a number (a user might say "5,18"
   while a tool returns "5.18"). The check tries both forms.
3. **Conservative.** False negatives (calling a grounded specific
   ungrounded) cost a slightly more cautious fallback message; false
   positives (calling an ungrounded specific grounded) ship fabricated
   data. We tune toward false negatives.
4. **Composable.** Returns the list of ungrounded specifics so the
   caller can log them, expose them in telemetry, and reason about
   coverage.

Scope: this module is intentionally narrow. It does NOT validate that
the SHAPE of a specific matches its tool source (a pace of 5:18/km in
the rewrite that appears as "pace_sec_per_km: 318" in the tool result
is treated as ungrounded; we cannot normalise across all tool schemas
here without spreading the agent loop's domain knowledge into a regex).
The agent system prompt is responsible for telling the model to quote
tool output verbatim.
"""

from __future__ import annotations

import re
from typing import Iterable


# Each pattern captures the full token (with surrounding context as
# needed) and a normalised canonical form used for lookup. Order
# matters: paces and heart rates must be checked BEFORE generic
# durations so we do not misclassify "5:18/km" as the duration "5:18".

# Pace: "5:18/km" or "5,18/km" or "5.18/km" with optional whitespace.
# Captures the numeric portion so the lookup can try every decimal
# style.
_PACE_PATTERN = re.compile(
    r"\b(\d{1,2})[:.,](\d{2})\s*/\s*km\b",
    re.IGNORECASE,
)

# HRV value: "40 ms" or "40ms". Two or three digits, optional space.
_HRV_PATTERN = re.compile(
    r"\b(\d{2,3})\s*ms\b",
    re.IGNORECASE,
)

# Distance: "22 km", "10.43km", "10,43 km". One to three digits, optional
# decimal portion with comma or dot.
_DISTANCE_PATTERN = re.compile(
    r"\b(\d{1,3}(?:[.,]\d{1,2})?)\s*km\b",
    re.IGNORECASE,
)

# Heart rate: "HR 152", "152 bpm", "152 HR". Two or three digits.
_HEART_RATE_PATTERN_LEAD = re.compile(
    r"\bHR\s*(\d{2,3})\b",
    re.IGNORECASE,
)
_HEART_RATE_PATTERN_TRAIL = re.compile(
    r"\b(\d{2,3})\s*(?:bpm|HR)\b",
    re.IGNORECASE,
)

# Duration: "42:48" or "1:23:45". MM:SS or HH:MM:SS. Must NOT be
# followed by "/km" (that would be a pace) and must NOT be immediately
# preceded by "HR" (that would be a heart rate range). The negative
# lookahead for "/km" handles the pace overlap.
_DURATION_PATTERN = re.compile(
    r"\b(\d{1,2}:\d{2}(?::\d{2})?)\b(?!\s*/\s*km)",
    re.IGNORECASE,
)


def _normalise_haystack(tool_results: Iterable[dict | str]) -> str:
    """Flatten *tool_results* into one big lowercase string for lookup.

    Accepts either dicts (typical agent tool_result payloads) or raw
    strings (handy for unit tests). Non-stringifiable values are
    coerced via ``str()`` so we never raise on an exotic payload.
    """
    parts: list[str] = []
    for item in tool_results:
        if item is None:
            continue
        if isinstance(item, str):
            parts.append(item)
        else:
            try:
                parts.append(str(item))
            except Exception:
                continue
    return " ".join(parts).lower()


def _decimal_variants(token: str) -> tuple[str, ...]:
    """Return the comma- and dot-decimal variants of *token*.

    "5.18" -> ("5.18", "5,18")
    "5,18" -> ("5,18", "5.18")
    "10"   -> ("10",)
    """
    if "," in token:
        return (token, token.replace(",", "."))
    if "." in token:
        return (token, token.replace(".", ","))
    return (token,)


def _pace_grounded(whole: re.Match[str], haystack: str) -> bool:
    """Check if a pace match appears in *haystack* in any decimal form.

    The pace separator (":", ".", or ",") is interchangeable: the tool
    might emit "5:18/km" while the model wrote "5,18/km". All three
    separators are checked.
    """
    minutes, seconds = whole.group(1), whole.group(2)
    for sep in (":", ",", "."):
        needle = f"{minutes}{sep}{seconds}"
        # The "/km" suffix is optional in the haystack: a tool might
        # emit "pace: 5:18" with the unit elsewhere in the payload.
        if needle in haystack:
            return True
    return False


def _hrv_grounded(match: re.Match[str], haystack: str) -> bool:
    """HRV ms values: integer lookup is exact."""
    value = match.group(1)
    return value in haystack


def _distance_grounded(match: re.Match[str], haystack: str) -> bool:
    """Distance: try both comma and dot decimal forms."""
    token = match.group(1)
    for variant in _decimal_variants(token):
        if variant in haystack:
            return True
    return False


def _heart_rate_grounded(match: re.Match[str], haystack: str) -> bool:
    """Heart rate: simple integer lookup."""
    value = match.group(1)
    return value in haystack


def _duration_grounded(match: re.Match[str], haystack: str) -> bool:
    """Duration MM:SS or HH:MM:SS: lookup of the literal token.

    Tool payloads typically emit "42:48" or "moving_time: 2568" (in
    seconds). The seconds-form is NOT cross-checked here; we are
    intentionally conservative.
    """
    token = match.group(1)
    return token in haystack


def specifics_grounded_in_tools(
    text: str,
    tool_results: Iterable[dict | str],
) -> tuple[bool, list[str]]:
    """Return ``(grounded, ungrounded_specifics)``.

    *text* is the candidate coach reply (typically the regen). For each
    numeric specific the function recognises, it checks whether the
    specific (or an equivalent decimal-form) appears anywhere in the
    concatenated tool-result payload string.

    Returns:
        A tuple ``(grounded, ungrounded)`` where ``grounded`` is True
        only if every recognised specific has a match. ``ungrounded``
        is a list of the original tokens, in the order they were found,
        suitable for telemetry and logs.
    """
    if not text:
        return True, []

    haystack = _normalise_haystack(tool_results)
    ungrounded: list[str] = []
    matched_spans: list[tuple[int, int]] = []

    # Iterate patterns in priority order. Each match's character span
    # is recorded so a later, more generic pattern (e.g. duration) does
    # not re-claim characters already explained by an earlier match
    # (e.g. pace).
    checks: tuple[tuple[re.Pattern[str], callable], ...] = (
        (_PACE_PATTERN, _pace_grounded),
        (_HEART_RATE_PATTERN_LEAD, _heart_rate_grounded),
        (_HEART_RATE_PATTERN_TRAIL, _heart_rate_grounded),
        (_HRV_PATTERN, _hrv_grounded),
        (_DISTANCE_PATTERN, _distance_grounded),
        (_DURATION_PATTERN, _duration_grounded),
    )

    for pattern, grounded_fn in checks:
        for m in pattern.finditer(text):
            span = m.span()
            if _overlaps_any(span, matched_spans):
                continue
            matched_spans.append(span)
            if not grounded_fn(m, haystack):
                ungrounded.append(m.group(0).strip())

    return (len(ungrounded) == 0), ungrounded


def _overlaps_any(span: tuple[int, int], spans: list[tuple[int, int]]) -> bool:
    """Return True if *span* overlaps any (start, end) in *spans*."""
    start, end = span
    for s, e in spans:
        if start < e and s < end:
            return True
    return False
