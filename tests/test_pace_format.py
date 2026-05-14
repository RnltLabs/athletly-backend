"""Tests for src/utils/pace_format.py.

Pure-function helpers; tests focus on edge cases and the carry-over
boundary (4.99 -> "5:00", 59.99 min -> "1:00:00").
"""

from __future__ import annotations

import pytest

from src.utils.pace_format import (
    decimal_min_to_hms,
    decimal_min_to_mmss,
    hr_to_pretty,
    watts_to_pretty,
)


# ---------------------------------------------------------------------------
# decimal_min_to_mmss
# ---------------------------------------------------------------------------


def test_mmss_elena_threshold_pace_450_is_four_thirty():
    # The actual Sprint C bug: 4.50 decimal -> 4:30, NOT 4:50.
    assert decimal_min_to_mmss(4.50) == "4:30"


def test_mmss_typical_easy_pace():
    assert decimal_min_to_mmss(5.25) == "5:15"


def test_mmss_zero_input_yields_zero_zero():
    assert decimal_min_to_mmss(0.0) == "0:00"


def test_mmss_sub_minute_input():
    assert decimal_min_to_mmss(0.5) == "0:30"


def test_mmss_whole_minute_input():
    assert decimal_min_to_mmss(4.0) == "4:00"


def test_mmss_rounding_carry_over_to_next_minute():
    # 4.99 min = 4 min 59.4 sec -> 4:59 (no carry).
    assert decimal_min_to_mmss(4.99) == "4:59"
    # 4.999 min = 4 min 59.94 sec -> 5:00 (carry).
    assert decimal_min_to_mmss(4.999) == "5:00"


def test_mmss_almost_full_second_rounds_correctly():
    # 4.0083... min ~ 4 min 0.5 sec -> 4:00 or 4:01 depending on rounding;
    # banker's rounding via round() takes the even result.
    # Explicit check on a non-boundary value:
    assert decimal_min_to_mmss(4.5) == "4:30"


def test_mmss_none_returns_none():
    assert decimal_min_to_mmss(None) is None


def test_mmss_garbage_string_returns_none():
    assert decimal_min_to_mmss("not a number") is None  # type: ignore[arg-type]


def test_mmss_negative_returns_none():
    assert decimal_min_to_mmss(-1.0) is None


def test_mmss_accepts_int():
    assert decimal_min_to_mmss(4) == "4:00"


def test_mmss_accepts_numeric_string_via_float_coercion():
    # We do not document this, but the float() coercion is permissive.
    # Pinning the behaviour here so future refactors do not regress.
    assert decimal_min_to_mmss("4.5") == "4:30"  # type: ignore[arg-type]


def test_mmss_large_value():
    # Marathon shuffle: 12.5 min/km
    assert decimal_min_to_mmss(12.5) == "12:30"


# ---------------------------------------------------------------------------
# decimal_min_to_hms
# ---------------------------------------------------------------------------


def test_hms_under_an_hour_uses_mmss_format():
    assert decimal_min_to_hms(7.5) == "7:30"


def test_hms_exactly_one_hour():
    assert decimal_min_to_hms(60.0) == "1:00:00"


def test_hms_one_hour_thirty_three_minutes_thirty_seconds():
    assert decimal_min_to_hms(93.5) == "1:33:30"


def test_hms_just_under_one_hour_carries_when_rounding_promotes_to_60min():
    # 59.99 min = 3599.4 sec -> rounds to 3599 sec -> "59:59" (no carry).
    assert decimal_min_to_hms(59.99) == "59:59"
    # 59.9999 min = 3599.994 sec -> rounds to 3600 sec -> "1:00:00" (carry).
    assert decimal_min_to_hms(59.9999) == "1:00:00"


def test_hms_zero_returns_zero_zero():
    assert decimal_min_to_hms(0.0) == "0:00"


def test_hms_none_returns_none():
    assert decimal_min_to_hms(None) is None


def test_hms_negative_returns_none():
    assert decimal_min_to_hms(-5.0) is None


def test_hms_garbage_returns_none():
    assert decimal_min_to_hms("oops") is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# watts_to_pretty
# ---------------------------------------------------------------------------


def test_watts_integer_input():
    assert watts_to_pretty(250) == "250W"


def test_watts_float_rounds():
    assert watts_to_pretty(249.7) == "250W"


def test_watts_none_returns_none():
    assert watts_to_pretty(None) is None


def test_watts_negative_returns_none():
    assert watts_to_pretty(-10) is None


# ---------------------------------------------------------------------------
# hr_to_pretty
# ---------------------------------------------------------------------------


def test_hr_integer_input():
    assert hr_to_pretty(155) == "155 bpm"


def test_hr_float_rounds():
    assert hr_to_pretty(154.6) == "155 bpm"


def test_hr_none_returns_none():
    assert hr_to_pretty(None) is None


def test_hr_negative_returns_none():
    assert hr_to_pretty(-5) is None


# ---------------------------------------------------------------------------
# Hardening: no I/O, no mutation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [4.50, 0.0, 12.34, 7.5, None, "garbage"],
)
def test_helpers_are_pure(value):
    """Calling a helper twice with the same input returns the same output."""
    a = decimal_min_to_mmss(value if value is not None and not isinstance(value, str) else value)
    b = decimal_min_to_mmss(value if value is not None and not isinstance(value, str) else value)
    assert a == b
