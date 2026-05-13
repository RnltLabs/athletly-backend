"""Tests for src/services/cache_telemetry.py — in-memory LLM telemetry.

Verifies:
- record + summary roundtrip
- cache_hit_rate computation with mixed cached/uncached records
- itpm window respects the time cutoff (time.time is patched)
- summary returns the documented dict shape
"""

from unittest.mock import patch

import pytest

from src.services.cache_telemetry import (
    CacheTelemetry,
    CallRecord,
    get_telemetry,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh() -> CacheTelemetry:
    """Return a fresh telemetry instance (independent of the singleton)."""
    return CacheTelemetry(max_records=200)


# ---------------------------------------------------------------------------
# Record + summary roundtrip
# ---------------------------------------------------------------------------


def test_record_and_summary_roundtrip():
    tel = _fresh()
    tel.record_call(
        model="anthropic/claude-haiku-4-5",
        input_tokens=100,
        cache_creation_tokens=2000,
        cache_read_tokens=0,
        output_tokens=50,
    )
    tel.record_call(
        model="anthropic/claude-haiku-4-5",
        input_tokens=80,
        cache_creation_tokens=0,
        cache_read_tokens=2000,
        output_tokens=40,
    )

    summary = tel.summary()
    assert summary["window_calls"] == 2
    assert "anthropic/claude-haiku-4-5" in summary["by_model"]

    by_model = summary["by_model"]["anthropic/claude-haiku-4-5"]
    assert by_model["calls"] == 2
    assert by_model["input_tokens"] == 180
    assert by_model["cache_creation_tokens"] == 2000
    assert by_model["cache_read_tokens"] == 2000
    assert by_model["output_tokens"] == 90


def test_summary_dict_structure_is_valid():
    tel = _fresh()
    tel.record_call(
        model="gemini/gemini-2.5-flash",
        input_tokens=10,
        cache_creation_tokens=0,
        cache_read_tokens=0,
        output_tokens=5,
    )
    summary = tel.summary()

    # All documented keys must be present.
    assert set(summary.keys()) == {
        "window_calls",
        "cache_hit_rate_last_20",
        "itpm_last_60s",
        "itpm_last_300s_per_minute",
        "by_model",
    }
    assert isinstance(summary["window_calls"], int)
    assert isinstance(summary["cache_hit_rate_last_20"], float)
    assert isinstance(summary["itpm_last_60s"], int)
    assert isinstance(summary["itpm_last_300s_per_minute"], int)
    assert isinstance(summary["by_model"], dict)


def test_summary_empty_buffer_returns_zeros():
    tel = _fresh()
    summary = tel.summary()
    assert summary["window_calls"] == 0
    assert summary["cache_hit_rate_last_20"] == 0.0
    assert summary["itpm_last_60s"] == 0
    assert summary["itpm_last_300s_per_minute"] == 0
    assert summary["by_model"] == {}


# ---------------------------------------------------------------------------
# Cache hit rate
# ---------------------------------------------------------------------------


def test_cache_hit_rate_zero_with_no_records():
    tel = _fresh()
    assert tel.cache_hit_rate(20) == 0.0


def test_cache_hit_rate_mixed_cached_and_uncached():
    tel = _fresh()
    # Call 1: all uncached input (1000 in, 0 read).
    tel.record_call(
        model="anthropic/claude-haiku-4-5",
        input_tokens=1000,
        cache_creation_tokens=0,
        cache_read_tokens=0,
        output_tokens=100,
    )
    # Call 2: full cache hit (0 fresh in, 1000 read).
    tel.record_call(
        model="anthropic/claude-haiku-4-5",
        input_tokens=0,
        cache_creation_tokens=0,
        cache_read_tokens=1000,
        output_tokens=100,
    )
    # total_input = 1000 + 1000 = 2000, cached = 1000, rate = 0.5
    assert tel.cache_hit_rate(20) == pytest.approx(0.5)


def test_cache_hit_rate_full_hit():
    tel = _fresh()
    tel.record_call(
        model="anthropic/claude-haiku-4-5",
        input_tokens=0,
        cache_creation_tokens=0,
        cache_read_tokens=5000,
        output_tokens=200,
    )
    assert tel.cache_hit_rate(20) == pytest.approx(1.0)


def test_cache_hit_rate_window_limits_records():
    tel = _fresh()
    # 5 uncached + 1 fully cached, window=1 should see only the last.
    for _ in range(5):
        tel.record_call(
            model="m",
            input_tokens=100,
            cache_creation_tokens=0,
            cache_read_tokens=0,
            output_tokens=10,
        )
    tel.record_call(
        model="m",
        input_tokens=0,
        cache_creation_tokens=0,
        cache_read_tokens=100,
        output_tokens=10,
    )
    assert tel.cache_hit_rate(1) == pytest.approx(1.0)
    # Across the whole window 6 records: 500 fresh + 100 cached, total 600.
    assert tel.cache_hit_rate(20) == pytest.approx(100 / 600)


# ---------------------------------------------------------------------------
# ITPM window with patched time.time
# ---------------------------------------------------------------------------


def test_itpm_respects_time_cutoff():
    tel = _fresh()

    # First call at t=1000, second at t=1100 (100s later).
    with patch("src.services.cache_telemetry.time.time", return_value=1000.0):
        tel.record_call(
            model="m",
            input_tokens=500,
            cache_creation_tokens=0,
            cache_read_tokens=0,
            output_tokens=10,
        )
    with patch("src.services.cache_telemetry.time.time", return_value=1100.0):
        tel.record_call(
            model="m",
            input_tokens=200,
            cache_creation_tokens=300,
            cache_read_tokens=999,
            output_tokens=10,
        )

    # "Now" = 1110, 60s window covers only the second call (recorded at 1100).
    with patch("src.services.cache_telemetry.time.time", return_value=1110.0):
        assert tel.itpm(60) == 500  # 200 + 300, cache_read excluded

    # 300s window covers both records.
    with patch("src.services.cache_telemetry.time.time", return_value=1110.0):
        assert tel.itpm(300) == 1000  # 500 + 500


def test_itpm_excludes_cache_read_tokens():
    tel = _fresh()
    tel.record_call(
        model="m",
        input_tokens=0,
        cache_creation_tokens=0,
        cache_read_tokens=10_000,
        output_tokens=10,
    )
    # cache_read tokens do not count toward ITPM-relevant input.
    assert tel.itpm(60) == 0


# ---------------------------------------------------------------------------
# CallRecord properties
# ---------------------------------------------------------------------------


def test_call_record_total_input_and_itpm_relevant():
    rec = CallRecord(
        ts=0.0,
        model="m",
        input_tokens=10,
        cache_creation_tokens=20,
        cache_read_tokens=30,
        output_tokens=5,
    )
    assert rec.total_input == 60
    assert rec.itpm_relevant == 30  # cache_read excluded


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


def test_get_telemetry_returns_singleton():
    a = get_telemetry()
    b = get_telemetry()
    assert a is b
