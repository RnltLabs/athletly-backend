"""Tests for src/services/prompt_metrics.py and /admin/prompt-metrics.

Covers:
- Per-detector pass/fail on known-good and known-bad strings.
- ``scan_response`` aggregates over all detectors.
- German transliteration detector only fires in German context.
- Singleton identity.
- Recording: response counts, by_rule, by_model, by_variant aggregates.
- ``violation_rate`` respects the time cutoff (patched time.time).
- WARN alert fires once when rate exceeds 5%, throttled inside 60s.
- /admin/prompt-metrics endpoint returns the documented JSON shape.
- A/B routing scaffold returns DEFAULT for any user.
- Performance smoke: 1000 scans of a 2KB response complete in < 1s.
"""

from __future__ import annotations

import logging
import time
from unittest.mock import patch

import pytest

from src.services.prompt_metrics import (
    PromptMetrics,
    PromptVariant,
    Violation,
    get_prompt_metrics,
    resolve_variant,
    scan_and_record,
    scan_response,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh() -> PromptMetrics:
    """A fresh store, independent of the singleton."""
    return PromptMetrics(max_records=200)


# ---------------------------------------------------------------------------
# Individual detectors
# ---------------------------------------------------------------------------


def test_em_dash_detected():
    text = "Heute war ein guter Lauf — wir machen morgen weiter."
    vios = scan_response(text)
    rule_ids = {v.rule_id for v in vios}
    assert "no_em_dash" in rule_ids


def test_en_dash_detected():
    text = "Mo – Mi macht 3 Tage."
    vios = scan_response(text)
    assert any(v.rule_id == "no_en_dash" for v in vios)


def test_plain_hyphen_not_a_dash_violation():
    text = "Mo - Mi macht 3 Tage."
    vios = scan_response(text)
    rule_ids = {v.rule_id for v in vios}
    assert "no_em_dash" not in rule_ids
    assert "no_en_dash" not in rule_ids


def test_markdown_header_detected():
    text = "# Heading\nplain text"
    vios = scan_response(text)
    assert any(v.rule_id == "no_markdown_header" for v in vios)


def test_markdown_bold_detected():
    text = "this is **bold** word"
    vios = scan_response(text)
    assert any(v.rule_id == "no_markdown_bold" for v in vios)


def test_markdown_italic_detected_as_warn():
    text = "this is *italic* word"
    vios = scan_response(text)
    italic = [v for v in vios if v.rule_id == "no_markdown_italic"]
    assert italic, "italic detector should fire"
    assert italic[0].severity == "warn"


def test_markdown_italic_not_confused_with_bold():
    # **bold** has two asterisks per side; the italic detector must NOT
    # also fire on it because that would double-count.
    text = "this is **bold** word"
    vios = scan_response(text)
    italic_hits = [v for v in vios if v.rule_id == "no_markdown_italic"]
    assert italic_hits == []


def test_clean_german_response_has_no_violations():
    text = (
        "Hey, dein Lauf gestern war sauber. Pace lag bei 4:42 pro Kilometer "
        "und die Herzfrequenz blieb stabil. Lass uns morgen ruhig laufen."
    )
    vios = scan_response(text, language_hint="de")
    assert vios == [], f"expected clean, got {vios}"


def test_clean_english_response_has_no_violations():
    text = (
        "Yesterday's run looked solid. Pace held at 4:42 per kilometer "
        "and heart rate stayed steady. Let's go easy tomorrow."
    )
    vios = scan_response(text, language_hint="en")
    assert vios == []


# ---------------------------------------------------------------------------
# German transliteration detector
# ---------------------------------------------------------------------------


def test_german_ascii_transliteration_with_hint():
    text = "Deine Praeferenzen liegen bei Tempo-Laeufen im Maerz."
    vios = scan_response(text, language_hint="de")
    assert any(v.rule_id == "german_ascii_transliteration" for v in vios)


def test_german_ascii_transliteration_auto_detected_without_hint():
    # Three or more German function words trigger the detector even
    # without a hint.
    text = (
        "Du hast doch noch nicht geschlafen und ich denke das ist ein "
        "ueberfaelliger Lauf fuer dich."
    )
    vios = scan_response(text)
    assert any(v.rule_id == "german_ascii_transliteration" for v in vios)


def test_english_text_does_not_fire_german_detector():
    # English prose with no German function words and no transliteration
    # candidates.
    text = "The user is configured. The feature works. Run it again."
    vios = scan_response(text)
    rule_ids = {v.rule_id for v in vios}
    assert "german_ascii_transliteration" not in rule_ids


# ---------------------------------------------------------------------------
# Decimal-pace leak detector (Sprint C)
# ---------------------------------------------------------------------------


def test_decimal_pace_leak_detects_basic_case():
    # Elena's actual failure mode: "deine Schwellen-Pace liegt bei 4.50/km".
    text = "Deine Schwellen-Pace liegt bei 4.50/km."
    vios = scan_response(text)
    assert any(v.rule_id == "decimal_pace_leak" for v in vios)


def test_decimal_pace_leak_detects_with_min_qualifier():
    text = "Threshold is 4.50 min/km."
    vios = scan_response(text)
    assert any(v.rule_id == "decimal_pace_leak" for v in vios)


def test_decimal_pace_leak_detects_no_space():
    text = "4.50min/km looks fast."
    vios = scan_response(text)
    assert any(v.rule_id == "decimal_pace_leak" for v in vios)


def test_decimal_pace_leak_detects_one_decimal_digit():
    # 4.5/km is also a decimal-pace leak (means 4 min 30 sec).
    text = "Run today was 4.5/km."
    vios = scan_response(text)
    assert any(v.rule_id == "decimal_pace_leak" for v in vios)


def test_decimal_pace_leak_does_not_fire_on_correct_mmss():
    # The correct format must NOT trigger the detector.
    text = "Deine Schwellen-Pace liegt bei 4:30/km."
    vios = scan_response(text)
    rule_ids = {v.rule_id for v in vios}
    assert "decimal_pace_leak" not in rule_ids


def test_decimal_pace_leak_does_not_fire_on_distance():
    # 4.5km is a distance, not a pace. No /km after a decimal pace
    # qualifier ("min/km", "/km"), so the detector must not fire.
    text = "I ran 4.5km this morning."
    vios = scan_response(text)
    rule_ids = {v.rule_id for v in vios}
    assert "decimal_pace_leak" not in rule_ids


def test_decimal_pace_leak_does_not_fire_on_unit_free_decimal():
    # "4.50" with no /km context cannot be confidently flagged.
    text = "The value was 4.50, which is high."
    vios = scan_response(text)
    rule_ids = {v.rule_id for v in vios}
    assert "decimal_pace_leak" not in rule_ids


def test_decimal_pace_leak_severity_is_strict():
    text = "4.50/km is wrong."
    vios = scan_response(text)
    leaks = [v for v in vios if v.rule_id == "decimal_pace_leak"]
    assert leaks
    assert leaks[0].severity == "strict"


def test_decimal_pace_leak_double_digit_minute():
    # Marathon-shuffle pace like 10.50/km should also trigger.
    text = "He averaged 10.50/km on the descent."
    vios = scan_response(text)
    assert any(v.rule_id == "decimal_pace_leak" for v in vios)


def test_decimal_pace_leak_case_insensitive():
    text = "Pace was 4.50 MIN/KM."
    vios = scan_response(text)
    assert any(v.rule_id == "decimal_pace_leak" for v in vios)


# ---------------------------------------------------------------------------
# Empty / boundary inputs
# ---------------------------------------------------------------------------


def test_empty_text_returns_no_violations():
    assert scan_response("") == []
    assert scan_response(None) == []  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# A/B scaffold
# ---------------------------------------------------------------------------


def test_resolve_variant_default_for_any_user():
    assert resolve_variant(None) is PromptVariant.DEFAULT
    assert resolve_variant("user-123") is PromptVariant.DEFAULT


def test_prompt_variant_string_value():
    # Endpoint serializes with .value, so this is load-bearing.
    assert PromptVariant.DEFAULT.value == "default"


# ---------------------------------------------------------------------------
# PromptMetrics store
# ---------------------------------------------------------------------------


def test_record_response_clean_increments_response_only():
    tel = _fresh()
    tel.record_response(model="gemini/gemini-2.5-flash", variant="default", text="hi", violations=[])
    s = tel.summary(window_seconds=60)
    assert s["total_responses_scanned"] == 1
    assert s["total_violations"] == 0
    assert s["violation_rate_pct"] == 0.0


def test_record_response_with_violations_aggregates_by_rule():
    tel = _fresh()
    vios = [
        Violation(rule_id="no_em_dash", severity="strict", snippet="...x—y...", char_offset=10),
        Violation(rule_id="no_em_dash", severity="strict", snippet="...a—b...", char_offset=22),
        Violation(rule_id="no_markdown_header", severity="strict", snippet="# foo", char_offset=0),
    ]
    tel.record_response(
        model="anthropic/claude-haiku-4-5",
        variant="default",
        text="some text",
        violations=vios,
    )
    s = tel.summary(window_seconds=60)
    assert s["total_responses_scanned"] == 1
    assert s["total_violations"] == 3
    assert s["by_rule"]["no_em_dash"]["count"] == 2
    assert s["by_rule"]["no_markdown_header"]["count"] == 1
    assert s["by_rule"]["no_em_dash"]["severity"] == "strict"


def test_summary_breakdowns_by_model_and_variant():
    tel = _fresh()
    vios = [Violation(rule_id="no_em_dash", severity="strict", snippet="", char_offset=0)]
    tel.record_response(model="m1", variant="default", text="a", violations=vios)
    tel.record_response(model="m1", variant="default", text="b", violations=[])
    tel.record_response(model="m2", variant="default", text="c", violations=[])

    s = tel.summary(window_seconds=60)
    assert s["by_model"]["m1"]["responses"] == 2
    assert s["by_model"]["m1"]["violations"] == 1
    assert s["by_model"]["m2"]["responses"] == 1
    assert s["by_model"]["m2"]["violations"] == 0
    assert s["by_variant"]["default"]["responses"] == 3
    assert s["by_variant"]["default"]["violations"] == 1


def test_violation_rate_respects_time_cutoff():
    tel = _fresh()

    # 2 responses at t=1000 (one with a violation), one at t=1200 (clean).
    with patch("src.services.prompt_metrics.time.time", return_value=1000.0):
        tel.record_response(
            model="m",
            variant="default",
            text="x",
            violations=[Violation(rule_id="no_em_dash", severity="strict", snippet="", char_offset=0)],
        )
        tel.record_response(model="m", variant="default", text="y", violations=[])

    with patch("src.services.prompt_metrics.time.time", return_value=1200.0):
        tel.record_response(model="m", variant="default", text="z", violations=[])

    # "Now" = 1210, window 60s sees only the last record (clean).
    with patch("src.services.prompt_metrics.time.time", return_value=1210.0):
        assert tel.violation_rate(60) == 0.0

    # Window 300s sees all 3 records and 1 violation.
    with patch("src.services.prompt_metrics.time.time", return_value=1210.0):
        assert tel.violation_rate(300) == pytest.approx(1 / 3)


def test_summary_empty_buffer_is_safe():
    tel = _fresh()
    s = tel.summary(window_seconds=60)
    assert s["total_responses_scanned"] == 0
    assert s["total_violations"] == 0
    assert s["violation_rate_pct"] == 0.0
    assert s["by_rule"] == {}
    assert s["by_model"] == {}
    assert s["by_variant"] == {}
    assert s["recent_samples"] == []


def test_summary_includes_recent_samples_capped_at_ten():
    tel = _fresh()
    vios = [
        Violation(rule_id="no_em_dash", severity="strict", snippet=f"snip-{i}", char_offset=i)
        for i in range(15)
    ]
    tel.record_response(model="m", variant="default", text="big text", violations=vios)
    s = tel.summary(window_seconds=60)
    assert len(s["recent_samples"]) == 10
    # Recent samples carry their snippet and ts.
    for sample in s["recent_samples"]:
        assert "snippet" in sample
        assert "ts" in sample


# ---------------------------------------------------------------------------
# Alerting
# ---------------------------------------------------------------------------


def test_alert_fires_when_rate_exceeds_threshold(caplog: pytest.LogCaptureFixture):
    tel = _fresh()
    # Record 11 responses, 7 of them with a violation.
    # Rate = 7/11 = 63%, well above the 5% threshold.
    # The MIN_SAMPLES gate is 10 so the alert fires on the 11th record.
    with caplog.at_level(logging.WARNING, logger="src.services.prompt_metrics"):
        with patch("src.services.prompt_metrics.time.time", return_value=2000.0):
            for i in range(11):
                vios = (
                    [Violation(rule_id="no_em_dash", severity="strict", snippet="", char_offset=0)]
                    if i < 7
                    else []
                )
                tel.record_response(model="m", variant="default", text="x", violations=vios)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, "expected at least one WARN log"
    assert "Prompt violation rate spike" in warnings[-1].message


def test_alert_throttled_within_60s(caplog: pytest.LogCaptureFixture):
    tel = _fresh()
    with caplog.at_level(logging.WARNING, logger="src.services.prompt_metrics"):
        # All inserts at the same timestamp; only one alert should fire.
        with patch("src.services.prompt_metrics.time.time", return_value=3000.0):
            for _ in range(30):
                tel.record_response(
                    model="m",
                    variant="default",
                    text="x",
                    violations=[
                        Violation(rule_id="no_em_dash", severity="strict", snippet="", char_offset=0)
                    ],
                )

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, f"expected exactly 1 throttled WARN, got {len(warnings)}"


def test_alert_does_not_fire_below_min_samples(caplog: pytest.LogCaptureFixture):
    tel = _fresh()
    with caplog.at_level(logging.WARNING, logger="src.services.prompt_metrics"):
        # 3 responses, all with violations -> rate=100% but sample too small.
        for _ in range(3):
            tel.record_response(
                model="m",
                variant="default",
                text="x",
                violations=[
                    Violation(rule_id="no_em_dash", severity="strict", snippet="", char_offset=0)
                ],
            )
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings == []


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


def test_get_prompt_metrics_returns_singleton():
    a = get_prompt_metrics()
    b = get_prompt_metrics()
    assert a is b


# ---------------------------------------------------------------------------
# scan_and_record convenience
# ---------------------------------------------------------------------------


def test_scan_and_record_returns_violations_and_records():
    # We can't reset the singleton mid-test without leaking state across
    # tests; instead, snapshot the response count before and after.
    before = get_prompt_metrics().summary(window_seconds=86400)["total_responses_scanned"]
    vios = scan_and_record(
        text="hello — world",
        model="gemini/gemini-2.5-flash",
        user_id=None,
    )
    after = get_prompt_metrics().summary(window_seconds=86400)["total_responses_scanned"]
    assert after == before + 1
    assert any(v.rule_id == "no_em_dash" for v in vios)


# ---------------------------------------------------------------------------
# /admin/prompt-metrics endpoint
# ---------------------------------------------------------------------------


def test_admin_prompt_metrics_endpoint_returns_documented_shape():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.api.routers.admin import router as admin_router

    app = FastAPI()
    app.include_router(admin_router, prefix="/admin")
    client = TestClient(app)

    resp = client.get("/admin/prompt-metrics?window_minutes=60")
    assert resp.status_code == 200, resp.text
    data = resp.json()

    # Documented top-level keys.
    expected_keys = {
        "window_minutes",
        "total_responses_scanned",
        "total_violations",
        "violation_rate_pct",
        "by_rule",
        "by_model",
        "by_variant",
        "recent_samples",
    }
    assert expected_keys.issubset(data.keys())

    # Types.
    assert isinstance(data["total_responses_scanned"], int)
    assert isinstance(data["total_violations"], int)
    assert isinstance(data["violation_rate_pct"], (int, float))
    assert isinstance(data["by_rule"], dict)
    assert isinstance(data["by_model"], dict)
    assert isinstance(data["by_variant"], dict)
    assert isinstance(data["recent_samples"], list)


def test_admin_prompt_metrics_endpoint_validates_window():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.api.routers.admin import router as admin_router

    app = FastAPI()
    app.include_router(admin_router, prefix="/admin")
    client = TestClient(app)

    # window_minutes=0 is below the ge=1 bound and must fail validation.
    resp = client.get("/admin/prompt-metrics?window_minutes=0")
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Performance smoke
# ---------------------------------------------------------------------------


def test_scan_response_is_fast_on_2kb_text():
    # 2KB of mixed-language prose, no violations.
    chunk = (
        "Yesterday I ran a steady 10k at 4:42 per kilometer, heart rate "
        "stayed in zone 2, and the weather was cool. "
    )
    text = chunk * 30  # ~2.1KB
    assert len(text.encode("utf-8")) > 1500

    start = time.perf_counter()
    for _ in range(1000):
        scan_response(text)
    elapsed = time.perf_counter() - start
    # Soft SLA: 1000 scans of a 2KB response in under 1 second.
    assert elapsed < 1.0, f"scan_response too slow: {elapsed:.3f}s for 1000 scans"
