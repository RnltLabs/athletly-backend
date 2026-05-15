"""Tests for the deterministic response-gate layer (Sprint D).

Covers:
- Each individual gate triggers on positive examples (German + English).
- Each individual gate passes on negative examples (no false positives).
- ``run_gates`` aggregates correctly (multiple fails -> combined action).
- Per-gate flags toggle individual gates.
- Master switch turns the whole layer off.
- Exception in a single gate fails open (other gates still run).
- Metrics record every outcome.
- /admin/gates-stats endpoint returns the documented shape.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.agent.response_gates import (
    GATES,
    GateBatchResult,
    GateContext,
    GateResult,
    TOOL_REQUIREMENTS,
    _compose_tool_force_instruction,
    _gate_holistic_alert,
    _gate_injury_persistence,
    _gate_language_mirror,
    _gate_stats_grounding,
    _gate_temporal_freshness,
    _looks_german,
    gate_by_id,
    is_gates_layer_enabled,
    needs_tool_forcing,
    record_regenerate_outcome,
    record_tool_forced_outcome,
    run_gates,
)
from src.services.gates_metrics import (
    GATE_IDS,
    GatesMetrics,
    get_gates_metrics,
    reset_gates_metrics,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_state():
    """Reset the metrics singleton between tests."""
    reset_gates_metrics()
    yield
    reset_gates_metrics()


def _ctx(
    user_message: str = "",
    response_text: str = "",
    tools_this_turn: tuple[str, ...] = (),
    tools_recent: tuple[str, ...] = (),
    alerts: tuple[str, ...] = (),
) -> GateContext:
    return GateContext(
        user_message=user_message,
        response_text=response_text,
        tools_called_this_turn=tools_this_turn,
        tools_called_recent=tools_recent,
        runtime_context_alerts=alerts,
    )


# ---------------------------------------------------------------------------
# Gate registry sanity
# ---------------------------------------------------------------------------


def test_five_gates_registered():
    assert len(GATES) == 5


def test_gate_ids_match_metrics_module():
    assert tuple(g.gate_id for g in GATES) == GATE_IDS


# ---------------------------------------------------------------------------
# Gate 1: temporal_freshness
# ---------------------------------------------------------------------------


def test_temporal_freshness_german_triggers_when_sync_missing():
    ctx = _ctx(user_message="ich war heute laufen", response_text="cool")
    r = _gate_temporal_freshness(ctx)
    assert not r.passed
    assert r.gate_id == "temporal_freshness"
    assert "sync_garmin_data" in (r.required_action or "")


def test_temporal_freshness_english_triggers_when_sync_missing():
    ctx = _ctx(user_message="I just finished my run!", response_text="nice")
    r = _gate_temporal_freshness(ctx)
    assert not r.passed


def test_temporal_freshness_passes_when_sync_and_activities_called():
    ctx = _ctx(
        user_message="ich war heute laufen",
        response_text="cool",
        tools_this_turn=("sync_garmin_data", "get_activities", "get_provider_status"),
    )
    r = _gate_temporal_freshness(ctx)
    assert r.passed


def test_temporal_freshness_passes_on_plain_question():
    ctx = _ctx(user_message="wie soll ich naechste woche trainieren?")
    r = _gate_temporal_freshness(ctx)
    assert r.passed


def test_temporal_freshness_passes_when_sport_noun_alone():
    ctx = _ctx(user_message="ich plane einen long run am sonntag")
    # Has "long run" but no temporal marker like eben/heute/gerade.
    r = _gate_temporal_freshness(ctx)
    assert r.passed


def test_temporal_freshness_passes_when_temporal_alone():
    ctx = _ctx(user_message="heute ist mein freier tag")
    # No sport noun.
    r = _gate_temporal_freshness(ctx)
    assert r.passed


def test_temporal_freshness_fails_with_only_sync_no_activities():
    ctx = _ctx(
        user_message="ich war gerade laufen",
        tools_this_turn=("sync_garmin_data",),
    )
    r = _gate_temporal_freshness(ctx)
    assert not r.passed


# ---------------------------------------------------------------------------
# Gate 2: injury_persistence
# ---------------------------------------------------------------------------


def test_injury_persistence_knee_triggers():
    ctx = _ctx(user_message="mein knie zwickt nach dem long run")
    r = _gate_injury_persistence(ctx)
    assert not r.passed
    assert "annotate_activity" in (r.required_action or "")
    assert "append_to_journal" in (r.required_action or "")


def test_injury_persistence_english_pain_triggers():
    ctx = _ctx(user_message="my achilles is sore today")
    r = _gate_injury_persistence(ctx)
    assert not r.passed


def test_injury_persistence_passes_when_annotated():
    ctx = _ctx(
        user_message="knie zwickt nach dem lauf",
        tools_this_turn=("annotate_activity", "get_activities"),
    )
    r = _gate_injury_persistence(ctx)
    assert r.passed


def test_injury_persistence_passes_when_journal_appended():
    ctx = _ctx(
        user_message="wadenheber tut weh",
        tools_this_turn=("append_to_journal",),
    )
    r = _gate_injury_persistence(ctx)
    assert r.passed


def test_injury_persistence_passes_when_journal_section_updated():
    ctx = _ctx(
        user_message="achilles schmerzen",
        tools_this_turn=("update_journal_section",),
    )
    r = _gate_injury_persistence(ctx)
    assert r.passed


def test_injury_persistence_passes_on_clean_message():
    ctx = _ctx(user_message="alles super, war ein guter lauf")
    r = _gate_injury_persistence(ctx)
    assert r.passed


def test_injury_persistence_wadenheber_triggers():
    """Persona bug: 'Wadenheber Zwicken' must trigger."""
    ctx = _ctx(user_message="der wadenheber zwickt nach dem long run")
    r = _gate_injury_persistence(ctx)
    assert not r.passed


# ---------------------------------------------------------------------------
# Gate 3: stats_grounding
# ---------------------------------------------------------------------------


def test_stats_grounding_pace_triggers():
    ctx = _ctx(response_text="Du bist heute 4:30/km gelaufen.")
    r = _gate_stats_grounding(ctx)
    assert not r.passed


def test_stats_grounding_hr_triggers():
    ctx = _ctx(response_text="Deine HR 145 ist im aeroben Bereich.")
    r = _gate_stats_grounding(ctx)
    assert not r.passed


def test_stats_grounding_hrv_triggers():
    ctx = _ctx(response_text="HRV 52ms zeigt gute Erholung.")
    r = _gate_stats_grounding(ctx)
    assert not r.passed


def test_stats_grounding_distance_triggers():
    ctx = _ctx(response_text="Dein 10km lauf war gut.")
    r = _gate_stats_grounding(ctx)
    assert not r.passed


def test_stats_grounding_duration_triggers():
    ctx = _ctx(response_text="Du hast 1h 45min gemacht.")
    r = _gate_stats_grounding(ctx)
    assert not r.passed


def test_stats_grounding_recovery_triggers():
    ctx = _ctx(response_text="Recovery score 78 sieht gut aus.")
    r = _gate_stats_grounding(ctx)
    assert not r.passed


def test_stats_grounding_passes_with_recent_read():
    ctx = _ctx(
        response_text="Du bist 4:30/km gelaufen.",
        tools_recent=("get_activities",),
    )
    r = _gate_stats_grounding(ctx)
    assert r.passed


def test_stats_grounding_passes_with_activity_details():
    ctx = _ctx(
        response_text="Dein 10km in 45min war gut.",
        tools_recent=("get_activity_details",),
    )
    r = _gate_stats_grounding(ctx)
    assert r.passed


def test_stats_grounding_passes_on_plain_text():
    ctx = _ctx(response_text="Wie geht es dir heute?")
    r = _gate_stats_grounding(ctx)
    assert r.passed


def test_stats_grounding_does_not_trigger_on_dates():
    ctx = _ctx(response_text="Wir sehen uns am 9. November.")
    r = _gate_stats_grounding(ctx)
    assert r.passed


# ---------------------------------------------------------------------------
# Gate 3 Sprint K: German comma decimal coverage
# ---------------------------------------------------------------------------
#
# These tests cover the Elena Iter 2 Turn 4 failure mode: a German-formatted
# coach response uses comma decimals ("HRV 56,5 ms") and the gate must
# still trigger. Each test exists in both German-comma and English-period
# forms so the detector cannot regress by being made too strict on one
# notation. See AUDIT.md for the full failure inventory.


def test_stats_grounding_german_hrv_comma_decimal_triggers():
    # The exact bug case from Elena Iter 2 Turn 4.
    ctx = _ctx(response_text="HRV 56,5 ms, RHR 50 bpm, Body Battery 74")
    r = _gate_stats_grounding(ctx)
    assert not r.passed


def test_stats_grounding_german_hrv_alone_triggers():
    ctx = _ctx(response_text="HRV 56,5 ms")
    r = _gate_stats_grounding(ctx)
    assert not r.passed


def test_stats_grounding_english_hrv_period_decimal_triggers():
    ctx = _ctx(response_text="HRV 56.5 ms")
    r = _gate_stats_grounding(ctx)
    assert not r.passed


def test_stats_grounding_german_vo2max_comma_decimal_triggers():
    ctx = _ctx(response_text="VO2max 56,5 ist solide.")
    r = _gate_stats_grounding(ctx)
    assert not r.passed


def test_stats_grounding_english_vo2max_period_decimal_triggers():
    ctx = _ctx(response_text="VO2max 56.5 looks solid.")
    r = _gate_stats_grounding(ctx)
    assert not r.passed


def test_stats_grounding_german_ftp_comma_decimal_triggers():
    ctx = _ctx(response_text="FTP 250,5 Watt fuer heute.")
    r = _gate_stats_grounding(ctx)
    assert not r.passed


def test_stats_grounding_german_recovery_comma_decimal_triggers():
    ctx = _ctx(response_text="Recovery score 78,5 sieht gut aus.")
    r = _gate_stats_grounding(ctx)
    assert not r.passed


def test_stats_grounding_german_pace_comma_as_colon_triggers():
    # German user might type comma where the canonical pace format uses
    # a colon: "4,30/km". This is a stat reference and must trigger.
    ctx = _ctx(response_text="Schwellenpace 4,30/km")
    r = _gate_stats_grounding(ctx)
    assert not r.passed


def test_stats_grounding_canonical_pace_colon_still_triggers():
    # Regression guard for the canonical mm:ss with colon.
    ctx = _ctx(response_text="Schwellenpace 4:30/km")
    r = _gate_stats_grounding(ctx)
    assert not r.passed


def test_stats_grounding_german_pace_decimal_min_per_km_triggers():
    # "4,5 min/km" is a German decimal pace; treat as a stat reference.
    # See AUDIT.md Phase 3 for the rationale (canonical form is mm:ss
    # and either notation is a stat that must be grounded).
    ctx = _ctx(response_text="Pace heute 4,5 min/km war ok.")
    r = _gate_stats_grounding(ctx)
    assert not r.passed


def test_stats_grounding_body_battery_integer_still_triggers():
    # Sanity: integer-only Body Battery still trips the detector. The
    # original Elena response combined this with the comma-decimal HRV,
    # so we keep an explicit integer-only assertion to prevent regressions.
    ctx = _ctx(response_text="Body Battery 74 ist top.")
    r = _gate_stats_grounding(ctx)
    assert not r.passed


def test_stats_grounding_distance_period_decimal_still_triggers():
    ctx = _ctx(response_text="Du bist heute 21.1 km gelaufen.")
    r = _gate_stats_grounding(ctx)
    assert not r.passed


def test_stats_grounding_distance_comma_decimal_still_triggers():
    ctx = _ctx(response_text="Du bist heute 21,1 km gelaufen.")
    r = _gate_stats_grounding(ctx)
    assert not r.passed


def test_stats_grounding_passes_with_recent_read_for_german_decimal():
    # The bug case: grounded by a recent tool call, so the gate must pass.
    ctx = _ctx(
        response_text="HRV 56,5 ms, Body Battery 74",
        tools_recent=("get_health_summary",),
    )
    r = _gate_stats_grounding(ctx)
    assert r.passed


def test_parse_number_accepts_both_separators():
    from src.agent.response_gates import _parse_number
    assert _parse_number("56,5") == 56.5
    assert _parse_number("56.5") == 56.5
    assert _parse_number("250") == 250.0


def test_numeric_re_matches_full_german_decimal_token():
    # Sanity: the helper captures the FULL number, not just the integer
    # part. This is the semantic correctness check that distinguishes
    # the fix from the prior accident (where ``\b`` happened to land on
    # the comma and the gate matched only the integer prefix).
    from src.agent.response_gates import _NUMERIC_RE
    m = _NUMERIC_RE.search("HRV 56,5 ms")
    assert m is not None
    assert m.group(0) == "56,5"


# ---------------------------------------------------------------------------
# Gate 4: holistic_alert
# ---------------------------------------------------------------------------


def test_holistic_alert_passes_when_no_alerts():
    ctx = _ctx(response_text="Alles gut bei mir!", alerts=())
    r = _gate_holistic_alert(ctx)
    assert r.passed


def test_holistic_alert_triggers_when_alert_active_and_not_mentioned():
    ctx = _ctx(
        response_text="Lauf locker eine Runde.",
        alerts=("critical:sleep_critical",),
    )
    r = _gate_holistic_alert(ctx)
    assert not r.passed


def test_holistic_alert_passes_when_recovery_mentioned():
    ctx = _ctx(
        response_text="Dein Schlaf war kurz, ruh dich aus.",
        alerts=("critical:sleep_critical",),
    )
    r = _gate_holistic_alert(ctx)
    assert r.passed


def test_holistic_alert_passes_when_hrv_mentioned():
    ctx = _ctx(
        response_text="HRV ist gefallen, leichte Einheit heute.",
        alerts=("warn:hrv_drop",),
    )
    r = _gate_holistic_alert(ctx)
    assert r.passed


def test_holistic_alert_passes_info_severity_only():
    ctx = _ctx(
        response_text="Cooler Lauf!",
        alerts=("info:body_battery_chronic",),
    )
    r = _gate_holistic_alert(ctx)
    assert r.passed


def test_holistic_alert_fails_on_empty_response_with_active_alert():
    ctx = _ctx(response_text="", alerts=("critical:sleep_critical",))
    r = _gate_holistic_alert(ctx)
    assert not r.passed


def test_holistic_alert_bare_pattern_id_treated_as_actionable():
    ctx = _ctx(response_text="Lauf 10km", alerts=("sleep_low_3d",))
    r = _gate_holistic_alert(ctx)
    assert not r.passed


# ---------------------------------------------------------------------------
# Gate 5: language_mirror
# ---------------------------------------------------------------------------


def test_language_mirror_passes_german_to_german():
    ctx = _ctx(
        user_message="wie soll ich das training planen mit der wade?",
        response_text="Ich wuerde leicht starten und schauen wie die wade reagiert.",
    )
    r = _gate_language_mirror(ctx)
    assert r.passed


def test_language_mirror_passes_english_to_english():
    ctx = _ctx(
        user_message="how should I train this week with my knee?",
        response_text="I would suggest taking it easy and seeing how it feels tomorrow.",
    )
    r = _gate_language_mirror(ctx)
    assert r.passed


def test_language_mirror_fails_german_to_english():
    ctx = _ctx(
        user_message="ich war heute laufen und das knie zwickt etwas",
        response_text="Take it easy this week and rest the knee.",
    )
    r = _gate_language_mirror(ctx)
    assert not r.passed


def test_language_mirror_passes_when_user_message_empty():
    ctx = _ctx(user_message="", response_text="anything")
    r = _gate_language_mirror(ctx)
    assert r.passed


def test_looks_german_helper():
    assert _looks_german("ich und du und der weg ist das ziel")
    assert not _looks_german("hello there how are you doing today")


# ---------------------------------------------------------------------------
# Aggregation: run_gates
# ---------------------------------------------------------------------------


def test_run_gates_all_pass_returns_true():
    ctx = _ctx(
        user_message="wie soll ich montag trainieren?",
        response_text="Locker 45min, drauf achten wie sich der koerper anfuehlt.",
    )
    batch = run_gates(ctx)
    assert batch.passed
    assert batch.failures == ()
    assert batch.combined_action is None


def test_run_gates_aggregates_multiple_failures():
    ctx = _ctx(
        user_message="ich war heute laufen und das knie zwickt",
        response_text="Take it easy this week. HR 145 was fine.",
        tools_this_turn=(),
        tools_recent=(),
    )
    batch = run_gates(ctx)
    assert not batch.passed
    # Expect: temporal_freshness, injury_persistence, stats_grounding,
    # language_mirror.
    fail_ids = batch.failure_ids()
    assert "temporal_freshness" in fail_ids
    assert "injury_persistence" in fail_ids
    assert "stats_grounding" in fail_ids
    assert "language_mirror" in fail_ids
    assert batch.combined_action
    # Combined action concatenates each failure's required action.
    assert "sync_garmin_data" in batch.combined_action
    assert "annotate_activity" in batch.combined_action


def test_run_gates_only_runs_enabled_gates():
    ctx = _ctx(
        user_message="ich war heute laufen",
        response_text="cool",
    )
    with patch("src.agent.response_gates.get_settings") as mock_get_settings:
        mock_settings = mock_get_settings.return_value
        mock_settings.response_gate_temporal_freshness = False
        mock_settings.response_gate_injury_persistence = True
        mock_settings.response_gate_stats_grounding = True
        mock_settings.response_gate_holistic_alert = True
        mock_settings.response_gate_language_mirror = True
        batch = run_gates(ctx)
    # Temporal would have failed but is disabled. Other gates pass on
    # this clean input. Net: passed.
    assert "temporal_freshness" not in batch.failure_ids()


def test_run_gates_fails_open_on_gate_exception():
    """If one gate raises, others still run and the layer survives."""
    from src.agent.response_gates import Gate

    def boom(_ctx: GateContext) -> GateResult:
        raise RuntimeError("boom from test")

    bad_gate = Gate(
        gate_id="temporal_freshness",
        setting_attr="response_gate_temporal_freshness",
        check=boom,
    )
    # Swap the entire GATES tuple with one good replacement plus the bad one.
    new_gates = (bad_gate,) + GATES[1:]
    with patch("src.agent.response_gates.GATES", new_gates):
        ctx = _ctx(
            user_message="hallo",
            response_text="hallo, alles gut?",
        )
        batch = run_gates(ctx)
    # Even with one gate broken, run_gates returns a result and the
    # other four gates ran.
    assert isinstance(batch, GateBatchResult)
    metrics = get_gates_metrics()
    summary = metrics.summary()
    assert summary["per_gate"]["temporal_freshness"]["gate_error"] >= 1


def test_is_gates_layer_enabled_default_true():
    # Default settings have response_gates_enabled=True.
    assert is_gates_layer_enabled() is True


def test_is_gates_layer_enabled_off_when_flag_false():
    with patch("src.agent.response_gates.get_settings") as mock_get_settings:
        mock_get_settings.return_value.response_gates_enabled = False
        assert is_gates_layer_enabled() is False


# ---------------------------------------------------------------------------
# Metrics integration
# ---------------------------------------------------------------------------


def test_metrics_record_pass_and_fail():
    metrics = get_gates_metrics()
    metrics.reset()

    ctx = _ctx(user_message="ich war heute laufen")
    run_gates(ctx)  # at least temporal_freshness fails
    summary = metrics.summary()
    assert summary["per_gate"]["temporal_freshness"]["fail"] >= 1


def test_metrics_drops_unknown_gate_id():
    metrics = GatesMetrics()
    metrics.record("not_a_real_gate", "pass")
    summary = metrics.summary()
    assert summary["window_records"] == 0


def test_metrics_drops_unknown_outcome():
    metrics = GatesMetrics()
    metrics.record("temporal_freshness", "weird_outcome")
    summary = metrics.summary()
    assert summary["window_records"] == 0


def test_metrics_summary_shape():
    metrics = GatesMetrics()
    metrics.record("temporal_freshness", "fail")
    metrics.record("temporal_freshness", "regenerate_success")
    summary = metrics.summary()
    assert "per_gate" in summary
    assert "rates" in summary
    assert "totals" in summary
    assert summary["totals"]["fails"] == 1
    assert summary["totals"]["regenerate_success"] == 1


def test_record_regenerate_outcome_writes_metrics():
    metrics = get_gates_metrics()
    metrics.reset()
    record_regenerate_outcome(("temporal_freshness", "injury_persistence"), success=True)
    summary = metrics.summary()
    assert summary["per_gate"]["temporal_freshness"]["regenerate_success"] == 1
    assert summary["per_gate"]["injury_persistence"]["regenerate_success"] == 1


def test_record_regenerate_outcome_failure_writes_metrics():
    metrics = get_gates_metrics()
    metrics.reset()
    record_regenerate_outcome(("stats_grounding",), success=False)
    summary = metrics.summary()
    assert summary["per_gate"]["stats_grounding"]["regenerate_failed"] == 1


# ---------------------------------------------------------------------------
# Admin endpoint shape
# ---------------------------------------------------------------------------


def test_gates_metrics_summary_includes_all_gate_ids():
    metrics = GatesMetrics()
    summary = metrics.summary()
    for gid in GATE_IDS:
        assert gid in summary["per_gate"]
        assert gid in summary["rates"]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_response_passes_stats_grounding():
    ctx = _ctx(response_text="")
    r = _gate_stats_grounding(ctx)
    assert r.passed


def test_empty_user_message_passes_temporal_freshness():
    ctx = _ctx(user_message="")
    r = _gate_temporal_freshness(ctx)
    assert r.passed


def test_empty_user_message_passes_injury_persistence():
    ctx = _ctx(user_message="")
    r = _gate_injury_persistence(ctx)
    assert r.passed


def test_persona_marco_turn_3_blocked():
    """Marco persona: turn 3 'ich war eben laufen, war ganz okay' must block."""
    ctx = _ctx(
        user_message="ich war eben laufen, war ganz okay",
        response_text="Schoen! Das hoert sich gut an.",
        tools_this_turn=(),
        tools_recent=(),
    )
    batch = run_gates(ctx)
    assert not batch.passed
    assert "temporal_freshness" in batch.failure_ids()


def test_persona_lisa_knee_pain_blocked():
    """Lisa persona: 'knee zwickt nach dem long run' must block."""
    ctx = _ctx(
        user_message="das knie zwickt nach dem long run vom sonntag",
        response_text="Hmm, lass uns das im Auge behalten.",
        tools_this_turn=("get_activities",),
    )
    batch = run_gates(ctx)
    assert not batch.passed
    assert "injury_persistence" in batch.failure_ids()


def test_persona_war_heute_laufen_blocked():
    """Bug report case: 'war heute laufen' must trigger proactive sync."""
    ctx = _ctx(
        user_message="war heute laufen",
        response_text="cool, wie war es?",
    )
    batch = run_gates(ctx)
    assert not batch.passed
    assert "temporal_freshness" in batch.failure_ids()


# ---------------------------------------------------------------------------
# Sprint J: holistic_alert end-to-end with Marco's persona seed
# ---------------------------------------------------------------------------


def _marco_alert_ids() -> tuple[str, ...]:
    """Drive Marco's persona through the recovery_alerts detector
    and encode the result the way agent_loop._detect_active_alert_ids_safe
    does. Returns a non-empty tuple by construction (Sprint J fix).
    """
    from src.services.recovery_alerts import detect_alerts_from_metrics
    from tools.persona_test._persona import load_persona
    from tools.persona_test.fake_metrics import generate

    persona = load_persona("marco")
    rows = generate(persona, user_id="marco-gates-test", days=30)
    metrics = [
        {
            "date": r.date,
            "sleep_minutes": r.sleep_duration_minutes,
            "hrv": r.hrv_avg,
            "resting_hr": r.resting_heart_rate,
            "stress": r.stress_avg,
            "body_battery_high": r.body_battery_high,
            "recovery_score": r.recovery_score,
        }
        for r in rows
    ]
    alerts = detect_alerts_from_metrics(metrics)
    return tuple(f"{a.severity}:{a.pattern}" for a in alerts)


def test_marco_persona_produces_alert_ids_for_gate():
    """Marco's seed must hand at least one actionable id to the gate."""
    ids = _marco_alert_ids()
    actionable = [i for i in ids if not i.startswith("info:")]
    assert actionable, (
        f"Sprint J regression: Marco's persona must produce actionable "
        f"alert ids, got {ids}"
    )


def test_holistic_alert_fires_for_marco_when_response_ignores_recovery():
    """End-to-end Sprint J check: given Marco's alert payload, an ignorant
    response must fail the holistic_alert gate.
    """
    ids = _marco_alert_ids()
    ctx = _ctx(
        user_message="kurz frage: mach ich heute den 10k oder den 5k?",
        response_text=(
            "Mach den 10k, dein Plan sieht das so vor."
        ),
        alerts=ids,
    )
    result = _gate_holistic_alert(ctx)
    assert not result.passed, (
        f"Gate should fail when Marco has alerts {ids} but the response "
        f"ignores recovery."
    )
    assert "Schlaf" in (result.required_action or "") or "Recovery" in (
        result.required_action or ""
    )


def test_holistic_alert_passes_for_marco_when_response_addresses_sleep():
    """Acknowledging the recovery domain in German must pass the gate."""
    ids = _marco_alert_ids()
    ctx = _ctx(
        user_message="kurz frage: mach ich heute den 10k oder den 5k?",
        response_text=(
            "Dein Schlaf ist seit Tagen knapp und HRV liegt unter dem "
            "Baseline. Heute lieber den 5k locker."
        ),
        alerts=ids,
    )
    result = _gate_holistic_alert(ctx)
    assert result.passed


def test_run_gates_records_holistic_alert_fail_for_marco():
    """Verify the metrics layer records a fail when Marco's alerts go
    unacknowledged. This is what the /admin/gates-stats endpoint reports.
    """
    ids = _marco_alert_ids()
    ctx = _ctx(
        user_message="hi",
        response_text="alles top, mach einfach weiter so.",
        alerts=ids,
        # Ground the stats gate so only holistic_alert fails.
        tools_recent=("get_activities",),
    )
    batch = run_gates(ctx)
    assert not batch.passed
    assert "holistic_alert" in batch.failure_ids()

    metrics = get_gates_metrics()
    summary = metrics.summary()
    holistic = summary["per_gate"]["holistic_alert"]
    assert holistic.get("fail", 0) >= 1, (
        f"Expected holistic_alert.fail to record >= 1 after Marco run, "
        f"got summary {summary}"
    )
    rates = summary["rates"]["holistic_alert"]
    assert rates["fail_rate"] > 0.0, (
        f"Sprint J expectation: holistic_alert fail_rate must be > 0 "
        f"after a Marco-style run; got {rates}"
    )
# Sprint F: Tool-Forcing Regenerate
# ---------------------------------------------------------------------------


def test_gate_taxonomy_needs_tools_flags():
    """Three gates require tools, two are text-only."""
    by_id = {g.gate_id: g for g in GATES}
    assert by_id["temporal_freshness"].needs_tools is True
    assert by_id["injury_persistence"].needs_tools is True
    assert by_id["stats_grounding"].needs_tools is True
    assert by_id["holistic_alert"].needs_tools is False
    assert by_id["language_mirror"].needs_tools is False


def test_gate_by_id_returns_registered_gate():
    g = gate_by_id("stats_grounding")
    assert g is not None
    assert g.gate_id == "stats_grounding"
    assert g.needs_tools is True


def test_gate_by_id_unknown_returns_none():
    assert gate_by_id("does_not_exist") is None


def test_tool_requirements_cover_every_tool_required_gate():
    """Every gate marked needs_tools=True has an entry in TOOL_REQUIREMENTS."""
    for g in GATES:
        if g.needs_tools:
            assert g.gate_id in TOOL_REQUIREMENTS, (
                f"Tool-required gate {g.gate_id!r} is missing from TOOL_REQUIREMENTS"
            )
            tools = TOOL_REQUIREMENTS[g.gate_id]
            assert len(tools) >= 1, f"{g.gate_id} has empty tool list"


def test_needs_tool_forcing_true_when_tool_required_gate_fails():
    failures = (
        GateResult(passed=False, gate_id="stats_grounding", reason="x"),
        GateResult(passed=False, gate_id="language_mirror", reason="y"),
    )
    assert needs_tool_forcing(failures) is True


def test_needs_tool_forcing_false_for_text_only_failures():
    failures = (
        GateResult(passed=False, gate_id="language_mirror", reason="y"),
        GateResult(passed=False, gate_id="holistic_alert", reason="z"),
    )
    assert needs_tool_forcing(failures) is False


def test_needs_tool_forcing_empty_failures_is_false():
    assert needs_tool_forcing(()) is False


def test_compose_tool_force_instruction_stats_grounding():
    failures = (
        GateResult(passed=False, gate_id="stats_grounding", reason="x"),
    )
    text = _compose_tool_force_instruction(failures)
    assert "SYSTEM REMINDER" in text
    assert "get_activities" in text
    assert "get_activity_details" in text
    assert "get_health_summary" in text
    assert "REQUIRED" in text


def test_compose_tool_force_instruction_temporal_freshness():
    failures = (
        GateResult(passed=False, gate_id="temporal_freshness", reason="x"),
    )
    text = _compose_tool_force_instruction(failures)
    assert "sync_garmin_data" in text
    assert "get_provider_status" in text
    assert "get_activities" in text


def test_compose_tool_force_instruction_injury_persistence():
    failures = (
        GateResult(passed=False, gate_id="injury_persistence", reason="x"),
    )
    text = _compose_tool_force_instruction(failures)
    assert "annotate_activity" in text
    assert "append_to_journal" in text


def test_compose_tool_force_instruction_combines_multiple_failures():
    failures = (
        GateResult(passed=False, gate_id="temporal_freshness", reason="x"),
        GateResult(passed=False, gate_id="stats_grounding", reason="y"),
    )
    text = _compose_tool_force_instruction(failures)
    assert "sync_garmin_data" in text
    assert "get_health_summary" in text


def test_compose_tool_force_instruction_skips_text_only_failures():
    """Text-only gates contribute no fragment to the tool-force prompt."""
    failures = (
        GateResult(passed=False, gate_id="stats_grounding", reason="x"),
        GateResult(passed=False, gate_id="language_mirror", reason="y"),
    )
    text = _compose_tool_force_instruction(failures)
    assert "get_activities" in text
    # The text-only fragment is not the source of the SYSTEM REMINDER and
    # carries no tool list; ensure the prompt does not invent one.
    assert "language_mirror" not in text


def test_compose_tool_force_instruction_idempotent_over_duplicates():
    """A failure list with duplicates yields the same instruction once."""
    failures = (
        GateResult(passed=False, gate_id="stats_grounding", reason="x"),
        GateResult(passed=False, gate_id="stats_grounding", reason="y"),
    )
    text = _compose_tool_force_instruction(failures)
    # Only one fragment for stats_grounding.
    assert text.count("get_activities") == 1


def test_compose_tool_force_instruction_uses_real_umlauts():
    """No ASCII transliteration in agent-facing instruction text."""
    failures = (
        GateResult(passed=False, gate_id="injury_persistence", reason="x"),
    )
    text = _compose_tool_force_instruction(failures)
    # Real umlauts present (German body-issue keyword).
    assert "ähnlich" in text
    # No em-dashes / en-dashes per project policy.
    assert "—" not in text
    assert "–" not in text


def test_compose_tool_force_instruction_empty_when_only_text_failures():
    """If only text-only gates fail, instruction has preamble + closer only."""
    failures = (
        GateResult(passed=False, gate_id="language_mirror", reason="y"),
    )
    text = _compose_tool_force_instruction(failures)
    assert "SYSTEM REMINDER" in text
    # No tool-required fragment leaked in.
    assert "sync_garmin_data" not in text
    assert "annotate_activity" not in text


# ---------------------------------------------------------------------------
# Sprint F: tool_forced metric outcomes
# ---------------------------------------------------------------------------


def test_record_tool_forced_outcome_success():
    record_tool_forced_outcome(("stats_grounding",), success=True)
    summary = get_gates_metrics().summary()
    bucket = summary["per_gate"]["stats_grounding"]
    assert bucket["tool_forced_regenerate_success"] == 1
    assert bucket["tool_forced_regenerate_failed"] == 0
    assert summary["totals"]["tool_forced_regenerate_success"] == 1


def test_record_tool_forced_outcome_failure():
    record_tool_forced_outcome(
        ("temporal_freshness", "stats_grounding"),
        success=False,
    )
    summary = get_gates_metrics().summary()
    assert summary["per_gate"]["temporal_freshness"]["tool_forced_regenerate_failed"] == 1
    assert summary["per_gate"]["stats_grounding"]["tool_forced_regenerate_failed"] == 1
    assert summary["totals"]["tool_forced_regenerate_failed"] == 2


def test_record_tool_forced_outcome_ignores_unknown_gate():
    record_tool_forced_outcome(("does_not_exist",), success=True)
    summary = get_gates_metrics().summary()
    assert summary["totals"]["tool_forced_regenerate_success"] == 0


# ---------------------------------------------------------------------------
# 14-day window: stats grounding allowlist additions
# ---------------------------------------------------------------------------
#
# When the coach proposes future sessions or reads the window, paces and
# durations in the reply are grounded by those tool calls. The
# stats_grounding gate must therefore accept ``propose_sessions`` and
# ``get_session_window`` as read-tools.


def test_stats_grounding_passes_with_propose_sessions_call():
    ctx = _ctx(
        response_text="Morgen 60 min easy, Z2 bei 5:30/km.",
        tools_recent=("propose_sessions",),
    )
    r = _gate_stats_grounding(ctx)
    assert r.passed


def test_stats_grounding_passes_with_get_session_window_call():
    ctx = _ctx(
        response_text="Diese Woche stehen 4 Einheiten an, gesamt 5h.",
        tools_recent=("get_session_window",),
    )
    r = _gate_stats_grounding(ctx)
    assert r.passed


def test_stats_grounding_allowlist_contains_new_window_tools():
    from src.agent.response_gates import _READ_TOOLS_FOR_STATS
    assert "propose_sessions" in _READ_TOOLS_FOR_STATS
    assert "get_session_window" in _READ_TOOLS_FOR_STATS
