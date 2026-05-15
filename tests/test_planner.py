"""Tests for the Plan-and-Execute trigger heuristic and save_plan wiring.

Sprint B: cover the bug surfaced by Lisa's 8-week build:
- The agent sent an inline plan with sessions[] only, no duration_weeks.
- The old heuristic inferred 2 weeks from session dates, 2 * 6 = 12 < 14,
  no trigger. Sonnet never ran.

The fix introduces three gates in ``should_use_plan_and_execute``:
explicit ``duration_weeks``, keyword backstop, legacy session-count.
These tests cover each gate plus the save_plan failure-surfacing path.
"""

from __future__ import annotations

import logging

import pytest

from src.agent.planner import (
    _LONG_HORIZON_WEEKS_THRESHOLD,
    _looks_like_long_horizon,
    _parse_explicit_weeks,
    should_use_plan_and_execute,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def lisa_profile() -> dict:
    """Lisa's production profile: triathlete, 6 training days, big horizon."""
    return {
        "name": "Lisa Brandt",
        "sports": ["running", "cycling", "swimming"],
        "constraints": {
            "training_days_per_week": 6,
            "max_session_minutes": 240,
            "available_sports": ["running", "cycling", "swimming"],
        },
        "goal": {
            "event": "Challenge Roth",
            "target_date": "2026-07-06",
            "target_time": "11:30:00",
        },
        "fitness": {
            "estimated_vo2max": 56,
            "threshold_pace_min_km": 4.25,
            "weekly_volume_km": 60,
        },
    }


@pytest.fixture
def minimal_profile() -> dict:
    return {
        "constraints": {"training_days_per_week": 5},
        "sports": ["running"],
    }


# ---------------------------------------------------------------------------
# Gate 1: explicit intent
# ---------------------------------------------------------------------------


def test_trigger_fires_on_explicit_duration_weeks_16(lisa_profile: dict) -> None:
    """16-week marathon plan with duration_weeks set: trigger MUST fire."""
    plan = {
        "duration_weeks": 16,
        "start_date": "2026-05-18",
        "goal_event": "Marathon Berlin",
        "goal_date": "2026-09-26",
    }
    assert should_use_plan_and_execute(plan, lisa_profile) is True


def test_trigger_fires_at_threshold_exactly(lisa_profile: dict) -> None:
    """duration_weeks == 4 is the documented threshold."""
    plan = {"duration_weeks": _LONG_HORIZON_WEEKS_THRESHOLD, "start_date": "2026-06-01"}
    assert should_use_plan_and_execute(plan, lisa_profile) is True


def test_trigger_does_not_fire_below_threshold_with_no_keywords(
    minimal_profile: dict,
) -> None:
    """3-week plan with no marathon/build keywords: legacy path."""
    plan = {"duration_weeks": 3, "start_date": "2026-06-01"}
    # Below long-horizon threshold AND below session count (3*5=15 >= 14
    # would fire under gate 3, so use 2 weeks to be clearly below).
    plan_short = {"duration_weeks": 2, "start_date": "2026-06-01"}
    assert should_use_plan_and_execute(plan_short, minimal_profile) is False


def test_trigger_fires_when_weeks_legacy_key_is_used(lisa_profile: dict) -> None:
    plan = {"weeks": 8, "start_date": "2026-06-01"}
    assert should_use_plan_and_execute(plan, lisa_profile) is True


def test_trigger_handles_string_duration_weeks(lisa_profile: dict) -> None:
    plan = {"duration_weeks": "16", "start_date": "2026-06-01"}
    assert should_use_plan_and_execute(plan, lisa_profile) is True


def test_trigger_ignores_bool_duration_weeks(minimal_profile: dict) -> None:
    """True is technically int(1); guard against accidental booleans."""
    plan = {"duration_weeks": True}
    # No keywords either, so this must NOT fire.
    assert should_use_plan_and_execute(plan, minimal_profile) is False


# ---------------------------------------------------------------------------
# Gate 2: keyword backstop
# ---------------------------------------------------------------------------


def test_trigger_fires_on_focus_marathon_keyword(lisa_profile: dict) -> None:
    """Free-text label says marathon: backstop catches it."""
    plan = {"focus": "16-week marathon build", "start_date": "2026-06-01"}
    assert should_use_plan_and_execute(plan, lisa_profile) is True


def test_trigger_fires_on_goal_event_ironman(minimal_profile: dict) -> None:
    plan = {"goal_event": "Ironman Frankfurt 2026"}
    assert should_use_plan_and_execute(plan, minimal_profile) is True


def test_trigger_fires_on_german_wochen_pattern(lisa_profile: dict) -> None:
    """The exact failure-mode focus string from Lisa."""
    plan = {
        "focus": "Build 2: Langdistanz-Spezifik (8 Wochen bis Roth)",
        "start_date": "2026-05-18",
    }
    assert should_use_plan_and_execute(plan, lisa_profile) is True


def test_trigger_fires_on_roth_keyword(minimal_profile: dict) -> None:
    plan = {"focus": "Roth race-week tune-up"}
    assert should_use_plan_and_execute(plan, minimal_profile) is True


def test_trigger_fires_on_70_3_keyword(minimal_profile: dict) -> None:
    plan = {"goal_event": "70.3 Hamburg"}
    assert should_use_plan_and_execute(plan, minimal_profile) is True


def test_trigger_does_not_fire_on_short_keyword_free_label(
    minimal_profile: dict,
) -> None:
    """No long-horizon keyword, no duration, single session: no trigger."""
    plan = {
        "focus": "Recovery week",
        "sessions": [{"day": "monday", "date": "2026-06-01"}],
        "start_date": "2026-06-01",
    }
    assert should_use_plan_and_execute(plan, minimal_profile) is False


def test_trigger_fires_on_nested_goal_event(minimal_profile: dict) -> None:
    """Some shapes nest a `goal` dict; backstop checks there too."""
    plan = {"goal": {"event": "Marathon Paris"}, "start_date": "2026-06-01"}
    assert should_use_plan_and_execute(plan, minimal_profile) is True


# ---------------------------------------------------------------------------
# Gate 3: legacy session-count
# ---------------------------------------------------------------------------


def test_trigger_fires_on_session_count_legacy(minimal_profile: dict) -> None:
    """No duration_weeks, no keyword: legacy gate fires when weeks * days >= 14."""
    # 4 weeks worth of sessions, training_days_per_week=5 -> 20 >= 14.
    sessions = [
        {"day": "monday", "date": f"2026-06-{1 + 7 * w:02d}"} for w in range(4)
    ]
    sessions += [
        {"day": "wednesday", "date": f"2026-06-{3 + 7 * w:02d}"} for w in range(4)
    ]
    plan = {"sessions": sessions, "start_date": "2026-06-01"}
    assert should_use_plan_and_execute(plan, minimal_profile) is True


def test_trigger_does_not_fire_on_adjustment(minimal_profile: dict) -> None:
    """Single-week adjustment plan: inline path."""
    plan = {
        "start_date": "2026-06-01",
        "sessions": [
            {"day": "monday", "date": "2026-06-01"},
            {"day": "wednesday", "date": "2026-06-03"},
            {"day": "friday", "date": "2026-06-05"},
        ],
    }
    assert should_use_plan_and_execute(plan, minimal_profile) is False


def test_trigger_does_not_fire_on_lisa_actual_bug_input(lisa_profile: dict) -> None:
    """The exact shape Lisa got: 16 sessions over 2 weeks, no duration_weeks.

    Under the OLD heuristic this returned False (2 * 6 = 12 < 14). Under
    the NEW heuristic the backstop must catch the "8 Wochen bis Roth"
    focus AND/OR the "Roth" keyword.
    """
    plan = {
        "start_date": "2026-05-18",
        "focus": "Build 2: Langdistanz-Spezifik (8 Wochen bis Roth)",
        "sessions": [{"day": "monday", "date": f"2026-05-{18 + i:02d}"} for i in range(14)],
    }
    assert should_use_plan_and_execute(plan, lisa_profile) is True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_parse_explicit_weeks_handles_missing() -> None:
    assert _parse_explicit_weeks({}) is None
    assert _parse_explicit_weeks({"focus": "x"}) is None


def test_parse_explicit_weeks_handles_invalid_string() -> None:
    assert _parse_explicit_weeks({"duration_weeks": "soon"}) is None


def test_parse_explicit_weeks_caps_at_max() -> None:
    val = _parse_explicit_weeks({"duration_weeks": 9999})
    assert val == 32  # _MAX_PLAN_WEEKS


def test_looks_like_long_horizon_empty() -> None:
    assert _looks_like_long_horizon({}) is False
    assert _looks_like_long_horizon({"focus": "test"}) is False


def test_looks_like_long_horizon_too_few_weeks_in_text() -> None:
    """3 weeks in free text is below the long-horizon threshold (4)."""
    assert _looks_like_long_horizon({"focus": "3 weeks of base"}) is False


def test_looks_like_long_horizon_singular_woche_match() -> None:
    """The numeric pattern allows '8 Woche' as well as '8 Wochen'."""
    assert _looks_like_long_horizon({"focus": "8 Woche Aufbau"}) is True


# ---------------------------------------------------------------------------
# save_plan integration tests removed: the 24-week save_plan flow was
# replaced by the 14-day rolling window in session_tools. The planner
# heuristic (should_use_plan_and_execute) is still tested above because
# it remains useful for the long-horizon intent layer registered by the
# typed-tables agent, but it is no longer wired into the coach loop.
# See tests/test_session_tools.py for the new lifecycle coverage.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Auditable logging
# ---------------------------------------------------------------------------


def test_planner_invocation_log_is_emitted_for_sonnet(caplog) -> None:
    """The audit log line must fire when the planner runs Sonnet."""
    from src.agent.planner import _log_planner_invocation

    caplog.set_level(logging.INFO, logger="src.agent.planner")
    _log_planner_invocation(
        stage="planner",
        model="anthropic/claude-sonnet-4-5-20250929",
        request={"duration_weeks": 16},
    )
    msgs = [r.getMessage() for r in caplog.records]
    assert any("planner_invocation stage=planner" in m for m in msgs)
    assert any("premium=True" in m for m in msgs)


def test_planner_invocation_log_marks_haiku_as_non_premium(caplog) -> None:
    from src.agent.planner import _log_planner_invocation

    caplog.set_level(logging.INFO, logger="src.agent.planner")
    _log_planner_invocation(
        stage="executor_week_1",
        model="anthropic/claude-haiku-4-5-20251001",
        request={"week": 1},
    )
    msgs = [r.getMessage() for r in caplog.records]
    assert any("planner_invocation stage=executor_week_1" in m for m in msgs)
    assert any("premium=False" in m for m in msgs)
