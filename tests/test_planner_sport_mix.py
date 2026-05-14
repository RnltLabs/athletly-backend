"""Tests for the multi-sport floor heuristic (Sprint H sport-mix fix).

The Lisa bug: an 8-week plan for Challenge Roth (Langdistanz) came back
with 36 run, 9 bike, 3 swim sessions across 48 slots. Roth is a full
Ironman distance race; a plan that under-trains the bike and barely
swims at all is broken.

This module pins:
- the discipline detector (Langdistanz / 70.3 / Sprint / unknown)
- the per-week floor validator
- the regenerate-on-violation flow (mocked planner)
- the executor sport-override (planner's sport_per_day wins)
- the outline schema validator catches missing sport_per_day for
  multi-sport profiles
"""

from __future__ import annotations

from datetime import date as _date
from typing import Any
from unittest.mock import patch

import pytest

from src.agent.planner import (
    PlannerResult,
    _MULTI_SPORT_FLOORS,
    _detect_triathlon_discipline,
    _sanitize_executor_sessions,
    _validate_outline,
    _validate_sport_distribution,
    generate_training_plan,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def lisa_profile() -> dict:
    """Lisa's production profile: triathlete, 6 days/week, Langdistanz."""
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
        },
    }


def _valid_outline_for_lisa() -> dict:
    """Build a valid 2-week outline with proper sport_per_day."""
    return {
        "duration_weeks": 2,
        "start_date": "2026-05-18",
        "goal_event": "Challenge Roth",
        "goal_date": "2026-07-06",
        "phases": [{"weeks": [1, 2], "phase": "Build", "focus": "x",
                    "weekly_volume_km": 80, "intensity_distribution": "75/25"}],
        "weekly_template": {
            "monday": "easy",       "tuesday": "quality",
            "wednesday": "easy",    "thursday": "quality",
            "friday": "rest",       "saturday": "long_or_quality",
            "sunday": "long",
        },
        "sport_per_day": {
            "monday": "running",    "tuesday": "cycling",
            "wednesday": "swimming","thursday": "running",
            "friday": "rest",       "saturday": "cycling",
            "sunday": "cycling",
        },
    }


def _build_plan(outline: dict, sessions: list[dict]) -> dict:
    return {
        "start_date": outline["start_date"],
        "focus": "test",
        "sessions": sessions,
        "outline": outline,
    }


def _sessions_for_week(
    week_index: int, start: str, sport_per_day: dict[str, str],
) -> list[dict]:
    """Build one well-formed session per non-rest day for a given week."""
    weekdays = ["monday", "tuesday", "wednesday", "thursday",
                "friday", "saturday", "sunday"]
    start_dt = _date.fromisoformat(start)
    out: list[dict] = []
    for i, day in enumerate(weekdays):
        sport = sport_per_day.get(day, "rest")
        if sport == "rest":
            continue
        date = start_dt.replace(day=start_dt.day + (week_index - 1) * 7 + i)
        out.append({
            "day": day,
            "date": date.isoformat(),
            "sport": sport,
            "name": "x",
            "description": "",
            "duration_minutes": 60,
            "intensity": "low",
        })
    return out


# ---------------------------------------------------------------------------
# 1. Discipline detector
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text,expected", [
    ("Challenge Roth", "long_course"),
    ("Ironman Frankfurt", "long_course"),
    ("Langdistanz Aufbau", "long_course"),
    ("Kona qualifier", "long_course"),
    ("140.6 prep", "long_course"),
    ("Ironman 70.3 Hamburg", "middle_distance"),
    ("Mitteldistanz Build", "middle_distance"),
    ("Halbdistanz Saisonziel", "middle_distance"),
    ("Sprint distance Erlangen", "short_course"),
    ("Olympische Distanz Hamburg", "short_course"),
    ("Kurzdistanz Berlin", "short_course"),
    ("just some random text", None),
])
def test_discipline_detection(text: str, expected: str | None) -> None:
    request = {"goal_event": text}
    assert _detect_triathlon_discipline(request, outline=None) == expected


def test_discipline_detection_70_3_beats_ironman_substring() -> None:
    """`Ironman 70.3` must map to middle_distance, NOT long_course."""
    request = {"goal_event": "Ironman 70.3 Hamburg"}
    assert _detect_triathlon_discipline(request, None) == "middle_distance"


# ---------------------------------------------------------------------------
# 2. Outline schema validation
# ---------------------------------------------------------------------------


def test_outline_validation_rejects_missing_sport_per_day_when_multi_sport(
    lisa_profile: dict,
) -> None:
    payload: dict[str, Any] = {
        "outline": {
            "duration_weeks": 2,
            "start_date": "2026-05-18",
            "phases": [{"weeks": [1, 2], "phase": "Build"}],
            "weekly_template": {
                "monday": "easy", "tuesday": "quality",
                "wednesday": "easy", "thursday": "quality",
                "friday": "rest", "saturday": "long_or_quality",
                "sunday": "long",
            },
            # sport_per_day intentionally missing
        },
        "constraints_acknowledged": {},
    }
    err = _validate_outline(payload, request={}, profile=lisa_profile)
    assert err is not None and "sport_per_day" in err


def test_outline_validation_accepts_valid_multi_sport(lisa_profile: dict) -> None:
    payload = {
        "outline": _valid_outline_for_lisa(),
        "constraints_acknowledged": {},
    }
    err = _validate_outline(payload, request={}, profile=lisa_profile)
    assert err is None


def test_outline_validation_rejects_rest_day_with_sport_assigned(
    lisa_profile: dict,
) -> None:
    outline = _valid_outline_for_lisa()
    outline["sport_per_day"]["friday"] = "running"  # weekly_template says rest
    payload = {"outline": outline, "constraints_acknowledged": {}}
    err = _validate_outline(payload, request={}, profile=lisa_profile)
    assert err is not None and "friday" in err


def test_outline_validation_rejects_sport_not_in_available(
    lisa_profile: dict,
) -> None:
    outline = _valid_outline_for_lisa()
    outline["sport_per_day"]["monday"] = "rowing"  # not in available_sports
    payload = {"outline": outline, "constraints_acknowledged": {}}
    err = _validate_outline(payload, request={}, profile=lisa_profile)
    assert err is not None and "rowing" in err


def test_outline_validation_skips_sport_per_day_for_single_sport_profile() -> None:
    single = {
        "sports": ["running"],
        "constraints": {"training_days_per_week": 6,
                        "available_sports": ["running"]},
    }
    payload = {
        "outline": {
            "duration_weeks": 2,
            "start_date": "2026-05-18",
            "phases": [{"weeks": [1, 2], "phase": "Build"}],
            "weekly_template": {
                "monday": "easy", "tuesday": "quality",
                "wednesday": "easy", "thursday": "quality",
                "friday": "rest", "saturday": "long_or_quality",
                "sunday": "long",
            },
        },
        "constraints_acknowledged": {},
    }
    err = _validate_outline(payload, request={}, profile=single)
    assert err is None


# ---------------------------------------------------------------------------
# 3. Post-assembly sport-distribution validator
# ---------------------------------------------------------------------------


def test_validate_distribution_passes_when_floors_met(lisa_profile: dict) -> None:
    """Lisa, Langdistanz, 1 swim + 3 bike + 2 run per week = all floors met."""
    outline = _valid_outline_for_lisa()
    sessions = (
        _sessions_for_week(1, outline["start_date"], outline["sport_per_day"]) +
        _sessions_for_week(2, outline["start_date"], outline["sport_per_day"])
    )
    plan = _build_plan(outline, sessions)
    errs = _validate_sport_distribution(
        plan, lisa_profile, outline, request={"goal_event": "Challenge Roth"},
    )
    assert errs == [], f"unexpected errors: {errs}"


def test_validate_distribution_flags_missing_swim_for_langdistanz(
    lisa_profile: dict,
) -> None:
    """The Lisa production failure shape: every session is running."""
    outline = _valid_outline_for_lisa()
    # Force every weekday's "real" sport to running -> 0 swims, 0 bikes.
    bad_map = {d: ("running" if outline["sport_per_day"][d] != "rest" else "rest")
               for d in outline["sport_per_day"]}
    sessions = (
        _sessions_for_week(1, outline["start_date"], bad_map) +
        _sessions_for_week(2, outline["start_date"], bad_map)
    )
    plan = _build_plan(outline, sessions)
    errs = _validate_sport_distribution(
        plan, lisa_profile, outline, request={"goal_event": "Challenge Roth"},
    )
    assert errs, "expected violations for 0 swim / 0 bike weeks"
    joined = " ".join(errs)
    assert "swimming" in joined
    assert "cycling" in joined
    # Discipline tag must be present in the header.
    assert "long_course" in joined


def test_validate_distribution_passes_for_single_sport_profile() -> None:
    """Single-sport profiles are not subject to multi-sport floors."""
    single = {
        "sports": ["running"],
        "constraints": {"training_days_per_week": 5,
                        "available_sports": ["running"]},
    }
    outline = {
        "duration_weeks": 2, "start_date": "2026-05-18",
        "phases": [{"weeks": [1, 2], "phase": "Build"}],
        "weekly_template": {
            "monday": "easy", "tuesday": "quality", "wednesday": "easy",
            "thursday": "quality", "friday": "rest",
            "saturday": "long_or_quality", "sunday": "long",
        },
        "sport_per_day": {
            "monday": "running", "tuesday": "running", "wednesday": "running",
            "thursday": "running", "friday": "rest",
            "saturday": "running", "sunday": "running",
        },
    }
    sessions = (
        _sessions_for_week(1, outline["start_date"], outline["sport_per_day"]) +
        _sessions_for_week(2, outline["start_date"], outline["sport_per_day"])
    )
    plan = _build_plan(outline, sessions)
    errs = _validate_sport_distribution(plan, single, outline, request={})
    assert errs == []


def test_validate_distribution_uses_unknown_triathlon_default_when_no_event(
    lisa_profile: dict,
) -> None:
    """No goal_event AND multi-sport -> at least one session per sport per week."""
    outline = _valid_outline_for_lisa()
    outline["goal_event"] = None
    # 0 swims this whole plan.
    bad_map = {d: outline["sport_per_day"][d] for d in outline["sport_per_day"]}
    bad_map["wednesday"] = "running"  # was swimming
    sessions = (
        _sessions_for_week(1, outline["start_date"], bad_map) +
        _sessions_for_week(2, outline["start_date"], bad_map)
    )
    plan = _build_plan(outline, sessions)
    errs = _validate_sport_distribution(
        plan, lisa_profile, outline, request={"focus": "off-season build"},
    )
    assert errs, "expected violation when swim is missing entirely"
    assert any("swimming" in e for e in errs)


# ---------------------------------------------------------------------------
# 4. Executor sport-override: sanitizer takes the planner's choice.
# ---------------------------------------------------------------------------


def test_sanitizer_overrides_executor_sport_with_planner_assignment() -> None:
    """Executor said `running`; planner said `swimming`; planner wins."""
    week_slot = {
        "weekday_template": {
            "monday": "easy", "tuesday": "easy", "wednesday": "easy",
            "thursday": "easy", "friday": "rest",
            "saturday": "easy", "sunday": "easy",
        },
        "week_start_date": "2026-05-18",
        "constraints": {
            "max_session_minutes": 240,
            "training_days_per_week": 6,
            "available_sports": ["running", "cycling", "swimming"],
        },
        "sport_per_day": {
            "monday": "swimming",   "tuesday": "cycling",
            "wednesday": "running", "thursday": "cycling",
            "friday": "rest",       "saturday": "cycling",
            "sunday": "running",
        },
    }
    sessions = [
        {"day": "monday",    "date": "2026-05-18", "sport": "running",
         "name": "x", "description": "", "duration_minutes": 60,
         "intensity": "low"},
        {"day": "tuesday",   "date": "2026-05-19", "sport": "running",
         "name": "x", "description": "", "duration_minutes": 60,
         "intensity": "low"},
        {"day": "wednesday", "date": "2026-05-20", "sport": "running",
         "name": "x", "description": "", "duration_minutes": 60,
         "intensity": "low"},
        {"day": "thursday",  "date": "2026-05-21", "sport": "running",
         "name": "x", "description": "", "duration_minutes": 60,
         "intensity": "low"},
        {"day": "saturday",  "date": "2026-05-23", "sport": "running",
         "name": "x", "description": "", "duration_minutes": 60,
         "intensity": "low"},
        {"day": "sunday",    "date": "2026-05-24", "sport": "running",
         "name": "x", "description": "", "duration_minutes": 60,
         "intensity": "low"},
    ]
    sanitized, err = _sanitize_executor_sessions(sessions, week_slot, profile={})
    assert err is None, f"unexpected err: {err}"
    by_day = {s["day"]: s["sport"] for s in sanitized}
    assert by_day["monday"] == "swimming"
    assert by_day["tuesday"] == "cycling"
    assert by_day["wednesday"] == "running"
    assert by_day["thursday"] == "cycling"
    assert by_day["saturday"] == "cycling"
    assert by_day["sunday"] == "running"


def test_sanitizer_falls_back_to_first_available_when_no_assignment() -> None:
    """Single-sport: no sport_per_day -> first available_sports is used."""
    week_slot = {
        "weekday_template": {
            "monday": "easy", "tuesday": "rest", "wednesday": "rest",
            "thursday": "rest", "friday": "rest",
            "saturday": "rest", "sunday": "rest",
        },
        "week_start_date": "2026-05-18",
        "constraints": {
            "max_session_minutes": 120,
            "training_days_per_week": 1,
            "available_sports": ["running"],
        },
    }
    sessions = [
        {"day": "monday", "date": "2026-05-18", "sport": "running",
         "name": "x", "description": "", "duration_minutes": 60,
         "intensity": "low"},
    ]
    sanitized, err = _sanitize_executor_sessions(sessions, week_slot, profile={})
    assert err is None
    assert sanitized[0]["sport"] == "running"


# ---------------------------------------------------------------------------
# 5. Floor table self-test
# ---------------------------------------------------------------------------


def test_floor_table_long_course_has_swim_bike_run() -> None:
    floors = _MULTI_SPORT_FLOORS["long_course"]
    assert floors["swimming"] >= 1
    assert floors["cycling"] >= 2
    assert floors["running"] >= 2


def test_floor_table_middle_distance_matches_long_course_minimums() -> None:
    """70.3 still needs all three disciplines weekly."""
    floors = _MULTI_SPORT_FLOORS["middle_distance"]
    assert floors["swimming"] >= 1
    assert floors["cycling"] >= 1
    assert floors["running"] >= 1


# ---------------------------------------------------------------------------
# 6. End-to-end: generate_training_plan retries planner once on violation.
# ---------------------------------------------------------------------------


class _FakeLLMResponse:
    """Minimal duck-typed litellm response."""
    def __init__(self, content: str) -> None:
        msg = type("M", (), {"content": content})()
        self.choices = [type("C", (), {"message": msg})()]
        self.usage = type("U", (), {
            "prompt_tokens": 1, "completion_tokens": 1,
            "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
        })()


def _planner_payload(sport_per_day: dict[str, str]) -> str:
    """Build a valid planner JSON string with the given sport assignment."""
    import json
    payload = {
        "outline": {
            "duration_weeks": 2,
            "start_date": "2026-05-18",
            "goal_event": "Challenge Roth",
            "goal_date": "2026-07-06",
            "phases": [{"weeks": [1, 2], "phase": "Build", "focus": "x",
                        "weekly_volume_km": 80,
                        "intensity_distribution": "75/25"}],
            "weekly_template": {
                "monday": "easy",       "tuesday": "quality",
                "wednesday": "easy",    "thursday": "quality",
                "friday": "rest",       "saturday": "long_or_quality",
                "sunday": "long",
            },
            "sport_per_day": sport_per_day,
        },
        "constraints_acknowledged": {
            "max_session_minutes": 240,
            "training_days_per_week": 6,
            "available_sports": ["running", "cycling", "swimming"],
        },
    }
    return json.dumps(payload)


def _executor_payload_for_week(
    week_index: int, start: str, sport_per_day: dict[str, str],
) -> str:
    """Build a valid executor JSON string for a given week."""
    import json
    sessions = _sessions_for_week(week_index, start, sport_per_day)
    return json.dumps({"sessions": sessions})


def test_generate_training_plan_retries_planner_on_sport_mix_violation(
    lisa_profile: dict, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First planner attempt produces an all-running plan; retry must produce a valid mix."""
    weekdays = ["monday", "tuesday", "wednesday", "thursday",
                "friday", "saturday", "sunday"]
    bad_sport_per_day = {d: ("running" if d != "friday" else "rest") for d in weekdays}
    good_sport_per_day = {
        "monday": "running",   "tuesday": "cycling",
        "wednesday": "swimming","thursday": "running",
        "friday": "rest",      "saturday": "cycling",
        "sunday": "cycling",
    }

    # State machine for the LLM stub:
    # call 0 = planner attempt 1 (bad sport_per_day)
    # call 1..2 = executor for weeks 1..2 (bad)
    # call 3 = planner attempt 2 (good)
    # call 4..5 = executor for weeks 1..2 (good)
    call_counter = {"n": 0}

    def fake_chat_completion(*args: Any, **kwargs: Any) -> _FakeLLMResponse:
        n = call_counter["n"]
        call_counter["n"] = n + 1
        # Planner calls have temperature 0.3; executor have 0.5.
        if kwargs.get("temperature") == 0.3:
            spd = bad_sport_per_day if n == 0 else good_sport_per_day
            return _FakeLLMResponse(_planner_payload(spd))
        # Executor.
        if n in (1, 4):
            spd = bad_sport_per_day if n == 1 else good_sport_per_day
            return _FakeLLMResponse(_executor_payload_for_week(1, "2026-05-18", spd))
        if n in (2, 5):
            spd = bad_sport_per_day if n == 2 else good_sport_per_day
            return _FakeLLMResponse(_executor_payload_for_week(2, "2026-05-18", spd))
        raise AssertionError(f"unexpected call {n} kwargs={kwargs!r}")

    monkeypatch.setattr("src.agent.planner.chat_completion", fake_chat_completion)

    request = {
        "duration_weeks": 2,
        "start_date": "2026-05-18",
        "goal_event": "Challenge Roth",
        "goal_date": "2026-07-06",
    }
    result = generate_training_plan(request, lisa_profile)
    assert result.mode == "plan_and_execute", (
        f"expected successful retry, got mode={result.mode} meta={result.meta}"
    )
    assert result.meta["sport_mix_retries"] == 1
    # Final plan must satisfy the floors.
    from collections import Counter
    counts = Counter(s["sport"] for s in result.plan["sessions"])
    assert counts.get("swimming", 0) >= 2  # 1 per week x 2 weeks
    assert counts.get("cycling", 0) >= 4   # 2 per week x 2 weeks
    assert counts.get("running", 0) >= 4   # 2 per week x 2 weeks


def test_generate_training_plan_falls_back_when_retry_also_violates(
    lisa_profile: dict, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two bad planner outputs -> inline_fallback with sport_mix_floor reason."""
    weekdays = ["monday", "tuesday", "wednesday", "thursday",
                "friday", "saturday", "sunday"]
    bad_sport_per_day = {d: ("running" if d != "friday" else "rest") for d in weekdays}

    def fake_chat_completion(*args: Any, **kwargs: Any) -> _FakeLLMResponse:
        if kwargs.get("temperature") == 0.3:
            return _FakeLLMResponse(_planner_payload(bad_sport_per_day))
        # Executor: emit running for every day (matches the bad planner).
        # We just emit week 1 sessions; sanitizer keys off date math, not
        # week_index. Use the start date to make week 1 fall in range.
        return _FakeLLMResponse(
            _executor_payload_for_week(1, "2026-05-18", bad_sport_per_day),
        )

    monkeypatch.setattr("src.agent.planner.chat_completion", fake_chat_completion)
    # Need a 1-week plan to keep executor wiring trivial (week_index range).
    request = {
        "duration_weeks": 1,
        "start_date": "2026-05-18",
        "goal_event": "Challenge Roth",
    }
    # The planner payload we built is 2-weeks, but generate_training_plan
    # uses outline.duration_weeks (=2) regardless of request, so the
    # executor stub MUST handle whatever week_index the loop uses. Both
    # week 1 and week 2 here get the same payload (the dates for week 2
    # are still inside the same plan range so the post-assembly
    # validation just sees all-running sessions in week 1 and either an
    # invalid date or empty week 2 - either way the sport-mix validator
    # fires).
    result = generate_training_plan(request, lisa_profile)
    assert result.mode == "inline_fallback"
    assert result.meta.get("fallback_reason") == "sport_mix_floor_violation"
    assert result.meta["sport_mix_retries"] == 1
