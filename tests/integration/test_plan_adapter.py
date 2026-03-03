"""Integration tests for plan_adapter.adapt_plan_to_weekly_format.

Covers:
- Day-keyed input → correct 7-day output structure
- Rest days (missing days) get empty sessions list
- Invalid intensity raises ValueError
- Session missing 'type' raises ValueError
- {days: {monday: ...}} wrapper format is unwrapped correctly
- Output always has exactly all 7 days regardless of input
"""

from __future__ import annotations

import pytest

from src.agent.plan_adapter import VALID_DAYS, adapt_plan_to_weekly_format


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session(
    session_type: str = "run",
    intensity: str = "moderate",
    duration_min: int = 45,
) -> dict:
    return {
        "type": session_type,
        "intensity": intensity,
        "duration_min": duration_min,
        "description": "Test session",
    }


# ---------------------------------------------------------------------------
# Basic structure
# ---------------------------------------------------------------------------


class TestBasicPlanAdaptation:
    def test_basic_plan_adaptation(self):
        """Day-keyed input produces a dict with all 7 days."""
        agent_plan = {
            "monday": {"sessions": [_make_session("run", "moderate", 45)]},
            "wednesday": {"sessions": [_make_session("bike", "high", 60)]},
        }
        result = adapt_plan_to_weekly_format(agent_plan)

        assert "monday" in result
        assert len(result["monday"]["sessions"]) == 1
        assert result["monday"]["sessions"][0]["type"] == "run"
        assert result["monday"]["sessions"][0]["intensity"] == "moderate"

        assert "wednesday" in result
        assert result["wednesday"]["sessions"][0]["type"] == "bike"

    def test_all_days_present(self):
        """Output always contains all 7 days, even if input has only 1."""
        agent_plan = {
            "tuesday": {"sessions": [_make_session()]},
        }
        result = adapt_plan_to_weekly_format(agent_plan)

        assert set(result.keys()) == set(VALID_DAYS)
        assert len(result) == 7

    def test_empty_input_produces_seven_rest_days(self):
        result = adapt_plan_to_weekly_format({})
        assert set(result.keys()) == set(VALID_DAYS)
        for day in VALID_DAYS:
            assert result[day]["sessions"] == []

    def test_session_fields_are_normalised(self):
        """Type and intensity are lowercased and stripped."""
        agent_plan = {
            "friday": {"sessions": [{"type": "  Swim  ", "intensity": "  HIGH  "}]},
        }
        result = adapt_plan_to_weekly_format(agent_plan)
        session = result["friday"]["sessions"][0]
        assert session["type"] == "swim"
        assert session["intensity"] == "high"


# ---------------------------------------------------------------------------
# Rest days
# ---------------------------------------------------------------------------


class TestRestDays:
    def test_rest_day_has_empty_sessions(self):
        """A day not present in input → sessions: []."""
        agent_plan = {
            "monday": {"sessions": [_make_session()]},
        }
        result = adapt_plan_to_weekly_format(agent_plan)

        for rest_day in ("tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"):
            assert result[rest_day]["sessions"] == [], (
                f"{rest_day} should be a rest day with empty sessions"
            )

    def test_explicit_empty_sessions(self):
        """Explicitly providing sessions: [] also results in a rest day."""
        agent_plan = {
            "monday": {"sessions": []},
        }
        result = adapt_plan_to_weekly_format(agent_plan)
        assert result["monday"]["sessions"] == []

    def test_missing_day_key_produces_rest_day(self):
        """Days absent from input dict become rest days."""
        agent_plan = {"saturday": {"sessions": [_make_session("swim", "low", 30)]}}
        result = adapt_plan_to_weekly_format(agent_plan)
        assert result["monday"]["sessions"] == []
        assert result["sunday"]["sessions"] == []


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidationErrors:
    def test_invalid_intensity_raises(self):
        """An unrecognized intensity value raises ValueError."""
        agent_plan = {
            "monday": {"sessions": [{"type": "run", "intensity": "extreme"}]},
        }
        with pytest.raises(ValueError, match="intensity"):
            adapt_plan_to_weekly_format(agent_plan)

    def test_missing_type_raises(self):
        """A session without a 'type' field raises ValueError."""
        agent_plan = {
            "tuesday": {"sessions": [{"intensity": "moderate", "duration_min": 30}]},
        }
        with pytest.raises(ValueError, match="type"):
            adapt_plan_to_weekly_format(agent_plan)

    def test_valid_intensities_accepted(self):
        """All three valid intensity values must be accepted without error."""
        for intensity in ("low", "moderate", "high"):
            agent_plan = {
                "monday": {"sessions": [{"type": "run", "intensity": intensity}]},
            }
            result = adapt_plan_to_weekly_format(agent_plan)
            assert result["monday"]["sessions"][0]["intensity"] == intensity

    def test_session_type_via_session_type_key(self):
        """'session_type' key is accepted as an alternative to 'type'."""
        agent_plan = {
            "thursday": {"sessions": [{"session_type": "yoga", "intensity": "low"}]},
        }
        result = adapt_plan_to_weekly_format(agent_plan)
        assert result["thursday"]["sessions"][0]["type"] == "yoga"


# ---------------------------------------------------------------------------
# {days: {...}} wrapper format
# ---------------------------------------------------------------------------


class TestDaysWrapperFormat:
    def test_days_wrapper_format(self):
        """Input with a top-level 'days' key is unwrapped correctly."""
        agent_plan = {
            "days": {
                "monday": {"sessions": [_make_session("run", "high", 50)]},
                "friday": {"sessions": [_make_session("swim", "low", 30)]},
            }
        }
        result = adapt_plan_to_weekly_format(agent_plan)

        assert set(result.keys()) == set(VALID_DAYS)
        assert result["monday"]["sessions"][0]["type"] == "run"
        assert result["friday"]["sessions"][0]["type"] == "swim"
        # Remaining days are rest days
        assert result["tuesday"]["sessions"] == []

    def test_days_wrapper_all_days_present(self):
        """Even with days wrapper, output always has all 7 days."""
        agent_plan = {"days": {"wednesday": {"sessions": [_make_session()]}}}
        result = adapt_plan_to_weekly_format(agent_plan)
        assert len(result) == 7
        assert set(result.keys()) == set(VALID_DAYS)

    def test_days_wrapper_case_insensitive(self):
        """Day names inside the wrapper are lowercased."""
        agent_plan = {
            "days": {
                "Monday": {"sessions": [_make_session("run", "moderate", 45)]},
            }
        }
        result = adapt_plan_to_weekly_format(agent_plan)
        assert result["monday"]["sessions"][0]["type"] == "run"


# ---------------------------------------------------------------------------
# Duration normalisation
# ---------------------------------------------------------------------------


class TestDurationNormalisation:
    def test_duration_min_field(self):
        """duration_min is preserved as int."""
        agent_plan = {
            "monday": {"sessions": [{"type": "run", "duration_min": 75}]},
        }
        result = adapt_plan_to_weekly_format(agent_plan)
        assert result["monday"]["sessions"][0]["duration_min"] == 75

    def test_duration_minutes_alias(self):
        """duration_minutes (alias) is normalised to duration_min."""
        agent_plan = {
            "monday": {"sessions": [{"type": "run", "duration_minutes": 90}]},
        }
        result = adapt_plan_to_weekly_format(agent_plan)
        assert result["monday"]["sessions"][0]["duration_min"] == 90

    def test_missing_duration_defaults_to_zero(self):
        """When duration is absent, duration_min defaults to 0."""
        agent_plan = {
            "monday": {"sessions": [{"type": "run"}]},
        }
        result = adapt_plan_to_weekly_format(agent_plan)
        assert result["monday"]["sessions"][0]["duration_min"] == 0
