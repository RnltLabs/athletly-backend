"""Unit tests for src.agent.dynamic_triggers — Dynamic trigger evaluation.

Covers:
- build_trigger_context() — session totals, per-sport breakdown, daily metrics,
  days_since_last_session, empty-data defaults
- evaluate_dynamic_triggers() — rule matching, non-matching, cooldown, invalid
  formula, no rules
- _parse_time() — ISO strings, datetime objects, None, invalid formats

All DB calls are mocked via unittest.mock.patch.  No real Supabase calls.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from src.agent.dynamic_triggers import (
    build_trigger_context,
    evaluate_dynamic_triggers,
    _parse_time,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

USER_ID = "test-user-dynamic-triggers"


# ---------------------------------------------------------------------------
# Helpers — immutable factory functions
# ---------------------------------------------------------------------------


def _make_activity(
    sport: str = "running",
    start_time: str | None = None,
    duration_seconds: int = 3600,
    trimp: float = 100.0,
) -> dict:
    """Create a test activity dict.  Does NOT mutate any shared state."""
    now = datetime.now(timezone.utc)
    return {
        "sport": sport,
        "start_time": start_time or (now - timedelta(hours=2)).isoformat(),
        "duration_seconds": duration_seconds,
        "trimp": trimp,
    }


def _make_daily_metric(
    date: str = "2026-03-04",
    hrv_avg: float | None = 55.0,
    sleep_score: float | None = 80.0,
    resting_heart_rate: float | None = 58.0,
    body_battery_high: float | None = 85.0,
    stress_avg: float | None = 30.0,
    recovery_score: float | None = 75.0,
) -> dict:
    """Create a test daily metric dict."""
    return {
        "date": date,
        "hrv_avg": hrv_avg,
        "sleep_score": sleep_score,
        "resting_heart_rate": resting_heart_rate,
        "body_battery_high": body_battery_high,
        "stress_avg": stress_avg,
        "recovery_score": recovery_score,
    }


def _make_rule(
    name: str = "high_fatigue",
    condition: str = "total_trimp_7d > 500",
    action: str = "Suggest a rest day",
    cooldown_hours: int = 24,
) -> dict:
    """Create a test trigger rule dict."""
    return {
        "name": name,
        "condition": condition,
        "action": action,
        "cooldown_hours": cooldown_hours,
    }


# ---------------------------------------------------------------------------
# Tests: build_trigger_context
# ---------------------------------------------------------------------------


class TestBuildTriggerContextTotals(unittest.TestCase):
    """Verify session count, total minutes, and total TRIMP calculations."""

    def test_totals_from_recent_activities(self) -> None:
        now = datetime.now(timezone.utc)
        activities = [
            _make_activity(
                start_time=(now - timedelta(hours=h)).isoformat(),
                duration_seconds=1800,
                trimp=50.0,
            )
            for h in (1, 12, 48)
        ]

        ctx = build_trigger_context(activities, [], {})

        self.assertEqual(ctx["total_sessions_7d"], 3.0)
        self.assertAlmostEqual(ctx["total_minutes_7d"], 90.0)  # 3 * 30 min
        self.assertAlmostEqual(ctx["total_trimp_7d"], 150.0)   # 3 * 50

    def test_excludes_activities_older_than_7_days(self) -> None:
        now = datetime.now(timezone.utc)
        recent = _make_activity(
            start_time=(now - timedelta(days=2)).isoformat(),
            duration_seconds=3600,
            trimp=100.0,
        )
        old = _make_activity(
            start_time=(now - timedelta(days=10)).isoformat(),
            duration_seconds=7200,
            trimp=200.0,
        )

        ctx = build_trigger_context([recent, old], [], {})

        self.assertEqual(ctx["total_sessions_7d"], 1.0)
        self.assertAlmostEqual(ctx["total_minutes_7d"], 60.0)
        self.assertAlmostEqual(ctx["total_trimp_7d"], 100.0)


class TestBuildTriggerContextPerSport(unittest.TestCase):
    """Verify per-sport session counts and TRIMP."""

    def test_per_sport_breakdown(self) -> None:
        now = datetime.now(timezone.utc)
        activities = [
            _make_activity(sport="running", start_time=(now - timedelta(hours=1)).isoformat(), trimp=80.0),
            _make_activity(sport="running", start_time=(now - timedelta(hours=12)).isoformat(), trimp=70.0),
            _make_activity(sport="cycling", start_time=(now - timedelta(hours=24)).isoformat(), trimp=120.0),
        ]

        ctx = build_trigger_context(activities, [], {})

        self.assertEqual(ctx["running_sessions_7d"], 2.0)
        self.assertAlmostEqual(ctx["running_trimp_7d"], 150.0)
        self.assertEqual(ctx["cycling_sessions_7d"], 1.0)
        self.assertAlmostEqual(ctx["cycling_trimp_7d"], 120.0)

    def test_sport_name_normalization(self) -> None:
        now = datetime.now(timezone.utc)
        activities = [
            _make_activity(sport="Trail Running", start_time=(now - timedelta(hours=1)).isoformat()),
        ]

        ctx = build_trigger_context(activities, [], {})

        # Spaces should be replaced with underscores, lowercase
        self.assertIn("trail_running_sessions_7d", ctx)
        self.assertEqual(ctx["trail_running_sessions_7d"], 1.0)


class TestBuildTriggerContextEmptyData(unittest.TestCase):
    """Verify sensible defaults with empty inputs."""

    def test_empty_activities_and_metrics(self) -> None:
        ctx = build_trigger_context([], [], {})

        self.assertEqual(ctx["total_sessions_7d"], 0.0)
        self.assertEqual(ctx["total_minutes_7d"], 0.0)
        self.assertEqual(ctx["total_trimp_7d"], 0.0)
        self.assertEqual(ctx["days_since_last_session"], 999.0)
        self.assertEqual(ctx["avg_hrv_7d"], 0.0)
        self.assertEqual(ctx["avg_sleep_score_7d"], 0.0)
        self.assertEqual(ctx["avg_resting_hr_7d"], 0.0)
        self.assertEqual(ctx["body_battery_latest"], 0.0)
        self.assertEqual(ctx["stress_avg_latest"], 0.0)
        self.assertEqual(ctx["recovery_score_latest"], 0.0)


class TestBuildTriggerContextDailyMetrics(unittest.TestCase):
    """Verify HRV, sleep, battery, and stress averages from daily metrics."""

    def test_averages_from_metrics(self) -> None:
        metrics = [
            _make_daily_metric(hrv_avg=50.0, sleep_score=70.0, resting_heart_rate=55.0,
                               body_battery_high=80.0, stress_avg=35.0, recovery_score=70.0),
            _make_daily_metric(hrv_avg=60.0, sleep_score=90.0, resting_heart_rate=61.0,
                               body_battery_high=90.0, stress_avg=25.0, recovery_score=80.0),
        ]

        ctx = build_trigger_context([], metrics, {})

        self.assertAlmostEqual(ctx["avg_hrv_7d"], 55.0)
        self.assertAlmostEqual(ctx["avg_sleep_score_7d"], 80.0)
        self.assertAlmostEqual(ctx["avg_resting_hr_7d"], 58.0)
        # Latest values come from first item (newest first)
        self.assertEqual(ctx["body_battery_latest"], 80.0)
        self.assertEqual(ctx["stress_avg_latest"], 35.0)
        self.assertEqual(ctx["recovery_score_latest"], 70.0)

    def test_handles_partial_metrics(self) -> None:
        """Metrics with some None fields still produce valid averages."""
        metrics = [
            _make_daily_metric(hrv_avg=50.0, sleep_score=None, resting_heart_rate=55.0),
            _make_daily_metric(hrv_avg=None, sleep_score=80.0, resting_heart_rate=None),
        ]

        ctx = build_trigger_context([], metrics, {})

        self.assertAlmostEqual(ctx["avg_hrv_7d"], 50.0)      # only one value
        self.assertAlmostEqual(ctx["avg_sleep_score_7d"], 80.0)  # only one value
        self.assertAlmostEqual(ctx["avg_resting_hr_7d"], 55.0)   # only one value


class TestBuildTriggerContextDaysSinceLast(unittest.TestCase):
    """Verify days_since_last_session calculation."""

    def test_days_since_last_session(self) -> None:
        now = datetime.now(timezone.utc)
        activities = [
            _make_activity(start_time=(now - timedelta(days=3)).isoformat()),
        ]

        ctx = build_trigger_context(activities, [], {})

        # Should be approximately 3.0 days (within tolerance for test execution time)
        self.assertAlmostEqual(ctx["days_since_last_session"], 3.0, delta=0.1)

    def test_days_since_last_uses_most_recent(self) -> None:
        now = datetime.now(timezone.utc)
        activities = [
            _make_activity(start_time=(now - timedelta(days=1)).isoformat()),
            _make_activity(start_time=(now - timedelta(days=5)).isoformat()),
        ]

        ctx = build_trigger_context(activities, [], {})

        self.assertAlmostEqual(ctx["days_since_last_session"], 1.0, delta=0.1)

    def test_no_activities_returns_999(self) -> None:
        ctx = build_trigger_context([], [], {})
        self.assertEqual(ctx["days_since_last_session"], 999.0)


# ---------------------------------------------------------------------------
# Tests: evaluate_dynamic_triggers
# ---------------------------------------------------------------------------


class TestEvaluateFiresMatchingRule(unittest.TestCase):
    """Rules whose condition evaluates truthy should fire."""

    @patch("src.agent.dynamic_triggers._check_cooldown", return_value=False)
    @patch("src.agent.dynamic_triggers.get_proactive_trigger_rules")
    def test_fires_when_condition_is_true(self, mock_rules, mock_cooldown) -> None:
        mock_rules.return_value = [
            _make_rule(
                name="high_load",
                condition="total_trimp_7d > 100",
                action="Take a rest day",
            ),
        ]

        now = datetime.now(timezone.utc)
        activities = [
            _make_activity(
                start_time=(now - timedelta(hours=1)).isoformat(),
                trimp=200.0,
            ),
        ]

        result = evaluate_dynamic_triggers(USER_ID, activities, [], {})

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["type"], "dynamic:high_load")
        self.assertEqual(result[0]["priority"], "medium")
        self.assertEqual(result[0]["data"]["rule_name"], "high_load")
        self.assertEqual(result[0]["data"]["action"], "Take a rest day")
        self.assertIn("context_snapshot", result[0]["data"])


class TestEvaluateSkipsNonMatchingRule(unittest.TestCase):
    """Rules whose condition evaluates falsy should NOT fire."""

    @patch("src.agent.dynamic_triggers._check_cooldown", return_value=False)
    @patch("src.agent.dynamic_triggers.get_proactive_trigger_rules")
    def test_skips_when_condition_is_false(self, mock_rules, mock_cooldown) -> None:
        mock_rules.return_value = [
            _make_rule(
                name="low_activity",
                condition="total_sessions_7d < 2",
                action="Get moving!",
            ),
        ]

        now = datetime.now(timezone.utc)
        activities = [
            _make_activity(start_time=(now - timedelta(hours=h)).isoformat())
            for h in (1, 12, 48)
        ]

        result = evaluate_dynamic_triggers(USER_ID, activities, [], {})

        self.assertEqual(len(result), 0)


class TestEvaluateRespectsCooldown(unittest.TestCase):
    """Rules in cooldown should be skipped even if condition matches."""

    @patch("src.agent.dynamic_triggers._check_cooldown", return_value=True)
    @patch("src.agent.dynamic_triggers.get_proactive_trigger_rules")
    def test_skips_rule_in_cooldown(self, mock_rules, mock_cooldown) -> None:
        mock_rules.return_value = [
            _make_rule(
                name="always_fires",
                condition="total_sessions_7d >= 0",  # always true
                action="Hello",
                cooldown_hours=48,
            ),
        ]

        now = datetime.now(timezone.utc)
        activities = [
            _make_activity(start_time=(now - timedelta(hours=1)).isoformat()),
        ]

        result = evaluate_dynamic_triggers(USER_ID, activities, [], {})

        self.assertEqual(len(result), 0)
        mock_cooldown.assert_called_once_with(USER_ID, "always_fires", 48)


class TestEvaluateHandlesInvalidFormula(unittest.TestCase):
    """Invalid formulas should be gracefully skipped (not crash)."""

    @patch("src.agent.dynamic_triggers._check_cooldown", return_value=False)
    @patch("src.agent.dynamic_triggers.get_proactive_trigger_rules")
    def test_invalid_formula_skipped(self, mock_rules, mock_cooldown) -> None:
        mock_rules.return_value = [
            _make_rule(
                name="bad_rule",
                condition="import os",  # not a valid CalcEngine expression
                action="Should not fire",
            ),
        ]

        result = evaluate_dynamic_triggers(USER_ID, [], [], {})

        self.assertEqual(len(result), 0)

    @patch("src.agent.dynamic_triggers._check_cooldown", return_value=False)
    @patch("src.agent.dynamic_triggers.get_proactive_trigger_rules")
    def test_empty_condition_skipped(self, mock_rules, mock_cooldown) -> None:
        mock_rules.return_value = [
            _make_rule(name="empty_cond", condition="", action="Nope"),
        ]

        result = evaluate_dynamic_triggers(USER_ID, [], [], {})

        self.assertEqual(len(result), 0)


class TestEvaluateNoRules(unittest.TestCase):
    """No rules defined should return empty list."""

    @patch("src.agent.dynamic_triggers.get_proactive_trigger_rules")
    def test_no_rules_returns_empty(self, mock_rules) -> None:
        mock_rules.return_value = []

        result = evaluate_dynamic_triggers(USER_ID, [], [], {})

        self.assertEqual(result, [])

    @patch("src.agent.dynamic_triggers.get_proactive_trigger_rules")
    def test_none_rules_returns_empty(self, mock_rules) -> None:
        mock_rules.return_value = None

        result = evaluate_dynamic_triggers(USER_ID, [], [], {})

        self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# Tests: _parse_time
# ---------------------------------------------------------------------------


class TestParseTimeVariousFormats(unittest.TestCase):
    """Verify _parse_time handles ISO strings, datetime objects, and edge cases."""

    def test_iso_string_with_timezone(self) -> None:
        result = _parse_time({"start_time": "2026-03-04T10:00:00+00:00"})
        self.assertIsInstance(result, datetime)
        self.assertIsNotNone(result.tzinfo)

    def test_iso_string_without_timezone(self) -> None:
        result = _parse_time({"start_time": "2026-03-04T10:00:00"})
        self.assertIsInstance(result, datetime)
        self.assertEqual(result.tzinfo, timezone.utc)

    def test_datetime_object_with_tz(self) -> None:
        dt = datetime(2026, 3, 4, 10, 0, tzinfo=timezone.utc)
        result = _parse_time({"start_time": dt})
        self.assertEqual(result, dt)

    def test_datetime_object_without_tz(self) -> None:
        dt = datetime(2026, 3, 4, 10, 0)
        result = _parse_time({"start_time": dt})
        self.assertIsNotNone(result.tzinfo)
        self.assertEqual(result.tzinfo, timezone.utc)

    def test_none_start_time(self) -> None:
        result = _parse_time({"start_time": None})
        self.assertIsNone(result)

    def test_missing_start_time(self) -> None:
        result = _parse_time({})
        self.assertIsNone(result)

    def test_invalid_string(self) -> None:
        result = _parse_time({"start_time": "not-a-date"})
        self.assertIsNone(result)

    def test_empty_string(self) -> None:
        result = _parse_time({"start_time": ""})
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# Tests: evaluate with multiple rules (mixed matching/non-matching)
# ---------------------------------------------------------------------------


class TestEvaluateMultipleRules(unittest.TestCase):
    """Verify correct behavior when multiple rules exist."""

    @patch("src.agent.dynamic_triggers._check_cooldown", return_value=False)
    @patch("src.agent.dynamic_triggers.get_proactive_trigger_rules")
    def test_only_matching_rules_fire(self, mock_rules, mock_cooldown) -> None:
        mock_rules.return_value = [
            _make_rule(name="matches", condition="total_sessions_7d > 0", action="A"),
            _make_rule(name="no_match", condition="total_sessions_7d > 100", action="B"),
            _make_rule(name="also_matches", condition="days_since_last_session < 999", action="C"),
        ]

        now = datetime.now(timezone.utc)
        activities = [
            _make_activity(start_time=(now - timedelta(hours=1)).isoformat()),
        ]

        result = evaluate_dynamic_triggers(USER_ID, activities, [], {})

        fired_names = [t["data"]["rule_name"] for t in result]
        self.assertIn("matches", fired_names)
        self.assertIn("also_matches", fired_names)
        self.assertNotIn("no_match", fired_names)
        self.assertEqual(len(result), 2)


# ---------------------------------------------------------------------------
# Tests: context_snapshot in trigger data
# ---------------------------------------------------------------------------


class TestContextSnapshot(unittest.TestCase):
    """Verify context_snapshot values are rounded to 2 decimal places."""

    @patch("src.agent.dynamic_triggers._check_cooldown", return_value=False)
    @patch("src.agent.dynamic_triggers.get_proactive_trigger_rules")
    def test_context_snapshot_values_rounded(self, mock_rules, mock_cooldown) -> None:
        mock_rules.return_value = [
            _make_rule(name="test", condition="total_sessions_7d >= 0", action="Hi"),
        ]

        now = datetime.now(timezone.utc)
        activities = [
            _make_activity(
                start_time=(now - timedelta(hours=1)).isoformat(),
                duration_seconds=3661,  # 61.0167 minutes
                trimp=33.333,
            ),
        ]

        result = evaluate_dynamic_triggers(USER_ID, activities, [], {})

        self.assertEqual(len(result), 1)
        snapshot = result[0]["data"]["context_snapshot"]
        # All values should be rounded to 2 decimal places
        for key, value in snapshot.items():
            # Check that the value has at most 2 decimal places
            self.assertEqual(value, round(value, 2), f"{key} not rounded: {value}")


# ---------------------------------------------------------------------------
# Tests: input immutability
# ---------------------------------------------------------------------------


class TestInputImmutability(unittest.TestCase):
    """Verify that input data is not mutated."""

    def test_build_context_does_not_mutate_activities(self) -> None:
        now = datetime.now(timezone.utc)
        activities = [
            _make_activity(start_time=(now - timedelta(hours=1)).isoformat()),
        ]
        original_activities = [dict(a) for a in activities]

        build_trigger_context(activities, [], {})

        self.assertEqual(activities, original_activities)

    def test_build_context_does_not_mutate_metrics(self) -> None:
        metrics = [_make_daily_metric()]
        original_metrics = [dict(m) for m in metrics]

        build_trigger_context([], metrics, {})

        self.assertEqual(metrics, original_metrics)


if __name__ == "__main__":
    unittest.main()
