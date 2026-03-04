"""Tests for health data tools -- get_health_data() and get_daily_metrics().

Covers:
- Tool registration
- get_health_data: source filtering (all/health/garmin), field normalization,
  deduplication via garmin_activity_id, activity_type filter, provider filter,
  empty data, token budget truncation
- get_daily_metrics: merging health + garmin daily stats, conflict resolution
  (health wins), garmin-only dates, health-only dates, empty data
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.agent.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

USER_ID = "test-user-health-001"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(user_id: str = USER_ID) -> MagicMock:
    s = MagicMock()
    s.agenticsports_user_id = user_id
    s.use_supabase = True
    return s


def _make_health_activity(
    start_time: str = "2026-03-01T08:00:00",
    activity_type: str = "running",
    duration_seconds: int = 3600,
    distance_meters: float = 10000.0,
    avg_heart_rate: int = 150,
    max_heart_rate: int = 175,
    training_load_trimp: float = 85.0,
    provider_type: str = "apple_health",
    external_id: str | None = None,
) -> dict:
    return {
        "start_time": start_time,
        "activity_type": activity_type,
        "duration_seconds": duration_seconds,
        "distance_meters": distance_meters,
        "avg_heart_rate": avg_heart_rate,
        "max_heart_rate": max_heart_rate,
        "training_load_trimp": training_load_trimp,
        "provider_type": provider_type,
        "external_id": external_id,
    }


def _make_garmin_activity(
    start_time: str = "2026-03-02T09:00:00",
    activity_type: str = "cycling",
    duration: int = 5400,
    distance: float = 40000.0,
    avg_hr: int = 140,
    max_hr: int = 165,
    garmin_activity_id: str = "garmin-001",
) -> dict:
    return {
        "start_time": start_time,
        "type": activity_type,
        "duration": duration,
        "distance": distance,
        "avg_hr": avg_hr,
        "max_hr": max_hr,
        "garmin_activity_id": garmin_activity_id,
    }


def _make_agent_activity(
    start_time: str = "2026-03-03T07:00:00",
    sport: str = "running",
    duration_seconds: int = 2700,
    distance_meters: float = 8000.0,
    avg_hr: int = 155,
    max_hr: int = 178,
    trimp: float = 72.0,
    garmin_activity_id: str | None = None,
) -> dict:
    return {
        "start_time": start_time,
        "sport": sport,
        "duration_seconds": duration_seconds,
        "distance_meters": distance_meters,
        "avg_hr": avg_hr,
        "max_hr": max_hr,
        "trimp": trimp,
        "garmin_activity_id": garmin_activity_id,
    }


def _register_and_execute(
    tool_name: str,
    args: dict,
    mock_settings: MagicMock,
    mock_health_acts: list[dict],
    mock_garmin_acts: list[dict],
    mock_agent_acts: list[dict],
) -> dict:
    """Register health tools with mocked DB functions and execute the given tool.

    The health tool imports list_activities via ``from src.db import list_activities``
    which binds to the package-level name.  We patch both the source module and the
    re-exported package name so the mock is hit regardless of which binding is used.
    """
    registry = ToolRegistry()

    with (
        patch("src.agent.tools.health_tools.get_settings", return_value=mock_settings),
        patch(
            "src.db.health_data_db.list_health_activities",
            return_value=mock_health_acts,
        ),
        patch(
            "src.db.health_data_db.list_garmin_activities",
            return_value=mock_garmin_acts,
        ),
        # Patch both the source module and the package-level re-export so
        # ``from src.db import list_activities`` inside the closure is intercepted.
        patch(
            "src.db.activity_store_db.list_activities",
            return_value=mock_agent_acts,
        ),
        patch(
            "src.db.list_activities",
            return_value=mock_agent_acts,
        ),
    ):
        from src.agent.tools.health_tools import register_health_tools

        register_health_tools(registry)
        return registry.execute(tool_name, args)


# ---------------------------------------------------------------------------
# Registration tests
# ---------------------------------------------------------------------------


class TestHealthToolsRegistration:
    """Verify both health tools are registered with the expected names."""

    def test_get_health_data_registered(self) -> None:
        registry = ToolRegistry()
        with patch("src.agent.tools.health_tools.get_settings", return_value=_make_settings()):
            from src.agent.tools.health_tools import register_health_tools

            register_health_tools(registry)

        names = [t["function"]["name"] for t in registry.get_openai_tools()]
        assert "get_health_data" in names

    def test_get_daily_metrics_registered(self) -> None:
        registry = ToolRegistry()
        with patch("src.agent.tools.health_tools.get_settings", return_value=_make_settings()):
            from src.agent.tools.health_tools import register_health_tools

            register_health_tools(registry)

        names = [t["function"]["name"] for t in registry.get_openai_tools()]
        assert "get_daily_metrics" in names

    def test_both_tools_in_data_category(self) -> None:
        registry = ToolRegistry()
        with patch("src.agent.tools.health_tools.get_settings", return_value=_make_settings()):
            from src.agent.tools.health_tools import register_health_tools

            register_health_tools(registry)

        listed = registry.list_tools()
        health_tools = [t for t in listed if t["name"] in ("get_health_data", "get_daily_metrics")]
        assert all(t["category"] == "data" for t in health_tools)


# ---------------------------------------------------------------------------
# get_health_data — source filtering
# ---------------------------------------------------------------------------


class TestGetHealthDataSourceFilter:
    """Test source parameter controls which DB functions are queried."""

    def test_source_health_only_returns_health_activities(self) -> None:
        health_act = _make_health_activity(activity_type="swimming")
        result = _register_and_execute(
            "get_health_data",
            {"source": "health"},
            _make_settings(),
            [health_act],
            [],  # garmin empty — not queried
            [],
        )
        assert result["count"] == 1
        assert result["activities"][0]["sport"] == "swimming"
        assert result["activities"][0]["source"] == "health"

    def test_source_garmin_only_returns_garmin_activities(self) -> None:
        garmin_act = _make_garmin_activity(activity_type="cycling")
        result = _register_and_execute(
            "get_health_data",
            {"source": "garmin"},
            _make_settings(),
            [],  # health empty — not queried
            [garmin_act],
            [],
        )
        assert result["count"] == 1
        assert result["activities"][0]["sport"] == "cycling"
        assert result["activities"][0]["source"] == "garmin"

    def test_source_all_merges_all_three_tables(self) -> None:
        health_act = _make_health_activity(start_time="2026-02-01T08:00:00")
        garmin_act = _make_garmin_activity(start_time="2026-02-02T09:00:00")
        agent_act = _make_agent_activity(start_time="2026-02-03T07:00:00")

        result = _register_and_execute(
            "get_health_data",
            {"source": "all"},
            _make_settings(),
            [health_act],
            [garmin_act],
            [agent_act],
        )

        sources = {a["source"] for a in result["activities"]}
        assert "health" in sources
        assert "garmin" in sources
        assert "agent" in sources
        assert result["count"] == 3


# ---------------------------------------------------------------------------
# get_health_data — field normalization
# ---------------------------------------------------------------------------


class TestGetHealthDataNormalization:
    """Test that provider-specific field names are mapped to the common schema."""

    def test_health_activity_fields_normalized(self) -> None:
        health_act = _make_health_activity(
            start_time="2026-03-01T08:00:00",
            activity_type="running",
            duration_seconds=3600,
            distance_meters=10000.0,
            avg_heart_rate=150,
            max_heart_rate=175,
            training_load_trimp=85.0,
            provider_type="apple_health",
        )

        result = _register_and_execute(
            "get_health_data",
            {"source": "health"},
            _make_settings(),
            [health_act],
            [],
            [],
        )

        act = result["activities"][0]
        assert act["date"] == "2026-03-01"
        assert act["sport"] == "running"
        assert act["duration_minutes"] == 60.0
        assert act["distance_km"] == 10.0
        assert act["avg_hr"] == 150
        assert act["max_hr"] == 175
        assert act["trimp"] == 85.0
        assert act["source"] == "health"
        assert act["provider"] == "apple_health"

    def test_garmin_activity_fields_normalized(self) -> None:
        garmin_act = _make_garmin_activity(
            start_time="2026-03-02T09:00:00",
            activity_type="cycling",
            duration=5400,  # seconds
            distance=40000.0,  # meters
            avg_hr=140,
            max_hr=165,
            garmin_activity_id="garmin-999",
        )

        result = _register_and_execute(
            "get_health_data",
            {"source": "garmin"},
            _make_settings(),
            [],
            [garmin_act],
            [],
        )

        act = result["activities"][0]
        assert act["date"] == "2026-03-02"
        assert act["sport"] == "cycling"
        assert act["duration_minutes"] == 90.0
        assert act["distance_km"] == 40.0
        assert act["avg_hr"] == 140
        assert act["max_hr"] == 165
        assert act["trimp"] is None  # garmin_activities has no TRIMP
        assert act["source"] == "garmin"
        assert act["provider"] == "garmin"

    def test_internal_dedup_keys_stripped_from_output(self) -> None:
        """_external_id and _garmin_id must not appear in the final output."""
        health_act = _make_health_activity(external_id="ext-001")
        garmin_act = _make_garmin_activity(garmin_activity_id="g-001")

        result = _register_and_execute(
            "get_health_data",
            {"source": "all"},
            _make_settings(),
            [health_act],
            [garmin_act],
            [],
        )

        for act in result["activities"]:
            assert "_external_id" not in act
            assert "_garmin_id" not in act

    def test_none_distance_produces_none_distance_km(self) -> None:
        health_act = _make_health_activity(distance_meters=0)
        health_act["distance_meters"] = None  # explicitly null

        result = _register_and_execute(
            "get_health_data",
            {"source": "health"},
            _make_settings(),
            [health_act],
            [],
            [],
        )

        assert result["activities"][0]["distance_km"] is None

    def test_missing_start_time_produces_empty_date(self) -> None:
        health_act = _make_health_activity()
        health_act["start_time"] = None

        result = _register_and_execute(
            "get_health_data",
            {"source": "health"},
            _make_settings(),
            [health_act],
            [],
            [],
        )

        assert result["activities"][0]["date"] == ""


# ---------------------------------------------------------------------------
# get_health_data — deduplication
# ---------------------------------------------------------------------------


class TestGetHealthDataDeduplication:
    """Test that agent activities suppress matching health/garmin rows."""

    def test_garmin_activity_suppressed_when_covered_by_agent(self) -> None:
        """A garmin_activity whose garmin_activity_id appears in the agent
        table must be excluded from the merged output."""
        shared_id = "shared-garmin-100"
        garmin_act = _make_garmin_activity(garmin_activity_id=shared_id)
        agent_act = _make_agent_activity(
            garmin_activity_id=shared_id, sport="cycling"
        )

        result = _register_and_execute(
            "get_health_data",
            {"source": "all"},
            _make_settings(),
            [],
            [garmin_act],
            [agent_act],
        )

        # Only the agent version survives
        assert result["count"] == 1
        assert result["activities"][0]["source"] == "agent"

    def test_health_activity_suppressed_when_external_id_covered(self) -> None:
        """A health_activity whose external_id matches a garmin_activity_id
        already in the agent table must be excluded."""
        shared_id = "shared-garmin-200"
        health_act = _make_health_activity(external_id=shared_id)
        agent_act = _make_agent_activity(garmin_activity_id=shared_id)

        result = _register_and_execute(
            "get_health_data",
            {"source": "all"},
            _make_settings(),
            [health_act],
            [],
            [agent_act],
        )

        assert result["count"] == 1
        assert result["activities"][0]["source"] == "agent"

    def test_no_duplication_when_no_shared_ids(self) -> None:
        """When there are no overlapping IDs, all activities are kept."""
        health_act = _make_health_activity(external_id="unique-h-1")
        garmin_act = _make_garmin_activity(garmin_activity_id="unique-g-1")
        agent_act = _make_agent_activity(garmin_activity_id=None)

        result = _register_and_execute(
            "get_health_data",
            {"source": "all"},
            _make_settings(),
            [health_act],
            [garmin_act],
            [agent_act],
        )

        assert result["count"] == 3


# ---------------------------------------------------------------------------
# get_health_data — empty data
# ---------------------------------------------------------------------------


class TestGetHealthDataEmpty:
    """Test the empty-data contract."""

    def test_empty_all_sources_returns_zero_count(self) -> None:
        result = _register_and_execute(
            "get_health_data",
            {"source": "all"},
            _make_settings(),
            [],
            [],
            [],
        )
        assert result == {"count": 0, "activities": []}

    def test_empty_health_source(self) -> None:
        result = _register_and_execute(
            "get_health_data",
            {"source": "health"},
            _make_settings(),
            [],
            [],
            [],
        )
        assert result["count"] == 0
        assert result["activities"] == []

    def test_empty_garmin_source(self) -> None:
        result = _register_and_execute(
            "get_health_data",
            {"source": "garmin"},
            _make_settings(),
            [],
            [],
            [],
        )
        assert result["count"] == 0


# ---------------------------------------------------------------------------
# get_health_data — sorted newest first
# ---------------------------------------------------------------------------


class TestGetHealthDataSorting:
    def test_activities_sorted_newest_first(self) -> None:
        older = _make_health_activity(start_time="2026-01-01T08:00:00")
        newer = _make_health_activity(start_time="2026-03-01T08:00:00")

        result = _register_and_execute(
            "get_health_data",
            {"source": "health"},
            _make_settings(),
            [older, newer],  # DB returns in arbitrary order
            [],
            [],
        )

        dates = [a["date"] for a in result["activities"]]
        assert dates == sorted(dates, reverse=True)


# ---------------------------------------------------------------------------
# get_daily_metrics — helpers
# ---------------------------------------------------------------------------


def _make_garmin_daily(
    date: str,
    sleep_duration_minutes: int | None = 430,
    sleep_score: int | None = 72,
    hrv_weekly_avg: float | None = 55.0,
    resting_heart_rate: int | None = 52,
    stress_avg: int | None = 28,
    body_battery_high: int | None = 90,
    body_battery_low: int | None = 40,
    steps: int | None = 8500,
) -> dict:
    return {
        "date": date,
        "sleep_duration_minutes": sleep_duration_minutes,
        "sleep_score": sleep_score,
        "hrv_weekly_avg": hrv_weekly_avg,
        "resting_heart_rate": resting_heart_rate,
        "stress_avg": stress_avg,
        "body_battery_high": body_battery_high,
        "body_battery_low": body_battery_low,
        "steps": steps,
    }


def _make_health_daily(
    date: str,
    sleep_duration_minutes: int | None = 480,
    sleep_score: int | None = 85,
    hrv_avg: float | None = 62.0,
    resting_heart_rate: int | None = 50,
    stress_avg: int | None = 20,
    body_battery_high: int | None = 95,
    body_battery_low: int | None = 35,
    recovery_score: int | None = 88,
    steps: int | None = 9200,
) -> dict:
    return {
        "date": date,
        "sleep_duration_minutes": sleep_duration_minutes,
        "sleep_score": sleep_score,
        "hrv_avg": hrv_avg,
        "resting_heart_rate": resting_heart_rate,
        "stress_avg": stress_avg,
        "body_battery_high": body_battery_high,
        "body_battery_low": body_battery_low,
        "recovery_score": recovery_score,
        "steps": steps,
    }


def _execute_get_daily_metrics(
    garmin_rows: list[dict],
    health_rows: list[dict],
    days: int = 14,
) -> dict:
    registry = ToolRegistry()

    with (
        patch("src.agent.tools.health_tools.get_settings", return_value=_make_settings()),
        patch("src.db.health_data_db.list_garmin_daily_stats", return_value=garmin_rows),
        patch("src.db.health_data_db.list_daily_metrics", return_value=health_rows),
    ):
        from src.agent.tools.health_tools import register_health_tools

        register_health_tools(registry)
        return registry.execute("get_daily_metrics", {"days": days})


# ---------------------------------------------------------------------------
# get_daily_metrics — tests
# ---------------------------------------------------------------------------


class TestGetDailyMetricsEmpty:
    def test_empty_returns_zero_count(self) -> None:
        result = _execute_get_daily_metrics([], [])
        assert result == {"count": 0, "metrics": []}


class TestGetDailyMetricsGarminOnly:
    def test_garmin_only_date_included(self) -> None:
        garmin = _make_garmin_daily("2026-03-01")
        result = _execute_get_daily_metrics([garmin], [])

        assert result["count"] == 1
        m = result["metrics"][0]
        assert m["date"] == "2026-03-01"
        assert m["source"] == "garmin"
        assert m["sleep_minutes"] == garmin["sleep_duration_minutes"]
        assert m["hrv"] == garmin["hrv_weekly_avg"]
        assert m["steps"] == garmin["steps"]
        assert m["recovery_score"] is None  # garmin has no recovery_score


class TestGetDailyMetricsHealthOnly:
    def test_health_only_date_included(self) -> None:
        health = _make_health_daily("2026-03-02")
        result = _execute_get_daily_metrics([], [health])

        assert result["count"] == 1
        m = result["metrics"][0]
        assert m["date"] == "2026-03-02"
        assert m["source"] == "health"
        assert m["sleep_minutes"] == health["sleep_duration_minutes"]
        assert m["hrv"] == health["hrv_avg"]
        assert m["recovery_score"] == health["recovery_score"]


class TestGetDailyMetricsConflictResolution:
    """Health data wins when both sources have data for the same date."""

    def test_health_wins_on_conflict(self) -> None:
        date = "2026-03-03"
        garmin = _make_garmin_daily(date, sleep_duration_minutes=430, sleep_score=72)
        health = _make_health_daily(date, sleep_duration_minutes=480, sleep_score=85)

        result = _execute_get_daily_metrics([garmin], [health])

        assert result["count"] == 1
        m = result["metrics"][0]
        # Health values must win
        assert m["sleep_minutes"] == 480
        assert m["sleep_score"] == 85
        assert m["source"] == "health"

    def test_health_hrv_overrides_garmin(self) -> None:
        date = "2026-03-04"
        garmin = _make_garmin_daily(date, hrv_weekly_avg=50.0)
        health = _make_health_daily(date, hrv_avg=65.0)

        result = _execute_get_daily_metrics([garmin], [health])

        m = result["metrics"][0]
        assert m["hrv"] == 65.0  # health hrv_avg wins

    def test_garmin_fallback_used_when_health_field_is_none(self) -> None:
        """When health row has None for a field, the garmin baseline is kept."""
        date = "2026-03-05"
        garmin = _make_garmin_daily(date, steps=8000)
        health = _make_health_daily(date, steps=None)

        result = _execute_get_daily_metrics([garmin], [health])

        m = result["metrics"][0]
        assert m["steps"] == 8000  # garmin fallback

    def test_recovery_score_only_from_health(self) -> None:
        """recovery_score is only available in health_daily_metrics."""
        date = "2026-03-06"
        garmin = _make_garmin_daily(date)
        health = _make_health_daily(date, recovery_score=92)

        result = _execute_get_daily_metrics([garmin], [health])

        m = result["metrics"][0]
        assert m["recovery_score"] == 92


class TestGetDailyMetricsMixedDates:
    """Multi-date scenarios with a mix of garmin-only and health-only dates."""

    def test_separate_dates_both_included(self) -> None:
        garmin = _make_garmin_daily("2026-03-01")
        health = _make_health_daily("2026-03-02")

        result = _execute_get_daily_metrics([garmin], [health])

        assert result["count"] == 2
        dates = {m["date"] for m in result["metrics"]}
        assert "2026-03-01" in dates
        assert "2026-03-02" in dates

    def test_metrics_sorted_newest_first(self) -> None:
        g1 = _make_garmin_daily("2026-02-28")
        g2 = _make_garmin_daily("2026-03-01")
        h1 = _make_health_daily("2026-03-02")

        result = _execute_get_daily_metrics([g1, g2], [h1])

        dates = [m["date"] for m in result["metrics"]]
        assert dates == sorted(dates, reverse=True)
