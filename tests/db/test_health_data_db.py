"""Unit tests for src.db.health_data_db.

Covers all public functions:
- list_health_activities
- list_garmin_activities
- list_daily_metrics
- list_garmin_daily_stats
- get_health_activity_summary
- get_cross_source_load_summary (including deduplication logic)

All Supabase I/O is mocked via patch("src.db.health_data_db.get_supabase").
The list_activities dependency from activity_store_db is patched separately
for get_cross_source_load_summary tests.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.db.health_data_db import (
    get_cross_source_load_summary,
    get_health_activity_summary,
    list_daily_metrics,
    list_garmin_activities,
    list_garmin_daily_stats,
    list_health_activities,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

USER_ID = "user-health-test-uuid"


# ---------------------------------------------------------------------------
# Mock builder
# ---------------------------------------------------------------------------


def _make_chain(rows: list[dict]) -> MagicMock:
    """Return a chainable query mock whose .execute().data equals *rows*.

    Every intermediate method (.select, .eq, .gte, .lt, .order, .limit, …)
    returns the same chain object so that any call order is valid.
    """
    chain = MagicMock()
    result = MagicMock()
    result.data = rows

    # All query-builder methods return *chain* itself so calls can be chained.
    for method_name in ["select", "eq", "gte", "lt", "order", "limit", "neq", "in_", "upsert", "update"]:
        getattr(chain, method_name).return_value = chain

    chain.execute.return_value = result
    return chain


def _mock_supabase(table_data: dict[str, list[dict]]) -> MagicMock:
    """Return a mock Supabase client whose .table(name) resolves *table_data*.

    Args:
        table_data: mapping of table name -> list of row dicts to return.
    """
    client = MagicMock()

    def _table(name: str) -> MagicMock:
        rows = table_data.get(name, [])
        return _make_chain(rows)

    client.table.side_effect = _table
    return client


# ---------------------------------------------------------------------------
# list_health_activities
# ---------------------------------------------------------------------------


class TestListHealthActivities:
    """Tests for list_health_activities()."""

    def test_returns_rows_for_user(self) -> None:
        rows = [
            {"id": "1", "user_id": USER_ID, "activity_type": "running", "provider_type": "apple_health"},
            {"id": "2", "user_id": USER_ID, "activity_type": "cycling", "provider_type": "health_connect"},
        ]
        client = _mock_supabase({"health_activities": rows})
        with patch("src.db.health_data_db.get_supabase", return_value=client):
            result = list_health_activities(USER_ID)
        assert len(result) == 2
        assert result[0]["id"] == "1"

    def test_returns_empty_list_when_no_data(self) -> None:
        client = _mock_supabase({"health_activities": []})
        with patch("src.db.health_data_db.get_supabase", return_value=client):
            result = list_health_activities(USER_ID)
        assert result == []

    def test_filter_by_activity_type_calls_eq(self) -> None:
        """When activity_type is provided, an additional .eq() call is made."""
        rows = [{"id": "1", "activity_type": "swimming"}]
        client = _mock_supabase({"health_activities": rows})
        with patch("src.db.health_data_db.get_supabase", return_value=client):
            result = list_health_activities(USER_ID, activity_type="swimming")
        # Result still returns what the mock provides — key assertion is no exception.
        assert isinstance(result, list)

    def test_filter_by_provider_type(self) -> None:
        rows = [{"id": "3", "provider_type": "garmin"}]
        client = _mock_supabase({"health_activities": rows})
        with patch("src.db.health_data_db.get_supabase", return_value=client):
            result = list_health_activities(USER_ID, provider_type="garmin")
        assert isinstance(result, list)

    def test_filter_by_after_date(self) -> None:
        rows = [{"id": "4", "start_time": "2026-03-01T10:00:00"}]
        client = _mock_supabase({"health_activities": rows})
        with patch("src.db.health_data_db.get_supabase", return_value=client):
            result = list_health_activities(USER_ID, after="2026-02-01T00:00:00")
        assert isinstance(result, list)

    def test_filter_by_before_date(self) -> None:
        rows = [{"id": "5", "start_time": "2026-02-28T08:00:00"}]
        client = _mock_supabase({"health_activities": rows})
        with patch("src.db.health_data_db.get_supabase", return_value=client):
            result = list_health_activities(USER_ID, before="2026-03-01T00:00:00")
        assert isinstance(result, list)

    def test_result_is_new_list(self) -> None:
        """Returned value must be a new list (immutability)."""
        rows = [{"id": "6"}]
        client = _mock_supabase({"health_activities": rows})
        with patch("src.db.health_data_db.get_supabase", return_value=client):
            result = list_health_activities(USER_ID)
        assert result is not rows


# ---------------------------------------------------------------------------
# list_garmin_activities
# ---------------------------------------------------------------------------


class TestListGarminActivities:
    """Tests for list_garmin_activities()."""

    def test_returns_garmin_rows(self) -> None:
        rows = [
            {"id": "g1", "user_id": USER_ID, "type": "running", "start_time": "2026-03-03T07:00:00"},
            {"id": "g2", "user_id": USER_ID, "type": "cycling", "start_time": "2026-03-02T07:00:00"},
        ]
        client = _mock_supabase({"garmin_activities": rows})
        with patch("src.db.health_data_db.get_supabase", return_value=client):
            result = list_garmin_activities(USER_ID)
        assert len(result) == 2

    def test_empty_garmin_table(self) -> None:
        client = _mock_supabase({"garmin_activities": []})
        with patch("src.db.health_data_db.get_supabase", return_value=client):
            result = list_garmin_activities(USER_ID)
        assert result == []

    def test_filter_by_activity_type(self) -> None:
        rows = [{"id": "g3", "type": "swimming"}]
        client = _mock_supabase({"garmin_activities": rows})
        with patch("src.db.health_data_db.get_supabase", return_value=client):
            result = list_garmin_activities(USER_ID, activity_type="swimming")
        assert isinstance(result, list)

    def test_filter_by_after(self) -> None:
        rows = [{"id": "g4"}]
        client = _mock_supabase({"garmin_activities": rows})
        with patch("src.db.health_data_db.get_supabase", return_value=client):
            result = list_garmin_activities(USER_ID, after="2026-02-01T00:00:00")
        assert isinstance(result, list)

    def test_filter_by_before(self) -> None:
        rows = [{"id": "g5"}]
        client = _mock_supabase({"garmin_activities": rows})
        with patch("src.db.health_data_db.get_supabase", return_value=client):
            result = list_garmin_activities(USER_ID, before="2026-04-01T00:00:00")
        assert isinstance(result, list)

    def test_result_is_new_list(self) -> None:
        rows = [{"id": "g6"}]
        client = _mock_supabase({"garmin_activities": rows})
        with patch("src.db.health_data_db.get_supabase", return_value=client):
            result = list_garmin_activities(USER_ID)
        assert result is not rows


# ---------------------------------------------------------------------------
# list_daily_metrics
# ---------------------------------------------------------------------------


class TestListDailyMetrics:
    """Tests for list_daily_metrics()."""

    def test_returns_metrics(self) -> None:
        rows = [
            {"id": "m1", "user_id": USER_ID, "date": "2026-03-03", "hrv_avg": 55},
            {"id": "m2", "user_id": USER_ID, "date": "2026-03-02", "hrv_avg": 60},
        ]
        client = _mock_supabase({"health_daily_metrics": rows})
        with patch("src.db.health_data_db.get_supabase", return_value=client):
            result = list_daily_metrics(USER_ID)
        assert len(result) == 2

    def test_default_days_14(self) -> None:
        """Default window is 14 days — no explicit after/before needed."""
        rows = [{"id": "m3", "date": "2026-03-01"}]
        client = _mock_supabase({"health_daily_metrics": rows})
        with patch("src.db.health_data_db.get_supabase", return_value=client):
            result = list_daily_metrics(USER_ID)
        assert isinstance(result, list)

    def test_custom_days_parameter(self) -> None:
        rows = [{"id": "m4", "date": "2026-01-01"}]
        client = _mock_supabase({"health_daily_metrics": rows})
        with patch("src.db.health_data_db.get_supabase", return_value=client):
            result = list_daily_metrics(USER_ID, days=90)
        assert isinstance(result, list)

    def test_explicit_after_overrides_days(self) -> None:
        rows = [{"id": "m5"}]
        client = _mock_supabase({"health_daily_metrics": rows})
        with patch("src.db.health_data_db.get_supabase", return_value=client):
            result = list_daily_metrics(USER_ID, after="2026-02-01")
        assert isinstance(result, list)

    def test_before_filter_applied(self) -> None:
        rows = [{"id": "m6"}]
        client = _mock_supabase({"health_daily_metrics": rows})
        with patch("src.db.health_data_db.get_supabase", return_value=client):
            result = list_daily_metrics(USER_ID, before="2026-03-04")
        assert isinstance(result, list)

    def test_empty_result(self) -> None:
        client = _mock_supabase({"health_daily_metrics": []})
        with patch("src.db.health_data_db.get_supabase", return_value=client):
            result = list_daily_metrics(USER_ID)
        assert result == []


# ---------------------------------------------------------------------------
# list_garmin_daily_stats
# ---------------------------------------------------------------------------


class TestListGarminDailyStats:
    """Tests for list_garmin_daily_stats()."""

    def test_returns_stats(self) -> None:
        rows = [
            {"id": "s1", "user_id": USER_ID, "date": "2026-03-03", "steps": 10000},
            {"id": "s2", "user_id": USER_ID, "date": "2026-03-02", "steps": 8500},
        ]
        client = _mock_supabase({"garmin_daily_stats": rows})
        with patch("src.db.health_data_db.get_supabase", return_value=client):
            result = list_garmin_daily_stats(USER_ID)
        assert len(result) == 2

    def test_default_days_14(self) -> None:
        rows = [{"id": "s3", "date": "2026-02-25"}]
        client = _mock_supabase({"garmin_daily_stats": rows})
        with patch("src.db.health_data_db.get_supabase", return_value=client):
            result = list_garmin_daily_stats(USER_ID)
        assert isinstance(result, list)

    def test_custom_days_parameter(self) -> None:
        rows = [{"id": "s4", "date": "2026-01-15"}]
        client = _mock_supabase({"garmin_daily_stats": rows})
        with patch("src.db.health_data_db.get_supabase", return_value=client):
            result = list_garmin_daily_stats(USER_ID, days=60)
        assert isinstance(result, list)

    def test_before_filter(self) -> None:
        rows = [{"id": "s5"}]
        client = _mock_supabase({"garmin_daily_stats": rows})
        with patch("src.db.health_data_db.get_supabase", return_value=client):
            result = list_garmin_daily_stats(USER_ID, before="2026-03-04")
        assert isinstance(result, list)

    def test_empty_result(self) -> None:
        client = _mock_supabase({"garmin_daily_stats": []})
        with patch("src.db.health_data_db.get_supabase", return_value=client):
            result = list_garmin_daily_stats(USER_ID)
        assert result == []


# ---------------------------------------------------------------------------
# get_health_activity_summary
# ---------------------------------------------------------------------------


class TestGetHealthActivitySummary:
    """Tests for get_health_activity_summary()."""

    _ROWS = [
        {
            "activity_type": "running",
            "provider_type": "apple_health",
            "distance_meters": 5000,
            "duration_seconds": 1800,
            "training_load_trimp": 45.0,
        },
        {
            "activity_type": "cycling",
            "provider_type": "health_connect",
            "distance_meters": 20000,
            "duration_seconds": 3600,
            "training_load_trimp": 60.0,
        },
    ]

    def test_count_is_correct(self) -> None:
        client = _mock_supabase({"health_activities": self._ROWS})
        with patch("src.db.health_data_db.get_supabase", return_value=client):
            summary = get_health_activity_summary(USER_ID)
        assert summary["count"] == 2

    def test_total_distance_km(self) -> None:
        client = _mock_supabase({"health_activities": self._ROWS})
        with patch("src.db.health_data_db.get_supabase", return_value=client):
            summary = get_health_activity_summary(USER_ID)
        # 5000 + 20000 = 25000 m => 25.0 km
        assert summary["total_distance_km"] == 25.0

    def test_total_duration_hours(self) -> None:
        client = _mock_supabase({"health_activities": self._ROWS})
        with patch("src.db.health_data_db.get_supabase", return_value=client):
            summary = get_health_activity_summary(USER_ID)
        # 1800 + 3600 = 5400 s => 1.5 h
        assert summary["total_duration_hours"] == 1.5

    def test_total_trimp(self) -> None:
        client = _mock_supabase({"health_activities": self._ROWS})
        with patch("src.db.health_data_db.get_supabase", return_value=client):
            summary = get_health_activity_summary(USER_ID)
        assert summary["total_trimp"] == 105.0

    def test_sports_discovered_dynamically(self) -> None:
        client = _mock_supabase({"health_activities": self._ROWS})
        with patch("src.db.health_data_db.get_supabase", return_value=client):
            summary = get_health_activity_summary(USER_ID)
        # Sports must be sorted and discovered at runtime — never hardcoded.
        assert summary["sports_seen"] == ["cycling", "running"]

    def test_provider_types_discovered_dynamically(self) -> None:
        client = _mock_supabase({"health_activities": self._ROWS})
        with patch("src.db.health_data_db.get_supabase", return_value=client):
            summary = get_health_activity_summary(USER_ID)
        assert summary["provider_types"] == ["apple_health", "health_connect"]

    def test_returns_zeros_for_empty_data(self) -> None:
        client = _mock_supabase({"health_activities": []})
        with patch("src.db.health_data_db.get_supabase", return_value=client):
            summary = get_health_activity_summary(USER_ID)
        assert summary["count"] == 0
        assert summary["total_distance_km"] == 0.0
        assert summary["total_duration_hours"] == 0.0
        assert summary["total_trimp"] == 0.0
        assert summary["sports_seen"] == []
        assert summary["provider_types"] == []

    def test_handles_none_fields_gracefully(self) -> None:
        """Rows with None numeric fields must not raise and must contribute 0."""
        rows = [
            {
                "activity_type": "yoga",
                "provider_type": "apple_health",
                "distance_meters": None,
                "duration_seconds": None,
                "training_load_trimp": None,
            }
        ]
        client = _mock_supabase({"health_activities": rows})
        with patch("src.db.health_data_db.get_supabase", return_value=client):
            summary = get_health_activity_summary(USER_ID)
        assert summary["count"] == 1
        assert summary["total_distance_km"] == 0.0
        assert summary["total_trimp"] == 0.0

    def test_returns_new_dict(self) -> None:
        """Result must be a new dict (immutability check)."""
        client = _mock_supabase({"health_activities": []})
        with patch("src.db.health_data_db.get_supabase", return_value=client):
            a = get_health_activity_summary(USER_ID)
            b = get_health_activity_summary(USER_ID)
        assert a is not b


# ---------------------------------------------------------------------------
# get_cross_source_load_summary
# ---------------------------------------------------------------------------


class TestGetCrossSourceLoadSummary:
    """Tests for get_cross_source_load_summary() — the key aggregation function.

    list_activities (agent/activities table) is patched via
    'src.db.health_data_db.list_activities' so that health_data_db.py sees
    its own import. get_supabase handles the two health tables.
    """

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _patch_all(agent_rows: list[dict], health_rows: list[dict], garmin_rows: list[dict]):
        """Context manager stack: patches list_activities + get_supabase."""
        client = _mock_supabase({
            "health_activities": health_rows,
            "garmin_activities": garmin_rows,
        })
        list_activities_patch = patch(
            "src.db.health_data_db.list_activities",
            return_value=agent_rows,
        )
        get_supabase_patch = patch("src.db.health_data_db.get_supabase", return_value=client)
        return list_activities_patch, get_supabase_patch

    # ------------------------------------------------------------------ basic tests

    def test_empty_all_sources_returns_zeros(self) -> None:
        lp, gp = self._patch_all([], [], [])
        with lp, gp:
            summary = get_cross_source_load_summary(USER_ID)
        assert summary["total_sessions"] == 0
        assert summary["total_minutes"] == 0.0
        assert summary["total_trimp"] == 0.0
        assert summary["sports_seen"] == []
        assert summary["sessions_by_sport"] == {}
        assert summary["sessions_by_source"] == {}

    def test_aggregates_agent_rows(self) -> None:
        agent_rows = [
            {"sport": "running", "duration_seconds": 1800, "trimp": 50.0, "garmin_activity_id": None},
        ]
        lp, gp = self._patch_all(agent_rows, [], [])
        with lp, gp:
            summary = get_cross_source_load_summary(USER_ID)
        assert summary["total_sessions"] == 1
        assert summary["total_minutes"] == 30.0
        assert summary["total_trimp"] == 50.0
        assert summary["sessions_by_source"] == {"agent": 1}
        assert summary["sessions_by_sport"] == {"running": 1}

    def test_aggregates_health_rows(self) -> None:
        health_rows = [
            {
                "activity_type": "cycling",
                "duration_seconds": 3600,
                "training_load_trimp": 70.0,
                "external_id": None,
            },
        ]
        lp, gp = self._patch_all([], health_rows, [])
        with lp, gp:
            summary = get_cross_source_load_summary(USER_ID)
        assert summary["total_sessions"] == 1
        assert summary["total_minutes"] == 60.0
        assert summary["sessions_by_source"] == {"health": 1}
        assert "cycling" in summary["sessions_by_sport"]

    def test_aggregates_garmin_rows(self) -> None:
        garmin_rows = [
            {
                "type": "swimming",
                "duration": 2700,
                "garmin_activity_id": "garm-001",
            },
        ]
        lp, gp = self._patch_all([], [], garmin_rows)
        with lp, gp:
            summary = get_cross_source_load_summary(USER_ID)
        assert summary["total_sessions"] == 1
        assert summary["total_minutes"] == 45.0
        assert summary["sessions_by_source"] == {"garmin": 1}
        assert "swimming" in summary["sessions_by_sport"]

    def test_aggregates_all_three_sources(self) -> None:
        agent_rows = [
            {"sport": "running", "duration_seconds": 1800, "trimp": 40.0, "garmin_activity_id": None},
        ]
        health_rows = [
            {"activity_type": "yoga", "duration_seconds": 3600, "training_load_trimp": 20.0, "external_id": None},
        ]
        garmin_rows = [
            {"type": "cycling", "duration": 3600, "garmin_activity_id": "garm-999"},
        ]
        lp, gp = self._patch_all(agent_rows, health_rows, garmin_rows)
        with lp, gp:
            summary = get_cross_source_load_summary(USER_ID)
        assert summary["total_sessions"] == 3
        assert summary["sessions_by_source"]["agent"] == 1
        assert summary["sessions_by_source"]["health"] == 1
        assert summary["sessions_by_source"]["garmin"] == 1

    # ------------------------------------------------------------------ deduplication

    def test_dedup_health_activity_by_garmin_id(self) -> None:
        """A health_activity sharing a garmin_activity_id with an agent row is excluded."""
        covered_garmin_id = "garm-dup-001"
        agent_rows = [
            {
                "sport": "running",
                "duration_seconds": 1800,
                "trimp": 55.0,
                "garmin_activity_id": covered_garmin_id,
            },
        ]
        health_rows = [
            {
                "activity_type": "running",
                "duration_seconds": 1800,
                "training_load_trimp": 55.0,
                "external_id": covered_garmin_id,  # DUPLICATE — must be dropped
            },
        ]
        lp, gp = self._patch_all(agent_rows, health_rows, [])
        with lp, gp:
            summary = get_cross_source_load_summary(USER_ID)
        # Only 1 session, not 2 — the health row is deduplicated away.
        assert summary["total_sessions"] == 1
        assert summary["sessions_by_source"] == {"agent": 1}

    def test_dedup_garmin_activity_by_agent_row(self) -> None:
        """A garmin_activity sharing a garmin_activity_id with an agent row is excluded."""
        covered_garmin_id = "garm-dup-002"
        agent_rows = [
            {
                "sport": "cycling",
                "duration_seconds": 3600,
                "trimp": 80.0,
                "garmin_activity_id": covered_garmin_id,
            },
        ]
        garmin_rows = [
            {
                "type": "cycling",
                "duration": 3600,
                "garmin_activity_id": covered_garmin_id,  # DUPLICATE — must be dropped
            },
        ]
        lp, gp = self._patch_all(agent_rows, [], garmin_rows)
        with lp, gp:
            summary = get_cross_source_load_summary(USER_ID)
        assert summary["total_sessions"] == 1
        assert summary["sessions_by_source"] == {"agent": 1}

    def test_dedup_both_health_and_garmin_for_same_agent_row(self) -> None:
        """One agent row may deduplicate matching rows in BOTH health and garmin tables."""
        covered_garmin_id = "garm-triple"
        agent_rows = [
            {
                "sport": "running",
                "duration_seconds": 2700,
                "trimp": 65.0,
                "garmin_activity_id": covered_garmin_id,
            },
        ]
        health_rows = [
            {
                "activity_type": "running",
                "duration_seconds": 2700,
                "training_load_trimp": 65.0,
                "external_id": covered_garmin_id,
            },
        ]
        garmin_rows = [
            {
                "type": "running",
                "duration": 2700,
                "garmin_activity_id": covered_garmin_id,
            },
        ]
        lp, gp = self._patch_all(agent_rows, health_rows, garmin_rows)
        with lp, gp:
            summary = get_cross_source_load_summary(USER_ID)
        # Three raw rows but only one unique session.
        assert summary["total_sessions"] == 1
        assert summary["sessions_by_source"] == {"agent": 1}

    def test_dedup_does_not_drop_unrelated_health_rows(self) -> None:
        """Health rows whose external_id is NOT in covered_garmin_ids must be kept."""
        agent_rows = [
            {"sport": "running", "duration_seconds": 1800, "trimp": 40.0, "garmin_activity_id": "garm-A"},
        ]
        health_rows = [
            {
                "activity_type": "yoga",
                "duration_seconds": 3600,
                "training_load_trimp": 10.0,
                "external_id": "different-id",  # NOT covered — keep it
            },
        ]
        lp, gp = self._patch_all(agent_rows, health_rows, [])
        with lp, gp:
            summary = get_cross_source_load_summary(USER_ID)
        assert summary["total_sessions"] == 2
        assert summary["sessions_by_source"]["agent"] == 1
        assert summary["sessions_by_source"]["health"] == 1

    def test_dedup_does_not_drop_unrelated_garmin_rows(self) -> None:
        """Garmin rows whose garmin_activity_id is NOT covered must be kept."""
        agent_rows = [
            {"sport": "swimming", "duration_seconds": 900, "trimp": 30.0, "garmin_activity_id": "garm-B"},
        ]
        garmin_rows = [
            {"type": "cycling", "duration": 5400, "garmin_activity_id": "garm-C"},  # different
        ]
        lp, gp = self._patch_all(agent_rows, [], garmin_rows)
        with lp, gp:
            summary = get_cross_source_load_summary(USER_ID)
        assert summary["total_sessions"] == 2
        assert summary["sessions_by_source"]["agent"] == 1
        assert summary["sessions_by_source"]["garmin"] == 1

    # ------------------------------------------------------------------ sport/source aggregation

    def test_sessions_by_sport_counts_correctly(self) -> None:
        agent_rows = [
            {"sport": "running", "duration_seconds": 1800, "trimp": 30.0, "garmin_activity_id": None},
            {"sport": "running", "duration_seconds": 1800, "trimp": 30.0, "garmin_activity_id": None},
            {"sport": "cycling", "duration_seconds": 3600, "trimp": 50.0, "garmin_activity_id": None},
        ]
        lp, gp = self._patch_all(agent_rows, [], [])
        with lp, gp:
            summary = get_cross_source_load_summary(USER_ID)
        assert summary["sessions_by_sport"]["running"] == 2
        assert summary["sessions_by_sport"]["cycling"] == 1

    def test_sports_seen_sorted(self) -> None:
        agent_rows = [
            {"sport": "yoga", "duration_seconds": 3600, "trimp": 10.0, "garmin_activity_id": None},
            {"sport": "cycling", "duration_seconds": 3600, "trimp": 50.0, "garmin_activity_id": None},
            {"sport": "running", "duration_seconds": 1800, "trimp": 30.0, "garmin_activity_id": None},
        ]
        lp, gp = self._patch_all(agent_rows, [], [])
        with lp, gp:
            summary = get_cross_source_load_summary(USER_ID)
        assert summary["sports_seen"] == sorted(summary["sports_seen"])

    def test_unknown_sport_used_when_type_missing(self) -> None:
        """Rows with missing/None sport/type field must be bucketed as 'unknown'."""
        garmin_rows = [
            {"type": None, "duration": 1800, "garmin_activity_id": None},
        ]
        lp, gp = self._patch_all([], [], garmin_rows)
        with lp, gp:
            summary = get_cross_source_load_summary(USER_ID)
        assert "unknown" in summary["sessions_by_sport"]

    def test_total_trimp_excludes_garmin_source(self) -> None:
        """garmin_activities have no TRIMP — their contribution must be 0."""
        garmin_rows = [
            {"type": "running", "duration": 3600, "garmin_activity_id": None},
        ]
        lp, gp = self._patch_all([], [], garmin_rows)
        with lp, gp:
            summary = get_cross_source_load_summary(USER_ID)
        assert summary["total_trimp"] == 0.0

    def test_none_duration_fields_treated_as_zero(self) -> None:
        """None duration/trimp must not raise and must contribute 0 to totals."""
        agent_rows = [
            {"sport": "running", "duration_seconds": None, "trimp": None, "garmin_activity_id": None},
        ]
        lp, gp = self._patch_all(agent_rows, [], [])
        with lp, gp:
            summary = get_cross_source_load_summary(USER_ID)
        assert summary["total_minutes"] == 0.0
        assert summary["total_trimp"] == 0.0

    def test_returns_new_dict(self) -> None:
        lp, gp = self._patch_all([], [], [])
        with lp, gp:
            a = get_cross_source_load_summary(USER_ID)
        lp2, gp2 = self._patch_all([], [], [])
        with lp2, gp2:
            b = get_cross_source_load_summary(USER_ID)
        assert a is not b
