"""Tests for rewritten analysis tools -- analyze_training_load() and
compare_plan_vs_actual().

Covers:
- analyze_training_load: aggregated multi-source data, no_data status,
  sessions_by_sport, data_sources dict, absence of deprecated detail fields
- compare_plan_vs_actual: Supabase-backed plan lookup (no file I/O),
  no_plan status, no_activities status, compliance rate math,
  health activities included, planned_by_sport and actual_by_sport dicts
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.agent.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

USER_ID = "test-user-analysis-001"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(user_id: str = USER_ID) -> MagicMock:
    s = MagicMock()
    s.agenticsports_user_id = user_id
    return s


def _make_load_summary(
    total_sessions: int = 10,
    total_minutes: float = 480.0,
    total_trimp: float = 650.0,
    sports_seen: list[str] | None = None,
    sessions_by_sport: dict | None = None,
    sessions_by_source: dict | None = None,
) -> dict:
    return {
        "total_sessions": total_sessions,
        "total_minutes": total_minutes,
        "total_trimp": total_trimp,
        "sports_seen": sports_seen or ["running", "cycling"],
        "sessions_by_sport": sessions_by_sport or {"running": 6, "cycling": 4},
        "sessions_by_source": sessions_by_source or {"agent": 5, "health": 3, "garmin": 2},
    }


def _register_analysis_tools() -> ToolRegistry:
    registry = ToolRegistry()
    from src.agent.tools.analysis_tools import register_analysis_tools

    register_analysis_tools(registry)
    return registry


def _execute_training_load(summary: dict, period_days: int = 28) -> dict:
    registry = _register_analysis_tools()
    with (
        patch("src.config.get_settings", return_value=_make_settings()),
        patch(
            "src.db.health_data_db.get_cross_source_load_summary",
            return_value=summary,
        ),
    ):
        return registry.execute("analyze_training_load", {"period_days": period_days})


def _make_plan_row(sessions: list[dict] | None = None) -> dict:
    """Return a fake plan DB row with the given sessions list."""
    return {
        "id": "plan-abc-123",
        "user_id": USER_ID,
        "active": True,
        "plan_data": {
            "sessions": sessions if sessions is not None else [
                {"sport": "running", "day": "Monday"},
                {"sport": "running", "day": "Wednesday"},
                {"sport": "cycling", "day": "Saturday"},
            ]
        },
    }


def _make_agent_activity(sport: str = "running", start_time: str = "2026-03-03T07:00:00") -> dict:
    return {
        "sport": sport,
        "start_time": start_time,
        "duration_seconds": 3600,
        "distance_meters": 10000.0,
        "source": "agent",
    }


def _make_health_activity(
    activity_type: str = "cycling", start_time: str = "2026-03-04T09:00:00"
) -> dict:
    return {
        "activity_type": activity_type,
        "start_time": start_time,
        "duration_seconds": 5400,
        "distance_meters": 40000.0,
    }


def _execute_compare(
    plan_row: dict | None,
    agent_acts: list[dict],
    health_acts: list[dict],
) -> dict:
    registry = _register_analysis_tools()
    with (
        patch("src.config.get_settings", return_value=_make_settings()),
        patch("src.db.plans_db.get_active_plan", return_value=plan_row),
        patch("src.db.activity_store_db.list_activities", return_value=agent_acts),
        patch("src.db.health_data_db.list_health_activities", return_value=health_acts),
    ):
        return registry.execute("compare_plan_vs_actual", {})


# ---------------------------------------------------------------------------
# Registration tests
# ---------------------------------------------------------------------------


class TestAnalysisToolsRegistration:
    def test_analyze_training_load_registered(self) -> None:
        registry = _register_analysis_tools()
        names = [t["function"]["name"] for t in registry.get_openai_tools()]
        assert "analyze_training_load" in names

    def test_compare_plan_vs_actual_registered(self) -> None:
        registry = _register_analysis_tools()
        names = [t["function"]["name"] for t in registry.get_openai_tools()]
        assert "compare_plan_vs_actual" in names

    def test_both_tools_in_analysis_category(self) -> None:
        registry = _register_analysis_tools()
        listed = registry.list_tools()
        analysis_tools = [
            t for t in listed
            if t["name"] in ("analyze_training_load", "compare_plan_vs_actual")
        ]
        assert all(t["category"] == "analysis" for t in analysis_tools)


# ---------------------------------------------------------------------------
# analyze_training_load — core behavior
# ---------------------------------------------------------------------------


class TestAnalyzeTrainingLoadNoData:
    def test_returns_no_data_status_when_zero_sessions(self) -> None:
        summary = _make_load_summary(total_sessions=0, total_minutes=0.0, total_trimp=0.0)
        result = _execute_training_load(summary)

        assert result["status"] == "no_data"
        assert "message" in result
        assert "recommendation" in result

    def test_no_data_does_not_include_totals(self) -> None:
        summary = _make_load_summary(total_sessions=0, total_minutes=0.0, total_trimp=0.0)
        result = _execute_training_load(summary)

        assert "total_sessions" not in result
        assert "sessions_per_week" not in result


class TestAnalyzeTrainingLoadAggregation:
    def test_returns_total_sessions(self) -> None:
        result = _execute_training_load(_make_load_summary(total_sessions=12))
        assert result["total_sessions"] == 12

    def test_sessions_per_week_calculated(self) -> None:
        # 28 days = 4 weeks, 12 sessions → 3.0 per week
        result = _execute_training_load(_make_load_summary(total_sessions=12), period_days=28)
        assert result["sessions_per_week"] == 3.0

    def test_minutes_per_week_calculated(self) -> None:
        # 28 days = 4 weeks, 480 minutes → 120 per week
        result = _execute_training_load(_make_load_summary(total_minutes=480.0), period_days=28)
        assert result["minutes_per_week"] == 120

    def test_trimp_per_week_calculated(self) -> None:
        # 28 days = 4 weeks, 800 trimp → 200 per week
        result = _execute_training_load(_make_load_summary(total_trimp=800.0), period_days=28)
        assert result["trimp_per_week"] == 200

    def test_period_days_returned(self) -> None:
        result = _execute_training_load(_make_load_summary(), period_days=14)
        assert result["period_days"] == 14


class TestAnalyzeTrainingLoadSportBreakdown:
    def test_sessions_by_sport_returned(self) -> None:
        summary = _make_load_summary(sessions_by_sport={"running": 7, "swimming": 3})
        result = _execute_training_load(summary)

        assert result["sessions_by_sport"] == {"running": 7, "swimming": 3}

    def test_sports_list_returned(self) -> None:
        summary = _make_load_summary(sports_seen=["cycling", "running", "swimming"])
        result = _execute_training_load(summary)

        assert result["sports"] == ["cycling", "running", "swimming"]

    def test_dynamically_discovered_sports(self) -> None:
        """Sports list is whatever the DB layer returns — no hardcoded list."""
        summary = _make_load_summary(
            sports_seen=["padel", "yoga", "kitesurfing"],
            sessions_by_sport={"padel": 3, "yoga": 2, "kitesurfing": 1},
        )
        result = _execute_training_load(summary)

        assert "padel" in result["sports"]
        assert "yoga" in result["sports"]
        assert "kitesurfing" in result["sports"]


class TestAnalyzeTrainingLoadDataSources:
    def test_data_sources_dict_returned(self) -> None:
        summary = _make_load_summary(
            sessions_by_source={"agent": 4, "health": 3, "garmin": 5}
        )
        result = _execute_training_load(summary)

        assert result["data_sources"] == {"agent": 4, "health": 3, "garmin": 5}

    def test_no_deprecated_detailed_context_field(self) -> None:
        """The old 'detailed_context' field must not appear in the response."""
        result = _execute_training_load(_make_load_summary())
        assert "detailed_context" not in result


class TestAnalyzeTrainingLoadMissingUserId:
    def test_missing_user_id_returns_error(self) -> None:
        settings = _make_settings(user_id="")
        registry = _register_analysis_tools()
        with (
            patch("src.config.get_settings", return_value=settings),
            patch(
                "src.db.health_data_db.get_cross_source_load_summary",
                return_value=_make_load_summary(),
            ),
        ):
            result = registry.execute("analyze_training_load", {})

        assert result["status"] == "error"
        assert "user_id" in result["message"]


# ---------------------------------------------------------------------------
# compare_plan_vs_actual — plan loading from Supabase
# ---------------------------------------------------------------------------


class TestComparePlanVsActualPlanLookup:
    def test_no_active_plan_returns_no_plan_status(self) -> None:
        result = _execute_compare(plan_row=None, agent_acts=[], health_acts=[])
        assert result["status"] == "no_plan"

    def test_plan_with_empty_sessions_returns_no_plan_status(self) -> None:
        plan = _make_plan_row(sessions=[])
        result = _execute_compare(plan_row=plan, agent_acts=[], health_acts=[])
        assert result["status"] == "no_plan"

    def test_reads_plan_from_supabase_not_file_system(self) -> None:
        """Verify that get_active_plan (Supabase) is called, not any file I/O.

        If this test passes, the implementation is correctly using the DB layer.
        The mock intercepts the DB call; if file I/O were used instead the mock
        would not prevent a FileNotFoundError.
        """
        plan = _make_plan_row()
        agent_act = _make_agent_activity()
        result = _execute_compare(plan_row=plan, agent_acts=[agent_act], health_acts=[])

        assert result["status"] == "ok"  # plan was found via DB mock


# ---------------------------------------------------------------------------
# compare_plan_vs_actual — no activities
# ---------------------------------------------------------------------------


class TestComparePlanVsActualNoActivities:
    def test_no_activities_returns_no_activities_status(self) -> None:
        plan = _make_plan_row()
        result = _execute_compare(plan_row=plan, agent_acts=[], health_acts=[])
        assert result["status"] == "no_activities"


# ---------------------------------------------------------------------------
# compare_plan_vs_actual — compliance rate
# ---------------------------------------------------------------------------


class TestComparePlanVsActualCompliance:
    def test_full_compliance_100_pct(self) -> None:
        """3 planned sessions, 3 actual → 100% compliance."""
        plan = _make_plan_row(sessions=[
            {"sport": "running"},
            {"sport": "running"},
            {"sport": "cycling"},
        ])
        agent_acts = [
            _make_agent_activity("running"),
            _make_agent_activity("running"),
            _make_agent_activity("cycling"),
        ]
        result = _execute_compare(plan_row=plan, agent_acts=agent_acts, health_acts=[])

        assert result["status"] == "ok"
        assert result["compliance_rate_pct"] == 100

    def test_partial_compliance(self) -> None:
        """4 planned sessions, 2 actual → 50% compliance."""
        plan = _make_plan_row(sessions=[
            {"sport": "running"},
            {"sport": "running"},
            {"sport": "cycling"},
            {"sport": "swimming"},
        ])
        agent_acts = [
            _make_agent_activity("running"),
            _make_agent_activity("cycling"),
        ]
        result = _execute_compare(plan_row=plan, agent_acts=agent_acts, health_acts=[])

        assert result["compliance_rate_pct"] == 50

    def test_compliance_capped_at_100_pct(self) -> None:
        """More actual sessions than planned must cap at 100%."""
        plan = _make_plan_row(sessions=[{"sport": "running"}])
        agent_acts = [
            _make_agent_activity("running"),
            _make_agent_activity("running"),
            _make_agent_activity("running"),
        ]
        result = _execute_compare(plan_row=plan, agent_acts=agent_acts, health_acts=[])

        assert result["compliance_rate_pct"] == 100

    def test_planned_and_actual_session_counts_returned(self) -> None:
        plan = _make_plan_row(sessions=[{"sport": "running"}, {"sport": "cycling"}])
        agent_acts = [_make_agent_activity("running")]

        result = _execute_compare(plan_row=plan, agent_acts=agent_acts, health_acts=[])

        assert result["planned_sessions"] == 2
        assert result["actual_sessions"] == 1


# ---------------------------------------------------------------------------
# compare_plan_vs_actual — health activities included
# ---------------------------------------------------------------------------


class TestComparePlanVsActualHealthActivities:
    def test_health_activities_counted_in_actual(self) -> None:
        """Health activities from health_activities table must be included."""
        plan = _make_plan_row(sessions=[
            {"sport": "running"},
            {"sport": "cycling"},
        ])
        agent_acts = [_make_agent_activity("running")]
        health_acts = [_make_health_activity("cycling")]

        result = _execute_compare(plan_row=plan, agent_acts=agent_acts, health_acts=health_acts)

        assert result["status"] == "ok"
        assert result["actual_sessions"] == 2  # 1 agent + 1 health

    def test_health_activities_in_actual_by_sport(self) -> None:
        plan = _make_plan_row(sessions=[{"sport": "cycling"}])
        health_acts = [_make_health_activity("cycling")]

        result = _execute_compare(plan_row=plan, agent_acts=[], health_acts=health_acts)

        assert "cycling" in result["actual_by_sport"]
        assert result["actual_by_sport"]["cycling"] == 1

    def test_data_sources_lists_health_when_present(self) -> None:
        plan = _make_plan_row(sessions=[{"sport": "running"}])
        health_acts = [_make_health_activity("running")]

        result = _execute_compare(plan_row=plan, agent_acts=[], health_acts=health_acts)

        assert "health" in result["data_sources"]

    def test_data_sources_lists_agent_when_present(self) -> None:
        plan = _make_plan_row(sessions=[{"sport": "running"}])
        agent_acts = [_make_agent_activity("running")]

        result = _execute_compare(plan_row=plan, agent_acts=agent_acts, health_acts=[])

        assert "agent" in result["data_sources"]


# ---------------------------------------------------------------------------
# compare_plan_vs_actual — sport breakdowns
# ---------------------------------------------------------------------------


class TestComparePlanVsActualSportBreakdowns:
    def test_planned_by_sport_aggregated_correctly(self) -> None:
        plan = _make_plan_row(sessions=[
            {"sport": "running"},
            {"sport": "running"},
            {"sport": "cycling"},
        ])
        agent_acts = [_make_agent_activity("running")]

        result = _execute_compare(plan_row=plan, agent_acts=agent_acts, health_acts=[])

        assert result["planned_by_sport"]["running"] == 2
        assert result["planned_by_sport"]["cycling"] == 1

    def test_actual_by_sport_aggregated_correctly(self) -> None:
        plan = _make_plan_row(sessions=[{"sport": "running"}])
        agent_acts = [
            _make_agent_activity("running"),
            _make_agent_activity("running"),
        ]

        result = _execute_compare(plan_row=plan, agent_acts=agent_acts, health_acts=[])

        assert result["actual_by_sport"]["running"] == 2

    def test_plan_supports_type_field_alias(self) -> None:
        """Plan sessions may use 'type' instead of 'sport'."""
        plan = {
            "id": "plan-xyz",
            "user_id": USER_ID,
            "active": True,
            "plan_data": {
                "sessions": [{"type": "swimming"}, {"type": "running"}]
            },
        }
        agent_acts = [_make_agent_activity("swimming")]

        result = _execute_compare(plan_row=plan, agent_acts=agent_acts, health_acts=[])

        assert "swimming" in result["planned_by_sport"]
        assert "running" in result["planned_by_sport"]

    def test_plan_supports_weekly_sessions_key(self) -> None:
        """plan_data may use 'weekly_sessions' instead of 'sessions'."""
        plan = {
            "id": "plan-xyz",
            "user_id": USER_ID,
            "active": True,
            "plan_data": {
                "weekly_sessions": [{"sport": "yoga"}, {"sport": "yoga"}]
            },
        }
        agent_acts = [_make_agent_activity("yoga")]

        result = _execute_compare(plan_row=plan, agent_acts=agent_acts, health_acts=[])

        assert result["planned_by_sport"]["yoga"] == 2

    def test_missing_user_id_returns_error(self) -> None:
        settings = _make_settings(user_id="")
        registry = _register_analysis_tools()
        with (
            patch("src.config.get_settings", return_value=settings),
            patch("src.db.plans_db.get_active_plan", return_value=_make_plan_row()),
            patch("src.db.activity_store_db.list_activities", return_value=[]),
            patch("src.db.health_data_db.list_health_activities", return_value=[]),
        ):
            result = registry.execute("compare_plan_vs_actual", {})

        assert result["status"] == "error"
        assert "user_id" in result["message"]
