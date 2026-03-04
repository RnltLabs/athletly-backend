"""Analysis tools -- compute insights from training data across all data sources.

These are the equivalent of Claude Code using Bash to run analysis commands.
The agent calls these when it needs computed insights, not raw data.
"""

from src.agent.tools.registry import Tool, ToolRegistry


def register_analysis_tools(registry: ToolRegistry):
    """Register all analysis tools."""

    def analyze_training_load(period_days: int = 28) -> dict:
        """Analyze training load, trends, and recovery status across all sources."""
        from src.config import get_settings
        from src.db.health_data_db import get_cross_source_load_summary

        settings = get_settings()
        user_id = settings.agenticsports_user_id
        if not user_id:
            return {"status": "error", "message": "No user_id configured."}

        summary = get_cross_source_load_summary(user_id, days=period_days)

        if summary["total_sessions"] == 0:
            return {
                "status": "no_data",
                "message": "No training data available from any source.",
                "recommendation": "Start conservative -- gather baseline data first.",
            }

        weeks = max(1, period_days / 7)
        return {
            "period_days": period_days,
            "total_sessions": summary["total_sessions"],
            "sessions_per_week": round(summary["total_sessions"] / weeks, 1),
            "total_minutes": summary["total_minutes"],
            "minutes_per_week": round(summary["total_minutes"] / weeks),
            "total_trimp": summary["total_trimp"],
            "trimp_per_week": round(summary["total_trimp"] / weeks),
            "sports": summary["sports_seen"],
            "sessions_by_sport": summary["sessions_by_sport"],
            "data_sources": summary["sessions_by_source"],
        }

    registry.register(Tool(
        name="analyze_training_load",
        description=(
            "Analyze training load over a period across all data sources (agent, "
            "Apple Health, Garmin): total sessions, weekly averages, TRIMP, sport "
            "breakdown, and source breakdown. Use this before creating a plan or "
            "when the athlete asks about their training. "
            "Returns 'no_data' status if no activities exist."
        ),
        handler=analyze_training_load,
        parameters={
            "type": "object",
            "properties": {
                "period_days": {
                    "type": "integer",
                    "description": "Analysis period in days (default 28)",
                },
            },
        },
        category="analysis",
    ))

    def compare_plan_vs_actual() -> dict:
        """Compare planned vs actual training this week across all data sources."""
        from src.config import get_settings
        from src.db.plans_db import get_active_plan
        from src.db.activity_store_db import list_activities
        from src.db.health_data_db import list_health_activities
        from datetime import datetime, timedelta, timezone

        settings = get_settings()
        user_id = settings.agenticsports_user_id
        if not user_id:
            return {"status": "error", "message": "No user_id configured."}

        plan_row = get_active_plan(user_id)
        if not plan_row:
            return {"status": "no_plan", "message": "No active plan to compare against."}

        plan_data = plan_row.get("plan_data", {})
        sessions = plan_data.get("sessions") or plan_data.get("weekly_sessions") or []
        if not sessions:
            return {"status": "no_plan", "message": "Active plan has no sessions."}

        week_start = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        agent_acts = list_activities(user_id, limit=100, after=week_start)
        health_acts = list_health_activities(user_id, limit=100, after=week_start)

        all_activities = []
        for a in agent_acts:
            all_activities.append({
                "sport": a.get("sport", "unknown"),
                "start_time": a.get("start_time"),
                "duration_seconds": a.get("duration_seconds", 0),
                "distance_meters": a.get("distance_meters"),
                "source": "agent",
            })
        for a in health_acts:
            all_activities.append({
                "sport": a.get("activity_type", "unknown"),
                "start_time": a.get("start_time"),
                "duration_seconds": a.get("duration_seconds", 0),
                "distance_meters": a.get("distance_meters"),
                "source": "health",
            })

        if not all_activities:
            return {"status": "no_activities", "message": "No activities this week to compare."}

        planned_by_sport: dict[str, int] = {}
        for s in sessions:
            sport = s.get("sport") or s.get("type") or "unknown"
            planned_by_sport[sport] = planned_by_sport.get(sport, 0) + 1

        actual_by_sport: dict[str, int] = {}
        for a in all_activities:
            sport = a.get("sport", "unknown")
            actual_by_sport[sport] = actual_by_sport.get(sport, 0) + 1

        total_planned = sum(planned_by_sport.values())
        total_actual = len(all_activities)
        compliance_rate = round(min(total_actual / max(total_planned, 1), 1.0) * 100)

        return {
            "status": "ok",
            "planned_sessions": total_planned,
            "actual_sessions": total_actual,
            "compliance_rate_pct": compliance_rate,
            "planned_by_sport": planned_by_sport,
            "actual_by_sport": actual_by_sport,
            "data_sources": list(set(a["source"] for a in all_activities)),
        }

    registry.register(Tool(
        name="compare_plan_vs_actual",
        description=(
            "Compare this week's planned training against actual activities from all "
            "data sources (agent, Apple Health, Garmin). Shows planned vs completed "
            "sessions by sport and overall compliance rate. Use this when the athlete "
            "asks about plan adherence or when assessing progress."
        ),
        handler=compare_plan_vs_actual,
        parameters={},
        category="analysis",
    ))

    def classify_activity(activity_id: str, sport: str) -> dict:
        """Classify an unknown activity by updating its sport type.

        After classification, the agent should consider defining sport-specific
        configs if this is a new sport (define_session_schema, define_metric).
        """
        from src.config import get_settings

        settings = get_settings()
        user_id = settings.agenticsports_user_id
        if not user_id:
            return {"status": "error", "message": "No user_id configured."}

        if not activity_id or not sport:
            return {"status": "error", "message": "activity_id and sport are required."}

        try:
            from src.db.client import get_supabase

            result = (
                get_supabase()
                .table("health_activities")
                .update({"activity_type": sport.lower().strip()})
                .eq("id", activity_id)
                .eq("user_id", user_id)
                .execute()
            )

            if not result.data:
                return {
                    "status": "not_found",
                    "message": f"Activity {activity_id} not found.",
                }

            return {
                "status": "classified",
                "activity_id": activity_id,
                "sport": sport.lower().strip(),
                "message": (
                    f"Activity classified as '{sport}'. "
                    f"If this is a new sport, consider using define_session_schema "
                    f"and define_metric to set up sport-specific configurations."
                ),
            }
        except Exception as exc:
            return {"status": "error", "message": f"Classification failed: {exc}"}

    registry.register(Tool(
        name="classify_activity",
        description=(
            "Classify an unknown or uncategorized activity by setting its sport type. "
            "Use this when you detect an activity with type 'unknown', 'other', or "
            "'uncategorized'. After classifying, consider defining sport-specific "
            "configs if this is a new sport for the athlete."
        ),
        handler=classify_activity,
        parameters={
            "type": "object",
            "properties": {
                "activity_id": {
                    "type": "string",
                    "description": "UUID of the health_activity to classify.",
                },
                "sport": {
                    "type": "string",
                    "description": "The correct sport type (e.g., 'running', 'cycling', 'swimming').",
                },
            },
            "required": ["activity_id", "sport"],
        },
        category="analysis",
    ))
