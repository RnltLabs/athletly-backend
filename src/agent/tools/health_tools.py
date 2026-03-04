"""Health data tools -- read-only access to provider health and daily metrics.

Provides the agent with cross-source activity data (Apple Health, Garmin,
Health Connect) and daily recovery metrics (sleep, HRV, stress, body battery).

These tools surface data from health_activities, garmin_activities,
health_daily_metrics, and garmin_daily_stats tables via the DB layer.
"""

from datetime import datetime, timedelta, timezone

from src.agent.tools.registry import Tool, ToolRegistry
from src.config import get_settings


def register_health_tools(registry: ToolRegistry) -> None:
    """Register all health data tools."""
    _settings = get_settings()

    def get_health_data(
        days: int = 28,
        activity_type: str | None = None,
        provider: str | None = None,
        source: str = "all",
    ) -> dict:
        """Get health activities from external providers."""
        from src.db.health_data_db import (
            list_health_activities,
            list_garmin_activities,
        )
        from src.db import list_activities as list_agent_activities

        user_id = _settings.agenticsports_user_id
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        activities: list[dict] = []

        if source in ("health", "all"):
            health_rows = list_health_activities(
                user_id,
                limit=200,
                activity_type=activity_type,
                provider_type=provider,
                after=cutoff,
            )
            for r in health_rows:
                activities.append({
                    "date": (r.get("start_time") or "")[:10],
                    "sport": r.get("activity_type") or "unknown",
                    "duration_minutes": round((r.get("duration_seconds") or 0) / 60, 1),
                    "distance_km": round((r.get("distance_meters") or 0) / 1000, 2) if r.get("distance_meters") else None,
                    "avg_hr": r.get("avg_heart_rate"),
                    "max_hr": r.get("max_heart_rate"),
                    "trimp": r.get("training_load_trimp"),
                    "source": "health",
                    "provider": r.get("provider_type"),
                    "_external_id": r.get("external_id"),
                })

        if source in ("garmin", "all"):
            garmin_rows = list_garmin_activities(
                user_id,
                limit=200,
                activity_type=activity_type,
                after=cutoff,
            )
            for r in garmin_rows:
                activities.append({
                    "date": (r.get("start_time") or "")[:10],
                    "sport": r.get("type") or "unknown",
                    "duration_minutes": round((r.get("duration") or 0) / 60, 1),
                    "distance_km": round((r.get("distance") or 0) / 1000, 2) if r.get("distance") else None,
                    "avg_hr": r.get("avg_hr"),
                    "max_hr": r.get("max_hr"),
                    "trimp": None,
                    "source": "garmin",
                    "provider": "garmin",
                    "_garmin_id": r.get("garmin_activity_id"),
                })

        if source == "all":
            agent_rows = list_agent_activities(user_id, limit=200, after=cutoff)
            covered_garmin_ids: set[str] = {
                str(r["garmin_activity_id"])
                for r in agent_rows
                if r.get("garmin_activity_id")
            }

            # Remove health/garmin rows already covered by agent activities
            activities = [
                a for a in activities
                if not (
                    a.get("_external_id") and str(a["_external_id"]) in covered_garmin_ids
                ) and not (
                    a.get("_garmin_id") and str(a["_garmin_id"]) in covered_garmin_ids
                )
            ]

            for r in agent_rows:
                activities.append({
                    "date": (r.get("start_time") or "")[:10],
                    "sport": r.get("sport") or "unknown",
                    "duration_minutes": round((r.get("duration_seconds") or 0) / 60, 1),
                    "distance_km": round((r.get("distance_meters") or 0) / 1000, 2) if r.get("distance_meters") else None,
                    "avg_hr": r.get("avg_hr"),
                    "max_hr": r.get("max_hr"),
                    "trimp": r.get("trimp"),
                    "source": "agent",
                    "provider": "agent",
                })

        # Strip internal dedup keys, sort newest first, apply token budget
        cleaned = [
            {k: v for k, v in a.items() if not k.startswith("_")}
            for a in sorted(activities, key=lambda a: a.get("date", ""), reverse=True)
        ]

        # Enforce 1500-char token budget by trimming activities list
        import json as _json
        budget_chars = 1500 * 4
        serialised = _json.dumps(cleaned)
        if len(serialised) > budget_chars:
            while cleaned and len(_json.dumps(cleaned)) > budget_chars:
                cleaned.pop()

        return {"count": len(cleaned), "activities": cleaned}

    registry.register(Tool(
        name="get_health_data",
        description=(
            "Get training activities from health providers (Apple Health, Garmin, "
            "Health Connect). Use this to see ALL training beyond FIT imports. "
            "Returns sport, duration, distance, HR, TRIMP per activity."
        ),
        handler=get_health_data,
        parameters={
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": "Look-back window in days (default 28).",
                },
                "activity_type": {
                    "type": "string",
                    "description": "Filter by sport/activity type (e.g. 'running'). Omit for all.",
                    "nullable": True,
                },
                "provider": {
                    "type": "string",
                    "description": "Filter by provider: 'apple_health', 'garmin', 'health_connect'. Omit for all.",
                    "nullable": True,
                },
                "source": {
                    "type": "string",
                    "description": "'health' (health_activities only), 'garmin' (garmin_activities only), or 'all' (merged, deduplicated). Default 'all'.",
                    "enum": ["health", "garmin", "all"],
                },
            },
        },
        category="data",
    ))

    def get_daily_metrics(days: int = 14) -> dict:
        """Get daily health metrics (sleep, HRV, stress, body battery, recovery)."""
        from src.db.health_data_db import list_daily_metrics, list_garmin_daily_stats

        user_id = _settings.agenticsports_user_id

        garmin_rows = list_garmin_daily_stats(user_id, days=days)
        health_rows = list_daily_metrics(user_id, days=days)

        # Build date-keyed dict from Garmin data as the baseline
        by_date: dict[str, dict] = {}
        for r in garmin_rows:
            date = r.get("date", "")[:10]
            by_date[date] = {
                "date": date,
                "sleep_minutes": r.get("sleep_duration_minutes"),
                "sleep_score": r.get("sleep_score"),
                "hrv": r.get("hrv_weekly_avg"),
                "resting_hr": r.get("resting_heart_rate"),
                "stress": r.get("stress_avg"),
                "body_battery_high": r.get("body_battery_high"),
                "body_battery_low": r.get("body_battery_low"),
                "recovery_score": None,
                "steps": r.get("steps"),
                "source": "garmin",
            }

        # Overlay health_daily_metrics (wins on conflict)
        for r in health_rows:
            date = r.get("date", "")[:10]
            existing = by_date.get(date, {})
            by_date[date] = {
                "date": date,
                "sleep_minutes": r.get("sleep_duration_minutes") if r.get("sleep_duration_minutes") is not None else existing.get("sleep_minutes"),
                "sleep_score": r.get("sleep_score") if r.get("sleep_score") is not None else existing.get("sleep_score"),
                "hrv": r.get("hrv_avg") if r.get("hrv_avg") is not None else existing.get("hrv"),
                "resting_hr": r.get("resting_heart_rate") if r.get("resting_heart_rate") is not None else existing.get("resting_hr"),
                "stress": r.get("stress_avg") if r.get("stress_avg") is not None else existing.get("stress"),
                "body_battery_high": r.get("body_battery_high") if r.get("body_battery_high") is not None else existing.get("body_battery_high"),
                "body_battery_low": r.get("body_battery_low") if r.get("body_battery_low") is not None else existing.get("body_battery_low"),
                "recovery_score": r.get("recovery_score") if r.get("recovery_score") is not None else existing.get("recovery_score"),
                "steps": r.get("steps") if r.get("steps") is not None else existing.get("steps"),
                "source": "health",
            }

        metrics = sorted(by_date.values(), key=lambda m: m.get("date", ""), reverse=True)

        return {"count": len(metrics), "metrics": metrics}

    registry.register(Tool(
        name="get_daily_metrics",
        description=(
            "Get daily health and recovery metrics: sleep, HRV, stress, body battery, "
            "recovery, steps. Use this to assess recovery status and readiness before "
            "recommending training."
        ),
        handler=get_daily_metrics,
        parameters={
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": "Look-back window in days (default 14).",
                },
            },
        },
        category="data",
    ))
