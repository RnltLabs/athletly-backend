"""Read-only DB layer for health data from Apple Health, Health Connect, and Garmin.

Tables: health_activities, garmin_activities, health_daily_metrics, garmin_daily_stats.

Usage::

    from src.db.health_data_db import list_health_activities, get_cross_source_load_summary

    rows = list_health_activities(user_id, limit=20, activity_type="running")
    summary = get_cross_source_load_summary(user_id, days=28)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from src.db.activity_store_db import list_activities
from src.db.client import get_supabase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Health activities (Apple Health / Health Connect)
# ---------------------------------------------------------------------------


def list_health_activities(
    user_id: str,
    limit: int = 50,
    activity_type: str | None = None,
    provider_type: str | None = None,
    after: str | None = None,
    before: str | None = None,
) -> list[dict]:
    """Query health_activities table. Returns newest first.

    Columns: id, user_id, provider_type, activity_type, start_time, end_time,
    duration_seconds, distance_meters, avg_heart_rate, max_heart_rate,
    calories, training_load_trimp, source_name, external_id, raw_data, created_at.
    """
    query = (
        get_supabase()
        .table("health_activities")
        .select("*")
        .eq("user_id", user_id)
        .order("start_time", desc=True)
        .limit(limit)
    )
    if activity_type:
        query = query.eq("activity_type", activity_type)
    if provider_type:
        query = query.eq("provider_type", provider_type)
    if after:
        query = query.gte("start_time", after)
    if before:
        query = query.lt("start_time", before)

    return list(query.execute().data)


# ---------------------------------------------------------------------------
# Garmin activities
# ---------------------------------------------------------------------------


def list_garmin_activities(
    user_id: str,
    limit: int = 50,
    activity_type: str | None = None,
    after: str | None = None,
    before: str | None = None,
) -> list[dict]:
    """Query garmin_activities table. Returns newest first.

    Columns: id, user_id, garmin_activity_id, type, start_time, duration,
    distance, avg_hr, max_hr, avg_speed, max_speed, elevation_gain,
    calories, training_effect, vo2max_running, raw_data, created_at.
    """
    query = (
        get_supabase()
        .table("garmin_activities")
        .select("*")
        .eq("user_id", user_id)
        .order("start_time", desc=True)
        .limit(limit)
    )
    if activity_type:
        query = query.eq("type", activity_type)
    if after:
        query = query.gte("start_time", after)
    if before:
        query = query.lt("start_time", before)

    return list(query.execute().data)


# ---------------------------------------------------------------------------
# Daily metrics (sleep, HRV, stress — multi-source)
# ---------------------------------------------------------------------------


def list_daily_metrics(
    user_id: str,
    days: int = 14,
    after: str | None = None,
    before: str | None = None,
) -> list[dict]:
    """Query health_daily_metrics table. Returns newest first.

    Columns: id, user_id, date, sleep_duration_minutes, sleep_score, hrv_avg,
    resting_heart_rate, steps, active_calories, total_calories, stress_avg,
    body_battery_high, body_battery_low, recovery_score, source, raw_data, created_at.
    """
    resolved_after = after or (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()

    query = (
        get_supabase()
        .table("health_daily_metrics")
        .select("*")
        .eq("user_id", user_id)
        .gte("date", resolved_after)
        .order("date", desc=True)
    )
    if before:
        query = query.lt("date", before)

    return list(query.execute().data)


# ---------------------------------------------------------------------------
# Garmin daily stats
# ---------------------------------------------------------------------------


def list_garmin_daily_stats(
    user_id: str,
    days: int = 14,
    after: str | None = None,
    before: str | None = None,
) -> list[dict]:
    """Query garmin_daily_stats table. Returns newest first.

    Columns: id, user_id, date, steps, total_distance_meters, active_calories,
    resting_calories, stress_avg, stress_max, body_battery_high, body_battery_low,
    resting_heart_rate, sleep_duration_minutes, sleep_score, hrv_weekly_avg,
    intensity_minutes, floors_climbed, raw_data, created_at.
    """
    resolved_after = after or (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()

    query = (
        get_supabase()
        .table("garmin_daily_stats")
        .select("*")
        .eq("user_id", user_id)
        .gte("date", resolved_after)
        .order("date", desc=True)
    )
    if before:
        query = query.lt("date", before)

    return list(query.execute().data)


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------


def get_health_activity_summary(user_id: str, days: int = 28) -> dict:
    """Aggregate summary over health_activities for the last *days* days.

    Returns dict with count, total_distance_km, total_duration_hours, total_trimp,
    sports_seen (list), and provider_types (list) — all discovered dynamically.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    rows = list_health_activities(user_id, limit=1000, after=cutoff)

    total_distance = sum(r.get("distance_meters") or 0 for r in rows)
    total_duration = sum(r.get("duration_seconds") or 0 for r in rows)
    total_trimp = sum(r.get("training_load_trimp") or 0 for r in rows)
    sports_seen = sorted({r.get("activity_type") or "unknown" for r in rows})
    provider_types = sorted({r.get("provider_type") or "unknown" for r in rows})

    return {
        "count": len(rows),
        "total_distance_km": round(total_distance / 1000, 1),
        "total_duration_hours": round(total_duration / 3600, 1),
        "total_trimp": round(total_trimp, 1),
        "sports_seen": sports_seen,
        "provider_types": provider_types,
    }


def get_cross_source_load_summary(user_id: str, days: int = 28) -> dict:
    """Aggregate training load across all three activity sources with deduplication.

    Priority: activities (agent) > health_activities > garmin_activities.
    Any garmin_activity_id already present in the agent table causes matching
    rows in health_activities (external_id) and garmin_activities (garmin_activity_id)
    to be dropped before merging.

    Returns dict with total_sessions, total_minutes, total_trimp,
    sports_seen (list), sessions_by_sport (dict), sessions_by_source (dict).
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    agent_rows = list_activities(user_id, limit=1000, after=cutoff)
    health_rows = list_health_activities(user_id, limit=1000, after=cutoff)
    garmin_rows = list_garmin_activities(user_id, limit=1000, after=cutoff)

    # Build the set of garmin_activity_ids already covered by the agent table.
    covered_garmin_ids: set[str] = {
        str(r["garmin_activity_id"])
        for r in agent_rows
        if r.get("garmin_activity_id")
    }

    # Filter out health_activities that duplicate an agent-owned Garmin activity.
    filtered_health = [
        r for r in health_rows
        if not (r.get("external_id") and str(r["external_id"]) in covered_garmin_ids)
    ]

    # Filter out garmin_activities that duplicate an agent-owned Garmin activity.
    filtered_garmin = [
        r for r in garmin_rows
        if not (r.get("garmin_activity_id") and str(r["garmin_activity_id"]) in covered_garmin_ids)
    ]

    # Normalise each source into a common shape for aggregation.
    def _norm_agent(r: dict) -> dict:
        return {
            "source": "agent",
            "sport": r.get("sport") or "unknown",
            "duration_seconds": r.get("duration_seconds") or 0,
            "trimp": r.get("trimp") or 0,
        }

    def _norm_health(r: dict) -> dict:
        return {
            "source": "health",
            "sport": r.get("activity_type") or "unknown",
            "duration_seconds": r.get("duration_seconds") or 0,
            "trimp": r.get("training_load_trimp") or 0,
        }

    def _norm_garmin(r: dict) -> dict:
        return {
            "source": "garmin",
            "sport": r.get("type") or "unknown",
            "duration_seconds": r.get("duration") or 0,
            "trimp": 0,  # garmin_activities has training_effect, not TRIMP
        }

    unified = (
        [_norm_agent(r) for r in agent_rows]
        + [_norm_health(r) for r in filtered_health]
        + [_norm_garmin(r) for r in filtered_garmin]
    )

    total_minutes = sum(r["duration_seconds"] for r in unified) / 60
    total_trimp = sum(r["trimp"] for r in unified)

    sessions_by_sport: dict[str, int] = {}
    sessions_by_source: dict[str, int] = {}
    for r in unified:
        sport = r["sport"]
        source = r["source"]
        sessions_by_sport[sport] = sessions_by_sport.get(sport, 0) + 1
        sessions_by_source[source] = sessions_by_source.get(source, 0) + 1

    return {
        "total_sessions": len(unified),
        "total_minutes": round(total_minutes, 1),
        "total_trimp": round(total_trimp, 1),
        "sports_seen": sorted(sessions_by_sport.keys()),
        "sessions_by_sport": sessions_by_sport,
        "sessions_by_source": sessions_by_source,
    }
