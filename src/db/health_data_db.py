"""Read-only DB layer for health data from Apple Health, Health Connect, and Garmin.

Tables: activities (with source column), health_daily_metrics (with source column).

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
    """Query activities table WHERE source IN ('apple_health', 'health_connect').

    The activities table uses columns: sport, start_time, duration_seconds,
    distance_meters, avg_hr, max_hr, trimp, source, garmin_activity_id.

    Results are mapped to the legacy field names for backward compatibility:
    activity_type ← sport, avg_heart_rate ← avg_hr, max_heart_rate ← max_hr,
    training_load_trimp ← trimp, provider_type ← source.
    """
    query = (
        get_supabase()
        .table("activities")
        .select("*")
        .eq("user_id", user_id)
        .in_("source", ["apple_health", "health_connect"])
        .order("start_time", desc=True)
        .limit(limit)
    )
    if activity_type:
        query = query.eq("sport", activity_type)
    if provider_type:
        query = query.eq("source", provider_type)
    if after:
        query = query.gte("start_time", after)
    if before:
        query = query.lt("start_time", before)

    rows = query.execute().data
    return [
        {
            **r,
            "activity_type": r.get("sport"),
            "avg_heart_rate": r.get("avg_hr"),
            "max_heart_rate": r.get("max_hr"),
            "training_load_trimp": r.get("trimp"),
            "provider_type": r.get("source"),
        }
        for r in rows
    ]


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
    """Query activities table WHERE source = 'garmin'. Returns newest first.

    The activities table uses columns: sport, start_time, duration_seconds,
    distance_meters, avg_hr, max_hr, trimp, source, garmin_activity_id.

    Results are mapped to the legacy field names for backward compatibility:
    type ← sport, duration ← duration_seconds, distance ← distance_meters.
    """
    query = (
        get_supabase()
        .table("activities")
        .select("*")
        .eq("user_id", user_id)
        .eq("source", "garmin")
        .order("start_time", desc=True)
        .limit(limit)
    )
    if activity_type:
        query = query.eq("sport", activity_type)
    if after:
        query = query.gte("start_time", after)
    if before:
        query = query.lt("start_time", before)

    rows = query.execute().data
    return [
        {
            **r,
            "type": r.get("sport"),
            "duration": r.get("duration_seconds"),
            "distance": r.get("distance_meters"),
        }
        for r in rows
    ]


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
# Merged daily metrics (single table with source column)
# ---------------------------------------------------------------------------


def get_merged_daily_metrics(
    user_id: str,
    days: int = 14,
) -> list[dict]:
    """Query health_daily_metrics and return a unified view.

    All sources (garmin, apple_health, health_connect) are now in one table
    distinguished by the ``source`` column. Returns list of dicts sorted
    newest-first with keys: date, sleep_minutes, sleep_score, hrv,
    resting_hr, stress, body_battery_high, body_battery_low, recovery_score,
    steps, source.
    """
    rows = list_daily_metrics(user_id, days=days)

    return [
        {
            "date": r.get("date", "")[:10],
            "sleep_minutes": r.get("sleep_duration_minutes"),
            "sleep_score": r.get("sleep_score"),
            "hrv": r.get("hrv_avg"),
            "resting_hr": r.get("resting_heart_rate"),
            "stress": r.get("stress_avg"),
            "body_battery_high": r.get("body_battery_high"),
            "body_battery_low": r.get("body_battery_low"),
            "recovery_score": r.get("recovery_score"),
            "steps": r.get("steps"),
            "source": r.get("source", "unknown"),
        }
        for r in rows
    ]


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
    """Aggregate training load across all activity sources.

    All activities now live in the consolidated ``activities`` table with a
    ``source`` column (garmin, apple_health, health_connect, manual).
    No deduplication is needed since there is only one table.

    Returns dict with total_sessions, total_minutes, total_trimp,
    sports_seen (list), sessions_by_sport (dict), sessions_by_source (dict).
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    rows = list_activities(user_id, limit=1000, after=cutoff)

    sessions_by_sport: dict[str, int] = {}
    sessions_by_source: dict[str, int] = {}
    total_duration = 0
    total_trimp = 0

    for r in rows:
        sport = r.get("sport") or "unknown"
        source = r.get("source") or "unknown"
        total_duration += r.get("duration_seconds") or 0
        total_trimp += r.get("trimp") or 0
        sessions_by_sport[sport] = sessions_by_sport.get(sport, 0) + 1
        sessions_by_source[source] = sessions_by_source.get(source, 0) + 1

    return {
        "total_sessions": len(rows),
        "total_minutes": round(total_duration / 60, 1),
        "total_trimp": round(total_trimp, 1),
        "sports_seen": sorted(sessions_by_sport.keys()),
        "sessions_by_sport": sessions_by_sport,
        "sessions_by_source": sessions_by_source,
    }
