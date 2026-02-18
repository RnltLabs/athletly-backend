"""Supabase-backed activity store replacing JSON file persistence.

Provides the same logical operations as src/tools/activity_store.py but
persists to the ``activities`` and ``import_manifest`` Supabase tables
instead of local JSON files.

Usage::

    from src.db.activity_store_db import store_activity, list_activities

    row = store_activity(user_id, {"sport": "running", ...})
    recent = list_activities(user_id, limit=20)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from src.db.client import get_supabase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Activity CRUD
# ---------------------------------------------------------------------------


def store_activity(user_id: str, activity: dict) -> dict:
    """Store an activity to Supabase.

    Args:
        user_id: UUID of the owning user.
        activity: Dict with keys matching the ``activities`` table columns
                  (sport, start_time, duration_seconds, distance_meters, ...).

    Returns:
        The inserted row as a dict (includes server-generated ``id`` and
        ``created_at``).
    """
    db = get_supabase()

    row: dict = {
        "user_id": user_id,
        "sport": activity.get("sport", "running"),
        "start_time": activity.get("start_time"),
        "duration_seconds": activity.get("duration_seconds"),
        "distance_meters": activity.get("distance_meters"),
        "avg_hr": activity.get("avg_hr"),
        "max_hr": activity.get("max_hr"),
        "avg_pace_min_km": activity.get("avg_pace_min_km"),
        "elevation_gain_m": activity.get("elevation_gain_m"),
        "trimp": activity.get("trimp"),
        "zone_distribution": activity.get("zone_distribution", {}),
        "laps": activity.get("laps", []),
        "raw_data": activity.get("raw_data", {}),
        "source": activity.get("source", "manual"),
        "garmin_activity_id": activity.get("garmin_activity_id"),
    }

    result = db.table("activities").insert(row).execute()
    return result.data[0]


def list_activities(
    user_id: str,
    limit: int = 50,
    sport: str | None = None,
    after: str | None = None,
    before: str | None = None,
) -> list[dict]:
    """List activities for a user, newest first.

    Args:
        user_id: UUID of the owning user.
        limit: Maximum number of activities to return.
        sport: Optional sport filter (e.g. ``"running"``).
        after: Only include activities with ``start_time >= after`` (ISO string).
        before: Only include activities with ``start_time < before`` (ISO string).

    Returns:
        List of activity dicts ordered by ``start_time`` descending.
    """
    query = (
        get_supabase()
        .table("activities")
        .select("*")
        .eq("user_id", user_id)
        .order("start_time", desc=True)
        .limit(limit)
    )
    if sport:
        query = query.eq("sport", sport)
    if after:
        query = query.gte("start_time", after)
    if before:
        query = query.lt("start_time", before)

    return query.execute().data


def get_activity(user_id: str, activity_id: str) -> dict | None:
    """Get a single activity by ID.

    Args:
        user_id: UUID of the owning user (ensures row-level access).
        activity_id: UUID of the activity.

    Returns:
        Activity dict or ``None`` if not found.
    """
    result = (
        get_supabase()
        .table("activities")
        .select("*")
        .eq("id", activity_id)
        .eq("user_id", user_id)
        .maybe_single()
        .execute()
    )
    # maybe_single().execute() returns None when no row is found
    return result.data if result is not None else None


# ---------------------------------------------------------------------------
# Import manifest (FIT file dedup)
# ---------------------------------------------------------------------------


def check_import_manifest(user_id: str, file_hash: str) -> bool:
    """Check if a FIT file has already been imported.

    Args:
        user_id: UUID of the owning user.
        file_hash: SHA-256 hex digest of the FIT file.

    Returns:
        ``True`` if the file was already imported, ``False`` otherwise.
    """
    result = (
        get_supabase()
        .table("import_manifest")
        .select("id")
        .eq("user_id", user_id)
        .eq("file_hash", file_hash)
        .maybe_single()
        .execute()
    )
    # maybe_single().execute() returns None when no row is found
    return result is not None and result.data is not None


def record_import(
    user_id: str,
    file_hash: str,
    file_name: str,
    activity_id: str,
) -> None:
    """Record a FIT file import in the manifest.

    Args:
        user_id: UUID of the owning user.
        file_hash: SHA-256 hex digest of the imported FIT file.
        file_name: Original filename (for display / debugging).
        activity_id: UUID of the stored activity row.
    """
    get_supabase().table("import_manifest").insert(
        {
            "user_id": user_id,
            "file_hash": file_hash,
            "file_name": file_name,
            "activity_id": activity_id,
        }
    ).execute()


# ---------------------------------------------------------------------------
# Summary / aggregation helpers
# ---------------------------------------------------------------------------


def get_activities_summary(user_id: str, days: int = 28) -> dict:
    """Get activity summary stats for the last *days* days.

    Args:
        user_id: UUID of the owning user.
        days: Look-back window in days (default 28).

    Returns:
        Dict with ``count``, ``total_distance_km``, ``total_duration_hours``,
        and the full list of matching ``activities``.
    """
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

    result = (
        get_supabase()
        .table("activities")
        .select("*")
        .eq("user_id", user_id)
        .gte("start_time", cutoff)
        .order("start_time", desc=True)
        .execute()
    )

    activities = result.data
    total_distance = sum(a.get("distance_meters", 0) or 0 for a in activities)
    total_duration = sum(a.get("duration_seconds", 0) or 0 for a in activities)

    return {
        "count": len(activities),
        "total_distance_km": round(total_distance / 1000, 1),
        "total_duration_hours": round(total_duration / 3600, 1),
        "activities": activities,
    }


def get_weekly_summary(user_id: str, activities: list[dict] | None = None) -> dict:
    """Summarize activities for a week, mirroring the legacy helper.

    If *activities* is ``None``, the last 7 days are fetched from Supabase.

    Returns:
        Dict with ``total_sessions``, ``total_duration_minutes``,
        ``total_distance_km``, ``avg_hr``, and ``sessions_by_sport``.
    """
    if activities is None:
        activities = list_activities(user_id, limit=100)
        cutoff = (datetime.utcnow() - timedelta(days=7)).isoformat()
        activities = [a for a in activities if (a.get("start_time") or "") >= cutoff]

    if not activities:
        return {
            "total_sessions": 0,
            "total_duration_minutes": 0,
            "total_distance_km": 0.0,
            "avg_hr": None,
            "sessions_by_sport": {},
        }

    total_duration_sec = 0
    total_distance_m = 0
    hr_sum = 0
    hr_count = 0
    sessions_by_sport: dict[str, int] = {}

    for act in activities:
        dur = act.get("duration_seconds")
        if dur:
            total_duration_sec += dur

        dist = act.get("distance_meters")
        if dist:
            total_distance_m += dist

        avg_hr = act.get("avg_hr")
        if avg_hr:
            hr_sum += avg_hr
            hr_count += 1

        sport = act.get("sport", "unknown")
        sessions_by_sport[sport] = sessions_by_sport.get(sport, 0) + 1

    return {
        "total_sessions": len(activities),
        "total_duration_minutes": round(total_duration_sec / 60, 1),
        "total_distance_km": round(total_distance_m / 1000, 2),
        "avg_hr": round(hr_sum / hr_count) if hr_count > 0 else None,
        "sessions_by_sport": sessions_by_sport,
    }
