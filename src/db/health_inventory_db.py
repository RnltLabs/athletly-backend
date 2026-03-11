"""DB layer for health data inventory — connected providers, available metrics, sport summary.

Provides a read-only view of what health data is available for a given user,
enabling the agent to understand data coverage before making recommendations.

Tables: provider_tokens, health_daily_metrics, activities.

Usage::

    from src.db.health_inventory_db import get_connected_providers, get_available_metric_types

    providers = get_connected_providers(user_id)
    metrics = get_available_metric_types(user_id)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from src.db.client import get_supabase

logger = logging.getLogger(__name__)

# Metric columns to check for data availability in health_daily_metrics.
_METRIC_COLUMNS = (
    "sleep_duration_minutes",
    "sleep_score",
    "hrv_avg",
    "resting_heart_rate",
    "stress_avg",
    "body_battery_high",
    "body_battery_low",
    "recovery_score",
    "steps",
    "intensity_minutes",
    "floors_climbed",
)

# Mapping from raw column names to unified metric names.
_UNIFIED_NAMES: dict[str, str] = {
    "sleep_duration_minutes": "sleep",
    "sleep_score": "sleep_score",
    "hrv_avg": "hrv",
    "resting_heart_rate": "resting_hr",
    "stress_avg": "stress",
    "body_battery_high": "body_battery",
    "body_battery_low": "body_battery",
    "recovery_score": "recovery",
    "steps": "steps",
    "intensity_minutes": "intensity_minutes",
    "floors_climbed": "floors_climbed",
}


def get_connected_providers(user_id: str) -> list[dict]:
    """Query the provider_tokens table for connected providers.

    Returns a list of dicts with keys: id, provider (mapped as provider_type),
    status, last_sync_at, created_at. Returns an empty list on error or no data.
    """
    try:
        result = (
            get_supabase()
            .table("provider_tokens")
            .select("id,provider,status,last_sync_at,created_at")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .execute()
        )
        return [
            {**r, "provider_type": r.get("provider")}
            for r in (result.data or [])
        ]
    except Exception as exc:
        logger.error("Failed to fetch connected providers for %s: %s", user_id, exc)
        return []


def get_available_metric_types(user_id: str) -> dict[str, bool]:
    """Check which metric types have non-null data for the user.

    Scans the last 30 days of health_daily_metrics (all sources are now
    consolidated into this single table with a ``source`` column).

    Returns a dict mapping unified metric names to booleans, e.g.
    ``{"sleep": True, "hrv": True, "stress": False, ...}``.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).date().isoformat()
    available: dict[str, bool] = {}

    try:
        result = (
            get_supabase()
            .table("health_daily_metrics")
            .select(",".join(_METRIC_COLUMNS))
            .eq("user_id", user_id)
            .gte("date", cutoff)
            .limit(30)
            .execute()
        )
        for row in (result.data or []):
            for col in _METRIC_COLUMNS:
                if row.get(col) is not None:
                    unified = _UNIFIED_NAMES.get(col, col)
                    available[unified] = True
    except Exception as exc:
        logger.debug("health_daily_metrics scan skipped: %s", exc)

    # Fill in False for known metrics not found
    all_unified = sorted(set(_UNIFIED_NAMES.values()))
    return {name: available.get(name, False) for name in all_unified}


def get_activity_sport_summary(user_id: str) -> list[dict]:
    """Aggregate sport types across the consolidated activities table.

    Returns a list of dicts:
    ``[{"sport": "running", "count": 12, "sources": ["garmin", "apple_health"]}, ...]``

    Sports are discovered dynamically -- never hardcoded.
    """
    sport_data: dict[str, dict] = {}  # sport -> {"count": int, "sources": set}

    try:
        result = (
            get_supabase()
            .table("activities")
            .select("sport,source")
            .eq("user_id", user_id)
            .limit(1000)
            .execute()
        )
        for row in (result.data or []):
            sport = (row.get("sport") or "unknown").lower()
            source = row.get("source") or "unknown"
            entry = sport_data.get(sport, {"count": 0, "sources": set()})
            sport_data[sport] = {
                "count": entry["count"] + 1,
                "sources": entry["sources"] | {source},
            }
    except Exception as exc:
        logger.debug("activities sport scan skipped: %s", exc)

    return sorted(
        [
            {"sport": sport, "count": info["count"], "sources": sorted(info["sources"])}
            for sport, info in sport_data.items()
        ],
        key=lambda x: x["count"],
        reverse=True,
    )
