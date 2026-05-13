"""Garmin agent tools - check sync status + trigger a fresh sync.

The agent uses these two tools together:

  1. ``get_provider_status`` to check connection status, last sync time,
     and how many activities are already imported. Cheap, read-only.
  2. ``sync_garmin_data`` to actually pull new activities, daily health
     stats, and sleep data from Garmin.

All provider-specific library imports are deferred inside handlers so
the external SDKs stay optional.
"""

from __future__ import annotations

import logging

from src.agent.tools.registry import Tool, ToolRegistry

logger = logging.getLogger(__name__)


def register_garmin_tools(registry: ToolRegistry, user_model=None) -> None:
    """Register Garmin status + sync tools on *registry*."""
    user_id: str | None = getattr(user_model, "user_id", None) if user_model else None

    # ------------------------------------------------------------------
    # get_provider_status
    # ------------------------------------------------------------------

    def get_provider_status(provider: str = "garmin") -> dict:
        """Return connection + sync status for the given provider."""
        if not user_id:
            return {"error": "No user_id available"}

        provider_norm = (provider or "garmin").lower().strip()
        if provider_norm not in {"garmin", "strava"}:
            return {
                "error": f"Unknown provider '{provider}'. Use 'garmin' or 'strava'."
            }

        from src.db.client import get_supabase

        sb = get_supabase()

        # Token row tells us if connected + last sync timestamp.
        token_rows = (
            sb.table("provider_tokens")
            .select("provider_user_id, last_sync_at, status, created_at")
            .eq("user_id", user_id)
            .eq("provider", provider_norm)
            .limit(1)
            .execute()
        )
        token = token_rows.data[0] if token_rows.data else None

        # Count activities already imported from this provider.
        activity_count = 0
        latest_activity_date: str | None = None
        try:
            count_res = (
                sb.table("activities")
                .select("id", count="exact")
                .eq("user_id", user_id)
                .eq("source", provider_norm)
                .execute()
            )
            activity_count = count_res.count or 0
        except Exception:
            logger.debug("activity count failed", exc_info=True)

        if activity_count > 0:
            try:
                latest = (
                    sb.table("activities")
                    .select("start_time")
                    .eq("user_id", user_id)
                    .eq("source", provider_norm)
                    .order("start_time", desc=True)
                    .limit(1)
                    .execute()
                )
                if latest.data:
                    latest_activity_date = latest.data[0].get("start_time")
            except Exception:
                logger.debug("latest activity lookup failed", exc_info=True)

        return {
            "provider": provider_norm,
            "connected": token is not None and (token.get("status") in (None, "active")),
            "provider_user": token.get("provider_user_id") if token else None,
            "connected_since": token.get("created_at") if token else None,
            "last_sync_at": token.get("last_sync_at") if token else None,
            "activity_count": activity_count,
            "latest_activity_date": latest_activity_date,
        }

    registry.register(Tool(
        name="get_provider_status",
        description=(
            "Check whether a wearable/training provider (garmin or strava) is "
            "connected, when the last sync happened, and how many activities "
            "are already imported. Returns {connected, provider_user, "
            "connected_since, last_sync_at, activity_count, "
            "latest_activity_date}. Cheap, read-only - call this BEFORE "
            "deciding whether you need to run sync_garmin_data or whether "
            "existing data is fresh enough. Default provider is 'garmin'."
        ),
        handler=get_provider_status,
        parameters={
            "type": "object",
            "properties": {
                "provider": {
                    "type": "string",
                    "description": "Provider to check. One of: garmin, strava.",
                    "enum": ["garmin", "strava"],
                },
            },
        },
        category="data",
    ))

    # ------------------------------------------------------------------
    # sync_garmin_data
    # ------------------------------------------------------------------

    def sync_garmin_data(days: int = 7) -> dict:
        """Sync recent activities, daily health stats, and sleep from Garmin."""
        if not user_id:
            return {"error": "No user_id available"}

        from src.services.garmin_sync import GarminSyncService

        activities = GarminSyncService.sync_activities(user_id, days)
        daily_stats = GarminSyncService.sync_daily_stats(user_id, days)
        sleep = GarminSyncService.sync_sleep(user_id, days)
        return {"activities": activities, "daily_stats": daily_stats, "sleep": sleep}

    registry.register(Tool(
        name="sync_garmin_data",
        description=(
            "Pull recent activities, daily health stats, and sleep data from "
            "the user's connected Garmin account. Returns a sync summary "
            "with counts of synced items. Call when get_provider_status "
            "shows stale or missing data, or when the user explicitly asks "
            "to refresh. Avoid spamming - one sync per chat turn is plenty."
        ),
        handler=sync_garmin_data,
        parameters={
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": "Number of days to sync (1-30, default 7)",
                    "default": 7,
                },
            },
        },
        category="data",
    ))
