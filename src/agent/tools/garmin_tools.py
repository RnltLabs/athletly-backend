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

    def sync_garmin_data(days: int = 30) -> dict:
        """Sync recent activities, daily health stats, and sleep from Garmin."""
        if not user_id:
            return {"error": "No user_id available"}

        # Hard cap to keep individual sync runtime sane. Daily-stats and
        # sleep loop one API call per day, so 365 days = ~365 calls.
        days = max(1, min(int(days or 30), 365))

        from src.services.garmin_sync import GarminSyncService

        activities = GarminSyncService.sync_activities(user_id, days)
        daily_stats = GarminSyncService.sync_daily_stats(user_id, days)
        sleep = GarminSyncService.sync_sleep(user_id, days)
        return {"activities": activities, "daily_stats": daily_stats, "sleep": sleep}

    registry.register(Tool(
        name="sync_garmin_data",
        description=(
            "Pull activities, daily health stats, and sleep from the user's "
            "connected Garmin account. Returns sync counts. "
            "Days argument: 1-365, default 30. For onboarding a NEW athlete, "
            "use days=90 to get 3 months of base context (training history, "
            "fitness trend, recent goals). For a returning athlete's regular "
            "refresh, 7-14 days is plenty. Avoid spamming - one sync per "
            "chat turn is enough; daily-stats and sleep loop one API call "
            "per day so 365 days takes ~30 sec."
        ),
        handler=sync_garmin_data,
        parameters={
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": (
                        "Number of days to sync (1-365). Default 30. "
                        "Recommended: 90 for onboarding, 7-14 for refresh."
                    ),
                    "default": 30,
                },
            },
        },
        category="data",
    ))
