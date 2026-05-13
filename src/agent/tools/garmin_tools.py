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

    def _resolve_sync_window(mode: str, days: int | None) -> tuple[int, str]:
        """Pick the day count based on requested mode + last_sync_at.

        Modes:
          - "full":  always 365 days (use for onboarding / migration)
          - "delta": since last_sync_at minus 3-day grace, capped at 30
          - "auto":  full if never synced or last_sync > 30 days ago, else delta

        An explicit `days` value overrides the mode entirely.
        """
        from datetime import datetime, timezone

        if days is not None:
            return (max(1, min(int(days), 365)), "explicit")

        last_sync_at: str | None = None
        try:
            from src.db.client import get_supabase
            sb = get_supabase()
            rows = (
                sb.table("provider_tokens")
                .select("last_sync_at")
                .eq("user_id", user_id)
                .eq("provider", "garmin")
                .limit(1)
                .execute()
            )
            if rows.data:
                last_sync_at = rows.data[0].get("last_sync_at")
        except Exception:
            pass

        days_since: int | None = None
        if last_sync_at:
            try:
                last_dt = datetime.fromisoformat(last_sync_at.replace("Z", "+00:00"))
                days_since = max(0, (datetime.now(timezone.utc) - last_dt).days)
            except Exception:
                days_since = None

        if mode == "full":
            return (365, "full")
        if mode == "delta":
            if days_since is None:
                return (14, "delta:no-history")
            return (max(2, min(days_since + 3, 30)), "delta")

        # auto
        if days_since is None or days_since > 30:
            return (365, "auto:full")
        return (max(2, min(days_since + 3, 30)), "auto:delta")

    def sync_garmin_data(mode: str = "auto", days: int | None = None) -> dict:
        """Sync activities, daily health stats, and sleep from Garmin.

        Picks the day window automatically based on last_sync_at:
          - First time / stale (no sync in >30 days): pulls 365 days.
          - Recent sync: pulls only the gap since last sync + 3-day grace.

        Use mode="full" to force a 365-day initial sync (onboarding).
        Use mode="delta" to force a short refresh.
        Use days=N to override entirely.
        """
        if not user_id:
            return {"error": "No user_id available"}

        effective_days, resolved_mode = _resolve_sync_window(mode, days)

        from src.services.garmin_sync import GarminSyncService

        activities = GarminSyncService.sync_activities(user_id, effective_days)
        daily_stats = GarminSyncService.sync_daily_stats(user_id, effective_days)
        sleep = GarminSyncService.sync_sleep(user_id, effective_days)
        return {
            "mode": resolved_mode,
            "days_synced": effective_days,
            "activities": activities,
            "daily_stats": daily_stats,
            "sleep": sleep,
        }

    registry.register(Tool(
        name="sync_garmin_data",
        description=(
            "Pull activities, daily health stats, and sleep from the user's "
            "connected Garmin account. AUTO mode (default) inspects "
            "last_sync_at: pulls 365 days the first time (or after a "
            "30-day gap), otherwise pulls only the delta since last sync "
            "with a 3-day grace. Use mode='full' to force a 365-day "
            "initial pull (e.g. during onboarding). Use mode='delta' to "
            "force a short refresh. Pass days=N to override entirely. "
            "Returns {mode, days_synced, activities, daily_stats, sleep}. "
            "365-day syncs take ~30 sec because daily-stats and sleep "
            "loop per day."
        ),
        handler=sync_garmin_data,
        parameters={
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "description": "auto (default), full, or delta",
                    "enum": ["auto", "full", "delta"],
                    "default": "auto",
                },
                "days": {
                    "type": "integer",
                    "description": "Explicit override 1-365. Omit for auto/full/delta logic.",
                },
            },
        },
        category="data",
    ))
