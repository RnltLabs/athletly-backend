"""Garmin agent tool — lets the agent trigger a Garmin data sync.

The agent calls ``sync_garmin_data`` when it detects that the user has a
connected Garmin account and fresh activity / health data would improve its
coaching response.

All garminconnect imports are deferred inside the handler so the library is an
optional dependency.
"""

from __future__ import annotations

import logging

from src.agent.tools.registry import Tool, ToolRegistry

logger = logging.getLogger(__name__)


def register_garmin_tools(registry: ToolRegistry, user_model=None) -> None:
    """Register Garmin sync tool on *registry*.

    Args:
        registry: The ToolRegistry to register onto.
        user_model: User model instance; used to extract user_id.
    """
    user_id: str | None = getattr(user_model, "user_id", None) if user_model else None

    def sync_garmin_data(days: int = 7) -> dict:
        """Sync recent activities, daily health stats, and sleep data from Garmin.

        Args:
            days: Number of days to sync (1–30, default 7).

        Returns:
            Dict with ``activities``, ``daily_stats``, and ``sleep``
            sub-results, each containing ``status``, ``synced``, and ``days``
            fields.
        """
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
            "Sync recent activities, daily health stats, and sleep data from the "
            "user's connected Garmin account. Returns a sync summary with counts "
            "of synced items. Use when the user asks to import Garmin data, or "
            "when fresh health metrics are needed for planning and recovery "
            "assessment."
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
