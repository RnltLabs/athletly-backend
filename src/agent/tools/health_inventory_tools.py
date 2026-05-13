"""Health inventory tool -- gives the agent a birds-eye view of available data.

The ``get_health_inventory`` tool returns:
- Connected providers and sync status
- Which metrics have data (sleep, HRV, stress, etc.)
- Which sport types exist across all activity sources
- Data coverage in days
- Gaps: metrics without data or with stale data (>7 days old)

This helps the agent decide which metrics are relevant before coaching.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from src.agent.tools.registry import Tool, ToolRegistry
from src.config import get_settings

logger = logging.getLogger(__name__)

# A provider is considered stale if no sync in the last 7 days.
_STALE_THRESHOLD_DAYS = 7


def register_health_inventory_tools(registry: ToolRegistry) -> None:
    """Register health inventory tools on the given *registry*."""
    settings = get_settings()

    def get_health_inventory() -> dict:
        """Return a comprehensive inventory of available health data."""
        from src.db.health_inventory_db import (
            get_activity_sport_summary,
            get_available_metric_types,
            get_connected_providers,
        )
        from src.db.health_data_db import get_merged_daily_metrics

        user_id = settings.agenticsports_user_id
        if not user_id:
            return {"error": "No user configured", "providers": [], "available_metrics": {}}

        # 1. Connected providers
        providers = get_connected_providers(user_id)

        # 2. Available metrics
        available_metrics = get_available_metric_types(user_id)

        # 3. Activity sport summary
        activity_sports = get_activity_sport_summary(user_id)

        # 4. Data coverage — count distinct days with merged metrics
        merged = get_merged_daily_metrics(user_id, days=90)
        data_coverage_days = len(merged)

        # 5. Gaps — metrics without data or providers with stale sync
        gaps = _compute_gaps(providers, available_metrics, merged)

        return {
            "providers": providers,
            "available_metrics": available_metrics,
            "activity_sports": activity_sports,
            "data_coverage_days": data_coverage_days,
            "gaps": gaps,
        }

    registry.register(Tool(
        name="get_health_inventory",
        description=(
            "Get a comprehensive inventory of connected health providers, "
            "available metric types, sport types, data coverage, and gaps. "
            "Use this to understand what data is available before coaching."
        ),
        handler=get_health_inventory,
        parameters={
            "type": "object",
            "properties": {},
        },
        category="data",
        display_label="Pruefe Health-Daten",
    ))


def _compute_gaps(
    providers: list[dict],
    available_metrics: dict[str, bool],
    merged_metrics: list[dict],
) -> list[str]:
    """Identify gaps in health data coverage.

    Returns a list of human-readable gap descriptions.
    """
    gaps: list[str] = []
    now = datetime.now(timezone.utc)
    stale_cutoff = (now - timedelta(days=_STALE_THRESHOLD_DAYS)).isoformat()

    # Stale providers
    for p in providers:
        last_sync = p.get("last_sync_at")
        if last_sync and str(last_sync) < stale_cutoff:
            provider_type = p.get("provider_type", "unknown")
            gaps.append(f"Provider '{provider_type}' last synced >7 days ago")
        elif not last_sync:
            provider_type = p.get("provider_type", "unknown")
            gaps.append(f"Provider '{provider_type}' has never synced")

    # Metrics without data
    for metric_name, has_data in available_metrics.items():
        if not has_data:
            gaps.append(f"No data for metric '{metric_name}'")

    # Recent data staleness — check if newest metric is >7 days old
    if merged_metrics:
        newest_date = merged_metrics[0].get("date", "")
        if newest_date:
            try:
                newest_dt = datetime.fromisoformat(newest_date)
                if newest_dt.tzinfo is None:
                    newest_dt = newest_dt.replace(tzinfo=timezone.utc)
                if (now - newest_dt).days > _STALE_THRESHOLD_DAYS:
                    gaps.append(
                        f"Most recent daily metrics are from {newest_date} (>{_STALE_THRESHOLD_DAYS} days old)"
                    )
            except (ValueError, TypeError):
                pass
    elif not merged_metrics and providers:
        gaps.append("No daily metrics data despite having connected providers")

    return gaps
