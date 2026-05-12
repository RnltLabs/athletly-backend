"""Provider abstraction for external data sources (Garmin, Strava, ...).

A Provider integrates with one external fitness/health platform and is
responsible for:

1. Authenticating the user (provider-specific flow).
2. Persisting session tokens in ``provider_tokens``.
3. Pulling normalized data into the local Supabase tables
   (``activities``, ``health_daily_metrics``).

The abstraction is intentionally narrow.  Auth flows are *not* part of the
ABC because they vary too much between providers (password+MFA for Garmin,
browser OAuth for Strava).  Each concrete provider exposes its own
``authenticate`` / ``exchange_code`` / ``refresh_token`` methods as needed.

What IS part of the ABC: the steady-state operations every provider must
support so that downstream code (athctl, agent tools, sync jobs) can treat
all providers uniformly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderCapabilities:
    """What data types this provider can deliver.

    Used by the coach/tools to decide which sync to call, and by athctl to
    surface "what's available" in `provider list`.
    """

    activities: bool = True
    daily_metrics: bool = False
    sleep: bool = False
    body_battery: bool = False
    vo2max: bool = False
    auth_flow: str = "unknown"


class Provider(ABC):
    """Base class for an external data provider.

    Concrete providers MUST set ``name`` and ``capabilities`` as class
    attributes and implement ``check_connection`` and ``sync_activities``.

    Optional capability-gated methods default to a no-op result. Subclasses
    override them only when the capability flag is set.
    """

    name: str = ""
    capabilities: ProviderCapabilities = ProviderCapabilities()

    @abstractmethod
    def check_connection(self, user_id: str) -> dict:
        """Return ``{connected: bool, status: str, ...}`` for this provider+user."""

    @abstractmethod
    def sync_activities(self, user_id: str, days: int = 7) -> dict:
        """Sync recent activities. Returns ``{status, synced, skipped, days}``."""

    def sync_daily_stats(self, user_id: str, days: int = 7) -> dict:
        """Sync daily metrics (HRV, RHR, steps, ...). Default: not supported."""
        return {
            "status": "skipped",
            "reason": f"{self.name} does not support daily_metrics",
        }

    def sync_sleep(self, user_id: str, days: int = 7) -> dict:
        """Sync sleep records. Default: not supported."""
        return {
            "status": "skipped",
            "reason": f"{self.name} does not support sleep",
        }

    def disconnect(self, user_id: str) -> dict:
        """Delete stored tokens for this provider. Default impl."""
        from src.db.provider_tokens_db import delete_token

        delete_token(user_id, self.name)
        return {"status": "disconnected"}
