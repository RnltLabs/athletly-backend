"""Backward-compat shim for the legacy ``GarminSyncService`` class.

New code should use :class:`src.services.providers.garmin.GarminProvider`
or :func:`src.services.providers.registry.get_provider` directly. This
module is kept so existing callers (HTTP router, agent tools) continue to
work without churn while we migrate.
"""

from __future__ import annotations

from src.db.provider_tokens_db import (  # noqa: F401 -- re-export for legacy test mocks
    get_token,
    store_token,
    update_last_sync,
    update_token_status,
)
from src.services.providers.garmin import GarminProvider

_provider = GarminProvider()


class GarminSyncService:
    """Static-method wrapper around the new :class:`GarminProvider`.

    Deprecated. Each method here just forwards to the singleton provider
    instance so the public surface stays identical to what the router and
    agent tools already import.
    """

    @staticmethod
    def authenticate(user_id: str, email: str, password: str) -> dict:
        return _provider.authenticate(user_id, email, password)

    @staticmethod
    def check_connection(user_id: str) -> dict:
        return _provider.check_connection(user_id)

    @staticmethod
    def sync_activities(user_id: str, days: int = 7) -> dict:
        return _provider.sync_activities(user_id, days=days)

    @staticmethod
    def sync_daily_stats(user_id: str, days: int = 7) -> dict:
        return _provider.sync_daily_stats(user_id, days=days)

    @staticmethod
    def sync_sleep(user_id: str, days: int = 7) -> dict:
        return _provider.sync_sleep(user_id, days=days)
