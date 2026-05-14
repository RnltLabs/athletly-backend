"""User-tier lookup service for the hybrid model router.

Single source of truth for "is this user on Free or Pro?". The result
gates Sonnet 4.6 access in :mod:`src.agent.model_router`.

Design notes
------------

- The lookup is intentionally fail-closed: any error or missing row
  returns the cost-safe default (typically "free"). Sonnet is never
  reachable for users whose tier we cannot prove.
- The result is cached per-process for ``user_tier_cache_ttl_seconds``
  (60s by default) so that a chatty agent loop does not hammer the DB
  on every tool round.
- Cache is keyed on user_id only; tier flips at most once per minute
  from the application's point of view. That is acceptable for a
  billing flag that changes at most a few times per user-lifetime.
- ``invalidate(user_id)`` is exposed so the upgrade/downgrade webhook
  can force-evict an entry when we add billing integration.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Literal

from src.config import get_settings

logger = logging.getLogger(__name__)

UserTier = Literal["free", "pro"]
VALID_TIERS: frozenset[str] = frozenset({"free", "pro"})


class _TierCache:
    """Thread-safe TTL cache for user_id -> tier resolutions.

    Kept small and explicit instead of pulling in ``functools.lru_cache``
    so we can flush a single entry on demand from a billing webhook.
    """

    def __init__(self) -> None:
        self._data: dict[str, tuple[float, UserTier]] = {}
        self._lock = threading.Lock()

    def get(self, key: str, ttl: float) -> UserTier | None:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if expires_at < time.time():
                # Expired; clean it up so the cache cannot grow without bound.
                del self._data[key]
                return None
            return value

    def put(self, key: str, value: UserTier, ttl: float) -> None:
        with self._lock:
            self._data[key] = (time.time() + ttl, value)

    def invalidate(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


_CACHE = _TierCache()


def _coerce_tier(value: object) -> UserTier:
    """Validate and normalize an arbitrary tier value.

    Anything not in :data:`VALID_TIERS` becomes "free". This is the
    cost-safe default: an unknown tier never gets Sonnet access.
    """
    if isinstance(value, str) and value in VALID_TIERS:
        return value  # type: ignore[return-value]
    return "free"


def _lookup_in_db(user_id: str) -> UserTier:
    """Fetch the tier column from Supabase ``profiles``.

    Returns "free" on any failure -- missing column, missing row,
    connectivity issue, malformed value. Logged at WARNING because a
    persistent failure means every Pro user is being downgraded.
    """
    try:
        from src.db.client import get_supabase
    except Exception:
        logger.warning(
            "user_tier: supabase client unavailable, defaulting to free"
        )
        return "free"

    try:
        client = get_supabase()
        result = (
            client.table("profiles")
            .select("tier")
            .eq("user_id", user_id)
            .maybe_single()
            .execute()
        )
    except Exception as exc:
        logger.warning(
            "user_tier: lookup failed user=%s err=%s -> defaulting to free",
            user_id, exc,
        )
        return "free"

    if result is None or not getattr(result, "data", None):
        return "free"
    return _coerce_tier(result.data.get("tier"))


def get_user_tier(user_id: str | None) -> UserTier:
    """Return the tier ("free" or "pro") for ``user_id``.

    No DB call is made for empty user_ids; we return
    ``settings.default_user_tier`` instead (cost-safe "free" by default).
    Otherwise the result is read-through cached for
    ``settings.user_tier_cache_ttl_seconds`` to keep chatty callers off
    the hot path of the DB.
    """
    settings = get_settings()

    if not user_id:
        return _coerce_tier(settings.default_user_tier)

    ttl = float(settings.user_tier_cache_ttl_seconds)
    cached = _CACHE.get(user_id, ttl)
    if cached is not None:
        return cached

    resolved = _lookup_in_db(user_id)
    _CACHE.put(user_id, resolved, ttl)
    return resolved


def invalidate(user_id: str) -> None:
    """Drop the cached tier for ``user_id``.

    Call this after a billing event (upgrade or downgrade) so the next
    chat turn sees the new tier immediately rather than waiting up to
    ``user_tier_cache_ttl_seconds``.
    """
    _CACHE.invalidate(user_id)


def reset_cache() -> None:
    """Clear the entire process-local tier cache. Test helper."""
    _CACHE.clear()
