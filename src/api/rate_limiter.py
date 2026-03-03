"""Rate limiting for the Athletly FastAPI backend.

Uses slowapi backed by Redis when available; falls back to in-memory
storage so the server stays functional without a Redis instance.

Key function uses remote IP address for rate limiting. Per-user limits
are enforced inside authenticated endpoints after JWT verification.
"""

import logging

from fastapi import Request
from limits.storage import MemoryStorage
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.config import get_settings

logger = logging.getLogger(__name__)


def _build_storage_uri() -> str | None:
    """Return the Redis URI from settings, or None when unconfigured."""
    url = get_settings().redis_url
    return url if url else None


def _make_limiter() -> Limiter:
    """Construct a Limiter with Redis backend, falling back to in-memory."""
    redis_url = _build_storage_uri()
    if redis_url:
        try:
            from limits.storage import RedisStorage
            RedisStorage(redis_url)
            logger.info("Rate limiter: using Redis backend (%s)", redis_url)
            return Limiter(
                key_func=get_remote_address,
                default_limits=["10/minute", "100/hour"],
                storage_uri=redis_url,
            )
        except Exception as exc:
            logger.warning(
                "Redis unavailable (%s) — falling back to in-memory rate limiter", exc
            )

    logger.info("Rate limiter: using in-memory backend")
    return Limiter(
        key_func=get_remote_address,
        default_limits=["10/minute", "100/hour"],
        storage=MemoryStorage(),  # type: ignore[call-arg]
    )


# -- Singleton ---------------------------------------------------------------

limiter: Limiter = _make_limiter()
