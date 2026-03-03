"""Supabase client singleton for AgenticSports.

Usage::

    from src.db.client import get_supabase, get_async_supabase

    # Synchronous (existing code, file-backed sessions, CLI)
    client = get_supabase()
    result = client.table("profiles").select("*").execute()

    # Asynchronous (FastAPI endpoints)
    async_client = await get_async_supabase()
"""

import asyncio
import logging
from functools import lru_cache

from supabase import Client, AsyncClient, create_client, acreate_client

from src.config import get_settings

logger = logging.getLogger(__name__)

# Module-level cache for the async client (lru_cache is incompatible with
# async factory functions, so we manage the reference manually).
_async_client: AsyncClient | None = None
_async_client_lock: asyncio.Lock = asyncio.Lock()


@lru_cache
def get_supabase() -> Client:
    """Return a cached singleton Supabase client.

    Uses the SERVICE_ROLE_KEY when available (bypasses RLS for backend
    operations), falling back to the ANON_KEY for client-level access.
    """
    settings = get_settings()
    key = settings.supabase_service_role_key or settings.supabase_anon_key
    if not key:
        raise RuntimeError(
            "No Supabase key configured. "
            "Set SUPABASE_SERVICE_ROLE_KEY or SUPABASE_ANON_KEY in your .env file."
        )
    return create_client(settings.supabase_url, key)


async def get_async_supabase() -> AsyncClient:
    """Return a cached singleton async Supabase client.

    Thread-safe: uses an asyncio.Lock so concurrent requests that arrive
    before the client is initialised do not create duplicate instances.

    Uses the SERVICE_ROLE_KEY when available (bypasses RLS for backend
    operations), falling back to the ANON_KEY for client-level access.

    Raises:
        RuntimeError: If no Supabase key is configured.
    """
    global _async_client  # noqa: PLW0603 -- intentional singleton

    if _async_client is not None:
        return _async_client

    async with _async_client_lock:
        # Double-checked locking: another coroutine may have initialised
        # the client while we waited for the lock.
        if _async_client is not None:
            return _async_client

        settings = get_settings()
        key = settings.supabase_service_role_key or settings.supabase_anon_key
        if not key:
            raise RuntimeError(
                "No Supabase key configured. "
                "Set SUPABASE_SERVICE_ROLE_KEY or SUPABASE_ANON_KEY in your .env file."
            )

        client = await acreate_client(settings.supabase_url, key)
        _async_client = client
        logger.info("Async Supabase client initialised (url=%s)", settings.supabase_url)
        return _async_client
