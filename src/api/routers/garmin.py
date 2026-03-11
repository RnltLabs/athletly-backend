"""Garmin router — connect, sync, and disconnect Garmin accounts.

Endpoints:
    POST   /garmin/connect     — authenticate with Garmin credentials
    GET    /garmin/status      — check connection status
    POST   /garmin/sync        — trigger manual sync (rate-limited)
    DELETE /garmin/disconnect  — remove stored Garmin tokens

All endpoints require a valid Supabase JWT in the Authorization header.
"""

from __future__ import annotations

import logging
from typing import Annotated

import redis.asyncio as aioredis
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.api.auth import get_user_id
from src.config import get_settings
from src.services.garmin_sync import GarminSyncService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["garmin"])


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class GarminConnectRequest(BaseModel):
    """Body for POST /garmin/connect."""

    email: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class GarminSyncRequest(BaseModel):
    """Body for POST /garmin/sync."""

    days: int = Field(default=7, ge=1, le=30)


# ---------------------------------------------------------------------------
# Rate limiting helper
# ---------------------------------------------------------------------------


async def _check_sync_cooldown(user_id: str) -> None:
    """Enforce Redis-based sync cooldown (default: 1 sync per 15 minutes).

    Raises:
        HTTPException(429): When the cooldown key is present in Redis.

    Note:
        If Redis is unavailable, the cooldown check is silently skipped so
        that a Redis outage does not break the sync endpoint.
    """
    settings = get_settings()
    cooldown = settings.garmin_sync_cooldown_seconds
    try:
        client = aioredis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=1,
        )
        key = f"garmin_sync:{user_id}"
        if await client.exists(key):
            await client.aclose()
            raise HTTPException(
                429,
                f"Sync cooldown active. Try again in {cooldown} seconds.",
            )
        await client.set(key, "1", ex=cooldown)
        await client.aclose()
    except HTTPException:
        raise
    except Exception:
        # Redis unavailable — skip cooldown check rather than blocking sync
        logger.debug("Redis unavailable for cooldown check; skipping")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/connect")
async def garmin_connect(
    body: GarminConnectRequest,
    user_id: Annotated[str, Depends(get_user_id)],
) -> dict:
    """Authenticate with Garmin Connect and store session tokens.

    Returns:
        ``{"status": "connected", "display_name": "..."}`` on success.

    Raises:
        HTTPException(400): If authentication fails.
    """
    result = GarminSyncService.authenticate(user_id, body.email, body.password)
    if result.get("status") == "error":
        raise HTTPException(400, result.get("error", "Authentication failed"))
    return result


@router.get("/status")
async def garmin_status(
    user_id: Annotated[str, Depends(get_user_id)],
) -> dict:
    """Check whether the user's stored Garmin tokens are still valid.

    Returns:
        Dict with ``connected`` (bool), ``provider``, ``status``, and
        optional ``last_sync_at`` / ``provider_user_id`` fields.
    """
    return GarminSyncService.check_connection(user_id)


@router.post("/sync")
async def garmin_sync(
    user_id: Annotated[str, Depends(get_user_id)],
    body: GarminSyncRequest | None = None,
) -> dict:
    """Trigger a manual Garmin sync (rate-limited to once per cooldown period).

    Syncs activities and daily health stats for the requested number of days.

    Returns:
        Dict with ``activities`` and ``daily_stats`` sub-results.

    Raises:
        HTTPException(429): When the sync cooldown is active.
    """
    await _check_sync_cooldown(user_id)
    days = body.days if body else 7
    activities_result = GarminSyncService.sync_activities(user_id, days)
    stats_result = GarminSyncService.sync_daily_stats(user_id, days)
    return {
        "activities": activities_result,
        "daily_stats": stats_result,
    }


@router.delete("/disconnect")
async def garmin_disconnect(
    user_id: Annotated[str, Depends(get_user_id)],
) -> dict:
    """Delete stored Garmin tokens and clear sync cooldown for the current user.

    Returns:
        ``{"status": "disconnected"}``
    """
    from src.db.provider_tokens_db import delete_token

    delete_token(user_id, "garmin")

    # Clear the sync cooldown so the user can sync immediately after reconnecting
    try:
        settings = get_settings()
        client = aioredis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=1,
        )
        await client.delete(f"garmin_sync:{user_id}")
        await client.aclose()
    except Exception:
        logger.debug("Redis unavailable when clearing cooldown on disconnect; skipping")

    return {"status": "disconnected"}
