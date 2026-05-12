"""Push token registration endpoint.

The mobile app calls POST /push/register once per session (or on token
rotation) to persist the Expo push token for the authenticated user.
The coach then uses send_notification to dispatch proactive messages.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.api.auth import get_user_id

logger = logging.getLogger(__name__)

router = APIRouter(tags=["push"])


class RegisterTokenRequest(BaseModel):
    """Body for POST /push/register."""

    token: str = Field(..., min_length=1)
    platform: str | None = Field(default=None)
    device_name: str | None = Field(default=None)


@router.post("/register")
async def register_push_token(
    body: RegisterTokenRequest,
    user_id: Annotated[str, Depends(get_user_id)],
) -> dict:
    """Upsert the Expo push token for the authenticated user."""
    from src.db.client import get_supabase

    client = get_supabase()
    try:
        client.table("push_tokens").upsert({
            "user_id": user_id,
            "token": body.token,
            "platform": body.platform,
            "device_name": body.device_name,
            "updated_at": "now()",
        }, on_conflict="user_id").execute()
    except Exception as exc:
        logger.exception("Failed to upsert push token")
        raise HTTPException(500, f"Could not register push token: {exc}")
    return {"status": "registered"}


@router.delete("/register")
async def delete_push_token(
    user_id: Annotated[str, Depends(get_user_id)],
) -> dict:
    """Remove the user's push token (e.g. on logout)."""
    from src.db.client import get_supabase

    client = get_supabase()
    try:
        client.table("push_tokens").delete().eq("user_id", user_id).execute()
    except Exception as exc:
        logger.exception("Failed to delete push token")
        raise HTTPException(500, f"Could not delete push token: {exc}")
    return {"status": "deleted"}
