"""Webhook router — receives activity events from external providers.

POST /webhook/activity
    Validates HMAC signature and timestamp, stores the activity, and
    optionally triggers an agent analysis.

Security model:
    - HMAC-SHA256 signature in ``X-Webhook-Signature`` header (hex digest).
    - Timestamp in ``X-Webhook-Timestamp`` header (Unix seconds as a string).
    - Events older than ``_MAX_AGE_SECONDS`` are rejected to prevent replay.
    - ``hmac.compare_digest`` is used to prevent timing attacks.
"""

import hashlib
import hmac
import logging
import time
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel, field_validator

from src.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Maximum age of an accepted webhook event (seconds).
_MAX_AGE_SECONDS = 300  # 5 minutes


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------


class ActivityWebhookPayload(BaseModel):
    """Body of an inbound activity webhook."""

    user_id: str
    activity_type: str
    data: dict[str, Any]

    @field_validator("user_id")
    @classmethod
    def user_id_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("user_id must not be empty")
        return v.strip()

    @field_validator("activity_type")
    @classmethod
    def activity_type_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("activity_type must not be empty")
        return v.strip()


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post("/activity", status_code=200)
async def receive_activity(
    request: Request,
    payload: ActivityWebhookPayload,
    x_webhook_signature: str = Header(default=""),
    x_webhook_timestamp: str = Header(default=""),
) -> dict:
    """Receive an activity event from an external provider.

    Validates the HMAC signature and timestamp before processing.
    Returns 401 on bad signature and 400 on stale timestamp.
    """
    _validate_timestamp(x_webhook_timestamp)
    raw_body = await request.body()
    _validate_signature(raw_body, x_webhook_signature)

    logger.info(
        "Webhook received: user=%s type=%s",
        payload.user_id,
        payload.activity_type,
    )

    stored = await _store_activity(payload)
    return {"status": "ok", "stored": stored}


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_timestamp(timestamp_header: str) -> None:
    """Reject events older than _MAX_AGE_SECONDS.

    Raises:
        HTTPException(400): If the timestamp is missing, non-numeric, or stale.
    """
    if not timestamp_header:
        raise HTTPException(status_code=400, detail="Missing X-Webhook-Timestamp header")

    try:
        event_ts = int(timestamp_header)
    except ValueError:
        raise HTTPException(status_code=400, detail="X-Webhook-Timestamp must be a Unix timestamp")

    age_seconds = time.time() - event_ts
    if age_seconds > _MAX_AGE_SECONDS or age_seconds < -60:
        raise HTTPException(
            status_code=400,
            detail=f"Webhook timestamp too old or too far in future (age={age_seconds:.0f}s)",
        )


def _validate_signature(body: bytes, signature_header: str) -> None:
    """Verify HMAC-SHA256 signature over the raw request body.

    Raises:
        HTTPException(401): If the signature is missing or does not match.
    """
    settings = get_settings()
    secret = settings.webhook_secret

    if not secret:
        logger.error("WEBHOOK_SECRET not configured — rejecting all webhook requests")
        raise HTTPException(status_code=503, detail="Webhook endpoint not configured")

    if not signature_header:
        raise HTTPException(status_code=401, detail="Missing X-Webhook-Signature header")

    expected = hmac.new(
        secret.encode("utf-8"),
        msg=body,
        digestmod=hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(expected, signature_header):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


async def _store_activity(payload: ActivityWebhookPayload) -> bool:
    """Persist the incoming activity to Supabase.

    Returns True on success, False when Supabase is unavailable.
    """
    try:
        from src.db.client import get_async_supabase

        client = await get_async_supabase()
        await (
            client.table("activities")
            .insert(
                {
                    "user_id": payload.user_id,
                    "activity_type": payload.activity_type,
                    "data": payload.data,
                    "source": "webhook",
                }
            )
            .execute()
        )
        return True
    except Exception as exc:
        logger.error("Failed to store webhook activity: %s", exc)
        return False
