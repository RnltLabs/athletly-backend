"""Supabase-backed proactive message queue.

Replaces the file-based _load_queue / _save_queue I/O in
``src/agent/proactive.py``.  All state lives in the ``proactive_queue``
table and is keyed by ``user_id`` for multi-tenancy.

Usage::

    from src.db.proactive_queue_db import (
        queue_message,
        get_pending_messages,
        deliver_message,
        record_engagement,
        expire_stale_messages,
    )

    row = queue_message(user_id, "low_activity", 0.8, {}, "Hey, time to train!")
    pending = get_pending_messages(user_id)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from src.db.client import get_supabase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Queue write operations
# ---------------------------------------------------------------------------


def queue_message(
    user_id: str,
    trigger_type: str,
    priority: float,
    data: dict,
    message_text: str,
) -> dict:
    """Insert or update a pending message in the proactive queue.

    A partial unique index on ``(user_id, trigger_type)`` WHERE
    ``status='pending'`` deduplicates at the DB level.  If a pending message
    of the same ``trigger_type`` already exists for the user, it is updated
    with the new ``priority``, ``data``, and ``message_text`` via upsert.

    Args:
        user_id: UUID of the owning user.
        trigger_type: Logical trigger identifier, e.g. ``"low_activity"``.
        priority: Float in [0, 1]; higher values are delivered first.
        data: Arbitrary JSON payload attached to the message.
        message_text: Human-readable message body.

    Returns:
        The inserted or updated row as a new dict.
    """
    db = get_supabase()

    row: dict = {
        "user_id": user_id,
        "trigger_type": trigger_type,
        "priority": priority,
        "data": data,
        "message_text": message_text,
        "status": "pending",
    }

    result = (
        db.table("proactive_queue")
        .upsert(row, on_conflict="user_id,trigger_type")
        .execute()
    )

    inserted = result.data[0]
    logger.info(
        "Queued proactive message user=%s trigger=%s id=%s",
        user_id,
        trigger_type,
        inserted.get("id"),
    )
    return dict(inserted)


# ---------------------------------------------------------------------------
# Queue read operations
# ---------------------------------------------------------------------------


def get_pending_messages(user_id: str) -> list[dict]:
    """Return all pending messages for a user, ordered by priority descending.

    Args:
        user_id: UUID of the owning user.

    Returns:
        New list of pending message dicts (may be empty).
    """
    result = (
        get_supabase()
        .table("proactive_queue")
        .select("*")
        .eq("user_id", user_id)
        .eq("status", "pending")
        .order("priority", desc=True)
        .execute()
    )
    return list(result.data)


# ---------------------------------------------------------------------------
# Delivery & engagement tracking
# ---------------------------------------------------------------------------


def deliver_message(user_id: str, message_id: str) -> dict | None:
    """Mark a message as delivered and stamp ``delivered_at``.

    Args:
        user_id: UUID of the owning user (scopes the update).
        message_id: UUID of the proactive_queue row.

    Returns:
        Updated row as a new dict, or ``None`` if not found.
    """
    result = (
        get_supabase()
        .table("proactive_queue")
        .update({"status": "delivered", "delivered_at": datetime.now(timezone.utc).isoformat()})
        .eq("id", message_id)
        .eq("user_id", user_id)
        .execute()
    )

    if not result.data:
        logger.warning("deliver_message: row not found id=%s user=%s", message_id, user_id)
        return None

    return dict(result.data[0])


def record_engagement(
    user_id: str,
    message_id: str,
    responded: bool = False,
    continued_session: bool = False,
    turns_after: int = 0,
) -> dict | None:
    """Persist engagement metrics on a delivered message.

    Builds the ``engagement_tracking`` JSONB payload and writes it back to
    the row identified by ``message_id``.

    Args:
        user_id: UUID of the owning user.
        message_id: UUID of the proactive_queue row.
        responded: ``True`` if the user replied to the message.
        continued_session: ``True`` if the user kept chatting after delivery.
        turns_after: Number of chat turns after the message was delivered.

    Returns:
        Updated row as a new dict, or ``None`` if not found.
    """
    engagement: dict = {
        "user_responded_at": datetime.now(timezone.utc).isoformat() if responded else None,
        "response_latency_seconds": None,  # caller may backfill if known
        "user_continued_session": continued_session,
        "session_turns_after_delivery": turns_after,
    }

    result = (
        get_supabase()
        .table("proactive_queue")
        .update({"engagement_tracking": engagement})
        .eq("id", message_id)
        .eq("user_id", user_id)
        .execute()
    )

    if not result.data:
        logger.warning(
            "record_engagement: row not found id=%s user=%s", message_id, user_id
        )
        return None

    return dict(result.data[0])


# ---------------------------------------------------------------------------
# Expiry / housekeeping
# ---------------------------------------------------------------------------


def expire_stale_messages(user_id: str, max_age_days: int = 7) -> list[dict]:
    """Set ``status='expired'`` on pending messages older than *max_age_days*.

    Args:
        user_id: UUID of the owning user.
        max_age_days: Messages whose ``created_at`` is older than this many
                      days will be expired (default 7).

    Returns:
        New list of expired rows (may be empty).
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).isoformat()

    result = (
        get_supabase()
        .table("proactive_queue")
        .update({"status": "expired"})
        .eq("user_id", user_id)
        .eq("status", "pending")
        .lt("created_at", cutoff)
        .execute()
    )

    expired = list(result.data)
    if expired:
        logger.info(
            "Expired %d stale proactive messages for user=%s (cutoff=%s)",
            len(expired),
            user_id,
            cutoff,
        )
    return expired
