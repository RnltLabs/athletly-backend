"""Supabase CRUD for pending_actions table.

Stores proposed plan changes that require user confirmation (checkpoint flow).

Usage::

    from src.db.pending_actions_db import (
        create_pending_action,
        get_pending_for_user,
        get_recently_resolved,
        resolve_pending_action,
        expire_stale_actions,
    )

    action = create_pending_action(user_id, "plan_restructure", "Swap Mon/Wed", {})
    pending = get_pending_for_user(user_id)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from src.db.client import get_supabase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Write operations
# ---------------------------------------------------------------------------


def create_pending_action(
    user_id: str,
    action_type: str,
    description: str,
    preview: dict,
    checkpoint_type: str = "HARD",
    session_id: str | None = None,
) -> dict:
    """Create a new pending action for user confirmation.

    Uses upsert with the unique partial index on ``(user_id, action_type)``
    WHERE ``status='pending'`` to avoid duplicates.

    Args:
        user_id: UUID of the owning user.
        action_type: Logical type, e.g. ``"plan_restructure"``.
        description: Human-readable description of the proposed change.
        preview: JSONB preview data showing before/after state.
        checkpoint_type: ``"HARD"`` (blocks until confirmed) or ``"SOFT"``.
        session_id: Optional session ID for traceability.

    Returns:
        The inserted or updated row as a new dict.
    """
    row: dict = {
        "user_id": user_id,
        "action_type": action_type,
        "description": description,
        "preview": preview,
        "checkpoint_type": checkpoint_type,
        "status": "pending",
    }
    if session_id:
        row["session_id"] = session_id

    result = (
        get_supabase()
        .table("pending_actions")
        .upsert(row, on_conflict="user_id,action_type")
        .execute()
    )
    inserted = result.data[0]
    logger.info(
        "Created pending action user=%s type=%s id=%s",
        user_id,
        action_type,
        inserted.get("id"),
    )
    return dict(inserted)


# ---------------------------------------------------------------------------
# Read operations
# ---------------------------------------------------------------------------


def get_pending_for_user(user_id: str) -> list[dict]:
    """Return all pending actions for a user, newest first.

    Args:
        user_id: UUID of the owning user.

    Returns:
        New list of pending action dicts (may be empty).
    """
    result = (
        get_supabase()
        .table("pending_actions")
        .select("*")
        .eq("user_id", user_id)
        .eq("status", "pending")
        .order("created_at", desc=True)
        .execute()
    )
    return list(result.data)


def get_recently_resolved(user_id: str, hours: int = 1) -> list[dict]:
    """Return actions resolved (confirmed/rejected) within the last N hours.

    Args:
        user_id: UUID of the owning user.
        hours: Look-back window in hours (default 1).

    Returns:
        New list of resolved action dicts (may be empty).
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    result = (
        get_supabase()
        .table("pending_actions")
        .select("*")
        .eq("user_id", user_id)
        .in_("status", ["confirmed", "rejected"])
        .gte("resolved_at", cutoff)
        .order("resolved_at", desc=True)
        .execute()
    )
    return list(result.data)


# ---------------------------------------------------------------------------
# Resolution & expiry
# ---------------------------------------------------------------------------


def resolve_pending_action(
    user_id: str,
    action_id: str,
    confirmed: bool,
) -> dict | None:
    """Resolve a pending action as confirmed or rejected.

    Args:
        user_id: UUID of the owning user (scopes the update).
        action_id: UUID of the pending_actions row.
        confirmed: ``True`` to confirm, ``False`` to reject.

    Returns:
        Updated row as a new dict, or ``None`` if not found.
    """
    new_status = "confirmed" if confirmed else "rejected"
    result = (
        get_supabase()
        .table("pending_actions")
        .update({
            "status": new_status,
            "resolved_at": datetime.now(timezone.utc).isoformat(),
        })
        .eq("id", action_id)
        .eq("user_id", user_id)
        .eq("status", "pending")
        .execute()
    )
    if not result.data:
        logger.warning(
            "resolve_pending_action: not found id=%s user=%s",
            action_id,
            user_id,
        )
        return None
    return dict(result.data[0])


def expire_stale_actions(user_id: str, max_age_hours: int = 24) -> list[dict]:
    """Expire pending actions older than *max_age_hours*.

    Args:
        user_id: UUID of the owning user.
        max_age_hours: Actions whose ``created_at`` is older than this many
                       hours will be expired (default 24).

    Returns:
        New list of expired rows (may be empty).
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=max_age_hours)).isoformat()
    result = (
        get_supabase()
        .table("pending_actions")
        .update({
            "status": "expired",
            "resolved_at": datetime.now(timezone.utc).isoformat(),
        })
        .eq("user_id", user_id)
        .eq("status", "pending")
        .lt("created_at", cutoff)
        .execute()
    )
    expired = list(result.data)
    if expired:
        logger.info(
            "Expired %d stale pending actions for user=%s",
            len(expired),
            user_id,
        )
    return expired
