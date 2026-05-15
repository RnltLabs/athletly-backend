"""Supabase-backed CRUD for the ``intents`` table.

Intents are high-mutation coach-maintained commitments. An athlete typically
has 1 to 5 active intents at any time (event + capacity + habit + recovery
in parallel). Coach reads them every turn, edits them often.

Pattern matches ``src/db/plans_db.py``: synchronous service-role client,
small focused functions, no exceptions leak out of read paths (return
sentinel value), but write paths raise so the caller learns about failures.

Usage::

    from src.db.intents_db import (
        list_intents, get_intent, insert_intent,
        update_intent_status, update_intent_priority, update_intent_payload,
    )
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from src.db.client import get_supabase

logger = logging.getLogger(__name__)


# Free text in the schema; these are the values the coach prompt suggests.
KNOWN_INTENT_TYPES: frozenset[str] = frozenset({
    "event", "capacity", "habit", "recovery",
})

KNOWN_INTENT_STATUSES: frozenset[str] = frozenset({
    "active", "paused", "achieved", "abandoned",
})


def _now_iso() -> str:
    """Return current UTC time as ISO 8601 string for ``updated_at`` writes."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


def list_intents(
    user_id: str,
    *,
    status_filter: str | None = "active",
) -> list[dict]:
    """Return intents for ``user_id`` filtered by status.

    Args:
        user_id: UUID of the owning user.
        status_filter: status to filter on, or ``None`` for all statuses.
            Defaults to ``"active"`` (the common coach query).

    Returns:
        List of intent dicts ordered by priority asc, then ``created_at`` desc.
        Empty list on any error.
    """
    try:
        query = (
            get_supabase()
            .table("intents")
            .select("*")
            .eq("user_id", user_id)
        )
        if status_filter is not None:
            query = query.eq("status", status_filter)

        result = (
            query
            .order("priority")
            .order("created_at", desc=True)
            .execute()
        )
        return list(result.data or [])
    except Exception:
        logger.exception("list_intents failed for user %s", user_id[:8])
        return []


def get_intent(intent_id: str) -> dict | None:
    """Return a single intent by id, or ``None`` if not found."""
    try:
        result = (
            get_supabase()
            .table("intents")
            .select("*")
            .eq("id", intent_id)
            .maybe_single()
            .execute()
        )
        return result.data if result is not None else None
    except Exception:
        logger.exception("get_intent failed for id %s", intent_id)
        return None


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------


def insert_intent(
    user_id: str,
    *,
    type: str,
    title: str,
    priority: int = 3,
    payload: dict | None = None,
    status: str = "active",
) -> dict:
    """Insert a new intent and return the inserted row.

    Args:
        user_id: UUID of the owning user.
        type: free text, hinted: ``event | capacity | habit | recovery``.
        title: short human-readable description (the athlete's commitment).
        priority: 1 to 5 (1 = highest). Clamped to that range.
        payload: optional type-specific data dict.
        status: initial status, defaults to ``active``.

    Returns:
        The inserted row as a dict.
    """
    safe_priority = max(1, min(5, int(priority)))

    row: dict = {
        "user_id": user_id,
        "type": (type or "").strip() or "other",
        "title": (title or "").strip(),
        "priority": safe_priority,
        "payload": dict(payload or {}),
        "status": (status or "active").strip() or "active",
    }
    result = (
        get_supabase()
        .table("intents")
        .insert(row)
        .execute()
    )
    return dict(result.data[0])


def update_intent_status(intent_id: str, status: str) -> dict | None:
    """Update only the ``status`` column on an intent.

    Returns the updated row, or ``None`` if not found.
    """
    payload = {
        "status": (status or "").strip() or "active",
        "updated_at": _now_iso(),
    }
    result = (
        get_supabase()
        .table("intents")
        .update(payload)
        .eq("id", intent_id)
        .execute()
    )
    if result.data:
        return dict(result.data[0])
    return None


def update_intent_priority(intent_id: str, priority: int) -> dict | None:
    """Update only the ``priority`` column on an intent (clamped 1 to 5)."""
    safe_priority = max(1, min(5, int(priority)))
    payload = {
        "priority": safe_priority,
        "updated_at": _now_iso(),
    }
    result = (
        get_supabase()
        .table("intents")
        .update(payload)
        .eq("id", intent_id)
        .execute()
    )
    if result.data:
        return dict(result.data[0])
    return None


def update_intent_payload(intent_id: str, payload: dict) -> dict | None:
    """Replace the ``payload`` JSONB blob on an intent.

    The payload is replaced wholesale (no merge): the coach passes the
    full new state. This keeps the atomic-mutation contract simple.
    """
    body = {
        "payload": dict(payload or {}),
        "updated_at": _now_iso(),
    }
    result = (
        get_supabase()
        .table("intents")
        .update(body)
        .eq("id", intent_id)
        .execute()
    )
    if result.data:
        return dict(result.data[0])
    return None
