"""Supabase-backed session persistence replacing JSONL file storage.

Provides equivalents for the ``AgentLoop`` session helpers
(``start_session``, ``_save_turn``, ``_load_session``) but backed by the
``sessions`` and ``session_messages`` Supabase tables.

Usage::

    from src.db.session_store_db import create_session, save_message

    sid = create_session(user_id)
    save_message(sid, user_id, "user", "Hello coach")
"""

from __future__ import annotations

import logging

from src.db.client import get_supabase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------


def create_session(user_id: str) -> str:
    """Create a new session row and return its UUID.

    Args:
        user_id: UUID of the owning user.

    Returns:
        The server-generated session UUID.
    """
    result = (
        get_supabase()
        .table("sessions")
        .insert({"user_id": user_id})
        .execute()
    )
    return result.data[0]["id"]


def get_session(session_id: str) -> dict | None:
    """Fetch a single session by ID.

    Args:
        session_id: UUID of the session.

    Returns:
        Session dict or ``None``.
    """
    result = (
        get_supabase()
        .table("sessions")
        .select("*")
        .eq("id", session_id)
        .maybe_single()
        .execute()
    )
    # maybe_single().execute() returns None when no row is found
    return result.data if result is not None else None


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


def save_message(
    session_id: str,
    user_id: str,
    role: str,
    content: str,
    meta: dict | None = None,
) -> None:
    """Persist a single message to the ``session_messages`` table.

    Also touches ``sessions.last_active`` so the session list stays sorted.

    Args:
        session_id: UUID of the parent session.
        user_id: UUID of the owning user.
        role: One of ``"user"``, ``"model"``, ``"tool_call"``, ``"system"``.
        content: Message text (truncated to 8 000 chars for safety).
        meta: Optional JSON metadata (tool args, timing info, etc.).
    """
    db = get_supabase()

    db.table("session_messages").insert(
        {
            "session_id": session_id,
            "user_id": user_id,
            "role": role,
            "content": content[:8000],
            "meta": meta or {},
        }
    ).execute()

    # Touch the session's last_active timestamp.
    # Using a raw SQL ``now()`` is not directly available via the PostgREST
    # client, so we pass a Python-side UTC timestamp instead.
    from datetime import datetime, timezone

    now_iso = datetime.now(timezone.utc).isoformat()
    db.table("sessions").update({"last_active": now_iso}).eq(
        "id", session_id
    ).execute()


def load_session_messages(session_id: str) -> list[dict]:
    """Load all messages for a session, ordered chronologically.

    Args:
        session_id: UUID of the session.

    Returns:
        List of message dicts (``role``, ``content``, ``meta``, ``created_at``).
    """
    result = (
        get_supabase()
        .table("session_messages")
        .select("*")
        .eq("session_id", session_id)
        .order("id")
        .execute()
    )
    return result.data


# ---------------------------------------------------------------------------
# Session listing / querying
# ---------------------------------------------------------------------------


def get_recent_sessions(user_id: str, limit: int = 10) -> list[dict]:
    """Get the most recent sessions for a user.

    Args:
        user_id: UUID of the owning user.
        limit: Maximum number of sessions to return.

    Returns:
        List of session dicts ordered by ``started_at`` descending.
    """
    result = (
        get_supabase()
        .table("sessions")
        .select("*")
        .eq("user_id", user_id)
        .order("started_at", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data


def update_session_summary(
    session_id: str,
    compressed_summary: str,
    turn_count: int,
    tool_calls_total: int,
) -> None:
    """Update aggregated session metadata (e.g. after compression).

    Args:
        session_id: UUID of the session.
        compressed_summary: Compressed conversation summary text.
        turn_count: Total user turns in the session so far.
        tool_calls_total: Cumulative tool calls across all turns.
    """
    get_supabase().table("sessions").update(
        {
            "compressed_summary": compressed_summary[:10000],
            "turn_count": turn_count,
            "tool_calls_total": tool_calls_total,
        }
    ).eq("id", session_id).execute()
