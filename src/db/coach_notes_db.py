"""Supabase-backed CRUD for the ``coach_notes`` table.

APPEND-ONLY by contract. The coach writes a new note every time something
changes; there is no update operation and no delete-via-API. This is a
load-bearing invariant: the history is itself signal (how the coach's
understanding of the athlete evolved).

Pattern matches ``src/db/plans_db.py``.

Usage::

    from src.db.coach_notes_db import insert_coach_note, list_coach_notes
"""

from __future__ import annotations

import logging

from src.db.client import get_supabase

logger = logging.getLogger(__name__)


# Free text in the schema; these are the suggested values for ``scope``.
KNOWN_NOTE_SCOPES: frozenset[str] = frozenset({
    "training_response", "life_pattern", "preference_learned",
})


# ---------------------------------------------------------------------------
# Writes (insert only)
# ---------------------------------------------------------------------------


def insert_coach_note(
    user_id: str,
    *,
    content: str,
    scope: str | None = None,
    source_session_id: str | None = None,
) -> dict:
    """Append a new coach note row and return it.

    Args:
        user_id: UUID of the owning user.
        content: free-form learning text (German or English, the
            athlete's language).
        scope: optional grouping label, e.g. ``training_response``.
        source_session_id: optional UUID of the session that prompted
            the note (may be ``None`` for off-session observations).

    Returns:
        The inserted row as a dict.
    """
    row: dict = {
        "user_id": user_id,
        "content": (content or "").strip(),
    }
    if scope is not None and str(scope).strip():
        row["scope"] = str(scope).strip()
    if source_session_id is not None and str(source_session_id).strip():
        row["source_session_id"] = str(source_session_id).strip()

    result = (
        get_supabase()
        .table("coach_notes")
        .insert(row)
        .execute()
    )
    return dict(result.data[0])


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


def list_coach_notes(
    user_id: str,
    *,
    limit: int = 20,
    scope_filter: str | None = None,
) -> list[dict]:
    """Return coach notes newest first.

    Args:
        user_id: UUID of the owning user.
        limit: maximum number of notes to return (default 20).
        scope_filter: when provided, restrict to that scope.

    Returns:
        List of note dicts ordered by ``created_at`` desc.
        Empty list on any error.
    """
    safe_limit = max(1, min(int(limit), 200))
    try:
        query = (
            get_supabase()
            .table("coach_notes")
            .select("*")
            .eq("user_id", user_id)
        )
        if scope_filter is not None and scope_filter.strip():
            query = query.eq("scope", scope_filter.strip())

        result = (
            query
            .order("created_at", desc=True)
            .limit(safe_limit)
            .execute()
        )
        return list(result.data or [])
    except Exception:
        logger.exception("list_coach_notes failed for user %s", user_id[:8])
        return []
