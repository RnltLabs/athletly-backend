"""Supabase-backed CRUD for the ``programs`` table.

Programs are persisting training structures (gym splits, run blocks,
swim blocks) the coach defines and the athlete reviews. They are
long-lived: the same program can be active for weeks or months.

No destructive delete: a program is retired by flipping ``active`` to
``False``. The historical row stays so the coach can refer back.

Pattern matches ``src/db/plans_db.py``.

Usage::

    from src.db.programs_db import (
        list_programs, insert_program, update_program,
    )
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from src.db.client import get_supabase

logger = logging.getLogger(__name__)


# Free text in the schema; these are the values the coach prompt suggests.
KNOWN_PROGRAM_KINDS: frozenset[str] = frozenset({
    "gym_split", "run_block", "swim_block", "bike_block",
})


def _now_iso() -> str:
    """Return current UTC time as ISO 8601 string for ``updated_at`` writes."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


def list_programs(user_id: str, *, active_only: bool = True) -> list[dict]:
    """Return programs for ``user_id``.

    Args:
        user_id: UUID of the owning user.
        active_only: when ``True`` (default), only ``active = TRUE`` rows.

    Returns:
        List of program dicts ordered by ``created_at`` desc.
        Empty list on any error.
    """
    try:
        query = (
            get_supabase()
            .table("programs")
            .select("*")
            .eq("user_id", user_id)
        )
        if active_only:
            query = query.eq("active", True)

        result = (
            query
            .order("created_at", desc=True)
            .execute()
        )
        return list(result.data or [])
    except Exception:
        logger.exception("list_programs failed for user %s", user_id[:8])
        return []


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------


def insert_program(
    user_id: str,
    *,
    kind: str,
    label: str = "",
    payload: dict | None = None,
    active: bool = True,
) -> dict:
    """Insert a new program and return the inserted row.

    Args:
        user_id: UUID of the owning user.
        kind: free text, hinted: ``gym_split | run_block | swim_block``.
        label: optional human-readable name (e.g. "Push/Pull/Legs").
        payload: optional kind-specific data dict.
        active: defaults to ``True``.

    Returns:
        The inserted row as a dict.
    """
    row: dict = {
        "user_id": user_id,
        "kind": (kind or "").strip() or "other",
        "label": (label or "").strip() or None,
        "payload": dict(payload or {}),
        "active": bool(active),
    }
    result = (
        get_supabase()
        .table("programs")
        .insert(row)
        .execute()
    )
    return dict(result.data[0])


def update_program(
    program_id: str,
    *,
    label: str | None = None,
    payload: dict | None = None,
    active: bool | None = None,
) -> dict | None:
    """Patch one or more mutable fields on a program.

    Args:
        program_id: UUID of the program.
        label: when not ``None``, replaces the label.
        payload: when not ``None``, replaces the JSONB payload entirely.
        active: when not ``None``, flips the active flag.

    Returns:
        The updated row, or ``None`` if the program does not exist.
        Returns the row unchanged if no fields were supplied.
    """
    body: dict = {"updated_at": _now_iso()}
    if label is not None:
        body["label"] = label.strip() or None
    if payload is not None:
        body["payload"] = dict(payload)
    if active is not None:
        body["active"] = bool(active)

    # If nothing besides updated_at would change, still apply it so callers
    # can use this as a touch operation.
    result = (
        get_supabase()
        .table("programs")
        .update(body)
        .eq("id", program_id)
        .execute()
    )
    if result.data:
        return dict(result.data[0])
    return None
