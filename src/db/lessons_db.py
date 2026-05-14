"""Supabase-backed CRUD for the Reflexion lesson store.

Two tables:

- ``coaching_lessons``: per-user lessons distilled by the reflection LLM.
  Stored with a 768-dim pgvector embedding for semantic retrieval.
- ``reflexion_runs``: idempotency log. Prevents double-running a
  reflection for the same (user_id, run_kind, period) tuple.

All functions are synchronous (Supabase client is sync). Call them via
``asyncio.to_thread`` from async contexts so the SSE event loop is not
blocked.

Usage::

    from src.db.lessons_db import (
        insert_lesson, list_active_lessons,
        match_lessons_by_embedding, mark_lessons_retrieved,
        record_reflexion_run, has_recent_pro_run_today,
    )

The module never raises on transient Supabase errors; it logs and
returns sentinel values (empty list, ``False``) so the calling service
can degrade gracefully.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from src.db.client import get_supabase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants matching the SQL CHECK constraint
# ---------------------------------------------------------------------------

LESSON_TOPICS: frozenset[str] = frozenset({
    "scheduling",
    "volume",
    "intensity",
    "recovery",
    "nutrition",
    "injury",
    "motivation",
    "preference",
    "constraint",
    "other",
})

RUN_KINDS: frozenset[str] = frozenset({"pro_session", "free_monthly"})


# ---------------------------------------------------------------------------
# coaching_lessons CRUD
# ---------------------------------------------------------------------------


def insert_lesson(
    user_id: str,
    *,
    topic: str,
    observation: str,
    lesson: str,
    applicable_to: list[str],
    evidence: str,
    confidence: float,
    embedding: list[float] | None,
    source_session_id: str | None = None,
) -> dict | None:
    """Insert a single lesson row.

    Returns the inserted row dict, or ``None`` on failure.
    Topic is coerced to ``other`` if it is not in :data:`LESSON_TOPICS`.
    """
    safe_topic = topic if topic in LESSON_TOPICS else "other"
    clamped_confidence = max(0.0, min(1.0, float(confidence)))

    row: dict = {
        "user_id": user_id,
        "topic": safe_topic,
        "observation": observation[:2000],
        "lesson": lesson[:2000],
        "applicable_to": list(applicable_to or []),
        "evidence": evidence[:2000],
        "confidence": clamped_confidence,
        "embedding": embedding,
        "source_session_id": source_session_id,
    }

    try:
        result = (
            get_supabase()
            .table("coaching_lessons")
            .insert(row)
            .execute()
        )
        return result.data[0] if result.data else None
    except Exception:
        logger.exception("insert_lesson failed for user %s", user_id[:8])
        return None


def list_active_lessons(
    user_id: str,
    *,
    limit: int = 50,
    topic: str | None = None,
) -> list[dict]:
    """Return active lessons for ``user_id`` ordered by recency."""
    try:
        query = (
            get_supabase()
            .table("coaching_lessons")
            .select("*")
            .eq("user_id", user_id)
            .eq("active", True)
        )
        if topic and topic in LESSON_TOPICS:
            query = query.eq("topic", topic)

        result = (
            query
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return result.data or []
    except Exception:
        logger.exception("list_active_lessons failed for user %s", user_id[:8])
        return []


def match_lessons_by_embedding(
    user_id: str,
    embedding: list[float],
    *,
    match_count: int = 5,
    min_confidence: float = 0.0,
) -> list[dict]:
    """Cosine-similarity search via the ``match_coaching_lessons`` RPC.

    Returns rows shaped like::

        {id, topic, lesson, applicable_to, confidence,
         reinforced_count, last_retrieved_at, created_at, similarity}

    On any error returns an empty list (caller falls back to recency).
    """
    if not embedding:
        return []

    try:
        result = (
            get_supabase()
            .rpc(
                "match_coaching_lessons",
                {
                    "p_user_id": user_id,
                    "p_embedding": embedding,
                    "p_match_count": match_count,
                    "p_min_confidence": min_confidence,
                },
            )
            .execute()
        )
        return result.data or []
    except Exception:
        logger.exception(
            "match_coaching_lessons RPC failed for user %s", user_id[:8],
        )
        return []


def mark_lessons_retrieved(lesson_ids: list[str]) -> None:
    """Bump ``reinforced_count`` and refresh ``last_retrieved_at``.

    Performs one UPDATE per id (no batch RPC). Failures are swallowed.
    """
    if not lesson_ids:
        return
    now_iso = datetime.now(timezone.utc).isoformat()
    db = get_supabase()
    for lesson_id in lesson_ids:
        try:
            # Two-step update because PostgREST cannot express
            # "reinforced_count = reinforced_count + 1" without an RPC.
            # We read-modify-write; concurrent retrievals are rare and a
            # small undercount is acceptable.
            current = (
                db.table("coaching_lessons")
                .select("reinforced_count")
                .eq("id", lesson_id)
                .maybe_single()
                .execute()
            )
            existing = (current.data if current is not None else None) or {}
            new_count = int(existing.get("reinforced_count") or 0) + 1
            db.table("coaching_lessons").update(
                {
                    "reinforced_count": new_count,
                    "last_retrieved_at": now_iso,
                    "updated_at": now_iso,
                }
            ).eq("id", lesson_id).execute()
        except Exception:
            logger.debug(
                "mark_lessons_retrieved failed for id %s", lesson_id,
                exc_info=True,
            )


def supersede_lesson(old_id: str, new_id: str) -> None:
    """Mark ``old_id`` as superseded by ``new_id``.

    Sets ``active = false``, ``archived_at = now()``, and
    ``superseded_by_id = new_id``. Failures are swallowed.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    try:
        get_supabase().table("coaching_lessons").update(
            {
                "active": False,
                "archived_at": now_iso,
                "superseded_by_id": new_id,
                "updated_at": now_iso,
            }
        ).eq("id", old_id).execute()
    except Exception:
        logger.debug("supersede_lesson failed for %s", old_id, exc_info=True)


# ---------------------------------------------------------------------------
# reflexion_runs (idempotency)
# ---------------------------------------------------------------------------


def record_reflexion_run(
    user_id: str,
    *,
    run_kind: str,
    period: str,
    lessons_added: int,
) -> bool:
    """Insert a row marking that the reflection has run.

    Returns ``True`` on success, ``False`` if the row already existed
    (idempotency violation) or on any other failure. Callers should
    interpret ``False`` as "skip - already done".
    """
    if run_kind not in RUN_KINDS:
        logger.warning("record_reflexion_run: bad run_kind %s", run_kind)
        return False

    row = {
        "user_id": user_id,
        "run_kind": run_kind,
        "period": period,
        "lessons_added": int(lessons_added),
    }

    try:
        get_supabase().table("reflexion_runs").insert(row).execute()
        return True
    except Exception:
        # Most common cause: PRIMARY KEY conflict, i.e. already done.
        # We do not differentiate at this layer; the caller treats both
        # "duplicate" and "DB down" the same way (skip).
        logger.debug(
            "record_reflexion_run conflict or failure for user=%s kind=%s period=%s",
            user_id[:8], run_kind, period,
            exc_info=True,
        )
        return False


def has_reflexion_run(
    user_id: str,
    *,
    run_kind: str,
    period: str,
) -> bool:
    """Return ``True`` if a run already exists for this triple."""
    try:
        result = (
            get_supabase()
            .table("reflexion_runs")
            .select("ran_at")
            .eq("user_id", user_id)
            .eq("run_kind", run_kind)
            .eq("period", period)
            .limit(1)
            .execute()
        )
        return bool(result.data)
    except Exception:
        logger.debug("has_reflexion_run check failed", exc_info=True)
        return False


def count_runs_since(
    user_id: str,
    *,
    since_iso: str,
) -> int:
    """Count reflection runs for ``user_id`` since ``since_iso``.

    Used to enforce the per-day rate limit ("at most one reflection per
    user per day, regardless of tier").
    """
    try:
        result = (
            get_supabase()
            .table("reflexion_runs")
            .select("user_id", count="exact")
            .eq("user_id", user_id)
            .gte("ran_at", since_iso)
            .execute()
        )
        # PostgREST exposes total in result.count when count="exact"
        count = getattr(result, "count", None)
        if count is None:
            count = len(result.data or [])
        return int(count)
    except Exception:
        logger.debug("count_runs_since failed", exc_info=True)
        return 0
