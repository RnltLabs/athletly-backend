"""Retrieve relevant coaching lessons for runtime-context injection.

Algorithm (see DESIGN.md "Retrieval at next session"):

1. Build a retrieval query string from the current runtime context
   (athlete name, sports, goal, recent training summary).
2. Generate a 768-dim embedding via Gemini ``text-embedding-004``.
3. pgvector cosine search via the ``match_coaching_lessons`` RPC.
4. Apply read-time decay: ``effective_confidence = confidence *
   exp(-age_days / 90)``, where ``age_days`` is measured from
   ``max(created_at, last_retrieved_at)``.
5. Re-rank by ``effective_confidence * similarity``.
6. Drop entries below the effective-confidence floor.
7. Format the survivors into a short bullet block.
8. Fire-and-forget update of ``reinforced_count`` and
   ``last_retrieved_at`` for the surfaced lessons.

Fallback paths:

- Embeddings disabled: return the top-N highest-confidence recent
  lessons.
- RPC fails: return [] (caller will simply skip the section).

The module exposes both a sync entry point (``fetch_relevant_lessons``)
used from ``build_runtime_context`` and an async fire-and-forget helper
for the reinforcement update.
"""

from __future__ import annotations

import logging
import math
import os
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


# Tuning knobs
_DEFAULT_TOP_K = 5
_MIN_SIMILARITY = 0.3
_MIN_EFFECTIVE_CONFIDENCE = 0.3
_DECAY_HALFLIFE_DAYS = 90.0
_MAX_BLOCK_CHARS = 2400  # ~600 tokens upper bound for the injected block


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def fetch_relevant_lessons(
    user_id: str,
    query_text: str,
    *,
    top_k: int = _DEFAULT_TOP_K,
) -> list[dict]:
    """Return up to ``top_k`` relevant lessons for ``user_id``.

    Pure read path. Safe to call synchronously from
    :func:`build_runtime_context`. All exceptions are caught and the
    function returns ``[]`` on failure.

    Args:
        user_id: Athlete UUID.
        query_text: Current context blurb to retrieve against.
        top_k: Max lessons to return.

    Returns:
        List of lesson dicts (each contains topic, lesson,
        applicable_to, similarity, effective_confidence, id). Empty list
        if no lessons exist or retrieval failed.
    """
    try:
        from src.db.lessons_db import (
            list_active_lessons,
            match_lessons_by_embedding,
        )

        embedding = _generate_embedding(query_text) if query_text else None

        if embedding is None:
            # Fallback path: recency + raw confidence.
            recent = list_active_lessons(user_id, limit=top_k * 2)
            scored = []
            for row in recent:
                effective = _effective_confidence(row)
                if effective < _MIN_EFFECTIVE_CONFIDENCE:
                    continue
                scored.append((effective, row))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [
                {
                    "id": r.get("id"),
                    "topic": r.get("topic"),
                    "lesson": r.get("lesson"),
                    "applicable_to": r.get("applicable_to") or [],
                    "similarity": 1.0,
                    "effective_confidence": eff,
                }
                for eff, r in scored[:top_k]
            ]

        matches = match_lessons_by_embedding(
            user_id, embedding,
            match_count=top_k * 2,
            min_confidence=0.0,
        )
        scored = []
        for row in matches:
            similarity = float(row.get("similarity") or 0.0)
            if similarity < _MIN_SIMILARITY:
                continue
            effective = _effective_confidence(row)
            if effective < _MIN_EFFECTIVE_CONFIDENCE:
                continue
            score = effective * similarity
            scored.append((score, similarity, effective, row))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [
            {
                "id": row.get("id"),
                "topic": row.get("topic"),
                "lesson": row.get("lesson"),
                "applicable_to": row.get("applicable_to") or [],
                "similarity": similarity,
                "effective_confidence": effective,
            }
            for _, similarity, effective, row in scored[:top_k]
        ]
    except Exception:
        logger.warning(
            "fetch_relevant_lessons failed for user %s",
            user_id[:8],
            exc_info=True,
        )
        return []


def format_lessons_block(lessons: list[dict]) -> str:
    """Render a list of lesson dicts as a runtime-context block.

    Returns an empty string when ``lessons`` is empty. Otherwise emits:

        # Lessons learned about this athlete
        - [topic] lesson text
        - [topic] lesson text

    The total block is capped at ``_MAX_BLOCK_CHARS`` chars; lessons
    beyond the cap are silently dropped.
    """
    if not lessons:
        return ""
    lines: list[str] = ["# Lessons learned about this athlete"]
    running = len(lines[0]) + 1
    for entry in lessons:
        topic = entry.get("topic") or "other"
        lesson_text = (entry.get("lesson") or "").strip()
        if not lesson_text:
            continue
        bullet = f"- [{topic}] {lesson_text}"
        if running + len(bullet) + 1 > _MAX_BLOCK_CHARS:
            break
        lines.append(bullet)
        running += len(bullet) + 1
    if len(lines) == 1:
        return ""
    return "\n".join(lines)


def build_query_text_from_runtime(
    *,
    athlete_name: str | None,
    sports: list[str] | None,
    goal_event: str | None,
    recent_summary: str | None,
) -> str:
    """Stitch a compact retrieval query from runtime-context fields."""
    parts: list[str] = []
    if athlete_name:
        parts.append(f"Athlete: {athlete_name}")
    if sports:
        parts.append("Sports: " + ", ".join(sports))
    if goal_event:
        parts.append(f"Goal: {goal_event}")
    if recent_summary:
        parts.append(recent_summary[:500])
    return "\n".join(parts).strip()


def update_reinforcement(lesson_ids: list[str]) -> None:
    """Best-effort update of reinforced_count + last_retrieved_at.

    Synchronous helper; call via ``asyncio.to_thread`` from async code.
    """
    if not lesson_ids:
        return
    try:
        from src.db.lessons_db import mark_lessons_retrieved
        mark_lessons_retrieved(lesson_ids)
    except Exception:
        logger.debug("update_reinforcement failed", exc_info=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _effective_confidence(row: dict) -> float:
    """Apply read-time decay to a lesson row's confidence.

    Age origin is ``max(created_at, last_retrieved_at)``. A lesson with
    no parseable timestamps gets full confidence (we assume it is brand
    new).
    """
    try:
        confidence = float(row.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    if confidence <= 0.0:
        return 0.0

    created_at = _parse_iso(row.get("created_at"))
    last_retrieved = _parse_iso(row.get("last_retrieved_at"))
    age_origin = max(filter(None, [created_at, last_retrieved]), default=None)
    if age_origin is None:
        return confidence

    now = datetime.now(timezone.utc)
    age_days = max(0.0, (now - age_origin).total_seconds() / 86400.0)
    decay = math.exp(-age_days / _DECAY_HALFLIFE_DAYS)
    return confidence * decay


def _parse_iso(value: object) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        # Supabase returns ISO strings like 2026-05-14T08:21:00+00:00 or
        # ending in Z. Both are handled by fromisoformat from Py 3.11+.
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _generate_embedding(text: str) -> list[float] | None:
    """Generate a 768-dim Gemini embedding for ``text``.

    Returns ``None`` when embeddings are disabled or the call fails. We
    duplicate this helper from ``user_model_db`` and ``reflexion`` so
    the retrieval path has no cross-module dependency on a class.

    TODO: extract a shared embedding service.
    """
    if os.environ.get("ATHLETLY_EMBEDDINGS_DISABLED") == "1":
        return None
    if not os.environ.get("GEMINI_API_KEY"):
        return None
    try:
        from src.agent.llm import get_client

        client = get_client()
        response = client.models.embed_content(
            model="models/text-embedding-004",
            contents=text or "",
        )
        return list(response.embeddings[0].values)
    except Exception:
        logger.debug("lesson retrieval embedding failed", exc_info=True)
        return None
