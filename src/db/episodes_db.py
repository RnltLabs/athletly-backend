"""Supabase-backed episode storage replacing local JSON files.

Mirrors the public API of ``src/memory/episodes.store_episode`` /
``list_episodes`` but persists to the ``episodes`` Supabase table.

Each episode is embedded at write time via
``src.services.embeddings.embed_text`` so the agent's per-turn replay
retrieval has vectors to search. Embedding failure is non-fatal - the
episode is stored without an embedding and replay simply skips it.

Usage::

    from src.db.episodes_db import store_episode, list_episodes

    ep = store_episode(user_id, {"summary": "Good week", ...})
    recent = list_episodes(user_id)
"""

from __future__ import annotations

import logging

from src.db.client import get_supabase

logger = logging.getLogger(__name__)


def _build_embedding_text(episode: dict) -> str:
    """Synthesize the text we embed for similarity search.

    Summary plus the first few insights. Keeps the signal dense without
    inflating cost.
    """
    parts: list[str] = []
    summary = (episode.get("summary") or "").strip()
    if summary:
        parts.append(summary)

    insights = episode.get("insights") or []
    if insights:
        lines = ["Insights:"]
        for ins in insights[:5]:
            text = str(ins).strip()
            if text:
                lines.append(f"- {text}")
        if len(lines) > 1:
            parts.append("\n".join(lines))

    return "\n\n".join(parts)


def store_episode(user_id: str, episode: dict) -> dict:
    """Store an episode (training-block reflection) to Supabase.

    Args:
        user_id: UUID of the owning user.
        episode: Dict containing at least ``summary``.  Optional keys:
                 ``episode_type``, ``period_start`` / ``period``,
                 ``period_end``, ``insights``, ``utility``.

    Returns:
        The inserted row as a dict (includes server-generated ``id`` and
        ``created_at``).
    """
    row: dict = {
        "user_id": user_id,
        "episode_type": episode.get("episode_type", "weekly_reflection"),
        "period_start": episode.get("period_start") or episode.get("period"),
        "period_end": episode.get("period_end") or episode.get("period"),
        "summary": episode.get("summary", ""),
        "insights": episode.get("insights", []),
    }

    # Optional utility passthrough (callers may pre-score).
    if "utility" in episode:
        try:
            row["utility"] = float(episode["utility"])
        except (TypeError, ValueError):
            pass

    # Embed at write time so replay retrieval has a vector to match. Any
    # failure here is silently absorbed - the episode is still stored.
    embedding = _maybe_generate_embedding(episode)
    if embedding is not None:
        row["embedding"] = embedding

    result = get_supabase().table("episodes").insert(row).execute()
    return result.data[0]


def _maybe_generate_embedding(episode: dict) -> list[float] | None:
    """Compute the embedding for *episode* if a provider is configured."""
    try:
        from src.services.embeddings import embed_text
    except Exception:
        logger.debug("embeddings service unavailable", exc_info=True)
        return None

    text = _build_embedding_text(episode)
    if not text:
        return None

    return embed_text(text)


def backfill_embedding(episode_id: str, user_id: str) -> bool:
    """Generate and persist an embedding for an existing episode.

    Useful for one-shot backfill scripts. Returns ``True`` on success,
    ``False`` if the episode was not found or embedding generation failed.
    """
    try:
        client = get_supabase()
        result = (
            client.table("episodes")
            .select("id, summary, insights")
            .eq("id", episode_id)
            .eq("user_id", user_id)
            .maybe_single()
            .execute()
        )
        episode = result.data if result is not None else None
        if not episode:
            return False

        embedding = _maybe_generate_embedding(episode)
        if embedding is None:
            return False

        client.table("episodes").update({"embedding": embedding}).eq(
            "id", episode_id
        ).eq("user_id", user_id).execute()
        return True
    except Exception:
        logger.warning(
            "backfill_embedding failed for episode=%s", episode_id, exc_info=True
        )
        return False


def list_episodes(user_id: str, limit: int = 10) -> list[dict]:
    """List stored episodes for a user, most recent period first.

    Args:
        user_id: UUID of the owning user.
        limit: Maximum number of episodes to return.

    Returns:
        List of episode dicts ordered by ``period_end`` descending.
    """
    result = (
        get_supabase()
        .table("episodes")
        .select("*")
        .eq("user_id", user_id)
        .order("period_end", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data


def get_episode(user_id: str, episode_id: str) -> dict | None:
    """Get a single episode by ID.

    Args:
        user_id: UUID of the owning user.
        episode_id: UUID of the episode.

    Returns:
        Episode dict or ``None`` if not found.
    """
    result = (
        get_supabase()
        .table("episodes")
        .select("*")
        .eq("id", episode_id)
        .eq("user_id", user_id)
        .maybe_single()
        .execute()
    )
    # maybe_single().execute() returns None when no row is found
    return result.data if result is not None else None


def list_episodes_for_period(
    user_id: str,
    episode_type: str,
    period_start: str,
    period_end: str,
) -> list[dict]:
    """List episodes of a given type within a date range.

    Args:
        user_id: UUID of the owning user.
        episode_type: E.g. ``"weekly_reflection"``.
        period_start: Start date (inclusive, ISO format).
        period_end: End date (exclusive, ISO format).

    Returns:
        List of episode dicts ordered by ``period_start`` ascending.
    """
    result = (
        get_supabase()
        .table("episodes")
        .select("*")
        .eq("user_id", user_id)
        .eq("episode_type", episode_type)
        .gte("period_start", period_start)
        .lt("period_start", period_end)
        .order("period_start", desc=False)
        .execute()
    )
    return result.data


def list_episodes_by_type(
    user_id: str,
    episode_type: str,
    limit: int = 10,
) -> list[dict]:
    """List episodes filtered by type.

    Args:
        user_id: UUID of the owning user.
        episode_type: E.g. ``"weekly_reflection"``, ``"monthly_review"``.
        limit: Maximum number of episodes to return.

    Returns:
        List of episode dicts ordered by ``period_end`` descending.
    """
    result = (
        get_supabase()
        .table("episodes")
        .select("*")
        .eq("user_id", user_id)
        .eq("episode_type", episode_type)
        .order("period_end", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data
