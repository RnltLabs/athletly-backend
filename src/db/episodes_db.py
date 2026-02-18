"""Supabase-backed episode storage replacing local JSON files.

Mirrors the public API of ``src/memory/episodes.store_episode`` /
``list_episodes`` but persists to the ``episodes`` Supabase table.

Usage::

    from src.db.episodes_db import store_episode, list_episodes

    ep = store_episode(user_id, {"summary": "Good week", ...})
    recent = list_episodes(user_id)
"""

from __future__ import annotations

import logging

from src.db.client import get_supabase

logger = logging.getLogger(__name__)


def store_episode(user_id: str, episode: dict) -> dict:
    """Store an episode (training-block reflection) to Supabase.

    Args:
        user_id: UUID of the owning user.
        episode: Dict containing at least ``summary``.  Optional keys:
                 ``episode_type``, ``period_start`` / ``period``,
                 ``period_end``, ``insights``.

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

    result = get_supabase().table("episodes").insert(row).execute()
    return result.data[0]


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
