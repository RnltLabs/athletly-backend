"""Episode Consolidation — monthly synthesis of weekly reflections.

Consolidates weekly episode reflections into monthly review summaries.
Patterns appearing in 3+ weeks are promoted to Beliefs with confidence 0.8.

Trigger: HeartbeatService checks daily for months with ≥3 weekly_reflections
that don't yet have a monthly_review.

Usage::

    from src.services.episode_consolidation import consolidate_month

    result = await consolidate_month(user_id, "2026-02")
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Minimum weekly reflections needed to produce a monthly review.
_MIN_WEEKLY_REFLECTIONS = 3

# LLM model for consolidation.
_CONSOLIDATION_MODEL = "gemini/gemini-2.5-flash"

_CONSOLIDATION_PROMPT = (
    "You are a sports coaching AI reviewing a month of training reflections.\n\n"
    "Below are weekly training reflections for {month}. Consolidate them into:\n"
    '1. A monthly_summary (3-5 sentences covering key themes, progress, and changes).\n'
    '2. recurring_patterns: list of patterns that appear in 3+ weeks.\n'
    '3. key_metrics: extracted numeric progress indicators.\n\n'
    "Output valid JSON only:\n"
    '{{"monthly_summary": "...", "recurring_patterns": ["..."], "key_metrics": {{"metric": value}}}}\n\n'
    "WEEKLY REFLECTIONS:\n{episodes}"
)


def _format_episodes(episodes: list[dict]) -> str:
    """Format weekly episodes into a readable text block."""
    lines = []
    for ep in episodes:
        period = ep.get("period_start", "?")
        summary = ep.get("summary", "")[:500]
        insights = ep.get("insights", [])
        lines.append(f"Week of {period}:\n{summary}")
        if insights:
            lines.append("Insights: " + "; ".join(str(i) for i in insights[:5]))
        lines.append("")
    return "\n".join(lines)


async def _generate_consolidation(episodes: list[dict], month: str) -> dict:
    """Call LLM to consolidate weekly episodes into a monthly review.

    Returns:
        Dict with ``monthly_summary``, ``recurring_patterns``, ``key_metrics``.
    """
    import json

    text = _format_episodes(episodes)
    prompt = _CONSOLIDATION_PROMPT.format(month=month, episodes=text)

    try:
        from src.agent.llm import chat_completion

        response = await asyncio.to_thread(
            chat_completion,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            model=_CONSOLIDATION_MODEL,
        )

        raw = (response.choices[0].message.content or "").strip()
        if not raw:
            return {"monthly_summary": "", "recurring_patterns": [], "key_metrics": {}}

        # Strip markdown fences
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        parsed = json.loads(raw)
        return {
            "monthly_summary": str(parsed.get("monthly_summary", "")),
            "recurring_patterns": list(parsed.get("recurring_patterns", [])),
            "key_metrics": dict(parsed.get("key_metrics", {})),
        }
    except Exception:
        logger.warning("Episode consolidation LLM call failed", exc_info=True)
        return {"monthly_summary": "", "recurring_patterns": [], "key_metrics": {}}


async def consolidate_month(user_id: str, month: str) -> dict | None:
    """Consolidate weekly reflections for a given month into a monthly review.

    Args:
        user_id: UUID of the user.
        month: Month string in "YYYY-MM" format.

    Returns:
        The consolidation result dict, or None if not enough data or on failure.
    """
    from src.db.episodes_db import list_episodes_for_period, store_episode

    # Load weekly reflections for this month
    period_start = f"{month}-01"
    # End at the first day of next month
    year, mon = int(month[:4]), int(month[5:7])
    if mon == 12:
        period_end = f"{year + 1}-01-01"
    else:
        period_end = f"{year}-{mon + 1:02d}-01"

    episodes = await asyncio.to_thread(
        list_episodes_for_period,
        user_id, "weekly_reflection", period_start, period_end,
    )

    if len(episodes) < _MIN_WEEKLY_REFLECTIONS:
        logger.debug(
            "Month %s has only %d weekly reflections (need %d) — skipping",
            month, len(episodes), _MIN_WEEKLY_REFLECTIONS,
        )
        return None

    result = await _generate_consolidation(episodes, month)

    if not result["monthly_summary"]:
        return None

    # Store as monthly_review episode
    review_episode = {
        "episode_type": "monthly_review",
        "period_start": period_start,
        "period_end": period_end,
        "summary": result["monthly_summary"],
        "insights": result["recurring_patterns"],
    }
    await asyncio.to_thread(store_episode, user_id, review_episode)

    # Promote recurring patterns to beliefs (patterns in 3+ weeks → confidence 0.8)
    promoted_count = 0
    if result["recurring_patterns"]:
        promoted_count = await _promote_patterns_to_beliefs(user_id, result["recurring_patterns"])

    # Record consolidation run in episode_consolidations table
    await _record_consolidation(
        user_id=user_id,
        month=month,
        weekly_episode_count=len(episodes),
        recurring_patterns=result["recurring_patterns"],
        key_metrics=result["key_metrics"],
        beliefs_promoted=promoted_count,
    )

    logger.info(
        "Monthly review for %s/%s: %d patterns, %d metrics, %d beliefs promoted",
        user_id[:8], month,
        len(result["recurring_patterns"]),
        len(result["key_metrics"]),
        promoted_count,
    )

    return result


async def _promote_patterns_to_beliefs(
    user_id: str,
    patterns: list[str],
) -> int:
    """Promote recurring patterns to beliefs with confidence 0.8.

    Returns:
        Number of beliefs created.
    """
    promoted = 0
    try:
        from src.db.user_model_db import UserModelDB

        user_model = await asyncio.to_thread(UserModelDB.load_or_create, user_id)
        existing_beliefs = user_model.get_active_beliefs()
        existing_texts = {b.get("text", "").lower().strip() for b in existing_beliefs}

        for pattern in patterns[:5]:  # Cap at 5 promotions per month
            if pattern.lower().strip() in existing_texts:
                continue
            user_model.add_belief(
                text=pattern,
                category="fitness",
                confidence=0.8,
                source="monthly_consolidation",
            )
            promoted += 1

        if promoted > 0:
            user_model.save()
            logger.info("Promoted %d patterns to beliefs for user %s", promoted, user_id[:8])
    except Exception:
        logger.warning("Pattern promotion failed", exc_info=True)

    return promoted


async def _record_consolidation(
    *,
    user_id: str,
    month: str,
    weekly_episode_count: int,
    recurring_patterns: list,
    key_metrics: dict,
    beliefs_promoted: int,
) -> None:
    """Upsert a row in episode_consolidations to track the consolidation run."""
    from src.db.client import get_supabase

    try:
        await asyncio.to_thread(
            lambda: get_supabase()
            .table("episode_consolidations")
            .upsert(
                {
                    "user_id": user_id,
                    "month": month,
                    "weekly_episode_count": weekly_episode_count,
                    "recurring_patterns": recurring_patterns,
                    "key_metrics": key_metrics,
                    "beliefs_promoted": beliefs_promoted,
                },
                on_conflict="user_id,month",
            )
            .execute()
        )
    except Exception:
        logger.warning("Failed to record consolidation for %s/%s", user_id[:8], month, exc_info=True)


async def get_unconsolidated_months(user_id: str) -> list[str]:
    """Find months that have ≥3 weekly reflections but no monthly review.

    Uses the ``episode_consolidations`` table for an efficient lookup of
    already-consolidated months instead of scanning all episodes.

    Returns:
        List of month strings in "YYYY-MM" format.
    """
    from src.db.client import get_supabase
    from src.db.episodes_db import list_episodes_by_type

    try:
        # Get all weekly reflections
        weekly = await asyncio.to_thread(
            list_episodes_by_type, user_id, "weekly_reflection", limit=100,
        )

        # Get already-consolidated months from the tracking table
        consolidated_result = await asyncio.to_thread(
            lambda: get_supabase()
            .table("episode_consolidations")
            .select("month")
            .eq("user_id", user_id)
            .execute()
        )
        consolidated_months = {row["month"] for row in consolidated_result.data}

        # Extract months from weekly reflections
        weekly_months: dict[str, int] = {}
        for ep in weekly:
            period = ep.get("period_start", "")
            if len(period) >= 7:
                month = period[:7]
                weekly_months[month] = weekly_months.get(month, 0) + 1

        # Find months with enough weeklies but no consolidation record
        unconsolidated = [
            month for month, count in sorted(weekly_months.items())
            if count >= _MIN_WEEKLY_REFLECTIONS and month not in consolidated_months
        ]

        return unconsolidated
    except Exception:
        logger.warning("get_unconsolidated_months failed", exc_info=True)
        return []
