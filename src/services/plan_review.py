"""Plan review trigger - kicks off an agent turn after a new activity syncs.

When a new training activity is ingested via ``POST /webhook/activity``, we
spawn a background agent turn that lets the agent decide whether the active
plan still fits. The decision (and any follow-up action: adjusting the plan,
sending a notification, doing nothing) is entirely the agent's own reasoning
from real data. This module is pure infrastructure: it fires the trigger,
nothing more.

Concurrency:
    Uses the same Redis lock key as ``src/api/routers/chat.py`` so the
    review never runs while the athlete is actively chatting. If the lock
    is held, the review is silently skipped - the agent will see the new
    activity the next time the user opens the chat.

Failure handling:
    Any exception is logged and swallowed. The caller (webhook handler)
    must always return 200 to the upstream provider.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import redis.asyncio as aioredis

from src.config import get_settings

logger = logging.getLogger(__name__)


# Lock key MUST match the one in src/api/routers/chat.py so chat and
# review serialize against each other for the same user.
_LOCK_KEY_TEMPLATE = "agent_loop:lock:{user_id}"
_LOCK_TTL_SECONDS = 300


def schedule_plan_review(user_id: str, activity_summary: dict[str, Any]) -> None:
    """Fire-and-forget entry point for the webhook handler.

    Schedules ``trigger_plan_review`` on the current asyncio loop without
    awaiting it. Safe to call from inside any async handler: exceptions in
    the spawned task are swallowed by ``trigger_plan_review`` itself.
    """
    try:
        asyncio.create_task(
            trigger_plan_review(user_id, activity_summary),
            name=f"plan_review_{user_id}",
        )
    except RuntimeError:
        # No running event loop - this should never happen from a FastAPI
        # handler, but be defensive: log and move on.
        logger.warning(
            "schedule_plan_review called without a running event loop "
            "(user=%s) - skipping",
            user_id,
        )


async def trigger_plan_review(
    user_id: str,
    activity_summary: dict[str, Any],
) -> None:
    """Spawn an agent turn that reviews the current plan against new data.

    The agent is given a neutral system-style nudge ("a new activity synced,
    look at the active plan, decide if anything should change"). All coaching
    judgment lives inside the agent's own reasoning loop. This function adds
    no thresholds, no rules, no domain logic.

    The turn runs to completion in the background. If the agent calls
    ``send_notification``, the existing push pipeline delivers it. No SSE
    events are emitted - this is server-internal.
    """
    try:
        await _run_review(user_id, activity_summary)
    except Exception:
        logger.exception(
            "Plan review failed for user=%s activity=%s",
            user_id,
            activity_summary.get("id"),
        )


async def _run_review(user_id: str, activity_summary: dict[str, Any]) -> None:
    """Inner worker: acquire lock, build prompt, run agent turn."""
    redis = await _get_redis()
    lock_acquired = await _acquire_lock(redis, user_id)
    if not lock_acquired:
        logger.info(
            "Plan review skipped for user=%s: agent loop lock is held "
            "(user is likely chatting)",
            user_id,
        )
        if redis is not None:
            try:
                await redis.close()
            except Exception:
                pass
        return

    try:
        prompt = _build_review_prompt(activity_summary)
        session_id = _build_session_id(activity_summary)
        await _run_agent_turn(user_id, session_id, prompt)
    finally:
        await _release_lock(redis, user_id)
        if redis is not None:
            try:
                await redis.close()
            except Exception:
                pass


def _build_session_id(activity_summary: dict[str, Any]) -> str:
    """Use a fresh session id so the review does not pollute the chat session."""
    activity_id = activity_summary.get("id") or "unknown"
    return f"review-{activity_id}"


def _build_review_prompt(activity_summary: dict[str, Any]) -> str:
    """Construct the neutral system-style message handed to the agent.

    The message tells the agent WHAT happened and WHICH tools are relevant,
    but contains no coaching prescription. The agent decides what (if
    anything) to do.
    """
    summary_line = _format_activity_line(activity_summary)
    return (
        "[System] A new activity just synced: "
        f"{summary_line}. "
        "Look at the current plan via get_active_plan and decide if it still "
        "fits. Adjust with save_plan if you think it should change. Send a "
        "push notification with send_notification if the athlete should be "
        "told. Otherwise do nothing."
    )


def _format_activity_line(activity_summary: dict[str, Any]) -> str:
    """One-line description of the new activity (id, sport, date, duration)."""
    parts: list[str] = []
    activity_id = activity_summary.get("id")
    if activity_id:
        parts.append(f"id={activity_id}")
    sport = activity_summary.get("sport") or activity_summary.get("activity_type")
    if sport:
        parts.append(f"sport={sport}")
    date = (
        activity_summary.get("date")
        or activity_summary.get("start_time")
        or activity_summary.get("created_at")
    )
    if date:
        parts.append(f"date={date}")
    duration = (
        activity_summary.get("duration_seconds")
        or activity_summary.get("duration_s")
        or activity_summary.get("duration")
    )
    if duration is not None:
        parts.append(f"duration_s={duration}")
    if not parts:
        return "new activity (no details supplied)"
    return ", ".join(parts)


async def _run_agent_turn(user_id: str, session_id: str, prompt: str) -> None:
    """Load the user, start a fresh session, and process a single agent turn.

    Uses ``AsyncAgentLoop`` with a no-op emit function so we get the same
    code path as ``/chat`` without producing SSE side effects.
    """
    from src.agent.agent_loop import AsyncAgentLoop
    from src.db.user_model_db import UserModelDB

    user_model = await asyncio.to_thread(UserModelDB.load_or_create, user_id)
    loop = AsyncAgentLoop(user_model=user_model, context="coach")

    # start_session() may hit Supabase - run in a worker thread.
    await asyncio.to_thread(loop.start_session, session_id)

    async def _noop_emit(event_type: str, data: dict) -> None:
        # Server-internal review: drop all SSE events on the floor.
        return None

    logger.info(
        "Starting plan review agent turn user=%s session=%s",
        user_id,
        session_id,
    )
    result = await loop.process_message_sse(prompt, _noop_emit)
    logger.info(
        "Plan review finished user=%s session=%s tool_calls=%d duration_ms=%d",
        user_id,
        session_id,
        getattr(result, "tool_calls_made", 0),
        getattr(result, "total_duration_ms", 0),
    )


# ---------------------------------------------------------------------------
# Redis lock helpers (mirror chat.py - keep the lock semantics identical).
# ---------------------------------------------------------------------------


async def _get_redis() -> aioredis.Redis | None:
    """Return an async Redis client, or None when Redis is unreachable."""
    try:
        redis_url = get_settings().redis_url
        client = aioredis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=1,
        )
        await client.ping()
        return client
    except Exception:
        logger.warning(
            "Redis unavailable for plan review - falling back to "
            "best-effort lock-free execution",
        )
        return None


async def _acquire_lock(redis: aioredis.Redis | None, user_id: str) -> bool:
    """Acquire the per-user agent loop lock.

    Returns True on success. When Redis is unavailable we cannot coordinate
    with chat sessions, so we proceed without a lock - chat and review may
    overlap in that degraded mode, but that is preferable to never running
    the review at all.
    """
    if redis is None:
        return True

    lock_key = _LOCK_KEY_TEMPLATE.format(user_id=user_id)
    try:
        acquired = await redis.set(lock_key, "1", nx=True, ex=_LOCK_TTL_SECONDS)
    except Exception:
        logger.warning(
            "Failed to acquire plan-review lock for user=%s; "
            "proceeding without a lock",
            user_id,
            exc_info=True,
        )
        return True
    return acquired is True


async def _release_lock(
    redis: aioredis.Redis | None,
    user_id: str,
) -> None:
    """Release the per-user agent loop lock (best-effort)."""
    if redis is None:
        return
    lock_key = _LOCK_KEY_TEMPLATE.format(user_id=user_id)
    try:
        await redis.delete(lock_key)
    except Exception:
        logger.warning(
            "Failed to release plan-review lock for user=%s",
            user_id,
            exc_info=True,
        )
