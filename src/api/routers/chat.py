"""Chat SSE endpoint for the Athletly FastAPI backend.

Exposes:
    POST /chat         — streams agent responses as Server-Sent Events
    POST /chat/confirm — stores a user confirmation for checkpoint flow

Architecture:
    - One AsyncAgentLoop per request (stateless HTTP + Supabase persistence)
    - Redis distributed lock prevents concurrent requests for the same user
    - Falls back to an in-process dict lock when Redis is unavailable
    - Error events are emitted over SSE but NEVER persisted to session history
"""

import asyncio
import json
import logging
import re
from typing import Annotated, AsyncGenerator, Awaitable, Callable

import redis.asyncio as aioredis
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse, ServerSentEvent

from src.agent.agent_loop import AsyncAgentLoop
from src.api.auth import get_current_user
from src.api.sse import SSEEmitter
from src.config import get_settings
from src.db.user_model_db import UserModelDB

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])

# ---------------------------------------------------------------------------
# In-process fallback stores (used when Redis is unavailable).
# ---------------------------------------------------------------------------
_in_process_locks: dict[str, bool] = {}
_in_process_confirmations: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    """Body for POST /chat."""

    message: str = Field(..., min_length=1, max_length=10_000)
    session_id: str | None = Field(default=None)
    context: str = Field(
        default="coach",
        pattern=r"^(coach|onboarding)$",
    )


class ConfirmRequest(BaseModel):
    """Body for POST /chat/confirm."""

    session_id: str = Field(..., min_length=1)
    action_id: str = Field(..., min_length=1)
    confirmed: bool


# ---------------------------------------------------------------------------
# Redis helpers
# ---------------------------------------------------------------------------


async def _get_redis() -> aioredis.Redis | None:
    """Return an async Redis client or None when Redis is unreachable."""
    try:
        redis_url = get_settings().redis_url
        client = aioredis.from_url(redis_url, decode_responses=True, socket_connect_timeout=1)
        await client.ping()
        return client
    except Exception:
        logger.warning("Redis unavailable — falling back to in-process lock")
        return None


async def _acquire_lock(redis: aioredis.Redis | None, user_id: str) -> bool:
    """Acquire an exclusive lock for *user_id*.

    Returns True on success, False when the user already has a request in
    flight (caller should immediately return an SSE error event).

    Lock TTL: 300 s (the maximum allowed agent processing time).
    """
    lock_key = f"agent_loop:lock:{user_id}"

    if redis is not None:
        acquired = await redis.set(lock_key, "1", nx=True, ex=300)
        return acquired is True

    if _in_process_locks.get(user_id):
        return False
    _in_process_locks[user_id] = True
    return True


async def _release_lock(redis: aioredis.Redis | None, user_id: str) -> None:
    """Release the exclusive lock for *user_id*."""
    lock_key = f"agent_loop:lock:{user_id}"

    if redis is not None:
        try:
            await redis.delete(lock_key)
        except Exception:
            logger.warning("Failed to release Redis lock for user %s", user_id)
        return

    _in_process_locks.pop(user_id, None)


# ---------------------------------------------------------------------------
# Assistant-text sanitizer
# ---------------------------------------------------------------------------
#
# The model sometimes leaks Markdown formatting (asterisks, headings) or
# em/en-dashes despite the system-prompt rules. The frontend renders chat
# text literally, so these characters appear as raw glyphs in the UI.
# Strip them server-side before emitting the `message` SSE event.
#
# Design choice: sanitize ONLY the outgoing SSE payload. The raw text is
# still persisted to `session_messages` by the agent loop, so the original
# model output is preserved for replay, debugging, and re-evaluation. The
# client never sees the raw form.

# **bold**  or  __bold__   (non-greedy, single line)
_BOLD_RE = re.compile(r"(?:\*\*|__)(.+?)(?:\*\*|__)")

# *italic*  -- single asterisks NOT acting as a list bullet.
# We require a non-space character on BOTH sides of the asterisks so that
# a line that starts with "* item" (Markdown bullet) is left alone.
_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\s)([^*\n]+?)(?<!\s)\*(?!\*)")

# `# heading`, `## heading`, ... at start of a line.
_HEADING_RE = re.compile(r"(?m)^[ \t]{0,3}#{1,6}[ \t]+")


def _sanitize_assistant_text(text: str) -> str:
    """Strip Markdown formatting and em/en-dashes from an assistant reply.

    The frontend renders SSE message text literally. Any Markdown bleed
    (``**bold**``, ``# heading``) or em/en-dash glyphs show up as raw
    characters. This helper normalises them to plain prose before the
    `message` event is emitted.

    Rules:
      - ``**bold**`` and ``__bold__`` -> inner text only.
      - ``*italic*`` -> inner text only, but a leading ``* `` bullet at
        the start of a line is left untouched.
      - ``#``..``######`` heading markers at start of line are removed.
      - U+2014 (em-dash) and U+2013 (en-dash) -> ``-`` (hyphen-minus).

    Args:
        text: The raw assistant text emitted by the agent loop.

    Returns:
        The sanitised text safe for SSE delivery to the client.
    """
    if not text:
        return text

    # Bold first so that the italic pass does not match the inner pair of
    # asterisks from a `**word**` construct.
    cleaned = _BOLD_RE.sub(r"\1", text)
    cleaned = _ITALIC_RE.sub(r"\1", cleaned)
    cleaned = _HEADING_RE.sub("", cleaned)
    cleaned = cleaned.replace("—", "-").replace("–", "-")
    return cleaned


# ---------------------------------------------------------------------------
# SSE event generator
# ---------------------------------------------------------------------------


def _make_sse_event(event_type: str, data: dict) -> ServerSentEvent:
    """Build a typed ServerSentEvent from a raw event_type / data pair."""
    if event_type == "thinking":
        return SSEEmitter.thinking(data.get("text", ""))
    if event_type in ("tool_call", "tool_hint"):
        return SSEEmitter.tool_hint(
            name=data.get("name", ""),
            args=data.get("args", {}),
            display_label=data.get("display_label", ""),
            group_id=data.get("group_id", ""),
        )
    if event_type == "tool_group_start":
        return SSEEmitter.tool_group_start(group_id=data.get("group_id", ""))
    if event_type == "tool_group_end":
        return SSEEmitter.tool_group_end(group_id=data.get("group_id", ""))
    if event_type == "tool_result":
        return ServerSentEvent(
            event="tool_result",
            data=json.dumps(data, ensure_ascii=False),
        )
    if event_type == "tool_error":
        return ServerSentEvent(
            event="tool_error",
            data=json.dumps(data, ensure_ascii=False),
        )
    if event_type == "message":
        return SSEEmitter.message(_sanitize_assistant_text(data.get("text", "")))
    if event_type == "error":
        return SSEEmitter.error(
            message=data.get("message", "Unknown error"),
            code=data.get("code", "internal_error"),
        )
    if event_type == "action_request":
        return SSEEmitter.action_request(
            action_type=data.get("action_type", ""),
            label=data.get("label", ""),
            payload=data.get("payload") or {},
        )
    if event_type == "ui_component":
        return SSEEmitter.ui_component(
            component_type=data.get("type", ""),
            ui_id=data.get("id", ""),
            props=data.get("props") or {},
        )
    if event_type == "critic_review":
        # Constitutional critique annotation (Feature 2). Emitted only
        # when the critic flagged a violation and the regenerated draft
        # still violated. Frontend should surface in a debug/admin panel,
        # not in the main chat thread.
        return ServerSentEvent(
            event="critic_review",
            data=json.dumps(data, ensure_ascii=False),
        )
    return ServerSentEvent(
        event=event_type,
        data=json.dumps(data, ensure_ascii=False),
    )


async def _chat_event_generator(
    user_message: str,
    session_id: str | None,
    user_id: str,
    redis: aioredis.Redis | None,
    context: str = "coach",
) -> AsyncGenerator[ServerSentEvent, None]:
    """Async generator that drives the agent and yields SSE frames in real-time.

    Uses an asyncio.Queue to stream events as they arrive from the agent,
    rather than buffering them all until completion.
    """
    lock_acquired = await _acquire_lock(redis, user_id)
    if not lock_acquired:
        yield SSEEmitter.error(
            message="Another request is already in progress. Please wait.",
            code="concurrent_request",
        )
        yield SSEEmitter.done()
        return

    try:
        user_model = await _load_user_model(user_id)
        loop = AsyncAgentLoop(user_model=user_model, context=context)
        resolved_session_id = loop.start_session(resume_session_id=session_id)
        logger.info(
            "Chat request: user=%s session=%s message_len=%d",
            user_id,
            resolved_session_id,
            len(user_message),
        )

        # Emit session_start before any agent events so the client
        # knows which session this stream belongs to.
        yield SSEEmitter.session_start(resolved_session_id)

        # Summarize the PREVIOUS session asynchronously (non-blocking).
        if session_id is None:
            # New session → summarize the last one in the background.
            prev_session_id: str | None = None
            try:
                from src.services.session_summarizer import summarize_previous_session
                from src.db.session_store_db import get_recent_sessions

                # Capture the previous session id BEFORE the new one is
                # written, so Reflexion runs against the conversation that
                # just ended. get_recent_sessions returns most-recent-first;
                # the new session row may already exist depending on
                # create_session timing, so we walk the list and pick the
                # first id that is NOT the freshly resolved one.
                _recent = await asyncio.to_thread(get_recent_sessions, user_id, 5)
                for _s in _recent:
                    _sid = _s.get("id")
                    if _sid and _sid != resolved_session_id:
                        prev_session_id = _sid
                        break

                asyncio.create_task(
                    summarize_previous_session(user_id),
                    name=f"summarize_{user_id}",
                )
            except Exception:
                logger.debug("Session summarization skipped", exc_info=True)

            # Reflexion loop: tier-aware self-reflection on the previous
            # session. Fire-and-forget; never blocks the SSE stream and
            # never surfaces failures to the user.
            try:
                from src.services.reflexion import maybe_run_reflexion
                asyncio.create_task(
                    maybe_run_reflexion(user_id, prev_session_id),
                    name=f"reflexion_{user_id}",
                )
            except Exception:
                logger.debug("Reflexion task scheduling skipped", exc_info=True)

            # Config GC: clean up stale configs in the background.
            try:
                from src.services.config_gc import run_config_gc
                asyncio.create_task(asyncio.to_thread(run_config_gc, user_id))
            except Exception:
                logger.debug("Config GC skipped", exc_info=True)

        # Inject checkpoint context (recently resolved pending actions)
        try:
            from src.db.pending_actions_db import get_recently_resolved
            resolved_actions = await asyncio.to_thread(
                get_recently_resolved, user_id, hours=1,
            )
            if resolved_actions:
                checkpoint_context = "\n".join(
                    f"- {a['action_type']}: {a['status']}"
                    for a in resolved_actions
                )
                user_message = (
                    f"[System: The user has responded to your previous proposals: "
                    f"{checkpoint_context}]\n\n{user_message}"
                )
        except Exception:
            logger.debug("Checkpoint context injection skipped", exc_info=True)

        # Real-time streaming via asyncio.Queue
        event_queue: asyncio.Queue[ServerSentEvent | None] = asyncio.Queue()

        async def emit_fn(event_type: str, data: dict) -> None:
            await event_queue.put(_make_sse_event(event_type, data))

        agent_result_holder: list = []

        async def run_agent() -> None:
            try:
                result = await loop.process_message_sse(user_message, emit_fn)
                agent_result_holder.append(result)
            except Exception:
                logger.exception("Agent error for user %s", user_id)
                await event_queue.put(SSEEmitter.error(
                    message="An unexpected error occurred. Please try again.",
                    code="internal_error",
                ))
            finally:
                await event_queue.put(None)  # sentinel

        agent_task = asyncio.create_task(run_agent())

        while True:
            evt = await event_queue.get()
            if evt is None:
                break
            yield evt

        await agent_task  # ensure clean completion

        # Emit onboarding_complete if the agent flagged it
        if agent_result_holder:
            agent_result = agent_result_holder[0]
            if getattr(agent_result, "onboarding_just_completed", False):
                yield SSEEmitter.onboarding_complete()

        # Emit token usage summary (best-effort)
        try:
            from src.agent.llm import MODEL as default_model
            yield SSEEmitter.usage(
                input_tokens=0,
                output_tokens=0,
                model=default_model,
            )
        except Exception:
            logger.debug("Could not emit usage event", exc_info=True)

        # Send push notification so the user can return to the chat
        # if they left the app while the agent was working.
        # The frontend suppresses this when the user is actively watching.
        if agent_result_holder:
            _result = agent_result_holder[0]
            _response_text = getattr(_result, "response_text", None)
            if _response_text:
                asyncio.create_task(
                    _send_coach_reply_notification(user_id, _response_text),
                    name=f"push_{user_id}",
                )

    except HTTPException:
        raise
    except Exception:
        logger.exception("Unhandled error in chat event generator for user %s", user_id)
        yield SSEEmitter.error(
            message="An unexpected error occurred. Please try again.",
            code="internal_error",
        )
    finally:
        await _release_lock(redis, user_id)
        yield SSEEmitter.done()


# ---------------------------------------------------------------------------
# User model loader
# ---------------------------------------------------------------------------


async def _load_user_model(user_id: str) -> UserModelDB:
    """Load (or create) a UserModelDB for *user_id* in a thread pool."""
    return await asyncio.to_thread(UserModelDB.load_or_create, user_id)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("")
async def post_chat(
    body: ChatRequest,
    current_user: Annotated[dict, Depends(get_current_user)],
) -> EventSourceResponse:
    """Stream an agent response for a single chat turn."""
    user_id: str = current_user["sub"]

    from src.services.usage_tracker import check_budget
    if not check_budget(user_id):
        raise HTTPException(status_code=429, detail="Daily token budget exceeded")

    redis = await _get_redis()

    # Frontend sends "new" to indicate a fresh session — normalize to None
    session_id = body.session_id if body.session_id and body.session_id != "new" else None

    generator = _chat_event_generator(
        user_message=body.message,
        session_id=session_id,
        user_id=user_id,
        redis=redis,
        context=body.context,
    )

    return EventSourceResponse(generator)


@router.post("/confirm")
async def post_chat_confirm(
    body: ConfirmRequest,
    current_user: Annotated[dict, Depends(get_current_user)],
) -> dict:
    """Store a user confirmation for the checkpoint / tool-approval flow."""
    user_id: str = current_user["sub"]
    confirm_key = f"confirm:{body.session_id}:{body.action_id}"
    value = json.dumps({"confirmed": body.confirmed, "user_id": user_id})

    redis = await _get_redis()

    if redis is not None:
        try:
            await redis.set(confirm_key, value, ex=600)
            logger.info(
                "Stored confirmation %s=%s for user %s",
                confirm_key,
                body.confirmed,
                user_id,
            )
        except Exception:
            logger.exception("Failed to store confirmation in Redis")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Confirmation store unavailable — please retry.",
            )
    else:
        _in_process_confirmations[confirm_key] = value
        logger.warning(
            "Redis unavailable; stored confirmation %s in-process (unreliable)",
            confirm_key,
        )

    # Also update the pending_actions table if it exists
    try:
        from src.db.pending_actions_db import resolve_pending_action
        await asyncio.to_thread(
            resolve_pending_action, user_id, body.action_id, body.confirmed,
        )
    except Exception:
        logger.debug("pending_actions update skipped", exc_info=True)

    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Push notification on agent reply
# ---------------------------------------------------------------------------


async def _send_coach_reply_notification(user_id: str, response_text: str) -> None:
    """Fire-and-forget push notification that the coach has replied.

    The frontend notification handler suppresses this when the user is
    actively streaming the response.  When the app is in the background
    iOS/Android show it as a banner notification.
    """
    from src.agent.tools.notification_tools import send_notification_async

    preview = response_text[:120]
    if len(response_text) > 120:
        preview += "…"

    try:
        await send_notification_async(
            user_id=user_id,
            title="Coach hat geantwortet",
            body=preview,
            data={"type": "coach_response", "navigate": "coach"},
        )
    except Exception:
        logger.debug("Coach reply push notification failed", exc_info=True)
