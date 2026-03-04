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
        )
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
        return SSEEmitter.message(data.get("text", ""))
    if event_type == "error":
        return SSEEmitter.error(
            message=data.get("message", "Unknown error"),
            code=data.get("code", "internal_error"),
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
    redis = await _get_redis()

    generator = _chat_event_generator(
        user_message=body.message,
        session_id=body.session_id,
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
