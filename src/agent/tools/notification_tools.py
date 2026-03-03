"""Notification tools — push notifications and background task spawning.

Provides two agent-callable tools:
    - send_notification: sends an Expo push notification to a user.
    - spawn_background_task: fires a sub-agent asynchronously (skeleton).

Both tools follow the immutable-result pattern: they return a new dict
describing the outcome without mutating any shared state.
"""

import logging
from typing import Any

import httpx

from src.agent.tools.registry import Tool, ToolRegistry

logger = logging.getLogger(__name__)

# Expo Push API endpoint.
_EXPO_PUSH_URL = "https://exp.host/--/api/v2/push/send"

# Timeout for Expo API requests (seconds).
_HTTP_TIMEOUT = 10.0


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_notification_tools(registry: ToolRegistry, user_id: str) -> None:
    """Register notification tools bound to the given user_id."""

    def send_notification(
        title: str,
        body: str,
        data: dict | None = None,
    ) -> dict:
        """Send a push notification to the current user."""
        import asyncio

        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop is not None and loop.is_running():
                future = asyncio.run_coroutine_threadsafe(
                    send_notification_async(user_id, title, body, data), loop
                )
                result = future.result(timeout=15)
            else:
                result = asyncio.run(
                    send_notification_async(user_id, title, body, data)
                )
        except Exception as exc:
            logger.error("send_notification failed: %s", exc)
            result = {"sent": False, "error": str(exc)}

        return {**result, "_sent_in_turn": True}

    registry.register(Tool(
        name="send_notification",
        description=(
            "Send a push notification to the user's device. "
            "Use this to alert the user about important insights, reminders, or "
            "actions they should take outside of the current chat session. "
            "Set _sent_in_turn=true in the result so the agent knows no further "
            "response is required."
        ),
        handler=send_notification,
        parameters={
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Short notification title (max ~50 chars)",
                },
                "body": {
                    "type": "string",
                    "description": "Notification body text (max ~200 chars)",
                },
                "data": {
                    "type": "object",
                    "description": "Optional extra payload forwarded to the app",
                    "nullable": True,
                },
            },
            "required": ["title", "body"],
        },
        category="meta",
    ))

    def spawn_background_task(
        instruction: str,
        max_iterations: int = 15,
    ) -> dict:
        """Spawn a background sub-agent for a long-running task."""
        import asyncio

        task_id = _make_task_id()
        logger.info(
            "Spawning background task %s for user %s: %.80s",
            task_id,
            user_id,
            instruction,
        )

        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop is not None and loop.is_running():
                loop.create_task(
                    _run_background_task(task_id, user_id, instruction, max_iterations),
                    name=f"bg_task_{task_id}",
                )
            else:
                # Sync context — we can't truly fire-and-forget here.
                logger.warning(
                    "spawn_background_task called outside async context; task %s deferred",
                    task_id,
                )
        except Exception as exc:
            logger.error("Failed to spawn background task %s: %s", task_id, exc)
            return {"spawned": False, "task_id": task_id, "error": str(exc)}

        return {"spawned": True, "task_id": task_id}

    registry.register(Tool(
        name="spawn_background_task",
        description=(
            "Spawn a background sub-agent to carry out a long-running task "
            "asynchronously (e.g. deep data analysis, plan regeneration). "
            "The task runs independently — results are delivered via notification. "
            "Use for work that would exceed the current turn's budget."
        ),
        handler=spawn_background_task,
        parameters={
            "type": "object",
            "properties": {
                "instruction": {
                    "type": "string",
                    "description": "Natural-language instruction for the sub-agent",
                },
                "max_iterations": {
                    "type": "integer",
                    "description": "Maximum tool-call rounds (default 15, max 30)",
                },
            },
            "required": ["instruction"],
        },
        category="meta",
    ))


# ---------------------------------------------------------------------------
# Push notification implementation
# ---------------------------------------------------------------------------


async def send_notification_async(
    user_id: str,
    title: str,
    body: str,
    data: dict[str, Any] | None = None,
) -> dict:
    """Send a push notification via the Expo Push API.

    Looks up the push token from the ``push_tokens`` table.  If no token
    exists for the user this is treated as a graceful no-op (not an error).

    Args:
        user_id: Supabase user UUID.
        title: Notification title.
        body: Notification body text.
        data: Optional extra payload delivered to the app.

    Returns:
        Dict with ``sent`` (bool) and details or ``error`` key.
    """
    push_token = await _lookup_push_token(user_id)
    if not push_token:
        logger.info("No push token for user %s — notification skipped", user_id)
        return {"sent": False, "reason": "no_push_token", "user_id": user_id}

    payload = _build_expo_payload(push_token, title, body, data)
    return await _post_to_expo(payload)


async def _lookup_push_token(user_id: str) -> str | None:
    """Fetch the Expo push token for a user from Supabase."""
    try:
        from src.db.client import get_async_supabase

        client = await get_async_supabase()
        result = await (
            client.table("push_tokens")
            .select("token")
            .eq("user_id", user_id)
            .maybe_single()
            .execute()
        )
        if result and result.data:
            return result.data.get("token")
        return None
    except Exception as exc:
        logger.error("Failed to look up push token for user %s: %s", user_id, exc)
        return None


def _build_expo_payload(
    token: str,
    title: str,
    body: str,
    data: dict[str, Any] | None,
) -> dict:
    """Build the Expo push message dict (immutable construction)."""
    base: dict = {
        "to": token,
        "title": title,
        "body": body,
        "sound": "default",
    }
    if data:
        return {**base, "data": data}
    return base


async def _post_to_expo(payload: dict) -> dict:
    """POST the notification payload to the Expo Push API."""
    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            response = await client.post(
                _EXPO_PUSH_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            response_data: dict = response.json()
            ticket = response_data.get("data", {})
            return {"sent": True, "ticket": ticket}
    except httpx.HTTPStatusError as exc:
        logger.error("Expo API HTTP error: %s", exc)
        return {"sent": False, "error": f"HTTP {exc.response.status_code}"}
    except Exception as exc:
        logger.error("Expo API request failed: %s", exc)
        return {"sent": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Background task runner (skeleton)
# ---------------------------------------------------------------------------


def _make_task_id() -> str:
    """Generate a short unique task identifier."""
    import uuid

    return f"bg_{uuid.uuid4().hex[:8]}"


async def _run_background_task(
    task_id: str,
    user_id: str,
    instruction: str,
    max_iterations: int,
) -> None:
    """Run a sub-agent with limited tools for a background task.

    TODO: Wire up a restricted agent loop here (data + calc tools only).
    The sub-agent result should be delivered back to the user via
    send_notification_async().
    """
    logger.info(
        "Background task %s started for user %s (max_iter=%d)",
        task_id,
        user_id,
        max_iterations,
    )
    # Placeholder — real implementation:
    # 1. Build a restricted ToolRegistry (data_tools + analysis_tools only).
    # 2. Run AgentLoop(user_id, tools=restricted_registry, max_tool_rounds=max_iterations).
    # 3. Send result notification via send_notification_async().
    logger.info("Background task %s completed (stub)", task_id)
