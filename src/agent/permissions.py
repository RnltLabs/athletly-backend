"""Permission system for destructive tool calls.

Mirrors Claude Code's permission flow: before a destructive tool runs,
the agent loop checks if the call needs confirmation. If yes, it creates
a ``pending_action`` row, returns a block decision to the agent, and the
human (mobile app or athctl) confirms via a separate endpoint.

Currently a small static list of destructive tools. Could be extended
later to per-user policies (e.g. "always require confirmation for X").
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.agent.hooks import HookDecision

logger = logging.getLogger(__name__)


# Hardcoded set of tool names that require user confirmation before running.
# Keep tight - adding tools here makes them block until confirmed, which
# slows the chat. Only include genuinely destructive operations.
DANGEROUS_TOOLS: frozenset[str] = frozenset({
    "disconnect_provider",  # wipes provider tokens
    "delete_plan",          # if such a tool exists
    "reset_user_data",      # wipes everything
})


def is_dangerous(tool_name: str) -> bool:
    """Return True if the tool requires explicit user confirmation."""
    return tool_name in DANGEROUS_TOOLS


def pre_permission_check(args: dict, ctx: "Any") -> HookDecision | None:
    """Pre-hook factory: blocks dangerous tools until user confirms.

    The hook creates a pending_action row in supabase and returns a
    block decision with the action_id. The agent then tells the user
    "I need confirmation to do X (action_id: Y)" and the user can
    approve via the chat/confirm endpoint or athctl.

    If the args contain ``confirmed_action_id`` (set when the agent
    retries after user approval) we check the row's resolved_at and
    let the call through if approved.
    """
    if not is_dangerous(ctx.tool_name):
        return None  # not dangerous - allow

    # If the agent passes a confirmed_action_id, verify the pending_action
    # was approved before letting the call through.
    confirmed_id = (args or {}).pop("confirmed_action_id", None)
    if confirmed_id:
        try:
            from src.db.client import get_supabase

            client = get_supabase()
            row_res = (
                client.table("pending_actions")
                .select("status, resolved_at")
                .eq("id", confirmed_id)
                .eq("user_id", ctx.user_id)
                .execute()
            )
            row = row_res.data[0] if row_res.data else None
            if row and row.get("status") == "approved":
                logger.info(
                    "Dangerous tool %s approved via pending_action %s",
                    ctx.tool_name, confirmed_id,
                )
                return HookDecision(action="allow")
            return HookDecision(
                action="block",
                reason=(
                    f"pending_action {confirmed_id} not approved "
                    f"(status={row.get('status') if row else 'missing'})"
                ),
            )
        except Exception as exc:
            logger.exception("permission verify failed")
            return HookDecision(action="block", reason=f"Verify failed: {exc}")

    # No confirmed_action_id - create a new pending_action and block.
    try:
        from src.db.client import get_supabase

        client = get_supabase()
        row = client.table("pending_actions").insert({
            "user_id": ctx.user_id,
            "action_type": ctx.tool_name,
            "description": f"Agent wants to call {ctx.tool_name}",
            "preview": json.loads(json.dumps(args or {}, default=str)),
            "status": "pending",
        }).execute()
        action_id = row.data[0]["id"] if row.data else None
    except Exception as exc:
        logger.exception("Could not create pending_action")
        return HookDecision(
            action="block",
            reason=f"Could not create confirmation: {exc}",
        )

    return HookDecision(
        action="block",
        reason=(
            f"Tool '{ctx.tool_name}' is destructive and needs user "
            f"approval. Pending action id: {action_id}. Tell the user "
            f"what you want to do and why; when they approve via the "
            f"mobile app or athctl, retry the call with "
            f"confirmed_action_id={action_id} in the args."
        ),
    )


def install_permission_hooks() -> None:
    """Register the permission pre-hook against every dangerous tool."""
    from src.agent.hooks import register_pre_hook

    for tool_name in DANGEROUS_TOOLS:
        register_pre_hook(tool_name, pre_permission_check)
