"""Checkpoint tools -- propose plan changes that need user confirmation.

The agent proposes changes, the user confirms/rejects via POST /chat/confirm,
and the agent reads the decisions on the next turn.

Usage::

    from src.agent.tools.checkpoint_tools import register_checkpoint_tools

    register_checkpoint_tools(registry, user_model)
"""

from __future__ import annotations

from src.agent.tools.registry import Tool, ToolRegistry


def register_checkpoint_tools(registry: ToolRegistry, user_model=None) -> None:
    """Register checkpoint/replanning tools into the registry."""

    def _resolve_user_id() -> str:
        """Resolve user_id from user_model (multi-tenant API) or settings (CLI)."""
        if user_model and hasattr(user_model, "user_id") and user_model.user_id:
            return user_model.user_id
        from src.config import get_settings
        return get_settings().agenticsports_user_id

    def propose_plan_change(
        action_type: str,
        description: str,
        preview: dict | None = None,
        checkpoint_type: str = "HARD",
    ) -> dict:
        """Propose a plan change that requires user confirmation.

        HARD checkpoint: blocks until user confirms (e.g., major plan restructure)
        SOFT checkpoint: proceeds after timeout (e.g., minor schedule adjustment)
        """
        from src.db.pending_actions_db import create_pending_action

        user_id = _resolve_user_id()
        if not user_id:
            return {"status": "error", "message": "No user_id configured."}

        if checkpoint_type not in ("HARD", "SOFT"):
            return {
                "status": "error",
                "message": "checkpoint_type must be HARD or SOFT",
            }

        action = create_pending_action(
            user_id=user_id,
            action_type=action_type,
            description=description,
            preview=preview or {},
            checkpoint_type=checkpoint_type,
        )

        return {
            "status": "proposed",
            "action_id": action.get("id"),
            "action_type": action_type,
            "checkpoint_type": checkpoint_type,
            "message": f"Proposed '{action_type}' — waiting for user confirmation.",
        }

    registry.register(
        Tool(
            name="propose_plan_change",
            description=(
                "Propose a plan change that needs user confirmation before execution. "
                "Use HARD checkpoint for major changes (plan restructure, goal change) "
                "and SOFT for minor adjustments (swap workout days, adjust intensity). "
                "The user will see the proposal and can confirm or reject it."
            ),
            handler=propose_plan_change,
            parameters={
                "type": "object",
                "properties": {
                    "action_type": {
                        "type": "string",
                        "description": (
                            "Type of change: 'plan_restructure', 'goal_adjustment', "
                            "'schedule_swap', 'intensity_change', etc."
                        ),
                    },
                    "description": {
                        "type": "string",
                        "description": "Human-readable description of the proposed change.",
                    },
                    "preview": {
                        "type": "object",
                        "description": "Preview data showing before/after state.",
                    },
                    "checkpoint_type": {
                        "type": "string",
                        "enum": ["HARD", "SOFT"],
                        "description": (
                            "HARD = blocks until confirmed, "
                            "SOFT = proceeds after timeout."
                        ),
                    },
                },
                "required": ["action_type", "description"],
            },
            category="planning",
        )
    )

    def get_pending_confirmations() -> dict:
        """Get the status of pending and recently resolved actions."""
        from src.db.pending_actions_db import (
            get_pending_for_user,
            get_recently_resolved,
        )

        user_id = _resolve_user_id()
        if not user_id:
            return {"status": "error", "message": "No user_id configured."}

        pending = get_pending_for_user(user_id)
        resolved = get_recently_resolved(user_id, hours=1)

        return {
            "status": "ok",
            "pending": [
                {
                    "action_id": a.get("id"),
                    "action_type": a.get("action_type"),
                    "description": a.get("description"),
                    "checkpoint_type": a.get("checkpoint_type"),
                    "created_at": a.get("created_at"),
                }
                for a in pending
            ],
            "recently_resolved": [
                {
                    "action_id": a.get("id"),
                    "action_type": a.get("action_type"),
                    "status": a.get("status"),
                    "resolved_at": a.get("resolved_at"),
                }
                for a in resolved
            ],
        }

    registry.register(
        Tool(
            name="get_pending_confirmations",
            description=(
                "Check the status of proposed plan changes. Returns pending proposals "
                "awaiting user decision and recently resolved ones (confirmed/rejected). "
                "Use this at the start of a conversation to check if the user has "
                "responded to any previous proposals."
            ),
            handler=get_pending_confirmations,
            parameters={},
            category="planning",
        )
    )
