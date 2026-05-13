"""Inline action tools - request the frontend to render action cards in the chat.

Actions are a separate concept from notifications and from pending_actions
(checkpoint confirmations). An action is a one-shot inline UI card with a
button. When the agent calls one of these tools the result dict carries a
special ``_action_request`` marker that the agent loop forwards to the SSE
stream as an ``action_request`` event. The frontend renders the card and
the user takes it from there (e.g. opens a Garmin login modal).

Convention: any tool that wants to surface an inline action card returns a
dict like::

    {
        "status": "requested",
        "_action_request": {
            "action_type": "garmin_connect",
            "label": "Garmin verbinden",
            "payload": {},
        },
        ...
    }

The agent loop inspects ``_action_request`` after the tool runs and emits a
matching SSE event via ``on_progress``. The key itself is stripped from the
result dict before it is added to the chat history so the model does not see
internal plumbing on its next turn.
"""

from __future__ import annotations

from src.agent.tools.registry import Tool, ToolRegistry


def register_action_tools(registry: ToolRegistry, user_model=None) -> None:
    """Register inline action tools into the registry."""
    del user_model  # currently unused; kept for signature parity

    def request_garmin_connect(label: str = "Garmin verbinden") -> dict:
        """Surface an inline 'Connect Garmin' card in the chat."""
        return {
            "status": "requested",
            "action_type": "garmin_connect",
            "label": label,
            "_action_request": {
                "action_type": "garmin_connect",
                "label": label,
                "payload": {},
            },
        }

    registry.register(Tool(
        name="request_garmin_connect",
        description=(
            "Render an inline 'Connect Garmin' action card in the chat. "
            "Use this when the athlete should connect their Garmin account so "
            "you can read their activities. The card has a button that opens "
            "the Garmin login modal; the user takes it from there. Call this "
            "once per onboarding flow; do not spam it."
        ),
        handler=request_garmin_connect,
        parameters={
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "description": (
                        "Button label shown on the action card. "
                        "Defaults to 'Garmin verbinden'."
                    ),
                },
            },
            "required": [],
        },
        category="meta",
    ))
