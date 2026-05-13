"""Inline action tools - request the frontend to render action cards in the chat.

Actions are a separate concept from notifications and from pending_actions
(checkpoint confirmations). Historically an action was a one-shot inline UI
card with a button: the tool returned a dict with an ``_action_request``
marker, the agent loop forwarded it as an ``action_request`` SSE event, and
the frontend rendered the card.

The ``action_request`` SSE plumbing is still wired in the agent loop and
remains available for future action types. Garmin Connect has migrated to
the generative UI ``text_input`` component (see ``ui_tools.py``): the
``request_garmin_connect`` tool now emits a ``_ui_component`` marker with
``on_submit: "garmin_connect"`` so the frontend renders an inline email +
password form and submits directly to ``POST /garmin/connect``.

Convention for any future action card tool::

    {
        "status": "requested",
        "_action_request": {
            "action_type": "<type>",
            "label": "<button label>",
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

import uuid

from src.agent.tools.registry import Tool, ToolRegistry


def register_action_tools(registry: ToolRegistry, user_model=None) -> None:
    """Register inline action tools into the registry."""
    del user_model  # currently unused; kept for signature parity

    def request_garmin_connect() -> dict:
        """Surface an inline Garmin email + password form in the chat."""
        return {
            "status": "rendered",
            "_ui_component": {
                "type": "text_input",
                "id": uuid.uuid4().hex,
                "props": {
                    "question": "Verbinde dein Garmin-Konto",
                    "fields": [
                        {
                            "name": "email",
                            "label": "Email",
                            "placeholder": "garmin@example.com",
                            "type": "email",
                        },
                        {
                            "name": "password",
                            "label": "Passwort",
                            "type": "password",
                            "isPassword": True,
                        },
                    ],
                    "submit_label": "Verbinden",
                    "on_submit": "garmin_connect",
                },
            },
        }

    registry.register(Tool(
        name="request_garmin_connect",
        description=(
            "Render an inline Garmin login form (email + password) in the "
            "chat. The frontend submits directly to POST /garmin/connect on "
            "success. Call this when the athlete should connect their "
            "Garmin account so you can read their activities. Call ONCE "
            "per onboarding flow; do not spam it."
        ),
        handler=request_garmin_connect,
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        category="meta",
    ))
