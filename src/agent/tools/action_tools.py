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

The tool guards against a no-op render: if Garmin is already connected for
the current user, the tool returns ``status: "already_connected"`` instead
of emitting the form, so the agent never re-asks an athlete who is already
paired (a common Haiku failure mode where status checks are skipped).
"""

from __future__ import annotations

import logging
import uuid

from src.agent.tools.registry import Tool, ToolRegistry

logger = logging.getLogger(__name__)


def _is_garmin_connected(user_id: str | None) -> bool:
    """Return True if there is an active garmin token row for *user_id*."""
    if not user_id:
        return False
    try:
        from src.db.client import get_supabase
        sb = get_supabase()
        rows = (
            sb.table("provider_tokens")
            .select("status")
            .eq("user_id", user_id)
            .eq("provider", "garmin")
            .limit(1)
            .execute()
        )
        if not rows.data:
            return False
        status = rows.data[0].get("status")
        return status in (None, "active")
    except Exception:
        logger.debug("garmin connection check failed", exc_info=True)
        return False


def register_action_tools(registry: ToolRegistry, user_model=None) -> None:
    """Register inline action tools into the registry."""
    user_id: str | None = (
        getattr(user_model, "user_id", None) if user_model else None
    )

    def request_garmin_connect() -> dict:
        """Surface an inline Garmin email + password form in the chat.

        Short-circuits when Garmin is already connected for the current
        user so we never ask an already-paired athlete to log in again.
        """
        if _is_garmin_connected(user_id):
            return {
                "status": "already_connected",
                "message": (
                    "Garmin ist bereits verbunden. Verwende "
                    "get_provider_status oder sync_garmin_data direkt - "
                    "kein neuer Login noetig."
                ),
            }

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
            "Garmin account so you can read their activities. "
            "IMPORTANT: this tool short-circuits if Garmin is already "
            "connected and returns status='already_connected' - in that "
            "case do NOT call request_garmin_connect again; use "
            "get_provider_status or sync_garmin_data instead. Call ONCE "
            "per onboarding flow; do not spam it."
        ),
        handler=request_garmin_connect,
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        category="meta",
        display_label="Frage nach Garmin-Verbindung",
    ))
