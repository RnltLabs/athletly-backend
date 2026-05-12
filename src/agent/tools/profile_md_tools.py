"""Tools for editing the user-controlled athlete memory section.

The ``profiles.athlete_md`` column holds a small markdown document the
athlete (or coach acting on their behalf) wants the coach to always
remember. The auto-generated ``build_athlete_md`` view renders this
verbatim under "Notes from athlete" each turn.

Only one tool here: ``edit_athlete_memory``. Reads and writes to the
column directly. The coach uses it whenever the user says something like
"please always remember that I have a sensitive Achilles" or "I prefer
morning sessions".
"""

from __future__ import annotations

import logging

from src.agent.tools.registry import Tool, ToolRegistry
from src.config import get_settings

logger = logging.getLogger(__name__)


def register_profile_md_tools(registry: ToolRegistry, user_model) -> None:
    """Register the edit_athlete_memory tool."""
    _settings = get_settings()
    _user_id = (
        getattr(user_model, "user_id", None)
        or _settings.agenticsports_user_id
    )

    def edit_athlete_memory(action: str, content: str = "") -> dict:
        """Read, append, or replace the user-controlled athlete memory note.

        Args:
            action: "read" returns current content. "append" adds *content*
                as a new bullet line. "replace" overwrites with *content*.
                "clear" removes all notes.
            content: New text (required for append/replace).

        Returns:
            ``{"status": "ok", "athlete_md": "<current text>"}``.
        """
        from src.db.client import get_supabase

        client = get_supabase()
        action = (action or "").lower().strip()

        try:
            res = (
                client.table("profiles")
                .select("athlete_md")
                .eq("user_id", _user_id)
                .execute()
            )
            current = (res.data[0].get("athlete_md") if res.data else "") or ""
        except Exception as exc:
            return {"error": f"Could not read profile: {exc}"}

        if action == "read":
            return {"status": "ok", "athlete_md": current}

        if action == "clear":
            new_value: str | None = None
        elif action == "append":
            if not content.strip():
                return {"error": "append requires non-empty content"}
            entry = content.strip()
            if not entry.startswith("- "):
                entry = f"- {entry}"
            new_value = (current + ("\n" if current else "") + entry).strip()
        elif action == "replace":
            new_value = content.strip() or None
        else:
            return {
                "error": (
                    "Unknown action. Use one of: read, append, replace, clear."
                )
            }

        try:
            client.table("profiles").update({"athlete_md": new_value}).eq(
                "user_id", _user_id
            ).execute()
        except Exception as exc:
            return {"error": f"Could not write profile: {exc}"}

        return {"status": "ok", "athlete_md": new_value or ""}

    # DEPRECATED: edit_athlete_memory is replaced by journal tools
    # (update_journal_section / append_to_journal). NOT registered anymore.
    _DISABLED_edit_athlete_memory = lambda: registry.register(Tool(
        name="edit_athlete_memory_disabled",
        description=(
            "Read, append, or rewrite the athlete's free-form memory notes - "
            "a small markdown document the coach sees every turn under "
            "'Notes from athlete'. Use this when the user shares something "
            "they want you to ALWAYS remember in the future: chronic "
            "injuries, recurring constraints, lifestyle facts, personal "
            "preferences that are not captured in structured profile "
            "fields or beliefs.\n\n"
            "WHEN to use:\n"
            "- User says 'please always remember that...'\n"
            "- User shares a permanent constraint: 'I have a sensitive "
            "Achilles', 'I cannot run on cobblestones'\n"
            "- User shares a personal preference that should outlast "
            "individual plans: 'I prefer morning sessions', 'I race best "
            "in cool weather'\n\n"
            "WHEN NOT to use:\n"
            "- For temporary state (use beliefs instead)\n"
            "- For structured fields like goal/sport/constraints (use "
            "update_profile)\n"
            "- For one-off training observations (use add_belief)\n\n"
            "Actions:\n"
            "- 'read': returns the current notes (no content arg needed)\n"
            "- 'append': adds a new bullet line at the end\n"
            "- 'replace': overwrites everything with the new content\n"
            "- 'clear': removes all notes\n\n"
            "Example:\n"
            "  edit_athlete_memory(action='append', content='Sensitive "
            "Achilles - no plyometrics or hill repeats over 5%')"
        ),
        handler=edit_athlete_memory,
        parameters={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["read", "append", "replace", "clear"],
                    "description": "What to do with the notes.",
                },
                "content": {
                    "type": "string",
                    "description": (
                        "New text for append or replace. Plain markdown, "
                        "no leading bullets needed - they are added "
                        "automatically for append."
                    ),
                    "nullable": True,
                },
            },
            "required": ["action"],
        },
        category="memory",
    ))
