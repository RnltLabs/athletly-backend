"""Tools that operate on the single-source-of-truth athlete journal.

Replaces the multi-store identity layer: update_profile (for identity
fields), add_belief, update_goal_history, edit_athlete_memory.

Five tools, one document per user:

* read_athlete_journal     - dump the whole markdown (rarely needed - it
                             is already in your runtime context)
* update_journal_section   - replace a section body entirely
* append_to_journal        - add a bullet to a section
* remove_from_journal      - remove bullets matching a substring
* annotate_activity        - per-activity note (e.g. injury during run)
"""

from __future__ import annotations

import logging
import re

from src.agent.athlete_journal import (
    append_to_section,
    read_journal,
    remove_from_section,
    update_section,
)
from src.agent.tools.registry import Tool, ToolRegistry
from src.config import get_settings

logger = logging.getLogger(__name__)


# Canonical UUID v1-v5 format check. Activity rows in Supabase use the
# uuid type; anything not matching this pattern is necessarily a
# hallucinated id and gets rejected with a recovery hint.
_UUID_PATTERN = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def _is_uuid(value: str) -> bool:
    """Return True iff *value* is a canonical UUID string.

    Defensive: rejects None, non-strings, and anything that does not
    match the 8-4-4-4-12 hex pattern. The point is to catch the agent
    inventing slug-like ids ("today-easy-run-0515") before we send a
    no-op update to Supabase.
    """
    if not isinstance(value, str):
        return False
    return bool(_UUID_PATTERN.match(value.strip()))


def register_journal_tools(registry: ToolRegistry, user_model) -> None:
    """Register the journal + activity-annotation tools."""
    _settings = get_settings()
    _user_id = (
        getattr(user_model, "user_id", None)
        or _settings.agenticsports_user_id
    )

    # -- read --------------------------------------------------------------

    def read_athlete_journal() -> dict:
        """Return the full athlete journal markdown for the active user."""
        return {"content": read_journal(_user_id)}

    # DEPRECATED: read_athlete_journal - journal already in system prompt. NOT registered.
    # registry.register(Tool(
    #     name="read_athlete_journal",
    #     description=(
    #         "Read the full athlete journal markdown. RARELY needed - "
    #         "the runtime context already injects the journal as your "
    #         "'# Athlete Profile' block, so you already see it."
    #     ),
    #     handler=read_athlete_journal,
    #     parameters={"type": "object", "properties": {}},
    #     category="memory",
    # ))

    # -- update_journal_section -------------------------------------------

    def update_journal_section(section: str, content: str) -> dict:
        """Replace the body of *section* in the journal (creates if missing).

        Used for sections where the FULL body is the truth: Identity,
        Current Goal, Preferences. For append-only sections (Goal
        Timeline, Open Threads, What we tried) use append_to_journal
        instead.

        Args:
            section: section name without leading "## " (e.g. "Identity").
            content: new section body in markdown.
        """
        new_doc = update_section(_user_id, section, content)
        return {"status": "ok", "section": section, "doc_length": len(new_doc)}

    registry.register(Tool(
        name="update_journal_section",
        description=(
            "Replace body of a journal section (source of truth for athlete "
            "facts). Canonical sections: 'Identity', 'Current Goal', 'What I "
            "know about my training', 'Preferences'. Use when info REPLACES "
            "prior section content or section needs cleanup. Avoid for "
            "appending one fact (use append_to_journal), goal target shifts "
            "(use update_goal), or per-activity notes (annotate_activity)."
        ),
        handler=update_journal_section,
        parameters={
            "type": "object",
            "properties": {
                "section": {
                    "type": "string",
                    "description": "Section name without '## ' prefix.",
                },
                "content": {
                    "type": "string",
                    "description": "New section body in markdown.",
                },
            },
            "required": ["section", "content"],
        },
        category="memory",
        display_label="Schreibe ins Journal",
    ))

    # -- append_to_journal -------------------------------------------------

    def append_to_journal(section: str, entry: str) -> dict:
        """Add *entry* as a new bullet to *section*.

        Section is created if missing. Entry gets a leading '- '
        automatically. Used for sections that accumulate over time:
        Goal Timeline, Open Threads, What we tried.
        """
        new_doc = append_to_section(_user_id, section, entry)
        return {"status": "ok", "section": section, "doc_length": len(new_doc)}

    registry.register(Tool(
        name="append_to_journal",
        description=(
            "Append a bullet to an append-only journal section. Sections: "
            "'Goal Timeline' (goal changes w/ date+reason), 'Open Threads' "
            "(things to check later), 'What we tried' (coach learnings). Use "
            "for new timeline events, threads, or recorded experiments. Avoid "
            "for full-section rewrites (use update_journal_section) or thread "
            "resolution (use remove_from_journal). Entry gets '- ' prefix auto."
        ),
        handler=append_to_journal,
        parameters={
            "type": "object",
            "properties": {
                "section": {
                    "type": "string",
                    "description": "Section name (e.g. 'Open Threads').",
                },
                "entry": {
                    "type": "string",
                    "description": "The bullet text (- is added automatically).",
                },
            },
            "required": ["section", "entry"],
        },
        category="memory",
        display_label="Notiere eine Beobachtung",
    ))

    # -- remove_from_journal ----------------------------------------------

    def remove_from_journal(section: str, contains: str) -> dict:
        """Strip bullets in *section* that contain *contains* substring."""
        new_doc = remove_from_section(_user_id, section, contains)
        return {"status": "ok", "section": section, "doc_length": len(new_doc)}

    registry.register(Tool(
        name="remove_from_journal",
        description=(
            "Remove bullets from a journal section that contain a given "
            "substring. Mainly used to RESOLVE open threads: when the "
            "athlete reports an issue is gone, take it off the watchlist.\n\n"
            "WHEN TO USE:\n"
            "- Open Thread is resolved (knee pain healed, decision made).\n"
            "- Stale or wrong entry needs cleanup.\n\n"
            "Match is case-insensitive substring. Be specific to avoid "
            "removing the wrong bullet.\n\n"
            "EXAMPLE:\n"
            "  remove_from_journal(section='Open Threads', contains='knee pain')\n"
            "  # then optionally:\n"
            "  append_to_journal(section='What we tried',\n"
            "    entry='2026-05-15: Knee pain episode resolved in 3 days, "
            "no chronic issue.')"
        ),
        handler=remove_from_journal,
        parameters={
            "type": "object",
            "properties": {
                "section": {"type": "string"},
                "contains": {"type": "string"},
            },
            "required": ["section", "contains"],
        },
        category="memory",
        display_label="Entferne aus dem Journal",
    ))

    # -- annotate_activity ------------------------------------------------

    def annotate_activity(activity_id: str, note: str) -> dict:
        """Attach a free-form note to a specific activity row.

        Used for context the coach wants attached to a single session:
        'knee pain during this run', 'felt great today', 'cut short due
        to weather'. Stored on activities.notes.

        Defensive validation (Iter 2 Sprint I): rejects non-UUID
        activity_id values BEFORE touching Supabase. The agent has been
        observed synthesising slug-like ids ('today-easy-run-0515') when
        forced to annotate without a prior get_activities call. The
        error response carries a recovery hint so the next round can
        self-correct.
        """
        if not isinstance(note, str) or not note.strip():
            return {
                "error": "note must be a non-empty string",
                "hint": "Provide a short free-form note describing what happened.",
            }
        if not _is_uuid(activity_id):
            return {
                "error": (
                    f"activity_id '{activity_id}' is not a valid UUID. "
                    f"Synthetic ids like 'today-easy-run-0515' are not "
                    f"accepted."
                ),
                "hint": (
                    "Call get_activities(limit=5) first to find the real "
                    "activity_id (uuid format, e.g. "
                    "'4f3c1b2a-90de-4abc-9123-fedcba987654'). Then retry "
                    "annotate_activity with that id."
                ),
            }

        from src.db.client import get_supabase

        client = get_supabase()
        try:
            client.table("activities").update({"notes": note.strip()}).eq(
                "id", activity_id
            ).execute()
        except Exception as exc:
            return {"error": f"Could not annotate activity: {exc}"}
        return {"status": "ok", "activity_id": activity_id}

    registry.register(Tool(
        name="annotate_activity",
        description=(
            "Attach a free-form note to a specific activity (pain, RPE, "
            "weather, cut-short reason). "
            "REQUIRED FIRST STEP: call get_activities(limit=5) to find "
            "the real activity_id. NEVER pass a synthetic id like "
            "'today-easy-run-0515' or 'last-run' - the handler rejects "
            "non-UUID ids and the call will fail. activity_id must be a "
            "canonical UUID (8-4-4-4-12 hex). "
            "Use when athlete reports something about a specific session "
            "or coach spots a trend-worthy observation. Follow-up: "
            "append_to_journal('Open Threads') if needs next-session "
            "check; update_journal_section('What I know...') for a "
            "lasting fitness fact."
        ),
        handler=annotate_activity,
        parameters={
            "type": "object",
            "properties": {
                "activity_id": {
                    "type": "string",
                    "description": (
                        "Canonical UUID of the activity row to annotate "
                        "(8-4-4-4-12 hex). Find it via "
                        "get_activities(limit=5) - never synthesise it."
                    ),
                },
                "note": {
                    "type": "string",
                    "description": "The annotation text.",
                },
            },
            "required": ["activity_id", "note"],
        },
        category="memory",
        display_label="Kommentiere ein Workout",
    ))
