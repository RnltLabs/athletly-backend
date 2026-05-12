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

from src.agent.athlete_journal import (
    append_to_section,
    read_journal,
    remove_from_section,
    update_section,
)
from src.agent.tools.registry import Tool, ToolRegistry
from src.config import get_settings

logger = logging.getLogger(__name__)


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

    registry.register(Tool(
        name="read_athlete_journal",
        description=(
            "Read the full athlete journal markdown. RARELY needed - "
            "the runtime context already injects the journal as your "
            "'# Athlete Profile' block, so you already see it. Use this "
            "tool only when you specifically need the raw text (e.g. to "
            "diff before and after a section update).\n\n"
            "Returns {\"content\": \"# Athlete Name\\n\\n## Identity\\n...\"}"
        ),
        handler=read_athlete_journal,
        parameters={"type": "object", "properties": {}},
        category="memory",
    ))

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
            "Replace the body of a section in the athlete journal. The "
            "journal is the single source of truth about who the athlete "
            "is - their identity, current goal, preferences, training "
            "knowledge.\n\n"
            "CANONICAL SECTIONS (use these names exactly):\n"
            "- 'Identity'                       static personal facts: "
            "name, age, location, sports, lifestyle, constraints.\n"
            "- 'Current Goal'                   the active target event + "
            "facts. Replaced completely on each update.\n"
            "- 'What I know about my training'  performance facts: paces, "
            "PBs, VO2max estimates.\n"
            "- 'Preferences'                    how the athlete wants to "
            "be coached: output style, time of day, surface, etc.\n\n"
            "WHEN TO USE:\n"
            "- The athlete shares something that REPLACES previous info "
            "for that section (new fitness baseline, updated identity).\n"
            "- The section needs structural cleanup after multiple appends.\n\n"
            "WHEN NOT TO USE:\n"
            "- Add a single fact to a list section: use append_to_journal.\n"
            "- Change just the target_date on a goal: use update_goal "
            "(atomic, also logs to Goal Timeline + archives macrocycle).\n"
            "- Annotate a single training activity: use annotate_activity.\n\n"
            "EXAMPLE:\n"
            "  update_journal_section(\n"
            "    section='Identity',\n"
            "    content='32, lives in Karlsruhe. Multi-sport: running, "
            "cycling, gym. Trains mornings. Max 90 min weekday, 4h+ "
            "weekend possible.'\n"
            "  )"
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
            "Add a new bullet to an APPEND-ONLY section of the athlete "
            "journal. Use for sections that accumulate facts over time:\n\n"
            "- 'Goal Timeline'  - every goal change (with date + reason).\n"
            "- 'Open Threads'   - things to check on later (e.g. 'knee "
            "pain reported 2026-05-12, ask next time').\n"
            "- 'What we tried'  - coach learnings: what worked, what did "
            "not.\n\n"
            "WHEN TO USE:\n"
            "- New event to add to the timeline.\n"
            "- New open thread the coach should remember to check.\n"
            "- A finished experiment / lesson worth recording.\n\n"
            "WHEN NOT TO USE:\n"
            "- Replacing an entire section: use update_journal_section.\n"
            "- Resolving an open thread: use remove_from_journal.\n\n"
            "EXAMPLE:\n"
            "  append_to_journal(\n"
            "    section='Open Threads',\n"
            "    entry='2026-05-12: Knee pain during easy run, athlete "
            "felt slower than usual. Check next session.'\n"
            "  )"
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
    ))

    # -- annotate_activity ------------------------------------------------

    def annotate_activity(activity_id: str, note: str) -> dict:
        """Attach a free-form note to a specific activity row.

        Used for context the coach wants attached to a single session:
        'knee pain during this run', 'felt great today', 'cut short due
        to weather'. Stored on activities.notes.
        """
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
            "Attach a free-form note to a specific activity. Lets the "
            "coach record per-session context that may matter later: "
            "pain, perceived effort, weather, mental state, cut short.\n\n"
            "WHEN TO USE:\n"
            "- Athlete reports something about a specific recent session: "
            "'I had knee pain on yesterday's run'.\n"
            "- Coach observes something worth recording for trend "
            "detection: 'completed long run but HR was unusually high'.\n\n"
            "AFTER calling annotate_activity, you almost always also "
            "want to:\n"
            "- append_to_journal(section='Open Threads', ...) if the "
            "issue needs follow-up next session, or\n"
            "- update_journal_section('What I know...') if the note "
            "reveals a lasting fitness fact.\n\n"
            "EXAMPLE workflow when athlete reports knee pain:\n"
            "1. annotate_activity(activity_id=..., note='Knee pain "
            "(L), slower than target pace, lasted ~30min run.')\n"
            "2. append_to_journal(section='Open Threads',\n"
            "   entry='2026-05-12: Left knee pain during easy run "
            "(activity X). Check at next session.')"
        ),
        handler=annotate_activity,
        parameters={
            "type": "object",
            "properties": {
                "activity_id": {
                    "type": "string",
                    "description": "UUID of the activity row to annotate.",
                },
                "note": {
                    "type": "string",
                    "description": "The annotation text.",
                },
            },
            "required": ["activity_id", "note"],
        },
        category="memory",
    ))
