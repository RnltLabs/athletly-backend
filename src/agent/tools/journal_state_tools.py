"""Atomic CRUD tools for typed journal state.

Three categories of coach-maintained state moved out of the markdown
``athlete_journal`` into typed tables (see ``20260515_typed_journal_state.sql``):

* **Intents**       - 1 to 5 active commitments (event/capacity/habit/recovery).
* **Programs**      - persisting training structures (gym split, run block).
* **Coach Notes**   - append-only learnings the coach writes over weeks.

Each tool is atomic: one row in, one row out. The markdown rendering for
the system prompt is handled by :mod:`src.agent.journal_state_md`; the
tools here never touch markdown directly.
"""

from __future__ import annotations

import logging
import re

from src.agent.tools.registry import Tool, ToolRegistry
from src.config import get_settings

logger = logging.getLogger(__name__)


# Canonical UUID v1-v5 format check. Matches the helper in journal_tools.py.
_UUID_PATTERN = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def _is_uuid(value: str) -> bool:
    """Return True iff *value* is a canonical UUID string."""
    if not isinstance(value, str):
        return False
    return bool(_UUID_PATTERN.match(value.strip()))


# Heuristic mapping from a free-text "reason" string to a terminal status.
# Used by ``retire_intent``: any cue suggesting completion routes to
# ``achieved``; everything else to ``abandoned``. Coach can override by
# passing the desired status directly.
_ACHIEVED_KEYWORDS: tuple[str, ...] = (
    "achieved", "done", "completed", "complete", "finished", "succeeded",
    "geschafft", "erreicht", "abgeschlossen", "fertig",
)


def _classify_retirement(reason: str) -> str:
    """Map a free-text reason to either ``achieved`` or ``abandoned``."""
    text = (reason or "").lower()
    for needle in _ACHIEVED_KEYWORDS:
        if needle in text:
            return "achieved"
    return "abandoned"


def register_journal_state_tools(registry: ToolRegistry, user_model) -> None:
    """Register the typed-state CRUD tools onto ``registry``."""
    _settings = get_settings()
    _user_id = (
        getattr(user_model, "user_id", None)
        or _settings.agenticsports_user_id
    )

    # ---------------------------------------------------------------
    # Intent tools
    # ---------------------------------------------------------------

    def add_intent(
        type: str,
        title: str,
        priority: int = 3,
        payload: dict | None = None,
    ) -> dict:
        from src.db.intents_db import insert_intent

        if not isinstance(title, str) or not title.strip():
            return {
                "error": "title must be a non-empty string",
                "hint": "Pass the athlete's commitment, e.g. 'Karlsruhe HM sub 1:45'.",
            }
        try:
            row = insert_intent(
                _user_id,
                type=type,
                title=title,
                priority=priority,
                payload=payload or {},
            )
        except Exception as exc:
            return {"error": f"Could not add intent: {exc}"}
        return {"status": "ok", "intent": row}

    registry.register(Tool(
        name="add_intent",
        description=(
            "Add an active athlete commitment to the Intents table. "
            "Use when the athlete signs up for a race, declares a "
            "capacity goal (e.g. squat 100kg), or commits to a habit "
            "(e.g. 4x/week movement). Type is free text but prefer "
            "one of: event, capacity, habit, recovery. Priority is "
            "1 to 5 (1 = top). Payload is type-specific JSON: "
            "{race_date, target_time} for events, {target_lift} for "
            "lifts, etc. The intent shows up in the next-turn system "
            "prompt under Active Intents. Do NOT use for one-off "
            "session decisions (those go in the journal Open Threads)."
        ),
        handler=add_intent,
        parameters={
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "description": (
                        "Intent type. Hinted values: event, capacity, "
                        "habit, recovery. Free text accepted."
                    ),
                },
                "title": {
                    "type": "string",
                    "description": (
                        "Short human-readable commitment, e.g. "
                        "'Karlsruhe HM 2026-04-12 sub 1:45'."
                    ),
                },
                "priority": {
                    "type": "integer",
                    "description": "1 (top) to 5. Defaults to 3.",
                },
                "payload": {
                    "type": "object",
                    "description": (
                        "Type-specific data, e.g. {race_date, target_time}."
                    ),
                },
            },
            "required": ["type", "title"],
        },
        category="memory",
        display_label="Lege ein Intent an",
    ))

    def retire_intent(intent_id: str, reason: str = "") -> dict:
        from src.db.intents_db import update_intent_status

        if not _is_uuid(intent_id):
            return {
                "error": (
                    f"intent_id '{intent_id}' is not a valid UUID. "
                    "Call list_intents first to find the real id."
                ),
                "hint": "Use list_intents(status='active') to look up the UUID.",
            }
        status = _classify_retirement(reason)
        try:
            row = update_intent_status(intent_id, status)
        except Exception as exc:
            return {"error": f"Could not retire intent: {exc}"}
        if row is None:
            return {"error": f"Intent {intent_id} not found."}
        return {"status": "ok", "intent": row, "new_status": status}

    registry.register(Tool(
        name="retire_intent",
        description=(
            "Mark an active intent as finished. Uses the reason text "
            "to decide between 'achieved' (athlete hit the target) "
            "and 'abandoned' (athlete dropped it). Pass a short reason "
            "string so the new status is correctly classified. "
            "REQUIRED FIRST STEP: call list_intents(status='active') "
            "to get the real intent_id (canonical UUID, 8-4-4-4-12 hex). "
            "Use when the athlete completes a race, gives up a habit, "
            "or pivots to a different focus."
        ),
        handler=retire_intent,
        parameters={
            "type": "object",
            "properties": {
                "intent_id": {
                    "type": "string",
                    "description": (
                        "Canonical UUID of the intent. Find via list_intents."
                    ),
                },
                "reason": {
                    "type": "string",
                    "description": (
                        "Short reason. Keywords like 'achieved', 'done', "
                        "'geschafft' route to status=achieved; anything "
                        "else routes to status=abandoned."
                    ),
                },
            },
            "required": ["intent_id"],
        },
        category="memory",
        display_label="Schliesse ein Intent ab",
    ))

    def update_intent(
        intent_id: str,
        priority: int | None = None,
        payload: dict | None = None,
    ) -> dict:
        from src.db.intents_db import (
            update_intent_payload,
            update_intent_priority,
        )

        if not _is_uuid(intent_id):
            return {
                "error": (
                    f"intent_id '{intent_id}' is not a valid UUID."
                ),
                "hint": "Use list_intents() to look up the UUID.",
            }
        if priority is None and payload is None:
            return {
                "error": "Pass at least one of priority or payload.",
                "hint": "Nothing to update otherwise.",
            }

        updated: dict | None = None
        try:
            if priority is not None:
                updated = update_intent_priority(intent_id, priority)
            if payload is not None:
                updated = update_intent_payload(intent_id, payload)
        except Exception as exc:
            return {"error": f"Could not update intent: {exc}"}
        if updated is None:
            return {"error": f"Intent {intent_id} not found."}
        return {"status": "ok", "intent": updated}

    registry.register(Tool(
        name="update_intent",
        description=(
            "Patch a live intent: change its priority, or replace its "
            "payload JSON wholesale (no merge). Use to bump priorities "
            "when the athlete reshuffles focus, or to refresh the "
            "payload (e.g. update target_time after a tune-up race). "
            "Title and type are intentionally immutable; if either is "
            "wrong, retire the old intent and add a new one. "
            "REQUIRED FIRST STEP: call list_intents() to get the UUID."
        ),
        handler=update_intent,
        parameters={
            "type": "object",
            "properties": {
                "intent_id": {"type": "string"},
                "priority": {
                    "type": "integer",
                    "description": "1 (top) to 5.",
                },
                "payload": {
                    "type": "object",
                    "description": "Full replacement payload (no merge).",
                },
            },
            "required": ["intent_id"],
        },
        category="memory",
        display_label="Aktualisiere ein Intent",
    ))

    def list_intents_tool(status: str = "active") -> dict:
        from src.db.intents_db import list_intents

        status_filter = status if status and status.strip() else None
        if status_filter == "all":
            status_filter = None
        rows = list_intents(_user_id, status_filter=status_filter)
        return {"intents": rows, "count": len(rows)}

    registry.register(Tool(
        name="list_intents",
        description=(
            "List athlete intents filtered by status. Defaults to "
            "'active'. Pass status='all' to see every intent ever "
            "filed (including 'achieved' and 'abandoned'). Returned "
            "rows include the canonical UUID needed by retire_intent / "
            "update_intent. The active intents are already injected "
            "into the system prompt under Active Intents, so call this "
            "only when you need a specific UUID or historical data."
        ),
        handler=list_intents_tool,
        parameters={
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "description": (
                        "active | paused | achieved | abandoned | all. "
                        "Defaults to active."
                    ),
                },
            },
        },
        category="memory",
        display_label="Liste Intents",
    ))

    # ---------------------------------------------------------------
    # Program tools
    # ---------------------------------------------------------------

    def add_program(
        kind: str,
        label: str = "",
        payload: dict | None = None,
    ) -> dict:
        from src.db.programs_db import insert_program

        if not isinstance(kind, str) or not kind.strip():
            return {
                "error": "kind must be a non-empty string",
                "hint": "Pass e.g. gym_split, run_block, swim_block.",
            }
        try:
            row = insert_program(
                _user_id,
                kind=kind,
                label=label,
                payload=payload or {},
            )
        except Exception as exc:
            return {"error": f"Could not add program: {exc}"}
        return {"status": "ok", "program": row}

    registry.register(Tool(
        name="add_program",
        description=(
            "Register a persisting training program (gym split, run "
            "block, swim block). Programs are long-lived structures "
            "that span weeks or months. Kind is free text, prefer "
            "gym_split | run_block | swim_block | bike_block. Label "
            "is the athlete-facing name (e.g. 'Push/Pull/Legs'). "
            "Payload holds the structure JSON (notes, sessions, "
            "weekly_hours, etc.). Programs render under Active Programs "
            "in the system prompt. Use ONCE per active structure; "
            "edits go through update_program."
        ),
        handler=add_program,
        parameters={
            "type": "object",
            "properties": {
                "kind": {
                    "type": "string",
                    "description": (
                        "Program type. Hinted: gym_split, run_block, "
                        "swim_block, bike_block. Free text accepted."
                    ),
                },
                "label": {
                    "type": "string",
                    "description": (
                        "Human-readable name, e.g. 'Push/Pull/Legs'."
                    ),
                },
                "payload": {
                    "type": "object",
                    "description": "Kind-specific structure JSON.",
                },
            },
            "required": ["kind"],
        },
        category="memory",
        display_label="Lege ein Programm an",
    ))

    def update_program_tool(
        program_id: str,
        label: str | None = None,
        payload: dict | None = None,
    ) -> dict:
        from src.db.programs_db import update_program

        if not _is_uuid(program_id):
            return {
                "error": (
                    f"program_id '{program_id}' is not a valid UUID."
                ),
                "hint": "Use list_programs() to look up the UUID.",
            }
        if label is None and payload is None:
            return {
                "error": "Pass at least one of label or payload.",
                "hint": "Nothing to update otherwise.",
            }
        try:
            row = update_program(
                program_id,
                label=label,
                payload=payload,
            )
        except Exception as exc:
            return {"error": f"Could not update program: {exc}"}
        if row is None:
            return {"error": f"Program {program_id} not found."}
        return {"status": "ok", "program": row}

    registry.register(Tool(
        name="update_program",
        description=(
            "Patch a live program's label or payload. Payload is "
            "replaced WHOLESALE (no JSON merge) - pass the full new "
            "state. Use when the structure evolves (added a 4th gym "
            "day, dropped a key session, updated weekly volume). "
            "REQUIRED FIRST STEP: call list_programs() for the UUID. "
            "Do NOT deactivate via this tool - use deactivate_program."
        ),
        handler=update_program_tool,
        parameters={
            "type": "object",
            "properties": {
                "program_id": {"type": "string"},
                "label": {"type": "string"},
                "payload": {
                    "type": "object",
                    "description": "Full replacement payload.",
                },
            },
            "required": ["program_id"],
        },
        category="memory",
        display_label="Aktualisiere ein Programm",
    ))

    def deactivate_program(program_id: str) -> dict:
        from src.db.programs_db import update_program

        if not _is_uuid(program_id):
            return {
                "error": (
                    f"program_id '{program_id}' is not a valid UUID."
                ),
                "hint": "Use list_programs() to look up the UUID.",
            }
        try:
            row = update_program(program_id, active=False)
        except Exception as exc:
            return {"error": f"Could not deactivate program: {exc}"}
        if row is None:
            return {"error": f"Program {program_id} not found."}
        return {"status": "ok", "program": row}

    registry.register(Tool(
        name="deactivate_program",
        description=(
            "Retire a program by flipping active=false. The historical "
            "row is preserved so the coach can refer back. Use when "
            "the athlete moves on to a different block (e.g. ends a "
            "marathon block and starts a strength phase). "
            "REQUIRED FIRST STEP: call list_programs() to find the UUID."
        ),
        handler=deactivate_program,
        parameters={
            "type": "object",
            "properties": {
                "program_id": {"type": "string"},
            },
            "required": ["program_id"],
        },
        category="memory",
        display_label="Beende ein Programm",
    ))

    def list_programs_tool(active_only: bool = True) -> dict:
        from src.db.programs_db import list_programs

        rows = list_programs(_user_id, active_only=bool(active_only))
        return {"programs": rows, "count": len(rows)}

    registry.register(Tool(
        name="list_programs",
        description=(
            "List the athlete's programs. Defaults to active_only=true "
            "(matches what the system prompt already injects). Pass "
            "active_only=false to see retired programs too. Use when "
            "you need a program UUID for update / deactivate, or want "
            "to audit history."
        ),
        handler=list_programs_tool,
        parameters={
            "type": "object",
            "properties": {
                "active_only": {
                    "type": "boolean",
                    "description": "Default true.",
                },
            },
        },
        category="memory",
        display_label="Liste Programme",
    ))

    # ---------------------------------------------------------------
    # Coach note tools (APPEND-ONLY)
    # ---------------------------------------------------------------

    def add_coach_note(
        content: str,
        scope: str = "",
        source_session_id: str | None = None,
    ) -> dict:
        from src.db.coach_notes_db import insert_coach_note

        if not isinstance(content, str) or not content.strip():
            return {
                "error": "content must be a non-empty string",
                "hint": "Pass the learning, e.g. 'HRV drops after 3 quality sessions'.",
            }
        if source_session_id is not None and not _is_uuid(source_session_id):
            return {
                "error": (
                    f"source_session_id '{source_session_id}' is not "
                    "a valid UUID."
                ),
                "hint": (
                    "Pass the canonical session UUID or leave it out "
                    "for off-session observations."
                ),
            }
        try:
            row = insert_coach_note(
                _user_id,
                content=content,
                scope=scope or None,
                source_session_id=source_session_id,
            )
        except Exception as exc:
            return {"error": f"Could not add coach note: {exc}"}
        return {"status": "ok", "coach_note": row}

    registry.register(Tool(
        name="add_coach_note",
        description=(
            "Append a learning about the athlete to the Coach Notes "
            "log. APPEND-ONLY: never edit a prior note. When something "
            "changes, write a new note. Use for durable observations "
            "the coach should remember across sessions (e.g. 'reacts "
            "well to 2x Threshold/week', 'always Easy after 2 nights "
            "<6h sleep'). Avoid for one-off session decisions (Open "
            "Threads in the journal). Optional scope groups the note: "
            "training_response, life_pattern, preference_learned, or "
            "a custom tag. source_session_id (optional) links to the "
            "session that prompted the note."
        ),
        handler=add_coach_note,
        parameters={
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Free-form note text.",
                },
                "scope": {
                    "type": "string",
                    "description": (
                        "Optional grouping label. Hinted: "
                        "training_response, life_pattern, "
                        "preference_learned."
                    ),
                },
                "source_session_id": {
                    "type": "string",
                    "description": (
                        "Optional UUID of the session that prompted "
                        "the note."
                    ),
                },
            },
            "required": ["content"],
        },
        category="memory",
        display_label="Notiere eine Lernerfahrung",
    ))

    def list_coach_notes_tool(
        limit: int = 20,
        scope: str | None = None,
    ) -> dict:
        from src.db.coach_notes_db import list_coach_notes

        scope_filter = scope if scope and scope.strip() else None
        rows = list_coach_notes(
            _user_id,
            limit=int(limit),
            scope_filter=scope_filter,
        )
        return {"coach_notes": rows, "count": len(rows)}

    registry.register(Tool(
        name="list_coach_notes",
        description=(
            "Return coach notes newest first. The system prompt "
            "already injects the last 10 notes under Coach Notes, so "
            "call this only when you need more (audit, look up older "
            "notes, filter by scope). Limit defaults to 20, max 200."
        ),
        handler=list_coach_notes_tool,
        parameters={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Number of notes (default 20).",
                },
                "scope": {
                    "type": "string",
                    "description": (
                        "Optional scope filter "
                        "(training_response, life_pattern, ...)."
                    ),
                },
            },
        },
        category="memory",
        display_label="Liste Coach-Notizen",
    ))
