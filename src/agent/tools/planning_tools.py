"""Planning tools: thin persistence + read access for training plans.

The agent composes plans inline using its own reasoning (no sub-LLM, no
hardcoded coaching rules). These tools only provide:

    - save_plan(plan): persist a plan dict the agent constructed
    - get_active_plan(): read the current plan so the agent can adjust it
    - get_plan_history(limit): list past plans for context

The plan schema is documented in the save_plan tool description so the
agent knows what shape to produce. No prompts, no scores, no rules.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path

from src.agent.tools.registry import Tool, ToolRegistry
from src.config import get_settings

logger = logging.getLogger(__name__)

# Cap the number of sessions embedded in the inline plan_preview UI payload
# so a 12-week plan does not bloat the SSE message. The full plan stays in
# the database; the UI just truncates the inline card.
_MAX_INLINE_PREVIEW_SESSIONS = 30


_SAVE_PLAN_DESCRIPTION = (
    "Persist a weekly training plan you composed AND render an inline "
    "preview card in the chat. Saving emits a plan_preview UI component "
    "the frontend renders immediately, so do NOT also describe the plan "
    "in prose. The frontend reads plan_data.sessions to render the plan "
    "view. Required shape:\n"
    "{\n"
    '  "start_date": "YYYY-MM-DD (Monday)",\n'
    '  "focus": "short free-text label",\n'
    '  "sessions": [\n'
    "    {\n"
    '      "day": "monday|tuesday|... or German equivalent",\n'
    '      "date": "YYYY-MM-DD",\n'
    '      "sport": "running|cycling|swimming|strength|...",\n'
    '      "name": "session title",\n'
    '      "description": "what to do in 1-2 sentences",\n'
    '      "duration_minutes": 60,\n'
    '      "intensity": "low|moderate|high",\n'
    '      "steps": [ /* optional structured intervals */ ],\n'
    '      "notes": "optional coaching notes"\n'
    "    }\n"
    "  ]\n"
    "}\n"
    "Use this whenever you decide to set or replace the athlete's plan. "
    "For adjustments (move a session, swap intensity, etc.) call "
    "get_active_plan first, mutate, then save_plan with the new dict."
)


_WEEKDAY_ORDER = (
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
)


def _coerce_sessions(plan: dict) -> list[dict]:
    """Extract sessions[] from a plan dict, tolerating older shapes.

    The canonical schema is ``plan["sessions"] = [{day, sport, ...}]`` but
    earlier agent versions sometimes produced ``plan["weekly_structure"] =
    {"monday": {...}, ...}``. We convert that on the fly so the
    plan_preview card renders the same way regardless of when the plan
    was saved.
    """
    sessions = plan.get("sessions")
    if isinstance(sessions, list) and sessions:
        return sessions

    weekly = plan.get("weekly_structure")
    if isinstance(weekly, dict) and weekly:
        derived: list[dict] = []
        # Sort by canonical weekday order so the UI shows Mon..Sun.
        for day in _WEEKDAY_ORDER:
            entry = weekly.get(day)
            if not isinstance(entry, dict):
                continue
            derived.append({
                "day": day,
                "sport": entry.get("sport"),
                "name": entry.get("focus") or entry.get("name"),
                "description": entry.get("notes") or entry.get("description"),
                "duration_minutes": entry.get("target_duration_min")
                or entry.get("duration_minutes"),
                "intensity": entry.get("intensity"),
            })
        # Also pick up any extra keys not in the canonical weekday list
        # (e.g. an athlete-specific custom label).
        for key, entry in weekly.items():
            if key in _WEEKDAY_ORDER or not isinstance(entry, dict):
                continue
            derived.append({
                "day": key,
                "sport": entry.get("sport"),
                "name": entry.get("focus") or entry.get("name"),
                "description": entry.get("notes") or entry.get("description"),
                "duration_minutes": entry.get("target_duration_min")
                or entry.get("duration_minutes"),
                "intensity": entry.get("intensity"),
            })
        return derived

    return []


def _build_plan_preview(plan: dict, plan_id: str | int | None) -> dict:
    """Construct the ``_ui_component`` payload for the plan_preview card.

    Drops fields that are absent from the plan dict rather than substituting
    nulls. The frontend treats an absent key as "the agent did not provide
    this" and renders accordingly.
    """
    sessions = _coerce_sessions(plan)
    truncated = len(sessions) > _MAX_INLINE_PREVIEW_SESSIONS
    inline_sessions = (
        list(sessions[:_MAX_INLINE_PREVIEW_SESSIONS]) if truncated else list(sessions)
    )

    props: dict = {}
    if plan_id is not None:
        props["plan_id"] = plan_id
    if "start_date" in plan:
        props["start_date"] = plan.get("start_date")
    elif "period_start" in plan:
        # Older plans used period_start; surface it under the canonical key.
        props["start_date"] = plan.get("period_start")
    if "focus" in plan:
        props["focus"] = plan.get("focus")
    elif "name" in plan:
        props["focus"] = plan.get("name")
    # Always include sessions (possibly empty list) so the card can render
    # an explicit "no sessions" state instead of breaking on a missing key.
    props["sessions"] = inline_sessions
    props["truncated_in_ui"] = truncated

    return {
        "type": "plan_preview",
        "id": uuid.uuid4().hex,
        "props": props,
    }


def register_planning_tools(registry: ToolRegistry, user_model) -> None:
    """Register plan persistence and read tools."""
    _settings = get_settings()

    def _resolve_user_id() -> str:
        return (
            getattr(user_model, "user_id", None)
            or _settings.agenticsports_user_id
        )

    def save_plan(plan: dict) -> dict:
        """Persist a training plan and emit an inline plan_preview card."""
        if not isinstance(plan, dict) or not plan:
            return {"error": "plan must be a non-empty dict matching the documented schema"}

        # Build a new dict instead of mutating the caller's plan.
        saved_plan = {**plan, "_saved_at": datetime.now().isoformat()}

        sessions = plan.get("sessions") or []
        truncated_in_ui = len(sessions) > _MAX_INLINE_PREVIEW_SESSIONS

        if _settings.use_supabase:
            from src.db import store_plan
            row = store_plan(_resolve_user_id(), saved_plan)
            plan_id = row["id"]
            result: dict = {"saved": True, "id": plan_id}
        else:
            plans_dir = Path("data/plans")
            plans_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            path = plans_dir / f"plan_{timestamp}.json"
            path.write_text(json.dumps(saved_plan, indent=2))
            plan_id = str(path)
            result = {"saved": True, "path": str(path)}

        if truncated_in_ui:
            # Surface truncation to the agent so it knows the inline card
            # only shows the first N sessions; the saved plan is full.
            result["truncated_in_ui"] = True

        # The agent loop strips ``_ui_component`` before the result goes
        # back to the LLM and forwards it as an SSE ``ui_component`` event
        # the frontend renders via the renderUIComponent registry.
        result["_ui_component"] = _build_plan_preview(plan, plan_id)
        return result

    registry.register(Tool(
        name="save_plan",
        description=_SAVE_PLAN_DESCRIPTION,
        handler=save_plan,
        parameters={
            "type": "object",
            "properties": {
                "plan": {
                    "type": "object",
                    "description": "The plan dict. See tool description for schema.",
                },
            },
            "required": ["plan"],
        },
        category="planning",
        display_label="Speichere den Plan",
    ))

    def get_active_plan() -> dict:
        """Return the most recent (active) plan or an empty result.

        Also emits an inline plan_preview UI card so the frontend can
        render the plan as a structured component instead of forcing
        the coach to describe it in prose. When the athlete asks
        "show me my plan" we want the visual, not a wall of text.
        """
        if not _settings.use_supabase:
            return {"plan": None, "message": "No Supabase configured."}
        from src.db.plans_db import get_active_plan as _get_active
        row = _get_active(_resolve_user_id())
        if not row:
            return {"plan": None, "message": "No active plan."}
        plan_data = row.get("plan_data") or {}
        plan_id = row.get("id")
        result: dict = {
            "plan": plan_data,
            "id": plan_id,
            "created_at": row.get("created_at"),
        }
        if plan_data:
            result["_ui_component"] = _build_plan_preview(plan_data, plan_id)
        return result

    registry.register(Tool(
        name="get_active_plan",
        description=(
            "Read the athlete's current active plan and render it as an "
            "inline plan_preview card. The card is the canonical surface "
            "for showing a plan in chat: do NOT re-describe sessions in "
            "prose after calling this. Use this before adjusting (move a "
            "session, change intensity, etc.) so you can mutate the dict "
            "and call save_plan with the new version."
        ),
        handler=get_active_plan,
        parameters={"type": "object", "properties": {}},
        category="planning",
        display_label="Lese deinen aktuellen Plan",
    ))

    def get_plan_history(limit: int = 5) -> dict:
        """List historical plans (compact summaries)."""
        if not _settings.use_supabase:
            return {"plans": [], "message": "No Supabase configured."}
        from src.db.plans_db import list_plans
        plans = list_plans(_resolve_user_id(), limit=min(max(limit, 1), 20))
        summaries = [
            {
                "id": p.get("id"),
                "created_at": p.get("created_at"),
                "active": p.get("active", False),
                "focus": (p.get("plan_data") or {}).get("focus"),
                "start_date": (p.get("plan_data") or {}).get("start_date"),
                "session_count": len((p.get("plan_data") or {}).get("sessions") or []),
            }
            for p in plans
        ]
        return {"plans": summaries, "count": len(summaries)}

    registry.register(Tool(
        name="get_plan_history",
        description=(
            "List historical training plans (compact summaries: id, "
            "created_at, focus, start_date, session_count). Use for "
            "context when planning a new week."
        ),
        handler=get_plan_history,
        parameters={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Max plans to return (default 5, max 20).",
                },
            },
        },
        category="planning",
        display_label="Schaue in deine alten Plaene",
    ))
