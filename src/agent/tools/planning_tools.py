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
from datetime import datetime
from pathlib import Path

from src.agent.tools.registry import Tool, ToolRegistry
from src.config import get_settings

logger = logging.getLogger(__name__)


_SAVE_PLAN_DESCRIPTION = (
    "Persist a weekly training plan you composed. The frontend reads "
    "plan_data.sessions to render the plan view. Required shape:\n"
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


def register_planning_tools(registry: ToolRegistry, user_model) -> None:
    """Register plan persistence and read tools."""
    _settings = get_settings()

    def _resolve_user_id() -> str:
        return (
            getattr(user_model, "user_id", None)
            or _settings.agenticsports_user_id
        )

    def save_plan(plan: dict) -> dict:
        """Persist a training plan to the active store."""
        if not isinstance(plan, dict) or not plan:
            return {"error": "plan must be a non-empty dict matching the documented schema"}

        plan = {**plan, "_saved_at": datetime.now().isoformat()}

        if _settings.use_supabase:
            from src.db import store_plan
            row = store_plan(_resolve_user_id(), plan)
            return {"saved": True, "id": row["id"]}

        plans_dir = Path("data/plans")
        plans_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        path = plans_dir / f"plan_{timestamp}.json"
        path.write_text(json.dumps(plan, indent=2))
        return {"saved": True, "path": str(path)}

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
        """Return the most recent (active) plan or an empty result."""
        if not _settings.use_supabase:
            return {"plan": None, "message": "No Supabase configured."}
        from src.db.plans_db import get_active_plan as _get_active
        row = _get_active(_resolve_user_id())
        if not row:
            return {"plan": None, "message": "No active plan."}
        return {"plan": row.get("plan_data"), "id": row.get("id"), "created_at": row.get("created_at")}

    registry.register(Tool(
        name="get_active_plan",
        description=(
            "Read the athlete's current active plan. Use this before "
            "adjusting (move a session, change intensity, etc.) so you "
            "can mutate the dict and call save_plan with the new version."
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
