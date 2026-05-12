"""Macrocycle planning tools -- create, view, and save multi-week training plans.

A macrocycle defines the high-level training structure across weeks/months,
including phases (base, build, peak, taper), weekly volume targets, and
intensity distribution. Weekly training plans are then derived from
the active macrocycle.

Tools:
- create_macrocycle_plan: Generate a macrocycle via LLM sub-agent
- get_macrocycle: Retrieve the active macrocycle from DB
- save_macrocycle: Persist a reviewed macrocycle to DB
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta

from src.agent.llm import chat_completion
from src.agent.json_utils import extract_json
from src.agent.tools.registry import Tool, ToolRegistry
from src.config import get_settings

logger = logging.getLogger(__name__)


def _resolve_user_id(user_model, settings) -> str:
    """Return the user ID from the model, falling back to settings."""
    if user_model and getattr(user_model, "user_id", None):
        return user_model.user_id
    return settings.agenticsports_user_id


def register_macrocycle_tools(registry: ToolRegistry, user_model) -> None:
    """Register macrocycle planning tools on the given *registry*."""
    _settings = get_settings()
    _user_id = _resolve_user_id(user_model, _settings)

    def create_macrocycle_plan(
        name: str,
        weeks: int = 12,
        periodization_model: str | None = None,
        start_date: str | None = None,
    ) -> dict:
        """Generate a multi-week macrocycle plan using LLM sub-agent.

        Args:
            name: Descriptive name for the macrocycle.
            weeks: Total weeks (4-52, default 12).
            periodization_model: Optional name of agent-defined periodization model.
            start_date: Optional start date (YYYY-MM-DD). Defaults to today.

        Returns:
            Dict with name, weeks plan, start_date -- ready for athlete review.
        """
        from src.agent.prompts import (
            MACROCYCLE_SYSTEM_PROMPT,
            build_macrocycle_prompt,
        )

        # Validate weeks range
        clamped_weeks = max(4, min(52, weeks))

        # Get athlete context
        profile = user_model.project_profile()
        beliefs = user_model.get_active_beliefs(min_confidence=0.6)

        # Load periodization model if specified
        model_data = None
        if periodization_model and _settings.use_supabase:
            try:
                from src.db.agent_config_db import get_periodization_model
                model_data = get_periodization_model(
                    _user_id,
                    periodization_model,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to load periodization model '%s': %s",
                    periodization_model,
                    exc,
                )

        # Load recent activities (28 days)
        activities = None
        if _settings.use_supabase:
            try:
                from src.db import list_activities as db_list_activities
                cutoff = (date.today() - timedelta(days=28)).isoformat()
                activities = db_list_activities(
                    _user_id,
                    limit=50,
                    after=cutoff,
                )
            except Exception as exc:
                logger.warning("Failed to load activities: %s", exc)

        # Load health summary
        health_summary = None
        if _settings.use_supabase:
            try:
                from src.services.health_context import build_health_summary
                health_summary = build_health_summary(
                    _user_id,
                    days=7,
                )
            except Exception as exc:
                logger.warning("Failed to load health summary: %s", exc)

        # Build prompt and call LLM
        user_prompt = build_macrocycle_prompt(
            profile=profile,
            total_weeks=clamped_weeks,
            beliefs=beliefs,
            activities=activities,
            health_summary=health_summary,
            periodization_model=model_data,
        )

        response = chat_completion(
            messages=[{"role": "user", "content": user_prompt}],
            system_prompt=MACROCYCLE_SYSTEM_PROMPT,
            temperature=0.7,
        )

        plan = extract_json(response.choices[0].message.content.strip())

        resolved_start = start_date or date.today().isoformat()

        draft = {
            "name": name,
            "total_weeks": clamped_weeks,
            "start_date": resolved_start,
            "weeks": plan.get("weeks", []),
            "periodization_model_name": periodization_model,
            "_generated_at": datetime.now().isoformat(),
            "_status": "draft",
        }

        # Cache the full draft in user_model.meta so that a subsequent
        # save_macrocycle() call can find it even if the agent forgets to
        # echo the weeks back. This is forgiveness, not laziness: it gives
        # the workflow "create -> save" semantics without round-tripping
        # the full plan through context.
        try:
            user_model.meta = {
                **user_model.meta,
                "_last_macrocycle_draft": draft,
            }
            user_model.save()
        except Exception as exc:
            logger.debug("Failed to cache macrocycle draft: %s", exc)

        return draft

    registry.register(Tool(
        name="create_macrocycle_plan",
        description=(
            "Draft a multi-week macrocycle (base/build/peak/taper, weekly "
            "volumes, intensity distribution, key sessions) via coach sub-agent. "
            "Use for goal events with a date, onboarding, or season reset. "
            "Avoid if active macrocycle still fits or for open-ended fitness. "
            "weeks clamped 4-52 (12-16 typical marathon, 8-12 half). "
            "periodization_model: optional name from define_periodization. "
            "Draft cached; commit via save_macrocycle() no-args. Present phases "
            "to athlete before saving; saving archives any prior macrocycle."
        ),
        handler=create_macrocycle_plan,
        parameters={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Descriptive name (e.g., 'Marathon Build 2026')",
                },
                "weeks": {
                    "type": "integer",
                    "description": "Total weeks (4-52, default 12)",
                },
                "periodization_model": {
                    "type": "string",
                    "description": "Name of agent-defined periodization model to follow",
                    "nullable": True,
                },
                "start_date": {
                    "type": "string",
                    "description": "Start date in YYYY-MM-DD format (default: today)",
                    "nullable": True,
                },
            },
            "required": ["name"],
        },
        category="planning",
    ))

    def get_macrocycle() -> dict:
        """Return the active macrocycle from the database.

        Returns:
            The active macrocycle dict, or an error dict if none exists.
        """
        if not _settings.use_supabase:
            return {"error": "Supabase not configured — macrocycle storage unavailable"}

        try:
            from src.db.macrocycle_db import get_active_macrocycle
            macrocycle = get_active_macrocycle(_user_id)
            if not macrocycle:
                return {"error": "No active macrocycle found. Use create_macrocycle_plan to create one."}
            return macrocycle
        except Exception as exc:
            logger.warning("Failed to load active macrocycle: %s", exc)
            return {"error": f"Failed to load macrocycle: {exc}"}

    registry.register(Tool(
        name="get_macrocycle",
        description=(
            "Return the active macrocycle: weeks array (phase, focus, volume, "
            "intensity distribution, key sessions) plus metadata (name, "
            "total_weeks, start_date). Use to answer 'what phase am I in?', "
            "'weeks to race?', or to decide whether to create a new macrocycle. "
            "Avoid for the weekly plan (use get_current_plan); archived cycles "
            "are not retrievable here. Returns error if none exists."
        ),
        handler=get_macrocycle,
        parameters={"type": "object", "properties": {}},
        category="planning",
    ))

    def save_macrocycle(macrocycle: dict | None = None) -> dict:
        """Persist a macrocycle plan to the database.

        If *macrocycle* is missing fields, falls back to the cached draft
        from the most recent ``create_macrocycle_plan`` call. This lets
        the agent issue a simple ``save_macrocycle()`` without re-echoing
        the entire weeks array.

        Args:
            macrocycle: Dict with name, total_weeks, start_date, weeks array,
                        and optional periodization_model_name. Optional - if
                        omitted, uses the last cached draft.

        Returns:
            Confirmation dict with saved status and macrocycle ID.
        """
        if not _settings.use_supabase:
            return {"error": "Supabase not configured - macrocycle storage unavailable"}

        provided = dict(macrocycle or {})
        cached = (user_model.meta or {}).get("_last_macrocycle_draft") or {}

        # Merge: explicit fields win, cached fills the gaps.
        merged: dict = {**cached, **{k: v for k, v in provided.items() if v not in (None, "", [])}}

        name = merged.get("name")
        if not name:
            return {
                "error": (
                    "Macrocycle must have a 'name' field. Either pass the "
                    "macrocycle dict returned by create_macrocycle_plan, or "
                    "call create_macrocycle_plan first so a draft is cached."
                )
            }

        weeks_data = merged.get("weeks", [])
        if not weeks_data:
            return {
                "error": (
                    "Macrocycle has no 'weeks' array. No draft is cached "
                    "either - call create_macrocycle_plan first, then "
                    "save_macrocycle without arguments (the draft is reused)."
                )
            }

        total_weeks = merged.get("total_weeks", len(weeks_data))
        start_date = merged.get("start_date", date.today().isoformat())
        period_model = merged.get("periodization_model_name")

        try:
            from src.db.macrocycle_db import store_macrocycle
            row = store_macrocycle(
                user_id=_user_id,
                name=name,
                total_weeks=total_weeks,
                start_date=start_date,
                weeks=weeks_data,
                periodization_model_name=period_model,
            )
            return {
                "saved": True,
                "id": row.get("id"),
                "name": name,
                "total_weeks": total_weeks,
                "status": "active",
            }
        except Exception as exc:
            logger.warning("Failed to save macrocycle: %s", exc)
            return {"error": f"Failed to save macrocycle: {exc}"}

    registry.register(Tool(
        name="save_macrocycle",
        description=(
            "Activate the cached macrocycle draft (auto-archives prior active). "
            "CALL save_macrocycle() with NO ARGS to commit the draft from "
            "create_macrocycle_plan (preferred, zero tokens). Pass "
            "macrocycle=<partial dict> only to override specific fields like "
            "start_date; cache fills gaps. Use only after athlete confirms; "
            "never auto-save. Missing name/weeks (both cache and arg empty) "
            "returns error; run create_macrocycle_plan first."
        ),
        handler=save_macrocycle,
        parameters={
            "type": "object",
            "properties": {
                "macrocycle": {
                    "type": "object",
                    "description": (
                        "OPTIONAL - the full macrocycle returned by "
                        "create_macrocycle_plan. If omitted, the cached "
                        "draft from the last create_macrocycle_plan call is "
                        "used instead. Prefer omitting to save context tokens."
                    ),
                    "nullable": True,
                },
            },
        },
        category="planning",
    ))
