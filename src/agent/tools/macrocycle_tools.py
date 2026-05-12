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
            "Generate a multi-week MACROCYCLE: the season-level training "
            "structure that defines which weeks are base, build, peak, taper, "
            "race; weekly volume targets; intensity distribution per week; key "
            "sessions per phase. Calls a specialized LLM sub-agent with the "
            "athlete profile, beliefs, last 28 days of activities, 7-day health "
            "summary, and (if specified) an agent-defined periodization model.\n\n"
            "WHEN TO USE:\n"
            "- Athlete commits to a target race or goal event with a date weeks "
            "or months away ('Berlin Marathon in October'): build the macrocycle "
            "first, then derive weekly plans from it.\n"
            "- Onboarding flow: after the athlete confirms a sport and goal, "
            "create a macrocycle so weekly planning has a frame.\n"
            "- Season reset: athlete finished a race or changed goals - draft a "
            "fresh macrocycle for the new cycle.\n"
            "- When create_training_plan(macrocycle_week=N) would be valuable "
            "but no active macrocycle exists.\n\n"
            "WHEN NOT TO USE:\n"
            "- Athlete just wants next week's sessions: use create_training_plan.\n"
            "- Goal is open-ended fitness (no event, no date): a macrocycle is "
            "overkill - lean on rolling weekly plans.\n"
            "- An active macrocycle already exists and the situation has not "
            "changed: use get_macrocycle to reference it, not create a new one. "
            "Saving a new macrocycle archives the previous one.\n\n"
            "ARGS GUIDE:\n"
            "- name: descriptive, includes goal + year ('Marathon Build 2026', "
            "'Off-season Base Q1 2026'). Used by the athlete to refer back to it.\n"
            "- weeks: clamped to 4-52. Match the time available until the goal "
            "event. 12-16 is common for marathon, 8-12 for a half, 4-8 for a "
            "short build between races.\n"
            "- periodization_model: name of a previously defined model "
            "(see define_periodization). Omit to let the sub-agent design phases "
            "from scratch using profile + goal.\n"
            "- start_date: YYYY-MM-DD. Defaults to today. Use a specific date for "
            "back-planning from a race (subtract weeks from race date).\n\n"
            "EXAMPLES:\n"
            "- create_macrocycle_plan(name='Berlin Marathon 2026', weeks=16, "
            "start_date='2026-06-22') - 16-week build culminating in early Oct.\n"
            "- create_macrocycle_plan(name='Off-season base block', weeks=8, "
            "periodization_model='linear_aerobic_base') - uses a saved model.\n\n"
            "SIDE EFFECT: the generated macrocycle is cached in user_model.meta "
            "as '_last_macrocycle_draft'. After athlete review, call "
            "save_macrocycle() with no args to commit from the cache without "
            "re-echoing the full weeks array.\n\n"
            "IMPORTANT: This drafts only. Nothing is saved or activated until "
            "save_macrocycle() is called. Always present the draft (phases + key "
            "weeks) to the athlete before saving."
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
            "Retrieve the currently active macrocycle (the persisted season-level "
            "training structure). Returns the full weeks array with phase, focus, "
            "volume targets, intensity distribution, and key sessions per week, "
            "plus top-level metadata (name, total_weeks, start_date).\n\n"
            "WHEN TO USE:\n"
            "- Before create_training_plan when you want to ground the weekly "
            "plan in the macrocycle context - you typically just pass "
            "macrocycle_week=N to create_training_plan and let it auto-fetch, "
            "but use get_macrocycle when you need to inspect or reference the "
            "structure in conversation.\n"
            "- When the athlete asks 'what phase am I in?', 'how many weeks left "
            "until race?', or 'what does my plan look like over the next month?'.\n"
            "- Before deciding whether to create_macrocycle_plan: check if an "
            "active one already exists and is still valid for the goal.\n\n"
            "WHEN NOT TO USE:\n"
            "- You just need the current weekly plan: use get_current_plan.\n"
            "- For an archived (inactive) macrocycle: this only returns the "
            "currently active one. Archived macrocycles are not retrievable via "
            "this tool.\n\n"
            "Returns {'error': 'No active macrocycle...'} if none exists - on "
            "that error, suggest create_macrocycle_plan to the athlete if their "
            "goal warrants it. Returns {'error': 'Supabase not configured...'} "
            "in local/CLI mode."
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
            "Commit a reviewed macrocycle as the new ACTIVE macrocycle. Archives "
            "the previously active macrocycle (if any) automatically - there is "
            "only ever one active at a time per user.\n\n"
            "WHEN TO USE:\n"
            "- After create_macrocycle_plan AND the athlete has confirmed they "
            "are happy with the structure (phases, total weeks, key sessions).\n"
            "- After substantive edits to a draft based on athlete feedback "
            "(usually achieved by calling create_macrocycle_plan again with "
            "adjusted args, then save_macrocycle again).\n\n"
            "WHEN NOT TO USE:\n"
            "- Before athlete review and confirmation. Activating a macrocycle "
            "is the foundation for all subsequent weekly plans - never auto-save "
            "without explicit user consent.\n"
            "- To make small tweaks to weekly volume: edit the cached draft "
            "before saving, do not save then re-save.\n\n"
            "CALLING CONVENTION (CRITICAL):\n"
            "- save_macrocycle() with NO arguments: persists the last draft "
            "cached by create_macrocycle_plan in user_model.meta. PREFERRED - "
            "zero context tokens, no need to echo the weeks array.\n"
            "- save_macrocycle(macrocycle={...}): persists the provided dict. "
            "Cached fields fill any gaps (so partial dicts work). Use this only "
            "if you need to override specific fields like start_date or name "
            "after caching.\n\n"
            "REQUIRED FIELDS (in dict or in cache): name (non-empty), weeks "
            "(non-empty array). Without them, returns {'error': ...} with "
            "guidance to call create_macrocycle_plan first.\n\n"
            "EXAMPLES:\n"
            "- Normal flow: create_macrocycle_plan(name='Marathon 2026', "
            "weeks=16) -> show draft to athlete -> athlete confirms -> "
            "save_macrocycle().\n"
            "- Override one field: save_macrocycle(macrocycle={'start_date': "
            "'2026-07-01'}) - merged with cached draft, only start_date is "
            "overridden.\n\n"
            "SIDE EFFECT: archives previous macrocycle to inactive status. Old "
            "macrocycles are not deleted but become inaccessible via "
            "get_macrocycle. Returns {'saved': true, 'id': ..., 'status': "
            "'active'} on success."
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
