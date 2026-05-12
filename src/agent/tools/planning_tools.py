"""Planning tools -- create and evaluate training plans.

These are the equivalent of Claude Code using Write/Edit tools to create code.
Plan creation uses a specialized LLM call (like a sub-agent) with a
coaching-specific system prompt.
"""

import json
from datetime import datetime
from pathlib import Path

import logging

from src.agent.llm import chat_completion
from src.agent.json_utils import extract_json
from src.agent.tools.registry import Tool, ToolRegistry
from src.config import get_settings

logger = logging.getLogger(__name__)


def _unwrap_plan(plan: dict) -> dict:
    """Unwrap nested plan structures that LLMs sometimes produce.

    Gemini occasionally wraps the actual plan in a `{"result": "```json ...```"}`
    envelope. This function detects and unwraps such wrappers so downstream
    code always receives a proper plan dict.
    """
    # Case 1: {"result": "<json string>"} wrapper
    if list(plan.keys()) == ["result"] and isinstance(plan.get("result"), str):
        raw = plan["result"]
        try:
            inner = extract_json(raw)
            if isinstance(inner, dict):
                logger.info("Unwrapped plan from 'result' string wrapper")
                return _unwrap_plan(inner)  # recurse in case of double-wrap
        except (ValueError, TypeError):
            pass

    # Case 2: Plan nested under a single top-level key like "plan" or "training_plan"
    # but only if the inner value is a dict with session-like content
    for wrapper_key in ("plan", "training_plan", "weekly_plan"):
        inner = plan.get(wrapper_key)
        if (
            isinstance(inner, dict)
            and len(plan) <= 3  # wrapper + maybe metadata keys
            and any(k in inner for k in ("sessions", "days", "s", "plan"))
        ):
            logger.info("Unwrapped plan from '%s' dict wrapper", wrapper_key)
            # Preserve any metadata keys from the outer wrapper
            result = {**inner}
            for k, v in plan.items():
                if k != wrapper_key and k not in result:
                    result[k] = v
            return result

    return plan


def _build_recovery_planning_context(user_id: str | None) -> str | None:
    """Build recovery-aware planning context from health data.

    Returns planning-relevant hints based on current recovery metrics,
    or None if no health data or on error.
    """
    if not user_id:
        return None

    try:
        from src.services.health_context import build_health_summary

        summary = build_health_summary(user_id, days=7)
        if not summary or not summary.get("data_available"):
            return None

        latest = summary.get("latest", {})
        avgs = summary.get("averages_7d", {})
        hints: list[str] = []

        # Sleep quality check
        sleep_score = latest.get("sleep_score")
        if sleep_score is not None and sleep_score < 65:
            hints.append(f"Sleep score is low ({sleep_score}/100). Reduce high-intensity sessions, add recovery work.")

        # HRV trend check
        hrv = latest.get("hrv")
        avg_hrv = avgs.get("hrv")
        if hrv is not None and avg_hrv is not None and avg_hrv > 0:
            hrv_deviation = ((hrv - avg_hrv) / avg_hrv) * 100
            if hrv_deviation < -15:
                hints.append(f"HRV is {abs(hrv_deviation):.0f}% below 7-day average ({hrv} vs {avg_hrv}). Signs of accumulated fatigue.")

        # Stress check
        stress = latest.get("stress")
        if stress is not None and stress > 50:
            hints.append(f"Stress level is elevated ({stress}). Consider lighter training load.")

        # Body battery check
        body_battery = latest.get("body_battery_high")
        if body_battery is not None and body_battery < 30:
            hints.append(f"Body battery is very low ({body_battery}). Rest day recommended.")

        if not hints:
            # Good recovery — note it
            parts = []
            if sleep_score is not None:
                parts.append(f"Sleep {sleep_score}")
            if hrv is not None:
                parts.append(f"HRV {hrv}")
            if parts:
                hints.append(f"Recovery looks good ({', '.join(parts)}). Normal training load appropriate.")

        return "\n".join(hints)

    except Exception:
        return None  # Non-critical — plan generation continues without recovery context


def _build_macrocycle_week_context(week_number: int, settings=None, user_id: str = None) -> str | None:
    """Load the active macrocycle and extract context for the given week.

    Returns a formatted string for prompt injection, or None on failure.
    """
    try:
        if settings is None:
            settings = get_settings()
        if not settings.use_supabase:
            return None

        uid = user_id or settings.agenticsports_user_id
        from src.db.macrocycle_db import get_active_macrocycle
        macrocycle = get_active_macrocycle(uid)
        if not macrocycle:
            return None

        weeks = macrocycle.get("weeks", [])
        # Find the matching week
        target_week = None
        for w in weeks:
            if w.get("week_number") == week_number:
                target_week = w
                break

        if not target_week:
            return None

        lines = [
            f"MACROCYCLE CONTEXT (Week {week_number} of {macrocycle.get('total_weeks', '?')}):",
            f"  Macrocycle: {macrocycle.get('name', 'Unknown')}",
            f"  Phase: {target_week.get('phase', 'Unknown')}",
            f"  Focus: {target_week.get('focus', 'Not specified')}",
        ]

        volume = target_week.get("volume_target", {})
        if volume:
            lines.append(f"  Volume target: {volume.get('total_minutes', '?')} min, {volume.get('total_sessions', '?')} sessions")

        intensity = target_week.get("intensity_distribution", {})
        if intensity:
            lines.append(
                f"  Intensity: {intensity.get('low', '?')}% low, "
                f"{intensity.get('moderate', '?')}% moderate, "
                f"{intensity.get('high', '?')}% high"
            )

        key_sessions = target_week.get("key_sessions", [])
        if key_sessions:
            lines.append(f"  Key sessions: {', '.join(key_sessions)}")

        notes = target_week.get("notes")
        if notes:
            lines.append(f"  Notes: {notes}")

        lines.append("")
        lines.append("Design this week's training plan to match the macrocycle targets above.")
        return "\n".join(lines)

    except Exception:
        return None  # Non-critical — plan generation continues without macrocycle context


def register_planning_tools(registry: ToolRegistry, user_model):
    """Register all planning tools."""
    _settings = get_settings()

    def create_training_plan(
        focus: str = None,
        feedback: str = None,
        sport_distribution: dict = None,
        macrocycle_week: int = None,
    ) -> dict:
        """Generate a training plan using the coach persona."""
        from src.agent.prompts import build_coach_system_prompt, build_plan_prompt

        profile = user_model.project_profile()
        beliefs = user_model.get_active_beliefs(min_confidence=0.6)
        uid = (getattr(user_model, "user_id", None) or _settings.agenticsports_user_id) if _settings.use_supabase else ""

        if _settings.use_supabase:
            from src.db import list_activities as db_list_activities
            from src.db import list_episodes as db_list_episodes
            activities = db_list_activities(uid, limit=50)
            episodes = db_list_episodes(uid, limit=10)
        else:
            from src.tools.activity_store import list_activities
            from src.memory.episodes import list_episodes
            activities = list_activities()
            episodes = list_episodes(limit=10)

        from src.memory.episodes import retrieve_relevant_episodes
        relevant_eps = retrieve_relevant_episodes(
            {"goal": profile.get("goal", {}), "sports": profile.get("sports", [])},
            episodes,
            max_results=5,
        )

        base_prompt = build_plan_prompt(
            profile, beliefs=beliefs, activities=activities,
            relevant_episodes=relevant_eps,
        )

        # Inject recovery context when available
        recovery_context = _build_recovery_planning_context(uid or None)
        if recovery_context:
            base_prompt += f"\n\nCURRENT RECOVERY STATUS:\n{recovery_context}"

        # Inject macrocycle week context when specified
        if macrocycle_week is not None:
            macro_context = _build_macrocycle_week_context(macrocycle_week, settings=_settings, user_id=uid)
            if macro_context:
                base_prompt += f"\n\n{macro_context}"

        if focus:
            base_prompt += f"\n\nFOCUS FOR THIS PLAN: {focus}"
        if feedback:
            base_prompt += f"\n\nPREVIOUS PLAN FEEDBACK (address these issues): {feedback}"
        if sport_distribution:
            base_prompt += f"\n\nREQUESTED SPORT DISTRIBUTION: {json.dumps(sport_distribution)}"

        response = chat_completion(
            messages=[{"role": "user", "content": base_prompt}],
            system_prompt=build_coach_system_prompt(uid),
            temperature=0.7,
        )

        plan = extract_json(response.choices[0].message.content.strip())
        plan = _unwrap_plan(plan)
        plan["_generated_at"] = datetime.now().isoformat()
        if macrocycle_week is not None:
            plan["_macrocycle_week"] = macrocycle_week

        # Cache the draft so save_plan() can save without the agent
        # re-echoing the full plan dict through context.
        try:
            user_model.meta = {**user_model.meta, "_last_plan_draft": plan}
            user_model.save()
        except Exception as exc:
            logger.debug("Failed to cache plan draft: %s", exc)

        return plan

    registry.register(Tool(
        name="create_training_plan",
        description=(
            "Generate a structured weekly training plan via a specialized coach "
            "sub-agent. Pulls in the athlete profile, active beliefs, last 50 "
            "activities, relevant past episodes, current recovery status from "
            "health data, and (if specified) the matching macrocycle week, then "
            "calls an LLM with the coach persona to produce a JSON plan of "
            "concrete sessions (sport, type, duration, intensity targets).\n\n"
            "WHEN TO USE:\n"
            "- Athlete asks for next week's plan, a new training week, or 'what "
            "should I do this week?'.\n"
            "- An existing plan was rejected by evaluate_plan and you have feedback "
            "to address - pass the feedback string here for the regenerated plan.\n"
            "- Athlete circumstances changed mid-cycle (injury, schedule change, "
            "race added) and the old plan is no longer fit for purpose.\n"
            "- After save_macrocycle: generate the first weekly plan grounded in "
            "the new multi-week structure by passing macrocycle_week=1.\n\n"
            "WHEN NOT TO USE:\n"
            "- Athlete wants the BIG picture (multiple weeks, race build, season "
            "structure): use create_macrocycle_plan instead.\n"
            "- Athlete wants tweaks to a specific session, not a fresh weekly plan: "
            "use the session-level adjust tools, not a full regen.\n"
            "- You just want to see the current plan: use get_current_plan.\n\n"
            "WORKFLOW (this is the canonical sequence, do not skip steps):\n"
            "  1. create_training_plan(...) - returns plan dict.\n"
            "  2. evaluate_plan(plan) - returns score, acceptable, issues, suggestions.\n"
            "  3. If acceptable=true (score >= 70): save_plan() with no args (uses "
            "cached draft).\n"
            "  4. If not acceptable: create_training_plan(feedback=<issues>) and "
            "loop. Max 2 retries before surfacing the issues to the athlete.\n\n"
            "EXAMPLES:\n"
            "- create_training_plan() - baseline week from profile + recent data.\n"
            "- create_training_plan(focus='base building, low intensity') for an "
            "intentional base block.\n"
            "- create_training_plan(focus='taper week', macrocycle_week=15) when "
            "week 15 of the active macrocycle is a taper.\n"
            "- create_training_plan(feedback='Too many hard days back-to-back, "
            "missing recovery between intervals') after evaluate_plan flagged it.\n"
            "- create_training_plan(sport_distribution={'running': 3, 'cycling': 2, "
            "'swimming': 1}) for an explicit multi-sport split.\n\n"
            "SIDE EFFECT: the generated plan is cached in user_model.meta as "
            "'_last_plan_draft' so that save_plan() called with no arguments will "
            "persist it. Do not echo the full plan dict back through context - call "
            "save_plan() to commit.\n\n"
            "IMPORTANT: This tool only DRAFTS. Nothing is saved until save_plan() is "
            "called. Always call evaluate_plan after creation; never save an "
            "unevaluated plan."
        ),
        handler=create_training_plan,
        parameters={
            "type": "object",
            "properties": {
                "focus": {
                    "type": "string",
                    "description": "Training focus or emphasis (e.g., 'base building', 'speed work', 'recovery week')",
                    "nullable": True,
                },
                "feedback": {
                    "type": "string",
                    "description": "Feedback from a previous plan evaluation -- issues to fix",
                    "nullable": True,
                },
                "sport_distribution": {
                    "type": "object",
                    "description": "Requested session counts per sport (e.g., {\"running\": 3, \"cycling\": 2})",
                    "nullable": True,
                },
                "macrocycle_week": {
                    "type": "integer",
                    "description": "Week number from active macrocycle to use as context for this plan",
                    "nullable": True,
                },
            },
        },
        category="planning",
    ))

    def evaluate_plan(plan: dict) -> dict:
        """Evaluate plan quality with an independent reviewer persona.

        Uses agent-defined eval criteria from DB. If no criteria are
        defined, the plan is accepted by default (score=100).
        """
        from src.agent.plan_evaluator import evaluate_plan as _evaluate

        profile = user_model.project_profile()
        beliefs = user_model.get_active_beliefs(min_confidence=0.6)

        user_id = getattr(user_model, "user_id", "unknown")

        evaluation = _evaluate(
            plan, profile, user_id=user_id, beliefs=beliefs,
        )

        return {
            "score": evaluation.score,
            "acceptable": evaluation.acceptable,
            "criteria": evaluation.criteria_scores,
            "issues": evaluation.issues,
            "suggestions": evaluation.suggestions,
        }

    registry.register(Tool(
        name="evaluate_plan",
        description=(
            "Have a training plan independently reviewed by a separate evaluator "
            "persona, scored against the athlete's agent-defined eval criteria "
            "(see define_eval_criteria). Returns a numeric score (0-100), an "
            "acceptable boolean, per-criterion sub-scores, a list of issues, and "
            "concrete suggestions to improve.\n\n"
            "WHEN TO USE:\n"
            "- IMMEDIATELY after every successful create_training_plan call. This "
            "is non-optional in the create -> evaluate -> save loop.\n"
            "- When the athlete asks 'is this plan good?' about an existing plan "
            "you have in hand - pass the plan dict directly.\n"
            "- Before save_plan: a plan must pass evaluation (acceptable=true) "
            "before persistence.\n\n"
            "WHEN NOT TO USE:\n"
            "- For evaluating a single session in isolation (the criteria are "
            "plan-level: balance, progression, recovery distribution).\n"
            "- To re-run on a plan you already evaluated and accepted - the result "
            "is deterministic enough that re-evaluation wastes tokens.\n\n"
            "DECISION RULES:\n"
            "- acceptable=true AND score >= 70 -> call save_plan() immediately.\n"
            "- acceptable=false OR score < 70 -> regenerate with "
            "create_training_plan(feedback=<concatenate issues + suggestions>). "
            "Max 2 retry loops, then surface the remaining issues to the athlete.\n\n"
            "DEFAULT BEHAVIOR: if no eval_criteria are defined for the user, the "
            "plan is auto-accepted with score=100. The agent is expected to "
            "define_eval_criteria during onboarding or first-plan-creation so "
            "evaluation has teeth.\n\n"
            "EXAMPLE:\n"
            "  draft = create_training_plan(focus='build phase')\n"
            "  eval = evaluate_plan(plan=draft)\n"
            "  if eval['acceptable']: save_plan()\n"
            "  else: create_training_plan(feedback=' '.join(eval['issues']))\n\n"
            "Returns {score, acceptable, criteria (dict of per-criterion scores), "
            "issues (list of strings), suggestions (list of strings)}."
        ),
        handler=evaluate_plan,
        parameters={
            "type": "object",
            "properties": {
                "plan": {
                    "type": "object",
                    "description": "The training plan to evaluate",
                },
            },
            "required": ["plan"],
        },
        category="planning",
    ))

    def save_plan(plan: dict | None = None) -> dict:
        """Save a training plan.

        If *plan* is omitted, uses the cached draft from the most recent
        ``create_training_plan`` call. This lets the agent persist a plan
        with a single ``save_plan()`` call after evaluation passes,
        without re-echoing the whole plan through context.
        """
        if plan is None or not isinstance(plan, dict) or not plan:
            plan = (user_model.meta or {}).get("_last_plan_draft")
            if not plan:
                return {
                    "error": (
                        "No plan provided and no draft is cached. Call "
                        "create_training_plan first, then save_plan() to "
                        "persist the draft."
                    )
                }
            # Use a copy so popping _evaluation does not mutate cache.
            plan = dict(plan)
        if _settings.use_supabase:
            from src.db import store_plan
            evaluation = plan.pop("_evaluation", None)
            score = evaluation.get("score") if evaluation else None
            feedback = evaluation.get("feedback") if evaluation else None
            row = store_plan(
                getattr(user_model, "user_id", None) or _settings.agenticsports_user_id, plan,
                evaluation_score=score, evaluation_feedback=feedback,
            )
            return {"saved": True, "id": row["id"]}
        else:
            plans_dir = Path("data/plans")
            plans_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            path = plans_dir / f"plan_{timestamp}.json"
            path.write_text(json.dumps(plan, indent=2))
            return {"saved": True, "path": str(path)}

    registry.register(Tool(
        name="save_plan",
        description=(
            "Persist the current training plan draft as the new active plan. "
            "Archives any previously active plan automatically. This is the "
            "commit step in the create -> evaluate -> save workflow.\n\n"
            "WHEN TO USE:\n"
            "- IMMEDIATELY after evaluate_plan returns acceptable=true. Do not "
            "wait for further user confirmation unless the user explicitly asked "
            "to review first.\n"
            "- After regeneration loop converges on an acceptable plan.\n\n"
            "WHEN NOT TO USE:\n"
            "- Before evaluate_plan has run and returned acceptable=true. Saving "
            "an unevaluated plan bypasses quality control - do not do it.\n"
            "- If the athlete is still editing the plan in conversation: wait for "
            "them to confirm before saving.\n"
            "- To save a plan from many turns ago: it is no longer the cached "
            "draft and the agent would need to pass it explicitly.\n\n"
            "CALLING CONVENTION (CRITICAL - read this):\n"
            "- save_plan() with NO arguments: persists the most recent draft "
            "cached by create_training_plan in user_model.meta. This is the "
            "preferred form - it costs ZERO context tokens because you do not "
            "have to echo the whole plan dict back.\n"
            "- save_plan(plan=<dict>): persists the provided plan, overriding "
            "the cache. Only use this if you genuinely need to save a different "
            "plan (e.g., one passed in from elsewhere) - normally avoid.\n\n"
            "If no plan is provided AND no draft is cached, returns "
            "{'error': '...'}. Diagnose: call create_training_plan first.\n\n"
            "EXAMPLE FLOW:\n"
            "  create_training_plan(focus='base') -> {plan dict, cached}\n"
            "  evaluate_plan(plan=<from step 1>) -> {acceptable: true, score: 82}\n"
            "  save_plan() -> {saved: true, id: '...'} - draft committed from cache."
        ),
        handler=save_plan,
        parameters={
            "type": "object",
            "properties": {
                "plan": {
                    "type": "object",
                    "description": (
                        "OPTIONAL - explicit plan dict. Omit to use the "
                        "cached draft from the last create_training_plan."
                    ),
                    "nullable": True,
                },
            },
        },
        category="planning",
    ))

    def get_plan_history(limit: int = 5) -> dict:
        """Get historical training plans for the athlete."""
        from src.db.plans_db import list_plans

        uid = getattr(user_model, "user_id", None) or _settings.agenticsports_user_id

        plans = list_plans(uid, limit=min(limit, 20))

        if not plans:
            return {"plans": [], "message": "No historical plans found."}

        from src.agent.plan_evaluator import extract_sessions_from_plan

        summaries = []
        for p in plans:
            plan_data = p.get("plan_data", {})
            sessions = extract_sessions_from_plan(plan_data)
            summaries.append({
                "id": p.get("id"),
                "created_at": p.get("created_at"),
                "active": p.get("active", False),
                "evaluation_score": p.get("evaluation_score"),
                "session_count": len(sessions),
                "week_start": plan_data.get("week_start"),
                "sports": list(set(
                    s.get("sport") or s.get("sp") or "unknown" for s in sessions
                )),
                "total_duration_min": sum(
                    s.get("total_duration_minutes") or s.get("duration_minutes") or s.get("dur_min") or s.get("dur") or 0
                    for s in sessions
                ),
            })

        return {"plans": summaries, "count": len(summaries)}

    registry.register(Tool(
        name="get_plan_history",
        description=(
            "Get historical training plans. Returns compact summaries of past "
            "plans with dates, scores, sports, and session counts."
        ),
        handler=get_plan_history,
        parameters={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of plans to return (default 5, max 20)",
                },
            },
        },
        category="planning",
    ))
