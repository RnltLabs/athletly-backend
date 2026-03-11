"""Planning tools -- create and evaluate training plans.

These are the equivalent of Claude Code using Write/Edit tools to create code.
Plan creation uses a specialized LLM call (like a sub-agent) with a
coaching-specific system prompt.
"""

import json
from datetime import datetime
from pathlib import Path

from src.agent.llm import chat_completion
from src.agent.json_utils import extract_json
from src.agent.tools.registry import Tool, ToolRegistry
from src.config import get_settings


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
        plan["_generated_at"] = datetime.now().isoformat()
        if macrocycle_week is not None:
            plan["_macrocycle_week"] = macrocycle_week
        return plan

    registry.register(Tool(
        name="create_training_plan",
        description=(
            "Generate a structured weekly training plan based on athlete profile, "
            "goals, and recent training data. The plan includes specific sessions "
            "with sport, type, duration, and intensity targets. "
            "Optionally pass focus (e.g., 'base building') or feedback from "
            "a previous evaluation to improve the plan. "
            "IMPORTANT: After creating a plan, ALWAYS use evaluate_plan to check quality."
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
            "Have a training plan independently reviewed by a sports science persona. "
            "Returns a quality score (0-100), specific issues found, and improvement "
            "suggestions. Use this AFTER create_training_plan. If score < 70, "
            "regenerate the plan with the feedback."
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

    def save_plan(plan: dict) -> dict:
        """Save a training plan."""
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
            "Save a finalized training plan to disk. Only use this AFTER the plan "
            "has passed evaluation (score >= 70). Returns the file path."
        ),
        handler=save_plan,
        parameters={
            "type": "object",
            "properties": {
                "plan": {
                    "type": "object",
                    "description": "The finalized training plan to save",
                },
            },
            "required": ["plan"],
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
