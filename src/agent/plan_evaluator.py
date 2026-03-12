"""Plan Evaluator: LLM-based scoring of generated training plans.

Priority 4 -- Audit finding #5 ("Kein Evaluator-Optimizer Loop"):
    v1.0 generates plans in a single shot with no feedback. Every LLM call
    is fire-and-forget. The evaluator-optimizer pattern (Anthropic's "Building
    Effective Agents") adds a feedback loop: generate -> evaluate -> accept or
    regenerate with evaluation feedback.

    This module implements the evaluator half. The cognitive loop in
    state_machine.py orchestrates the generate -> evaluate cycle by selecting
    the evaluate_plan action after generate_plan.

Evaluation criteria are loaded from the DB (eval_criteria table) per user.
If no criteria are defined, science-based DEFAULT_EVAL_CRITERIA are used
so every plan receives a real quality evaluation.
"""

import json
import logging
from dataclasses import dataclass, field

from src.agent.json_utils import extract_json
from src.agent.llm import chat_completion

logger = logging.getLogger(__name__)


# Threshold for accepting a plan without regeneration
PLAN_ACCEPTANCE_THRESHOLD = 70

# Maximum regeneration attempts before accepting best available
MAX_PLAN_ITERATIONS = 3

# Default evaluation criteria used when no DB criteria exist for the user.
# Sport-agnostic, grounded in exercise science fundamentals.
DEFAULT_EVAL_CRITERIA = [
    {
        "name": "progressive_overload",
        "weight": 2.0,
        "description": (
            "Hard-easy sequencing: no back-to-back high-intensity days. "
            "Volume increases max ~10% per week. Deload weeks present every 3-4 weeks."
        ),
    },
    {
        "name": "intensity_distribution",
        "weight": 2.0,
        "description": (
            "Approximately 80% of training time at low intensity, 20% at moderate-to-high. "
            "Polarized or pyramidal distribution preferred over threshold-heavy plans."
        ),
    },
    {
        "name": "recovery_adequacy",
        "weight": 1.5,
        "description": (
            "At least one full rest day per week. Recovery sessions placed after hard days. "
            "No more than two consecutive training days without a lower-intensity day."
        ),
    },
    {
        "name": "goal_alignment",
        "weight": 1.5,
        "description": (
            "Session types and volume match the athlete's stated goal event. "
            "Specificity increases as the target date approaches."
        ),
    },
    {
        "name": "constraint_compliance",
        "weight": 1.0,
        "description": (
            "Respects the athlete's stated training days per week, max session duration, "
            "sport preferences, and scheduling constraints."
        ),
    },
]


@dataclass
class PlanEvaluation:
    """Result of evaluating a training plan."""

    score: int                           # 0-100 overall score
    criteria_scores: dict[str, int]      # per-criterion scores
    issues: list[str]                    # specific problems found
    suggestions: list[str]               # how to improve
    acceptable: bool = False             # score >= threshold

    def __post_init__(self):
        self.acceptable = self.score >= PLAN_ACCEPTANCE_THRESHOLD


def evaluate_plan(
    plan: dict,
    profile: dict,
    user_id: str,
    beliefs: list[dict] | None = None,
    assessment: dict | None = None,
) -> PlanEvaluation:
    """Score a plan using eval criteria from the DB, or science-based defaults.

    When no eval_criteria are defined for the user, DEFAULT_EVAL_CRITERIA
    are used so every plan receives a real evaluation.

    Args:
        plan: The generated training plan dict.
        profile: Athlete profile dict.
        user_id: The user's ID for fetching eval criteria from DB.
        beliefs: Optional active beliefs for preference checking.
        assessment: Optional recent assessment for context.

    Returns:
        PlanEvaluation with score, criteria, issues, and suggestions.
    """
    # Unwrap malformed plan structures before evaluation
    plan = _unwrap_plan_for_eval(plan)

    # Pre-check: if no sessions can be extracted, return early with actionable feedback
    sessions = extract_sessions_from_plan(plan)
    if not sessions:
        logger.warning(
            "evaluate_plan: no sessions found in plan (keys: %s). "
            "Returning structural failure score.",
            list(plan.keys()),
        )
        return PlanEvaluation(
            score=20,
            criteria_scores={},
            issues=[
                "Plan structure is malformed — no sessions could be extracted. "
                f"Plan keys found: {list(plan.keys())}. "
                "Regenerate with the standard schema: {\"sessions\": [...]}."
            ],
            suggestions=[
                "Use the canonical plan format with a top-level 'sessions' array.",
                "Each session needs: day, date, sport, type, total_duration_minutes, steps.",
            ],
        )

    from src.db.agent_config_db import get_eval_criteria

    try:
        db_criteria = get_eval_criteria(user_id)
    except Exception:
        db_criteria = []

    criteria = db_criteria if db_criteria else DEFAULT_EVAL_CRITERIA

    system_prompt = _build_dynamic_system_prompt(criteria)
    prompt = _build_evaluation_prompt(plan, profile, beliefs, assessment)

    response = chat_completion(
        messages=[{"role": "user", "content": prompt}],
        system_prompt=system_prompt,
        temperature=0.2,
    )

    text = response.choices[0].message.content.strip()
    try:
        result = extract_json(text)
    except ValueError:
        logger.warning("evaluate_plan: could not parse evaluator response")
        return PlanEvaluation(
            score=50,
            criteria_scores={},
            issues=["Evaluation response was malformed — could not parse scores."],
            suggestions=["Try saving the plan as-is or regenerate."],
        )

    score = result.get("overall_score", 0)

    # Guard against all-zero scores (indicates evaluator couldn't parse the plan)
    all_criteria_zero = (
        result.get("criteria")
        and isinstance(result["criteria"], dict)
        and all(v == 0 for v in result["criteria"].values())
    )
    if score == 0 and all_criteria_zero:
        logger.warning("evaluate_plan: evaluator returned all-zero scores, likely structural issue")
        return PlanEvaluation(
            score=30,
            criteria_scores=result.get("criteria", {}),
            issues=[
                "Evaluator could not parse the plan structure (all scores = 0). "
                "Regenerate with the standard 'sessions' array format."
            ],
            suggestions=["Use create_training_plan with focus parameter to regenerate."],
        )

    return PlanEvaluation(
        score=score,
        criteria_scores=result.get("criteria", {}),
        issues=result.get("issues", []),
        suggestions=result.get("suggestions", []),
    )


def _unwrap_plan_for_eval(plan: dict) -> dict:
    """Unwrap nested plan wrappers before evaluation.

    Handles LLM drift patterns like {"result": "```json ...```"}.
    """
    # Unwrap {"result": "<json string>"} wrappers
    if list(plan.keys()) == ["result"] and isinstance(plan.get("result"), str):
        try:
            inner = extract_json(plan["result"])
            if isinstance(inner, dict):
                logger.info("evaluate_plan: unwrapped 'result' string wrapper")
                return _unwrap_plan_for_eval(inner)
        except (ValueError, TypeError):
            pass

    # Unwrap {"plan": {...}} wrappers
    for wrapper_key in ("plan", "training_plan", "weekly_plan"):
        inner = plan.get(wrapper_key)
        if isinstance(inner, dict) and any(
            k in inner for k in ("sessions", "days", "s")
        ):
            logger.info("evaluate_plan: unwrapped '%s' dict wrapper", wrapper_key)
            return inner

    return plan


def _build_dynamic_system_prompt(criteria: list[dict]) -> str:
    """Build an evaluation system prompt from agent-defined criteria."""
    # Normalize weights to percentages
    total_weight = sum(c.get("weight", 1.0) for c in criteria)
    if total_weight == 0:
        total_weight = 1.0

    criteria_lines = []
    criteria_names = []
    for c in criteria:
        name = c.get("name", "unnamed")
        desc = c.get("description", "No description")
        weight = c.get("weight", 1.0)
        pct = round((weight / total_weight) * 100)
        criteria_names.append(name)
        criteria_lines.append(f"- {name} ({pct}%): {desc}")

    criteria_json = ",\n        ".join(f'"{n}": 75' for n in criteria_names)

    return f"""\
You are evaluating a training plan for an athlete (any sport). Score each criterion 0-100.

Be STRICT -- a perfect plan is rare.

You MUST respond with ONLY a valid JSON object. No markdown, no explanation.

{{
    "overall_score": 72,
    "criteria": {{
        {criteria_json}
    }},
    "issues": [
        "Specific issue with the plan (reference session days/sports)"
    ],
    "suggestions": [
        "Specific improvement to make"
    ]
}}

Scoring guide (score each 0-100):
{chr(10).join(criteria_lines)}

The overall_score should be a weighted average of the individual criterion scores,
using the percentages shown above.
"""


def extract_sessions_from_plan(plan: dict) -> list[dict]:
    """Extract a flat list of sessions from any LLM-generated plan structure.

    LLMs drift from the prescribed schema over multiple iterations, producing
    varied top-level keys and nesting patterns.  This function handles all
    observed variants:

        1. plan["sessions"]          — canonical flat list
        2. plan["days"] (list)       — list of day objects, each with nested sessions
        3. plan["days"] (dict)       — day-name keyed dict with session lists
        4. plan["plan"] (list)       — same as days but under "plan" key
        5. plan["s"] (list)          — abbreviated sessions list
        6. Deeply nested combos      — e.g. plan["days"][i]["s"], plan["plan"][i]["sessions"]
        7. {"result": "json string"} — LLM wrapper with embedded JSON

    Returns a list of session dicts.  Each session is guaranteed to have at
    least "sport" (or "sp") and "type" (or "t"/"ty") keys so callers can
    format them for display.
    """
    # 0. Unwrap {"result": "<json string>"} wrappers first
    plan = _unwrap_plan_for_eval(plan)

    # 1. Canonical: flat "sessions" list
    sessions = plan.get("sessions", [])
    if sessions and isinstance(sessions, list) and isinstance(sessions[0], dict):
        return sessions

    # Helper: extract sessions from a list of day/date objects
    def _sessions_from_day_list(day_list: list) -> list[dict]:
        result = []
        for day_obj in day_list:
            if not isinstance(day_obj, dict):
                continue
            # Day object might itself be a session (has "sport"/"sp")
            if day_obj.get("sport") or day_obj.get("sp"):
                result.append(day_obj)
                continue
            # Look for nested session lists under various keys
            for key in ("sessions", "s"):
                nested = day_obj.get(key, [])
                if isinstance(nested, list):
                    for s in nested:
                        if isinstance(s, dict):
                            # Inherit date/day from parent if missing
                            enriched = {**s}
                            if not enriched.get("day") and not enriched.get("d"):
                                enriched.setdefault("day", day_obj.get("day", day_obj.get("d", day_obj.get("date", day_obj.get("dt", "?")))))
                            if not enriched.get("date") and not enriched.get("dt"):
                                enriched.setdefault("date", day_obj.get("date", day_obj.get("dt", "")))
                            result.append(enriched)
        return result

    # 2-4. Try "days", "plan", "s" as list of day objects
    for key in ("days", "plan", "s"):
        candidate = plan.get(key)
        if isinstance(candidate, list) and candidate:
            extracted = _sessions_from_day_list(candidate)
            if extracted:
                return extracted

    # 5. "days" as dict (day_name -> {sessions: [...]})
    days_dict = plan.get("days")
    if isinstance(days_dict, dict):
        result = []
        for day_name, day_data in days_dict.items():
            if isinstance(day_data, dict):
                for s in day_data.get("sessions", day_data.get("s", [])):
                    if isinstance(s, dict):
                        s_copy = {**s, "day": day_name}
                        result.append(s_copy)
        if result:
            return result

    # 6. Last resort: scan all list-valued top-level keys for session-like dicts
    for key, value in plan.items():
        if isinstance(value, list) and value and isinstance(value[0], dict):
            if value[0].get("sport") or value[0].get("sp") or value[0].get("type") or value[0].get("t"):
                return value

    logger.warning(
        "extract_sessions_from_plan found 0 sessions. Plan keys: %s",
        list(plan.keys()),
    )
    return []


def _format_session_line(s: dict) -> str:
    """Format a single session dict into a human-readable evaluation line.

    Handles both canonical and abbreviated key names.
    """
    day = s.get("day") or s.get("d") or s.get("date") or s.get("dt") or "?"
    sport = s.get("sport") or s.get("sp") or s.get("s") or "?"
    stype = s.get("type") or s.get("t") or s.get("ty") or "?"
    dur = (
        s.get("total_duration_minutes")
        or s.get("duration_minutes")
        or s.get("dur_min")
        or s.get("dur")
        or s.get("duration")
        or "?"
    )

    targets = ""
    steps = s.get("steps") or s.get("st") or []
    if steps and isinstance(steps, list):
        target_parts = []
        for step in steps:
            if isinstance(step, dict):
                tgt = step.get("targets") or step.get("tg") or step.get("trg")
                if tgt:
                    target_parts.append(str(tgt))
        if target_parts:
            targets = f" | targets: {'; '.join(target_parts[:2])}"
    elif s.get("targets"):
        targets = f" | targets: {s['targets']}"

    return f"  - {day}: {sport} {stype} ({dur}min){targets}"


def _build_evaluation_prompt(
    plan: dict,
    profile: dict,
    beliefs: list[dict] | None = None,
    assessment: dict | None = None,
) -> str:
    """Build the evaluation prompt with plan and athlete context."""
    goal = profile.get("goal", {})
    constraints = profile.get("constraints", {})

    # Extract sessions from any plan structure variant
    sessions = extract_sessions_from_plan(plan)
    session_lines = [_format_session_line(s) for s in sessions]

    # Format beliefs
    beliefs_text = ""
    if beliefs:
        scheduling = [b for b in beliefs if b.get("category") == "scheduling"]
        preference = [b for b in beliefs if b.get("category") == "preference"]
        if scheduling or preference:
            beliefs_lines = []
            for b in (scheduling + preference)[:6]:
                beliefs_lines.append(f"  - {b.get('text', '')}")
            beliefs_text = f"\nATHLETE PREFERENCES:\n" + "\n".join(beliefs_lines)

    assessment_text = ""
    if assessment:
        assess = assessment.get("assessment", {})
        assessment_text = (
            f"\nRECENT ASSESSMENT:\n"
            f"  Compliance: {assess.get('compliance', '?')}\n"
            f"  Fatigue: {assess.get('fatigue_level', '?')}\n"
            f"  Trend: {assess.get('fitness_trend', '?')}"
        )

    return f"""\
Evaluate this training plan:

ATHLETE PROFILE:
  Goal: {goal.get('event', 'General')} by {goal.get('target_date', 'N/A')}
  Target time: {goal.get('target_time', 'N/A')}
  Sports: {profile.get('sports', [])}
  Training days/week: {constraints.get('training_days_per_week', '?')}
  Max session: {constraints.get('max_session_minutes', '?')} min
{beliefs_text}
{assessment_text}
GENERATED PLAN ({len(sessions)} sessions):
{chr(10).join(session_lines) if session_lines else '  No sessions'}

Score each criterion 0-100 and provide specific issues and suggestions.
"""
