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
If no criteria are defined, the plan is accepted by default (score=100).
"""

import json
from dataclasses import dataclass, field

from src.agent.json_utils import extract_json
from src.agent.llm import chat_completion


# Threshold for accepting a plan without regeneration
PLAN_ACCEPTANCE_THRESHOLD = 70

# Maximum regeneration attempts before accepting best available
MAX_PLAN_ITERATIONS = 3


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
    """Score a plan using agent-defined eval criteria from the DB.

    When no eval_criteria are defined for the user, the plan is accepted
    by default (score=100, no issues or suggestions).

    Args:
        plan: The generated training plan dict.
        profile: Athlete profile dict.
        user_id: The user's ID for fetching eval criteria from DB.
        beliefs: Optional active beliefs for preference checking.
        assessment: Optional recent assessment for context.

    Returns:
        PlanEvaluation with score, criteria, issues, and suggestions.
    """
    from src.db.agent_config_db import get_eval_criteria

    try:
        db_criteria = get_eval_criteria(user_id)
    except Exception:
        db_criteria = []

    if not db_criteria:
        return PlanEvaluation(
            score=100,
            criteria_scores={},
            issues=[],
            suggestions=[],
            acceptable=True,
        )

    system_prompt = _build_dynamic_system_prompt(db_criteria)
    prompt = _build_evaluation_prompt(plan, profile, beliefs, assessment)

    response = chat_completion(
        messages=[{"role": "user", "content": prompt}],
        system_prompt=system_prompt,
        temperature=0.2,
    )

    text = response.choices[0].message.content.strip()
    result = extract_json(text)

    return PlanEvaluation(
        score=result.get("overall_score", 0),
        criteria_scores=result.get("criteria", {}),
        issues=result.get("issues", []),
        suggestions=result.get("suggestions", []),
    )


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


def _build_evaluation_prompt(
    plan: dict,
    profile: dict,
    beliefs: list[dict] | None = None,
    assessment: dict | None = None,
) -> str:
    """Build the evaluation prompt with plan and athlete context."""
    goal = profile.get("goal", {})
    constraints = profile.get("constraints", {})

    # Format plan sessions concisely
    sessions = plan.get("sessions", [])
    session_lines = []
    for s in sessions:
        targets = ""
        if s.get("steps"):
            target_parts = []
            for step in s["steps"]:
                if isinstance(step, dict) and step.get("targets"):
                    target_parts.append(str(step["targets"]))
            if target_parts:
                targets = f" | targets: {'; '.join(target_parts[:2])}"
        elif s.get("targets"):
            targets = f" | targets: {s['targets']}"

        dur = s.get("total_duration_minutes") or s.get("duration_minutes", "?")
        session_lines.append(
            f"  - {s.get('day', '?')}: {s.get('sport', '?')} {s.get('type', '?')} "
            f"({dur}min){targets}"
        )

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
