"""Plan Evaluator: LLM-based scoring of generated training plans.

Priority 4 -- Audit finding #5 ("Kein Evaluator-Optimizer Loop"):
    v1.0 generates plans in a single shot with no feedback. Every LLM call
    is fire-and-forget. The evaluator-optimizer pattern (Anthropic's "Building
    Effective Agents") adds a feedback loop: generate -> evaluate -> accept or
    regenerate with evaluation feedback.

    This module implements the evaluator half. The cognitive loop in
    state_machine.py orchestrates the generate -> evaluate cycle by selecting
    the evaluate_plan action after generate_plan.

Evaluation criteria (scored 0-100):
    - Sport Distribution (25%): sessions match athlete's sport preferences
    - Target Specificity (20%): each session has pace/HR/power targets
    - Constraint Compliance (20%): respects training days and duration limits
    - Volume Progression (15%): appropriate for training phase
    - Session Variety (10%): mix of session types (easy, tempo, intervals, long)
    - Recovery Balance (10%): adequate rest between hard sessions
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


EVALUATION_SYSTEM_PROMPT = """\
You are evaluating a training plan for an athlete (any sport). Score each criterion 0-100.

Be STRICT -- a perfect plan is rare. Common issues to penalize:
- Sessions without specific pace/HR/power targets (generic "easy run" with no zones)
- Sport distribution that doesn't match athlete preferences
- Too many hard sessions back-to-back without recovery
- Session durations exceeding the athlete's stated max
- Missing variety (all sessions same type)

You MUST respond with ONLY a valid JSON object. No markdown, no explanation.

{
    "overall_score": 72,
    "criteria": {
        "sport_distribution": 90,
        "target_specificity": 65,
        "constraint_compliance": 80,
        "volume_progression": 70,
        "session_variety": 75,
        "recovery_balance": 60
    },
    "issues": [
        "Specific issue with the plan (reference session days/sports)"
    ],
    "suggestions": [
        "Specific improvement to make"
    ]
}

Scoring guide:
- sport_distribution (25%): Do session sport counts match athlete's preferences?
  90-100: exact match. 50-89: close but off by 1. <50: major mismatch.
- target_specificity (20%): Does each session have concrete pace/HR/power targets?
  90-100: all sessions have targets. 50-89: most have targets. <50: many generic.
- constraint_compliance (20%): Respects max_session_minutes and training_days_per_week?
  90-100: all within limits. 50-89: minor violations. <50: major violations.
- volume_progression (15%): Appropriate weekly volume for athlete's level?
  90-100: well-matched. 50-89: slightly off. <50: way too much or too little.
- session_variety (10%): Mix of session types (easy, tempo, intervals, long)?
  90-100: good variety. 50-89: limited. <50: all same type.
- recovery_balance (10%): Adequate rest between hard sessions?
  90-100: well-spaced. 50-89: some back-to-back. <50: hard sessions stacked.
"""


def evaluate_plan(
    plan: dict,
    profile: dict,
    beliefs: list[dict] | None = None,
    assessment: dict | None = None,
) -> PlanEvaluation:
    """Score a generated training plan against quality criteria.

    Uses a separate LLM call (not self-evaluation) to score the plan.
    Low temperature (0.2) for consistent scoring.

    Args:
        plan: The generated training plan dict
        profile: Athlete profile dict
        beliefs: Optional active beliefs for preference checking
        assessment: Optional recent assessment for context

    Returns:
        PlanEvaluation with score, criteria, issues, and suggestions.
    """
    prompt = _build_evaluation_prompt(plan, profile, beliefs, assessment)

    response = chat_completion(
        messages=[{"role": "user", "content": prompt}],
        system_prompt=EVALUATION_SYSTEM_PROMPT,
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
