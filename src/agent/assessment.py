"""Training assessment: compares actual training vs prescribed plan using LiteLLM."""

import json
import re

from src.agent.json_utils import extract_json
from src.agent.llm import chat_completion

ASSESSMENT_SYSTEM_PROMPT = """\
You are an expert endurance sports coach analyzing an athlete's training data.

Compare the athlete's actual training activities against their prescribed plan.
Identify compliance, patterns, and areas for adjustment.

You MUST respond with ONLY a valid JSON object. No markdown, no explanation, no code fences.

The JSON must follow this exact structure:
{
    "assessment": {
        "compliance": 0.85,
        "observations": [
            "Specific observation about training pattern or deviation"
        ],
        "fitness_trend": "improving|stable|declining",
        "fatigue_level": "low|moderate|high",
        "injury_risk": "low|moderate|high"
    },
    "recommended_adjustments": [
        {
            "type": "pace_target|volume|session_type|rest|intensity|other",
            "description": "What to change and why",
            "impact": "low|medium|high",
            "autonomous": true
        }
    ]
}

Rules:
- compliance is a float 0.0-1.0 representing % of prescribed sessions completed
- observations should be specific, data-driven (reference actual HR, pace, duration)
- fitness_trend: base on HR/pace trends across sessions
- fatigue_level: base on HR drift, missed sessions, declining performance
- injury_risk: base on sudden volume increases, consistent overexertion
- Each adjustment must have a clear impact level:
  - low: single session target tweak (pace, HR zone)
  - medium: volume change >15%, adding/removing rest day
  - high: restructuring plan, flagging injury risk
- autonomous: true for low impact, false for medium/high
"""


def assess_training(
    profile: dict,
    plan: dict,
    activities: list[dict],
    conversation_context: str | None = None,
    beliefs: list[dict] | None = None,
) -> dict:
    """Send plan + actual activities to LLM for structured assessment.

    Args:
        profile: Athlete profile dict.
        plan: Current training plan dict.
        activities: List of activity dicts.
        conversation_context: Optional text from recent conversations (e.g. "I've been
                              sleeping badly") to provide subjective context beyond FIT data.
        beliefs: Optional list of active beliefs to inject as coach's notes.

    Returns a dict with assessment and recommended_adjustments.
    """
    prompt = _build_assessment_prompt(
        profile, plan, activities,
        conversation_context=conversation_context,
        beliefs=beliefs,
    )

    response = chat_completion(
        messages=[{"role": "user", "content": prompt}],
        system_prompt=ASSESSMENT_SYSTEM_PROMPT,
        temperature=0.4,
    )

    text = response.choices[0].message.content.strip()
    return extract_json(text)


def _format_conversation_context(
    conversation_context: str | None, beliefs: list[dict] | None
) -> str:
    """Format conversation context and beliefs for the assessment prompt."""
    sections = []

    if conversation_context:
        sections.append(
            f"ATHLETE'S RECENT SELF-REPORTED CONTEXT:\n{conversation_context}"
        )

    if beliefs:
        from src.agent.prompts import _format_beliefs_section
        beliefs_text = _format_beliefs_section(beliefs)
        if beliefs_text:
            sections.append(beliefs_text.strip())

    return "\n\n".join(sections) + "\n" if sections else ""


def _build_assessment_prompt(
    profile: dict,
    plan: dict,
    activities: list[dict],
    conversation_context: str | None = None,
    beliefs: list[dict] | None = None,
) -> str:
    """Build the user prompt for training assessment."""
    goal = profile.get("goal", {})
    fitness = profile.get("fitness", {})

    # Summarize prescribed sessions
    prescribed = plan.get("sessions", [])
    prescribed_summary = []
    for s in prescribed:
        prescribed_summary.append(
            f"  - {s.get('day', '?')}: {s.get('sport', '?')} {s.get('type', '?')} "
            f"({s.get('duration_minutes', '?')}min)"
        )

    # Summarize actual activities
    actual_summary = []
    for a in activities:
        hr = a.get("heart_rate", {})
        dur_min = round(a["duration_seconds"] / 60) if a.get("duration_seconds") else "?"
        dist_km = round(a["distance_meters"] / 1000, 1) if a.get("distance_meters") else "N/A"
        avg_hr = hr.get("avg", "N/A") if hr else "N/A"
        pace = a.get("pace", {})
        pace_str = f"{pace.get('avg_min_per_km', 'N/A')} min/km" if pace else "N/A"
        trimp = a.get("trimp", "N/A")

        actual_summary.append(
            f"  - {a.get('start_time', '?')}: {a.get('sport', '?')} | "
            f"{dur_min}min | {dist_km}km | HR {avg_hr} | Pace {pace_str} | TRIMP {trimp}"
        )

    subjective_section = _format_conversation_context(conversation_context, beliefs)

    return f"""\
Assess this athlete's training week:

ATHLETE GOAL: {goal.get('event', 'General')} by {goal.get('target_date', 'N/A')}
TARGET TIME: {goal.get('target_time', 'N/A')}

PRESCRIBED PLAN ({len(prescribed)} sessions):
{chr(10).join(prescribed_summary) if prescribed_summary else '  No plan available'}

ACTUAL TRAINING ({len(activities)} activities):
{chr(10).join(actual_summary) if actual_summary else '  No activities recorded'}

{subjective_section}Analyze compliance, identify patterns, assess fitness trend and fatigue, and recommend adjustments.
"""
