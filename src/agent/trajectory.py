"""Trajectory assessment: evaluates if current training leads to the goal."""

import json
from datetime import datetime, date

from src.agent.json_utils import extract_json
from src.agent.llm import chat_completion

TRAJECTORY_SYSTEM_PROMPT = """\
You are an expert endurance sports coach evaluating an athlete's long-term training trajectory.

Based on the athlete's profile, training history, and past reflections, assess whether they are on track to achieve their goal.

You MUST respond with ONLY a valid JSON object. No markdown, no explanation, no code fences.

The JSON must follow this exact structure:
{
    "current_fitness": {
        "estimated_vo2max": 48.5,
        "current_race_prediction": "1:52:00",
        "weekly_volume_km": 38,
        "trend": "improving|stable|declining"
    },
    "trajectory": {
        "on_track": true,
        "predicted_race_time": "1:43:00 - 1:48:00",
        "key_milestones": [
            {
                "date": "2026-04-01",
                "milestone": "Description of milestone",
                "status": "on_track|at_risk|achieved|too_early_to_assess"
            }
        ]
    },
    "recommendations": [
        "Specific, actionable recommendation based on actual data"
    ],
    "risks": [
        {
            "risk": "Description of risk",
            "probability": "low|medium|high",
            "mitigation": "How to address this risk"
        }
    ]
}

Rules:
- Be specific and data-driven. Reference actual numbers.
- Race predictions should be ranges, not precise times.
- Include 2-4 key milestones between now and race day.
- Recommendations should be actionable and based on observed patterns.
- Risks should consider overtraining, undertraining, and injury potential.
"""


def assess_trajectory(
    athlete_profile: dict,
    recent_activities: list[dict],
    episodes: list[dict],
    current_plan: dict,
) -> dict:
    """Assess whether current training trajectory leads to the goal.

    Sends all context to LLM and returns a trajectory assessment
    with confidence scoring.
    """
    prompt = _build_trajectory_prompt(athlete_profile, recent_activities, episodes, current_plan)

    response = chat_completion(
        messages=[{"role": "user", "content": prompt}],
        system_prompt=TRAJECTORY_SYSTEM_PROMPT,
        temperature=0.4,
    )

    text = response.choices[0].message.content.strip()
    result = extract_json(text)

    # Calculate and overlay our own confidence score
    weeks_of_data = len(episodes)
    total_activities = len(recent_activities)
    avg_compliance = _calculate_avg_compliance(episodes)

    confidence = calculate_confidence(
        data_points=total_activities,
        consistency=avg_compliance,
        weeks_of_data=weeks_of_data,
    )

    # Build complete trajectory result
    goal = athlete_profile.get("goal", {})
    target_date = goal.get("target_date")
    weeks_remaining = _weeks_until(target_date) if target_date else None

    trajectory = {
        "goal": {
            "event": goal.get("event", "Unknown"),
            "target_date": goal.get("target_date", "Unknown"),
            "target_time": goal.get("target_time", "Unknown"),
            "weeks_remaining": weeks_remaining,
        },
        "current_fitness": result.get("current_fitness", {}),
        "trajectory": result.get("trajectory", {}),
        "confidence": confidence,
        "confidence_explanation": _explain_confidence(confidence, weeks_of_data, avg_compliance),
        "recommendations": result.get("recommendations", []),
        "risks": result.get("risks", []),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    return trajectory


def calculate_confidence(
    data_points: int,
    consistency: float,
    weeks_of_data: int,
) -> float:
    """Calculate confidence in trajectory projection.

    Rules:
    - <4 weeks data: confidence capped at 0.5
    - <8 weeks data: confidence capped at 0.75
    - Inconsistent training (compliance <70%): confidence reduced by 0.2
    - More data points increase base confidence
    """
    # Base confidence from amount of data
    if weeks_of_data >= 12:
        base = 0.9
    elif weeks_of_data >= 8:
        base = 0.75
    elif weeks_of_data >= 4:
        base = 0.6
    elif weeks_of_data >= 2:
        base = 0.4
    else:
        base = 0.25

    # Data point bonus (more activities = slightly more confident)
    if data_points >= 20:
        base = min(base + 0.1, 1.0)

    # Consistency penalty
    if consistency < 0.7:
        base = max(base - 0.2, 0.1)

    # Cap by data duration
    if weeks_of_data < 4:
        base = min(base, 0.5)
    elif weeks_of_data < 8:
        base = min(base, 0.75)

    return round(base, 2)


def _calculate_avg_compliance(episodes: list[dict]) -> float:
    """Calculate average compliance rate across episodes."""
    if not episodes:
        return 0.0
    rates = [ep.get("compliance_rate", 0) for ep in episodes]
    return sum(rates) / len(rates)


def _weeks_until(target_date_str: str) -> int | None:
    """Calculate weeks until target date."""
    try:
        target = date.fromisoformat(target_date_str)
        today = date.today()
        delta = (target - today).days
        return max(0, delta // 7)
    except (ValueError, TypeError):
        return None


def _explain_confidence(confidence: float, weeks: int, consistency: float) -> str:
    """Generate a human-readable confidence explanation."""
    parts = []
    if weeks < 4:
        parts.append(f"Only {weeks} weeks of data available (need 4+ for reliable predictions)")
    elif weeks < 8:
        parts.append(f"{weeks} weeks of data (still building prediction reliability)")

    if consistency < 0.7:
        parts.append(f"Training consistency at {consistency:.0%} is below target (70%+)")

    if confidence >= 0.75:
        parts.append("Good data foundation for trajectory prediction")
    elif confidence >= 0.5:
        parts.append("Prediction range is moderate")
    else:
        parts.append("Not enough data for reliable prediction")

    return ". ".join(parts) + "."


def _build_trajectory_prompt(
    profile: dict,
    activities: list[dict],
    episodes: list[dict],
    plan: dict,
) -> str:
    """Build the prompt for trajectory assessment."""
    goal = profile.get("goal", {})
    target_date = goal.get("target_date", "N/A")
    weeks_remaining = _weeks_until(target_date) if target_date != "N/A" else "?"

    # Activity summary
    total_distance = sum((a.get("distance_meters", 0) or 0) / 1000 for a in activities)
    total_sessions = len(activities)
    avg_hr_values = [a["heart_rate"]["avg"] for a in activities if a.get("heart_rate", {}).get("avg")]
    avg_hr = round(sum(avg_hr_values) / len(avg_hr_values)) if avg_hr_values else "N/A"

    # Episode lessons
    all_lessons = []
    for ep in episodes:
        for lesson in ep.get("lessons", []):
            all_lessons.append(f"  - {lesson}")

    all_patterns = []
    for ep in episodes:
        for pattern in ep.get("patterns_detected", []):
            all_patterns.append(f"  - {pattern}")

    return f"""\
Assess the long-term training trajectory for this athlete:

GOAL: {goal.get('event', 'Unknown')} by {target_date}
TARGET TIME: {goal.get('target_time', 'N/A')}
WEEKS REMAINING: {weeks_remaining}

TRAINING HISTORY:
- Total sessions analyzed: {total_sessions}
- Total distance: {total_distance:.1f}km
- Average HR across sessions: {avg_hr}
- Weeks of data: {len(episodes)}

CURRENT PLAN:
- Sessions/week: {len(plan.get('sessions', []))}
- Focus: {plan.get('weekly_summary', {}).get('focus', 'N/A')}

LESSONS FROM PAST REFLECTIONS:
{chr(10).join(all_lessons) if all_lessons else '  No past lessons available'}

PATTERNS DETECTED:
{chr(10).join(all_patterns) if all_patterns else '  No patterns detected yet'}

Based on all this data, assess whether the athlete is on track to achieve their goal.
Provide a race time prediction range, milestones, recommendations, and risks.
"""
