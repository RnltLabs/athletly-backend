"""Episodic memory: generate, store, and retrieve training reflections."""

import json
import re
from datetime import datetime
from pathlib import Path

from src.agent.json_utils import extract_json
from src.agent.llm import chat_completion

DATA_DIR = Path(__file__).parent.parent.parent / "data"
EPISODES_DIR = DATA_DIR / "episodes"

REFLECTION_SYSTEM_PROMPT = """\
You are an expert endurance sports coach reflecting on a completed training block.

Analyze the prescribed plan, actual activities, and assessment to generate a structured reflection.
Focus on actionable lessons, patterns, and specific observations grounded in data.

You MUST respond with ONLY a valid JSON object. No markdown, no explanation, no code fences.

The JSON must follow this exact structure:
{
    "key_observations": [
        "Specific data-driven observation (reference actual HR, pace, duration numbers)"
    ],
    "lessons": [
        "Actionable lesson that should inform future planning"
    ],
    "patterns_detected": [
        "Recurring pattern observed across sessions or weeks"
    ],
    "fitness_delta": {
        "estimated_vo2max_change": "+0.5 or -0.3 or stable",
        "threshold_pace_change": "-0:05/km or +0:03/km or stable",
        "weekly_volume_trend": "increasing|stable|decreasing"
    },
    "confidence": 0.8
}

Rules:
- key_observations must reference actual numbers from the data (HR values, pace, distances)
- lessons must be actionable (e.g., "move Thursday interval to Wednesday" not "train smarter")
- patterns_detected should identify recurring behaviors across the training block
- confidence is 0.0-1.0 based on how much data was available
- Be specific, not generic. Reference actual session data.
"""


def generate_reflection(
    plan: dict,
    activities: list[dict],
    assessment: dict,
    athlete_profile: dict,
    conversation_context: str | None = None,
    beliefs: list[dict] | None = None,
) -> dict:
    """Generate a structured reflection for a completed training block.

    Sends plan + activities + assessment to LLM.
    Returns a complete episode dict ready for storage.

    Args:
        plan: The training plan that was followed.
        activities: Actual activities performed.
        assessment: Assessment results.
        athlete_profile: Athlete profile dict.
        conversation_context: Optional text from recent conversations providing
                              subjective context about the training period.
        beliefs: Optional list of active beliefs to inject as coach's notes.
    """
    prompt = _build_reflection_prompt(
        plan, activities, assessment, athlete_profile,
        conversation_context=conversation_context,
        beliefs=beliefs,
    )

    response = chat_completion(
        messages=[{"role": "user", "content": prompt}],
        system_prompt=REFLECTION_SYSTEM_PROMPT,
        temperature=0.4,
    )

    text = response.choices[0].message.content.strip()
    reflection = extract_json(text)

    # Build complete episode
    week_start = plan.get("week_start", "unknown")
    week_num = plan.get("week_number", "?")

    total_prescribed_km = 0
    for s in plan.get("sessions", []):
        # Rough estimate: duration * typical pace
        dur = s.get("duration_minutes", 0)
        total_prescribed_km += dur / 6  # ~6 min/km average

    total_actual_km = sum(
        (a.get("distance_meters", 0) or 0) / 1000 for a in activities
    )

    # Generate a unique episode ID
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    ep_id = f"ep_{week_start}" if week_start != "unknown" else f"ep_{ts}"

    episode = {
        "id": ep_id,
        "block": f"2026-W{week_num:02d}" if isinstance(week_num, int) else f"week-{week_num}",
        "period": week_start,
        "prescribed_sessions": len(plan.get("sessions", [])),
        "actual_sessions": len(activities),
        "prescribed_volume_km": round(total_prescribed_km, 1),
        "actual_volume_km": round(total_actual_km, 1),
        "compliance_rate": assessment.get("assessment", {}).get("compliance", 0),
        "key_observations": reflection.get("key_observations", []),
        "lessons": reflection.get("lessons", []),
        "patterns_detected": reflection.get("patterns_detected", []),
        "fitness_delta": reflection.get("fitness_delta", {}),
        "confidence": reflection.get("confidence", 0.5),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    return episode


def store_episode(episode: dict, storage_dir: str | Path | None = None) -> Path:
    """Store a reflection episode as JSON. Returns the file path."""
    dest = Path(storage_dir) if storage_dir else EPISODES_DIR
    dest.mkdir(parents=True, exist_ok=True)

    ep_id = episode.get("id", f"ep_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    filename = f"{ep_id}.json"
    path = dest / filename

    # Avoid overwriting existing episodes -- append timestamp suffix
    if path.exists():
        ts = datetime.now().strftime("%H%M%S_%f")
        filename = f"{ep_id}_{ts}.json"
        path = dest / filename

    path.write_text(json.dumps(episode, indent=2))
    return path


def list_episodes(
    storage_dir: str | Path | None = None,
    limit: int = 10,
) -> list[dict]:
    """List stored episodes, most recent first."""
    src = Path(storage_dir) if storage_dir else EPISODES_DIR
    if not src.exists():
        return []

    episodes = []
    for path in sorted(src.glob("ep_*.json"), reverse=True):
        episodes.append(json.loads(path.read_text()))
        if len(episodes) >= limit:
            break

    return episodes


def retrieve_relevant_episodes(
    current_context: dict,
    episodes: list[dict],
    max_results: int = 5,
) -> list[dict]:
    """Retrieve episodes relevant to current planning context.

    Uses keyword matching (0.4) + recency (0.3) + utility (0.3) scoring.
    Episodes with proven lessons (high utility) are preferred over
    recent-but-unproven episodes (P6: active memory).
    """
    if not episodes:
        return []

    # Extract keywords from context
    context_keywords = _extract_keywords(current_context)

    scored = []
    for i, ep in enumerate(episodes):
        # Keyword matching across observations, lessons, patterns
        ep_text = " ".join(
            ep.get("key_observations", [])
            + ep.get("lessons", [])
            + ep.get("patterns_detected", [])
        ).lower()

        keyword_score = 0.0
        for kw in context_keywords:
            if kw in ep_text:
                keyword_score += 1.0
        # Normalize keyword score to 0-1 range
        max_kw = max(len(context_keywords), 1)
        keyword_score = min(1.0, keyword_score / max_kw)

        # Recency: more recent episodes score higher (already sorted most-recent-first)
        recency_score = max(0, (len(episodes) - i)) / len(episodes)

        # Utility: episodes with proven lessons score higher (P6)
        utility_score = ep.get("utility", 0.0)

        # Weighted combination
        score = keyword_score * 0.4 + recency_score * 0.3 + utility_score * 0.3

        scored.append((score, ep))

    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    return [ep for _, ep in scored[:max_results]]


def record_episode_outcome(
    episode_id: str,
    utility_delta: float,
    storage_dir: str | Path | None = None,
) -> dict | None:
    """Update an episode's utility score based on coaching outcome.

    Args:
        episode_id: The episode ID to update.
        utility_delta: Amount to add to utility (positive = lesson was useful).
        storage_dir: Where episodes are stored.

    Returns:
        Updated episode dict, or None if not found.
    """
    src = Path(storage_dir) if storage_dir else EPISODES_DIR
    if not src.exists():
        return None

    for path in src.glob("ep_*.json"):
        try:
            ep = json.loads(path.read_text())
        except (json.JSONDecodeError, ValueError):
            continue

        if ep.get("id") == episode_id:
            ep.setdefault("utility", 0.0)
            ep.setdefault("referenced_count", 0)
            ep.setdefault("outcome_validated", False)

            ep["utility"] = max(0.0, min(1.0, ep["utility"] + utility_delta))
            ep["referenced_count"] += 1
            if utility_delta > 0:
                ep["outcome_validated"] = True

            path.write_text(json.dumps(ep, indent=2))
            return ep

    return None


META_REFLECTION_PROMPT = """\
You are an expert coach reflecting on your own coaching effectiveness.
Given the training block reflection below, evaluate your coaching decisions and generate meta-insights.

You MUST respond with ONLY a valid JSON object. No markdown, no explanation, no code fences.

The JSON must follow this exact structure:
{
    "meta_beliefs": [
        {
            "text": "A specific coaching insight (e.g., 'This athlete responds better to longer easy runs than frequent short ones')",
            "category": "meta",
            "confidence": 0.7,
            "reasoning": "Why this insight matters for future coaching"
        }
    ]
}

Rules:
- Focus on actionable coaching insights, not generic observations
- Each meta-belief should inform future plan generation or conversation style
- Examples of good meta-beliefs:
  - "Athlete runs easy sessions too fast -- need to emphasize zone discipline in plans"
  - "Motivation increases when sessions include race-specific elements"
  - "Plan compliance drops on Thursdays -- consider lighter sessions mid-week"
  - "Athlete prefers direct feedback over gentle suggestions"
- confidence: 0.0-1.0 based on how much supporting data exists
- Empty array if no coaching insights are warranted
"""


def extract_meta_beliefs(episode: dict) -> list[dict]:
    """Extract meta-beliefs about coaching effectiveness from a reflection episode.

    Meta-beliefs capture the agent's self-evaluation of its coaching decisions.
    They are stored as regular beliefs with category="meta" and influence
    future behavior through the PrefEval injection pattern.

    Args:
        episode: A completed reflection episode dict.

    Returns:
        List of meta-belief dicts with text, category, confidence, reasoning.
    """
    # Build context from the episode
    observations = episode.get("key_observations", [])
    lessons = episode.get("lessons", [])
    patterns = episode.get("patterns_detected", [])
    compliance = episode.get("compliance_rate", "?")

    prompt = f"""\
TRAINING BLOCK: {episode.get('block', 'unknown')}
COMPLIANCE: {compliance}

OBSERVATIONS:
{chr(10).join(f'  - {o}' for o in observations) if observations else '  None'}

LESSONS LEARNED:
{chr(10).join(f'  - {l}' for l in lessons) if lessons else '  None'}

PATTERNS DETECTED:
{chr(10).join(f'  - {p}' for p in patterns) if patterns else '  None'}

Based on this reflection, what coaching insights should I remember for future plans and conversations with this athlete?
"""

    response = chat_completion(
        messages=[{"role": "user", "content": prompt}],
        system_prompt=META_REFLECTION_PROMPT,
        temperature=0.4,
    )

    try:
        result = extract_json(response.choices[0].message.content)
        return result.get("meta_beliefs", [])
    except (ValueError, AttributeError):
        return []


def _extract_keywords(context: dict) -> list[str]:
    """Extract search keywords from a planning context."""
    keywords = []

    # From goal
    goal = context.get("goal", {})
    if goal.get("event"):
        keywords.extend(goal["event"].lower().split())

    # From sports
    for sport in context.get("sports", []):
        keywords.append(sport.lower())

    # From fitness
    fitness = context.get("fitness", {})
    if fitness.get("trend"):
        keywords.append(fitness["trend"].lower())

    # Common training keywords
    keywords.extend(["pace", "hr", "fatigue", "volume", "easy", "interval", "long run", "threshold"])

    return keywords


def _build_reflection_prompt(
    plan: dict,
    activities: list[dict],
    assessment: dict,
    profile: dict,
    conversation_context: str | None = None,
    beliefs: list[dict] | None = None,
) -> str:
    """Build the prompt for reflection generation."""
    goal = profile.get("goal", {})
    assess = assessment.get("assessment", {})

    # Prescribed sessions
    prescribed = plan.get("sessions", [])
    prescribed_text = []
    for s in prescribed:
        prescribed_text.append(
            f"  - {s.get('day', '?')}: {s.get('type', '?')} "
            f"({s.get('duration_minutes', '?')}min, {s.get('target_hr_zone', '?')})"
        )

    # Actual activities
    actual_text = []
    for a in activities:
        hr = a.get("heart_rate", {})
        dur_min = round(a["duration_seconds"] / 60) if a.get("duration_seconds") else "?"
        dist_km = round(a["distance_meters"] / 1000, 1) if a.get("distance_meters") else "N/A"
        avg_hr = hr.get("avg", "N/A") if hr else "N/A"

        actual_text.append(
            f"  - {a.get('start_time', '?')}: {a.get('sport', '?')} | "
            f"{dur_min}min | {dist_km}km | HR {avg_hr}"
        )

    # Assessment summary
    obs = assess.get("observations", [])
    obs_text = "\n".join(f"  - {o}" for o in obs) if obs else "  None"

    # Conversation context and beliefs
    subjective_section = ""
    if conversation_context:
        subjective_section += (
            f"\nATHLETE'S SELF-REPORTED CONTEXT FROM CONVERSATIONS:\n"
            f"{conversation_context}\n"
        )
    if beliefs:
        from src.agent.prompts import _format_beliefs_section
        beliefs_text = _format_beliefs_section(beliefs)
        if beliefs_text:
            subjective_section += beliefs_text

    return f"""\
Reflect on this completed training block:

GOAL: {goal.get('event', 'General')} by {goal.get('target_date', 'N/A')}

PRESCRIBED PLAN ({len(prescribed)} sessions):
{chr(10).join(prescribed_text)}

ACTUAL TRAINING ({len(activities)} activities):
{chr(10).join(actual_text)}

ASSESSMENT:
- Compliance: {assess.get('compliance', 'N/A')}
- Fitness trend: {assess.get('fitness_trend', 'N/A')}
- Fatigue: {assess.get('fatigue_level', 'N/A')}
- Observations:
{obs_text}
{subjective_section}
Generate specific, data-driven observations, actionable lessons, and patterns detected.
"""
