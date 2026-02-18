"""Startup coaching pipeline: goal type inference, assessment, and greeting composition.

Computes deterministic assessment data for the startup greeting. All arithmetic
is done here so the LLM never computes numbers -- it only interprets and coaches.
The greeting LLM call synthesizes the computed data into a coherent coaching message.

P7 enhancement: Goal type inference uses LLM instead of keyword matching,
with keyword fallback for reliability.

Public API:
    infer_goal_type(goal) -> str
    compute_startup_assessment(plan, activities, goal) -> dict
    compose_startup_greeting(user_model, plan, imported) -> str
"""

import logging
from datetime import datetime, timedelta, timezone

from src.agent.json_utils import extract_json
from src.agent.llm import chat_completion

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Goal type classification
# ---------------------------------------------------------------------------

GOAL_TYPES = {"race_target", "performance_target", "routine", "general"}

GOAL_TYPE_PROMPT = """\
Classify this athlete's goal into exactly ONE category.

Categories:
- race_target: Training for a specific race with a target time and date (e.g., "Berlin Marathon in 3:30 on 2026-09-27")
- performance_target: Working toward a measurable performance goal without a specific race date (e.g., "Run a sub-20 5K", "Improve FTP to 300W")
- routine: Maintaining regular training without a specific performance target (e.g., "Run 3 times per week", "Stay consistent")
- general: Everything else, vague or no specific goal

Goal event: "{event}"
Target time: {target_time}
Target date: {target_date}

You MUST respond with ONLY a valid JSON object:
{{"goal_type": "race_target|performance_target|routine|general"}}
"""


def infer_goal_type(goal: dict, use_llm: bool = True) -> str:
    """Infer goal type from structured_core.goal fields.

    Uses LLM-based inference (P7) with keyword fallback.

    Args:
        goal: Dict with event, target_time, target_date fields.
        use_llm: Whether to attempt LLM inference (disable for testing fallback).
    """
    event = (goal.get("event") or "").lower()

    # Primary: LLM-based inference (P7)
    if use_llm and event:
        result = _infer_goal_type_llm(goal)
        if result in GOAL_TYPES:
            return result

    # Fallback: keyword heuristic
    return _infer_goal_type_keywords(goal)


def _infer_goal_type_llm(goal: dict) -> str | None:
    """Infer goal type using LLM. Returns None on failure."""
    try:
        prompt = GOAL_TYPE_PROMPT.format(
            event=goal.get("event", "none"),
            target_time=goal.get("target_time", "none"),
            target_date=goal.get("target_date", "none"),
        )

        response = chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )

        result = extract_json(response.choices[0].message.content.strip())
        goal_type = result.get("goal_type", "")
        if goal_type in GOAL_TYPES:
            return goal_type
    except Exception:
        pass

    return None


def _infer_goal_type_keywords(goal: dict) -> str:
    """Fallback: infer goal type using keyword matching."""
    event = (goal.get("event") or "").lower()
    has_target_time = bool(goal.get("target_time"))
    has_target_date = bool(goal.get("target_date"))

    if has_target_time and has_target_date and event:
        return "race_target"

    performance_words = {
        "marathon", "half", "10k", "5k", "triathlon", "ironman",
        "century", "gran fondo", "time trial", "ftp", "improve",
    }
    if event and any(w in event for w in performance_words):
        return "performance_target"

    routine_words = {
        "per week", "x run", "x bike", "x swim", "times a week",
        "consistency", "routine", "regular", "maintain",
    }
    if event and any(w in event for w in routine_words):
        return "routine"

    return "general"


# ---------------------------------------------------------------------------
# Startup assessment computation
# ---------------------------------------------------------------------------


def compute_startup_assessment(
    plan: dict | None,
    activities: list[dict],
    goal: dict,
) -> dict:
    """Compute the full startup assessment for the greeting pipeline.

    Aggregates session matching, weekly compliance, intensity summary, trends,
    goal type, goal summary, and triggers into one dict. All arithmetic is done
    here so the LLM never computes numbers.

    Args:
        plan: Current training plan dict (may be None).
        activities: Full list of activity dicts from activity store.
        goal: structured_core.goal dict from user model.

    Returns:
        Assessment dict matching the StartupAssessment structure.
    """
    now = datetime.now(timezone.utc)
    cutoff_14 = (now - timedelta(days=14)).isoformat()
    recent_activities = [
        a for a in activities
        if a.get("start_time", "") >= cutoff_14
    ]

    # Goal type
    goal_type = goal.get("goal_type") or infer_goal_type(goal)

    # Goal summary
    goal_summary = _build_goal_summary(goal, goal_type)

    # Session matching and weekly compliance
    session_matching = None
    weekly_compliance = None
    intensity_summary = None

    if plan and plan.get("sessions"):
        # Lazy import following established pattern from 02-02
        from src.tools.activity_context import match_plan_sessions

        match_result = match_plan_sessions(plan, recent_activities)
        session_matching = match_result

        # Weekly compliance
        volume_actual_min = round(
            sum(a.get("duration_seconds", 0) for a in recent_activities) / 60
        )
        volume_planned_min = round(
            sum(s.get("duration_minutes", 0) for s in plan.get("sessions", []))
        )
        volume_delta_pct = (
            round(((volume_actual_min - volume_planned_min) / volume_planned_min) * 100, 1)
            if volume_planned_min > 0
            else 0
        )

        weekly_compliance = {
            "compliance_rate": match_result["compliance_rate"],
            "matched_count": match_result["matched_count"],
            "planned_count": match_result["planned_count"],
            "volume_actual_min": volume_actual_min,
            "volume_planned_min": volume_planned_min,
            "volume_delta_pct": volume_delta_pct,
        }

        # Intensity summary from matched sessions
        intensity_summary = _compute_intensity_summary(match_result["matched"])

    # Trends
    trends = _compute_trends(recent_activities)

    # Triggers
    triggers = _detect_triggers(
        weekly_compliance=weekly_compliance,
        trends=trends,
        recent_activities=recent_activities,
    )

    return {
        "goal_type": goal_type,
        "goal_summary": goal_summary,
        "session_matching": session_matching,
        "weekly_compliance": weekly_compliance,
        "intensity_summary": intensity_summary,
        "trends": trends,
        "triggers": triggers,
    }


def _build_goal_summary(goal: dict, goal_type: str) -> str:
    """Build a human-readable one-line goal summary."""
    event = goal.get("event") or "General fitness"
    target_date = goal.get("target_date")
    target_time = goal.get("target_time")

    parts = [event]

    if target_time:
        parts.append(f"under {target_time}")

    if target_date:
        parts.append(f"by {target_date}")
        try:
            target_dt = datetime.fromisoformat(target_date)
            now = datetime.now()
            # Handle date-only strings (no timezone)
            weeks_remaining = max(0, (target_dt - now).days // 7)
            parts.append(f"({weeks_remaining} weeks remaining)")
        except ValueError:
            pass

    if not target_time and not target_date:
        parts.append(f"({goal_type} goal)")

    return " ".join(parts)


def _compute_intensity_summary(matched: list[dict]) -> dict:
    """Compute intensity match summary from matched session pairs."""
    on_target = 0
    lower = 0
    higher = 0

    for m in matched:
        match_val = m.get("intensity_match", "unknown")
        if match_val == "on_target":
            on_target += 1
        elif match_val == "lower_than_planned":
            lower += 1
        elif match_val == "higher_than_planned":
            higher += 1

    total_known = on_target + lower + higher

    if total_known == 0:
        overall = "unknown"
    elif on_target >= total_known / 2:
        overall = "on_target"
    elif higher > lower:
        overall = "slightly_harder_than_planned"
    else:
        overall = "slightly_easier_than_planned"

    return {
        "on_target": on_target,
        "lower_than_planned": lower,
        "higher_than_planned": higher,
        "match": overall,
    }


def _compute_trends(recent_activities: list[dict]) -> dict:
    """Compute volume and TRIMP trends from recent activities."""
    # Lazy import following established pattern
    from src.tools.activity_context import compute_weekly_trends

    weeks = compute_weekly_trends(recent_activities)

    if not weeks:
        return {
            "volume_direction": "no data",
            "trimp_direction": "no data",
            "sessions_per_week": [],
        }

    sessions_per_week = [w["sessions"] for w in weeks]

    # Compute trend direction using first-half vs second-half comparison
    if len(weeks) >= 2:
        half = len(weeks) // 2
        first_half = weeks[:half] if half > 0 else weeks[:1]
        second_half = weeks[half:] if half > 0 else weeks[1:]

        def avg_metric(wks: list[dict], key: str) -> float:
            vals = [w[key] for w in wks]
            return sum(vals) / len(vals) if vals else 0

        vol_first = avg_metric(first_half, "duration_min")
        vol_second = avg_metric(second_half, "duration_min")
        trimp_first = avg_metric(first_half, "trimp")
        trimp_second = avg_metric(second_half, "trimp")

        def trend_word(first: float, second: float) -> str:
            if first == 0:
                return "increasing" if second > 0 else "stable"
            change_pct = ((second - first) / first) * 100
            if change_pct >= 15:
                return f"increasing (+{change_pct:.0f}%)"
            elif change_pct <= -15:
                return f"decreasing ({change_pct:.0f}%)"
            else:
                return "stable"

        volume_direction = trend_word(vol_first, vol_second)
        trimp_direction = trend_word(trimp_first, trimp_second)
    else:
        volume_direction = "stable"
        trimp_direction = "stable"

    return {
        "volume_direction": volume_direction,
        "trimp_direction": trimp_direction,
        "sessions_per_week": sessions_per_week,
    }


TRIGGER_EVALUATION_PROMPT = """\
You are a coaching AI evaluating training data for proactive insights.

Given the following training metrics, decide which coaching triggers are warranted.
Only flag triggers that are genuinely significant — avoid false positives.

Training data:
- TRIMP trend: {trimp_direction}
- Volume trend: {volume_direction}
- Sessions per week: {sessions_per_week}
- Compliance: {compliance_info}

For each trigger found, return it with a priority.

You MUST respond with ONLY a valid JSON object:
{{"triggers": [
    {{"type": "fatigue_warning|fitness_improving|compliance_low|great_consistency",
      "priority": "high|medium|low",
      "reasoning": "why this is significant"}}
]}}

Rules:
- fatigue_warning: TRIMP increasing significantly (consider athlete's baseline)
- fitness_improving: consistent training with positive volume trend
- compliance_low: substantially missing planned sessions
- great_consistency: exceptional plan adherence
- Return empty triggers array if nothing is noteworthy
"""


def _detect_triggers(
    weekly_compliance: dict | None,
    trends: dict,
    recent_activities: list[dict],
    use_llm: bool = True,
) -> list[dict]:
    """Detect data-driven triggers from computed assessment data.

    P7 enhancement: Uses LLM to evaluate significance of training data
    patterns, avoiding false positives from fixed thresholds. Falls back
    to heuristic thresholds if LLM is unavailable.

    Each trigger is: {"type": str, "priority": "high"|"medium"|"low", "data": {...}}
    """
    # Primary: LLM-based trigger evaluation (P7)
    if use_llm:
        result = _detect_triggers_llm(weekly_compliance, trends)
        if result is not None:
            return result

    # Fallback: heuristic thresholds
    return _detect_triggers_heuristic(weekly_compliance, trends)


def _detect_triggers_llm(
    weekly_compliance: dict | None,
    trends: dict,
) -> list[dict] | None:
    """Detect triggers using LLM evaluation. Returns None on failure."""
    try:
        compliance_info = "no plan" if not weekly_compliance else (
            f"rate={weekly_compliance.get('compliance_rate', '?')}, "
            f"matched={weekly_compliance.get('matched_count', '?')}/{weekly_compliance.get('planned_count', '?')}"
        )

        prompt = TRIGGER_EVALUATION_PROMPT.format(
            trimp_direction=trends.get("trimp_direction", "no data"),
            volume_direction=trends.get("volume_direction", "no data"),
            sessions_per_week=trends.get("sessions_per_week", []),
            compliance_info=compliance_info,
        )

        response = chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        result = extract_json(response.choices[0].message.content.strip())
        raw_triggers = result.get("triggers", [])

        # Convert to standard trigger format
        valid_types = {"fatigue_warning", "fitness_improving", "compliance_low", "great_consistency"}
        triggers = []
        for t in raw_triggers:
            if t.get("type") in valid_types:
                triggers.append({
                    "type": t["type"],
                    "priority": t.get("priority", "medium"),
                    "data": {"reasoning": t.get("reasoning", "")},
                })
        return triggers
    except Exception:
        return None


def _detect_triggers_heuristic(
    weekly_compliance: dict | None,
    trends: dict,
) -> list[dict]:
    """Fallback: detect triggers using fixed thresholds."""
    triggers = []

    trimp_dir = trends.get("trimp_direction", "")
    if "increasing" in trimp_dir:
        try:
            pct_str = trimp_dir.split("(")[1].rstrip("%)").lstrip("+")
            pct_val = float(pct_str)
            if pct_val > 30:
                triggers.append({
                    "type": "fatigue_warning",
                    "priority": "high",
                    "data": {"trimp_change": trimp_dir},
                })
        except (IndexError, ValueError):
            pass

    vol_dir = trends.get("volume_direction", "")
    sessions = trends.get("sessions_per_week", [])
    if "increasing" in vol_dir and len(sessions) >= 2:
        if all(s > 0 for s in sessions):
            triggers.append({
                "type": "fitness_improving",
                "priority": "low",
                "data": {"volume_trend": vol_dir, "sessions_per_week": sessions},
            })

    if weekly_compliance:
        rate = weekly_compliance.get("compliance_rate", 0)
        if rate < 0.6:
            triggers.append({
                "type": "compliance_low",
                "priority": "medium",
                "data": {
                    "compliance_rate": rate,
                    "matched": weekly_compliance.get("matched_count", 0),
                    "planned": weekly_compliance.get("planned_count", 0),
                },
            })
        elif rate >= 0.9:
            triggers.append({
                "type": "great_consistency",
                "priority": "low",
                "data": {
                    "compliance_rate": rate,
                    "matched": weekly_compliance.get("matched_count", 0),
                    "planned": weekly_compliance.get("planned_count", 0),
                },
            })

    return triggers


# ---------------------------------------------------------------------------
# Greeting composition (LLM call + fallback)
# ---------------------------------------------------------------------------


def compose_startup_greeting(
    user_model,
    plan: dict | None,
    imported: list[dict],
    reflection_result: dict | None = None,
) -> str:
    """Compose a coherent coaching greeting for session startup.

    Makes one LLM call to produce a warm, data-informed greeting. Falls back
    to a Python-formatted summary if the LLM call fails for any reason.

    Args:
        user_model: UserModel instance (for goal, structured_core).
        plan: Current training plan dict (may be None).
        imported: List of newly imported activity dicts from run_import().
        reflection_result: Optional reflection pipeline result dict. When
            present, belief updates are formatted and injected into the
            greeting prompt so the LLM can communicate them transparently.

    Returns:
        A greeting string. Never raises, never returns empty.
    """
    goal = user_model.structured_core.get("goal", {})

    # Lazy import to avoid circular dependency
    from src.tools.activity_store import list_activities

    activities = list_activities()

    # Compute the assessment
    assessment = compute_startup_assessment(plan, activities, goal)

    # Build import summary inline (avoid circular import from cli)
    import_summary = _format_import_summary(imported)

    # Build reflection update summary if available
    reflection_summary = ""
    if reflection_result:
        from src.agent.reflection import format_belief_update_summary
        reflection_summary = format_belief_update_summary(reflection_result)

    # Edge case: no plan AND no activities -> minimal greeting
    if not plan and not activities:
        return "Welcome! I'm your training coach. Let's get started -- tell me about your goals or import some training data."

    # Edge case: plan exists but no activities
    if plan and not activities:
        goal_summary = assessment.get("goal_summary", "your training")
        return (
            f"I see you have a plan set up for {goal_summary}. "
            "Once you start logging activities, I'll be able to track your progress and give tailored advice. "
            "Ready to get started?"
        )

    # Build the LLM user message from assessment data
    user_message = _build_greeting_user_message(
        assessment, import_summary, reflection_summary=reflection_summary,
    )

    # Try LLM call
    try:
        from src.agent.prompts import GREETING_SYSTEM_PROMPT
        from src.agent.json_utils import extract_json

        response = chat_completion(
            messages=[{"role": "user", "content": user_message}],
            system_prompt=GREETING_SYSTEM_PROMPT,
            temperature=0.7,
        )

        result = extract_json(response.choices[0].message.content)
        greeting = result.get("greeting", "")
        if greeting and len(greeting) > 10:
            return greeting

        log.warning("LLM greeting was empty or too short, using fallback")
    except Exception as exc:
        log.warning("LLM greeting failed (%s), using fallback", exc)

    # Fallback: Python-formatted greeting
    return _build_fallback_greeting(
        assessment, import_summary, reflection_summary=reflection_summary,
    )


def _format_import_summary(imported: list[dict]) -> str:
    """Build a human-readable import summary string (inline, no cli import).

    Examples:
        "Found 3 new activities: Running 8.2km, Cycling 45.0km, Strength 60min"
        "" (empty string if nothing imported)
    """
    if not imported:
        return ""

    summaries = []
    for act in imported:
        sport = (act.get("sport") or "unknown").title()
        distance = act.get("distance_meters")
        duration_sec = act.get("duration_seconds")

        if distance and distance > 0:
            summaries.append(f"{sport} {distance / 1000:.1f}km")
        elif duration_sec and duration_sec > 0:
            summaries.append(f"{sport} {round(duration_sec / 60)}min")
        else:
            summaries.append(sport)

    count = len(imported)
    noun = "activity" if count == 1 else "activities"
    return f"Found {count} new {noun}: {', '.join(summaries)}"


def _build_greeting_user_message(
    assessment: dict,
    import_summary: str,
    reflection_summary: str = "",
) -> str:
    """Build the structured user message for the greeting LLM call."""
    parts = []

    # Goal info
    goal_type = assessment.get("goal_type", "general")
    goal_summary = assessment.get("goal_summary", "General fitness")
    parts.append(f"GOAL TYPE: {goal_type}")
    parts.append(f"GOAL: {goal_summary}")

    # Import info
    if import_summary:
        parts.append(f"\nIMPORTED: {import_summary}")

    # Session matching
    sm = assessment.get("session_matching")
    if sm:
        parts.append(f"\nSESSION MATCHING:")
        parts.append(f"  Matched: {sm.get('matched_count', 0)} of {sm.get('planned_count', 0)} planned sessions")
        parts.append(f"  Missed: {sm.get('missed_count', 0)}")
        parts.append(f"  Unplanned: {sm.get('unplanned_count', 0)}")
        matched_details = sm.get("matched", [])
        if matched_details:
            for m in matched_details[:5]:  # limit detail lines
                parts.append(
                    f"  - {m.get('sport', '?')} on {m.get('date', '?')}: "
                    f"duration delta {m.get('duration_delta_pct', 0):+.0f}%, "
                    f"intensity {m.get('intensity_match', 'unknown')}"
                )

    # Weekly compliance
    wc = assessment.get("weekly_compliance")
    if wc:
        parts.append(f"\nWEEKLY COMPLIANCE:")
        parts.append(f"  Rate: {wc.get('compliance_rate', 0):.0%}")
        parts.append(f"  Volume actual: {wc.get('volume_actual_min', 0)} min")
        parts.append(f"  Volume planned: {wc.get('volume_planned_min', 0)} min")
        parts.append(f"  Volume delta: {wc.get('volume_delta_pct', 0):+.1f}%")

    # Intensity summary
    intensity = assessment.get("intensity_summary")
    if intensity:
        parts.append(f"\nINTENSITY: {intensity.get('match', 'unknown')}")
        parts.append(
            f"  On target: {intensity.get('on_target', 0)}, "
            f"Lower: {intensity.get('lower_than_planned', 0)}, "
            f"Higher: {intensity.get('higher_than_planned', 0)}"
        )

    # Trends
    trends = assessment.get("trends", {})
    parts.append(f"\nTRENDS:")
    parts.append(f"  Volume: {trends.get('volume_direction', 'no data')}")
    parts.append(f"  TRIMP (training load): {trends.get('trimp_direction', 'no data')}")
    spw = trends.get("sessions_per_week", [])
    if spw:
        parts.append(f"  Sessions per week (recent): {spw}")

    # Triggers
    triggers = assessment.get("triggers", [])
    if triggers:
        parts.append(f"\nTRIGGERS:")
        for t in triggers:
            parts.append(f"  - [{t.get('priority', 'low')}] {t.get('type', '?')}: {t.get('data', {})}")

    # Reflection updates (injected after triggers so LLM can weave them in)
    if reflection_summary:
        parts.append(f"\n{reflection_summary}")

    return "\n".join(parts)


def _build_fallback_greeting(
    assessment: dict,
    import_summary: str,
    reflection_summary: str = "",
) -> str:
    """Build a Python-formatted fallback greeting when LLM is unavailable.

    Produces a readable message from computed assessment data without any LLM call.
    """
    parts = []

    goal_type = assessment.get("goal_type", "general")

    # Opening
    wc = assessment.get("weekly_compliance")
    if wc:
        rate = wc.get("compliance_rate", 0)
        matched = wc.get("matched_count", 0)
        planned = wc.get("planned_count", 0)
        parts.append(
            f"Welcome back! You completed {matched} of {planned} planned sessions "
            f"({rate:.0%} compliance)."
        )
    else:
        parts.append("Welcome back!")

    # Import mention
    if import_summary:
        parts.append(import_summary + ".")

    # Trend mention
    trends = assessment.get("trends", {})
    vol = trends.get("volume_direction", "no data")
    if vol != "no data" and vol != "stable":
        parts.append(f"Your training volume is {vol}.")
    elif vol == "stable":
        parts.append("Your training volume has been steady.")

    # Trigger mention (pick first one)
    triggers = assessment.get("triggers", [])
    if triggers:
        t = triggers[0]
        ttype = t.get("type", "")
        if ttype == "fatigue_warning":
            parts.append("Your training load has been climbing -- keep an eye on recovery.")
        elif ttype == "compliance_low":
            parts.append("Looks like you've been missing some sessions. Let's talk about what's going on.")
        elif ttype == "great_consistency":
            parts.append("Great consistency lately -- keep it up!")
        elif ttype == "fitness_improving":
            parts.append("Your fitness trend is looking positive!")

    # Reflection updates
    if reflection_summary:
        parts.append(reflection_summary)

    # Closing
    if goal_type == "race_target":
        goal_summary = assessment.get("goal_summary", "your race")
        parts.append(f"Let's chat about your preparation for {goal_summary}.")
    else:
        parts.append("What would you like to work on today?")

    return " ".join(parts)
