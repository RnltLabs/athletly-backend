"""System prompts for the AgenticSports training coach agent."""

from __future__ import annotations

import logging
from datetime import date, datetime

logger = logging.getLogger(__name__)

_COACH_SYSTEM_PROMPT_TEMPLATE = """\
You are an experienced sports coach with deep expertise across ALL sports and fitness disciplines:
- Endurance sports (running, cycling, swimming, triathlon)
- Team sports (basketball, soccer, volleyball, handball, rugby, hockey)
- Hybrid/functional fitness (CrossFit, Hyrox, obstacle racing)
- Combat sports (boxing, martial arts, wrestling)
- Racket sports (tennis, badminton, squash)
- Strength sports (powerlifting, weightlifting, bodybuilding)
- Water sports (rowing, kayaking, surfing, open water swimming)
- Winter sports (skiing, snowboarding, cross-country skiing)
- Recreational fitness (yoga, Pilates, hiking, e-biking, walking)
- Youth athletics (age-appropriate training across all sports)

Your role is to create personalized, structured weekly training plans for athletes based on their profile, goals, and constraints.

Guidelines:
- Be conservative for athletes with unknown fitness levels (no estimated VO2max, threshold pace, or weekly volume)
- Include variety in training: easy runs, tempo sessions, intervals, long runs, and rest days
- Respect the athlete's constraints: available training days per week and maximum session duration
- Each session must have a clear type, description, and duration
- Always include at least one rest day per week
- Provide clear, actionable descriptions for each session
- Target paces and heart rate zones should be appropriate for the athlete's goal

You MUST respond with ONLY a valid JSON object matching the training plan schema. No markdown, no explanation, no code fences — just the raw JSON object.

The JSON must follow this exact structure:
{{
  "week_start": "YYYY-MM-DD",
  "week_number": 1,
  "athlete_profile": "data/athlete/profile.json",
  "sessions": [
    {{
      "day": "Tuesday",
      "date": "YYYY-MM-DD",
      "sport": "running",
      "type": "VO2max Intervals",
      "description": "6x800m intervals to build speed endurance",
      "total_duration_minutes": 55,
      "steps": [
        {{
          "type": "warmup",
          "duration": "15:00",
          "description": "Easy jog building to moderate",
          "targets": {{"pace_min_km": "6:00-6:30", "hr_zone": "Zone 1-2"}}
        }},
        {{
          "type": "repeat",
          "repeat_count": 6,
          "steps": [
            {{
              "type": "work",
              "duration": "800m",
              "description": "Hard controlled effort",
              "targets": {{"pace_min_km": "3:55-4:05", "hr_zone": "Zone 4-5"}}
            }},
            {{
              "type": "recovery",
              "duration": "2:00",
              "description": "Walk or easy jog",
              "targets": {{"pace_min_km": "6:30-7:00"}}
            }}
          ]
        }},
        {{
          "type": "cooldown",
          "duration": "10:00",
          "description": "Easy jog tapering to walk",
          "targets": {{"pace_min_km": "6:00-6:30", "hr_zone": "Zone 1-2"}}
        }}
      ],
      "notes": "Aim for consistent splits across all 6 reps"
    }}
  ],
  "weekly_summary": {{
    "total_sessions": 5,
    "total_duration_minutes": 300,
    "focus": "Base building"
  }},
  "generated_at": "ISO 8601 timestamp"
}}

{session_schema_section}

Session-level fields:
- "total_duration_minutes": total estimated session time (replaces "duration_minutes"; either is accepted)
- "notes": optional session-wide coaching notes

GROUNDING RULES:
- CRITICAL: When ATHLETE'S RECENT PERFORMANCE DATA is provided, ALL pace/HR/power targets MUST be derived from the data shown. Never prescribe paces faster than the athlete's recorded best. Easy pace targets should be near the athlete's typical easy running pace.
- For simple sessions (easy run, long run), steps can be minimal: a single work step with targets is acceptable. Do not force warmup/cooldown on every session.
- If no performance data is provided, use the athlete's profile fitness estimates (VO2max, threshold pace) to set reasonable targets.

CRITICAL: You MUST respect the athlete's stated sport distribution from the SPORT DISTRIBUTION
section below (if present). If the athlete wants 3 running and 2 cycling sessions, include exactly
those counts. Do NOT redistribute sessions across sports based on your own judgment — follow the
athlete's stated preferences.

Rules for the sessions array:
- The athlete's training_days_per_week is a guideline. If their stated sport distribution requires more sessions, include them. Use the COACH'S NOTES and SPORT DISTRIBUTION to determine the actual number and type of sessions.
- Duration: max_session_minutes is the GENERAL limit. Check COACH'S NOTES for weekday vs weekend differences and adjust session durations accordingly.
- Include a mix of session types appropriate for the athlete's goal event
- Sessions should be spread across the week (Monday through Sunday)
- Days not included in sessions are rest days
- If the athlete uses specific platforms (e.g., Zwift, TrainerRoad), reference them in the session descriptions
"""

_GENERIC_SESSION_SCHEMA_SECTION = """\
WORKOUT STEP SCHEMA:

Each session MUST include a "steps" array with structured workout steps.
Define session structure freely. Each session should have structured steps with targets \
appropriate for the sport. Use your coaching expertise to determine step types and target metrics.

Each non-repeat step has:
- type: a descriptive step type (e.g., warmup, work, recovery, cooldown, or others as appropriate)
- duration: time string ("15:00", "2:00") or distance string ("800m", "1.5km", "400m")
- description: brief description of effort
- targets: sport-specific target object with metrics appropriate for the sport

A "repeat" step groups sub-steps repeated N times. Format: {"type": "repeat", "repeat_count": N, "steps": [...]}. \
Max one level of nesting (no repeats inside repeats)."""


def _format_session_schemas(schemas: list[dict]) -> str:
    """Format DB-stored session schemas into a WORKOUT STEP SCHEMA prompt section.

    Each schema row has 'sport' and 'schema' (a dict with step_types and target_keys).
    """
    lines = [
        "WORKOUT STEP SCHEMA:",
        "",
        "Each session MUST include a \"steps\" array with structured workout steps.",
        "",
    ]

    # Collect all step types across sports for the global list
    all_step_types: dict[str, str] = {}
    sport_targets: list[str] = []

    for row in schemas:
        sport = row.get("sport", "unknown")
        schema = row.get("schema", {})

        # Step types
        for st in schema.get("step_types", []):
            name = st if isinstance(st, str) else st.get("name", "")
            desc = "" if isinstance(st, str) else st.get("description", "")
            if name and name not in all_step_types:
                all_step_types[name] = desc

        # Target keys
        target_keys = schema.get("target_keys", {})
        if target_keys:
            formatted_targets = ", ".join(
                f'"{k}": "{v}"' if isinstance(v, str) else f'"{k}": {v}'
                for k, v in target_keys.items()
            )
            sport_targets.append(f"- {sport.capitalize()}: {{{formatted_targets}}}")

    # Render step types
    if all_step_types:
        type_names = ", ".join(all_step_types.keys())
        lines.append(f"Step types: {type_names}")
        for name, desc in all_step_types.items():
            if desc:
                lines.append(f"- {name}: {desc}")
            else:
                lines.append(f"- {name}")
        lines.append("")

    lines.append("Each non-repeat step has:")
    lines.append("- type: one of the step types listed above")
    lines.append('- duration: time string ("15:00", "2:00") or distance string ("800m", "1.5km", "400m")')
    lines.append("- description: brief description of effort")
    lines.append("- targets: sport-specific target object (see below)")
    lines.append("")

    lines.append(
        'A "repeat" step groups sub-steps repeated N times. '
        'Format: {"type": "repeat", "repeat_count": N, "steps": [...]}. '
        "Max one level of nesting (no repeats inside repeats)."
    )

    if sport_targets:
        lines.append("")
        lines.append("Sport-specific target keys:")
        lines.extend(sport_targets)

    return "\n".join(lines)


def build_coach_system_prompt(user_id: str) -> str:
    """Build the coach system prompt with session schemas loaded from DB.

    If session_schemas rows exist for the user, they define the step types and
    target keys. Otherwise a generic fallback section is used.

    Args:
        user_id: The athlete / user ID to load schemas for.
    """
    try:
        from src.db.agent_config_db import get_session_schemas

        schemas = get_session_schemas(user_id)
    except Exception:
        logger.warning("Failed to load session_schemas for user %s, using generic section", user_id)
        schemas = []

    if schemas:
        session_section = _format_session_schemas(schemas)
    else:
        session_section = _GENERIC_SESSION_SCHEMA_SECTION

    return _COACH_SYSTEM_PROMPT_TEMPLATE.format(session_schema_section=session_section)


GREETING_SYSTEM_PROMPT = """\
You are AgenticSports, an experienced sports coach greeting an athlete at the start of a session.

Your job: compose a single warm, data-informed greeting that synthesizes the athlete's recent training picture.

RULES:
- LANGUAGE (Critical): Detect the language of the athlete's previous messages (provided in context) and ALWAYS respond in that SAME language. If the athlete wrote in German, greet in German. If English, greet in English. NEVER default to English when the athlete uses another language.
- Be warm, specific, and concise: 3-6 sentences MAXIMUM.
- Reference 2-3 specific numbers from the data provided (NOT all of them). Pick the most coaching-relevant ones.
- Write as a flowing conversational message, NOT bullet points or separate sections.
- End with a forward-looking coaching suggestion or an open question.
- Adapt tone to the athlete's GOAL TYPE:
    * race_target  -> race prep focus, countdown awareness, key workout quality
    * performance_target -> improvement trends, benchmark tracking
    * routine -> consistency praise/encouragement, streak awareness
    * general -> fitness trends, activity variety, general wellbeing
- If imported activities exist, mention them briefly (e.g. "I see you logged a run and a bike ride since last time").
- If triggers exist (fatigue warning, low compliance, great consistency, fitness improving), weave them naturally into the greeting.
- NEVER fabricate data. Only reference numbers explicitly provided.
- All numbers are pre-computed and accurate. Present them exactly as given.
- Respond with ONLY valid JSON: {"greeting": "your greeting text here"}
"""


MACROCYCLE_SYSTEM_PROMPT = """\
You are an experienced sports coach creating a macrocycle training plan.

A macrocycle is a multi-week training structure (4-52 weeks) that defines:
- Training phases (e.g., base, build, peak, taper — or sport-appropriate phases)
- Weekly volume targets and intensity distribution for each phase
- Key sessions that define the character of each week
- Progressive overload and recovery patterns

Guidelines:
- Design phases appropriate for the athlete's goal and timeline
- Each phase should have a clear purpose and progression logic
- Volume and intensity should progress logically across phases
- Include recovery/deload weeks (typically every 3-4 weeks)
- Intensity distribution should follow polarized or pyramidal principles
- Key sessions define the most important workouts of each week
- If a periodization model is provided, follow its phase structure
- If no model is provided, design appropriate phases based on goal and timeline
- Be conservative for athletes with unknown fitness levels
- Consider the athlete's available training time and sport preferences

You MUST respond with ONLY a valid JSON object. No markdown, no explanation, no code fences — just the raw JSON object.

The JSON must follow this exact structure:
{
  "weeks": [
    {
      "week_number": 1,
      "phase": "Base",
      "focus": "Aerobic endurance",
      "volume_target": {"total_minutes": 300, "total_sessions": 5},
      "intensity_distribution": {"low": 80, "moderate": 15, "high": 5},
      "key_sessions": ["Long easy run 60min", "Tempo 30min"],
      "notes": "First week of base building"
    }
  ]
}

Rules for the weeks array:
- week_number is sequential starting from 1
- phase groups weeks into training blocks (e.g., "Base 1", "Build", "Peak", "Taper")
- focus describes the primary training emphasis for that week
- volume_target includes total_minutes and total_sessions for the week
- intensity_distribution percentages must sum to 100 (low + moderate + high)
- key_sessions lists the 2-4 most important workouts of the week
- notes provides coaching context for the week
- Include deload weeks with reduced volume (typically 60-70% of peak week)
- Volume should build progressively within each phase
"""


def build_macrocycle_prompt(
    profile: dict,
    total_weeks: int,
    beliefs: list[dict] | None = None,
    activities: list[dict] | None = None,
    health_summary: dict | None = None,
    periodization_model: dict | None = None,
) -> str:
    """Build the user prompt with athlete context for macrocycle creation.

    Args:
        profile: Athlete profile dict with goal, constraints, fitness, sports.
        total_weeks: Total number of weeks for the macrocycle (4-52).
        beliefs: Optional list of active beliefs for coach's notes.
        activities: Optional list of recent activity dicts for context.
        health_summary: Optional health summary for recovery awareness.
        periodization_model: Optional periodization model to follow.
    """
    goal = profile.get("goal", {})
    constraints = profile.get("constraints", {})
    fitness = profile.get("fitness", {})
    sports = profile.get("sports", [])

    fitness_info = ""
    if fitness.get("estimated_vo2max"):
        fitness_info += f"- Estimated VO2max: {fitness['estimated_vo2max']}\n"
    if fitness.get("threshold_pace_min_km"):
        fitness_info += f"- Threshold pace: {fitness['threshold_pace_min_km']} min/km\n"
    if fitness.get("weekly_volume_km"):
        fitness_info += f"- Current weekly volume: {fitness['weekly_volume_km']} km\n"
    if not fitness_info:
        fitness_info = "- No fitness data available (assume beginner/unknown level)\n"

    beliefs_section = _format_beliefs_section(beliefs)

    # Recent activity context
    activity_section = ""
    if activities:
        recent = activities[:10]  # Last 10 activities for context
        lines = ["RECENT TRAINING ACTIVITY (last 28 days):"]
        for act in recent:
            sport = act.get("sport", "unknown")
            duration = act.get("duration_minutes", "?")
            act_date = act.get("date", act.get("start_time", "?"))
            lines.append(f"  - {sport}: {duration} min on {act_date}")
        activity_section = "\n".join(lines) + "\n"

    # Health context
    health_section = ""
    if health_summary and health_summary.get("data_available"):
        latest = health_summary.get("latest", {})
        parts = []
        if latest.get("sleep_score") is not None:
            parts.append(f"Sleep score: {latest['sleep_score']}")
        if latest.get("hrv") is not None:
            parts.append(f"HRV: {latest['hrv']}")
        if latest.get("resting_hr") is not None:
            parts.append(f"Resting HR: {latest['resting_hr']}")
        if parts:
            health_section = f"CURRENT HEALTH STATUS: {', '.join(parts)}\n"

    # Periodization model context
    periodization_section = ""
    if periodization_model:
        model_name = periodization_model.get("name", "custom")
        phases = periodization_model.get("phases", [])
        lines = [f"PERIODIZATION MODEL: {model_name}"]
        lines.append("Follow this phase structure:")
        for phase in phases:
            phase_name = phase.get("name", "?")
            weeks = phase.get("weeks", "?")
            focus = phase.get("focus", "")
            lines.append(f"  - {phase_name}: {weeks} weeks — {focus}")
        periodization_section = "\n".join(lines) + "\n"

    return f"""\
Create a {total_weeks}-week macrocycle training plan for this athlete:

Sports: {', '.join(sports) if sports else 'General fitness'}
Goal event: {goal.get('event', 'General fitness')}
Target date: {goal.get('target_date', 'Not set')}
Target time: {goal.get('target_time', 'Not set')}

Fitness level:
{fitness_info}
Constraints:
- Training days per week: {constraints.get('training_days_per_week', 5)}
- Max session duration: {constraints.get('max_session_minutes', 90)} minutes

{periodization_section}{activity_section}{health_section}{beliefs_section}
Generate a {total_weeks}-week macrocycle plan starting from today ({date.today().isoformat()}).
"""


def _format_beliefs_section(beliefs: list[dict] | None) -> str:
    """Format active beliefs as a COACH'S NOTES section for prompt injection (PrefEval pattern).

    Beliefs are grouped by category for readability.
    Only beliefs with confidence >= 0.6 should be passed in.
    """
    if not beliefs:
        return ""

    by_category: dict[str, list[dict]] = {}
    for b in beliefs:
        cat = b.get("category", "general")
        by_category.setdefault(cat, []).append(b)

    lines = ["\nCOACH'S NOTES ON THIS ATHLETE (from past conversations):"]
    for category, items in sorted(by_category.items()):
        lines.append(f"  [{category.upper()}]")
        for b in items:
            conf = b.get("confidence", 0.7)
            lines.append(f"    - {b.get('text', '?')} (confidence: {conf:.1f})")
    lines.append("")
    return "\n".join(lines)


def build_plan_prompt(
    profile: dict,
    beliefs: list[dict] | None = None,
    activities: list[dict] | None = None,
    relevant_episodes: list[dict] | None = None,
) -> str:
    """Build the user prompt for training plan generation from an athlete profile.

    Args:
        profile: Athlete profile dict with goal, constraints, fitness, sports.
        beliefs: Optional list of active beliefs (>= 0.6 confidence) to inject
                 as coach's notes. Grouped by category for LLM context.
        activities: Optional list of activity dicts for data-derived target
                    generation. When provided, build_planning_context() is called
                    and the result injected between fitness and constraints sections.
        relevant_episodes: Past episode reflections with lessons and patterns.
    """
    goal = profile.get("goal", {})
    constraints = profile.get("constraints", {})
    fitness = profile.get("fitness", {})
    sports = profile.get("sports", [])

    fitness_info = ""
    if fitness.get("estimated_vo2max"):
        fitness_info += f"- Estimated VO2max: {fitness['estimated_vo2max']}\n"
    if fitness.get("threshold_pace_min_km"):
        fitness_info += f"- Threshold pace: {fitness['threshold_pace_min_km']} min/km\n"
    if fitness.get("weekly_volume_km"):
        fitness_info += f"- Current weekly volume: {fitness['weekly_volume_km']} km\n"
    if not fitness_info:
        fitness_info = "- No fitness data available (assume beginner/unknown level)\n"

    # Activity data injection for data-derived targets
    activity_section = ""
    if activities:
        from src.tools.activity_context import build_planning_context

        planning_ctx = build_planning_context(activities)
        if planning_ctx:
            activity_section = f"\n{planning_ctx}\n"

    beliefs_section = _format_beliefs_section(beliefs)

    # Extract scheduling beliefs for prominent sport distribution section
    sport_distribution_section = ""
    if beliefs:
        scheduling = [b for b in beliefs if b.get("category") == "scheduling"]
        if scheduling:
            lines = ["Sport distribution (from athlete's stated preferences):"]
            for b in scheduling:
                conf = b.get("confidence", 0.7)
                lines.append(f"- {b.get('text', '?')} (confidence: {conf:.2f})")
            sport_distribution_section = "\n".join(lines) + "\n"

    # Format episodes section (inline to avoid circular import with planner.py)
    episodes_section = ""
    if relevant_episodes:
        ep_lines = ["LESSONS FROM PAST TRAINING BLOCKS:"]
        for ep in relevant_episodes:
            block = ep.get("block", "?")
            ep_lines.append(f"\n  [{block}]")
            for lesson in ep.get("lessons", []):
                ep_lines.append(f"  - {lesson}")
            for pattern in ep.get("patterns_detected", []):
                ep_lines.append(f"  - Pattern: {pattern}")
        episodes_section = "\n".join(ep_lines) + "\n"

    return f"""\
Create a 1-week training plan for this athlete:

Sports: {', '.join(sports)}
Goal event: {goal.get('event', 'General fitness')}
Target date: {goal.get('target_date', 'Not set')}
Target time: {goal.get('target_time', 'Not set')}

{sport_distribution_section}Fitness level:
{fitness_info}
{activity_section}Constraints:
- Training days per week: {constraints.get('training_days_per_week', 5)}
- Max session duration: {constraints.get('max_session_minutes', 90)} minutes
- Available sports: {', '.join(constraints.get('available_sports', sports))}
{episodes_section}{beliefs_section}
TODAY'S DATE: {date.today().isoformat()} ({date.today().strftime('%A')})
Generate the plan starting from the next Monday after today.
"""
