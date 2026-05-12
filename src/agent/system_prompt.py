"""System prompt for Athletly -- the agent's brain.

NanoBot pattern:
  STATIC_SYSTEM_PROMPT  -- cacheable, sent as LLM `system` message (identical for ALL users/requests)
  build_runtime_context -- per-request user message injected before the athlete's first turn
  build_system_prompt   -- returns ONLY the static prompt (for LLM provider caching)

The static prompt defines WHO the agent is, HOW it uses tools, and WHAT rules it follows.
It contains ZERO runtime data (no date, no user info, no sport-specific rules).
All runtime data lives in build_runtime_context().

The agent defines ALL sport-specific knowledge (metrics, formulas, periodization,
evaluation criteria) at runtime via Agent Config Store tools. The system prompt
is a blank slate -- a generalist coach that learns everything via tools.
"""

from datetime import date as _date_cls


# ---------------------------------------------------------------------------
# 1. STATIC SYSTEM PROMPT -- NO f-strings, NO runtime data, NO sport-specific knowledge
# ---------------------------------------------------------------------------

STATIC_SYSTEM_PROMPT = """\
You are Athletly, an autonomous AI coach for any sport. You coach via
natural conversation, ground every claim in real data, and remember
what you learn about each athlete across sessions.

You are a GENERALIST. Sport-specific knowledge (metrics, formulas, eval
criteria, periodization, triggers) is defined at RUNTIME by you via
the `define_config` tool. When a new sport appears: research via
`web_search` (or `spawn_subagent` for deeper research), then persist
your definitions.

## How You Work

You have tools + skills. Tools = atomic actions. Skills = multi-step
playbooks (open via `invoke_skill`). When a deferred tool is needed,
the model lists available tools - the search retrieves it on demand.

Default workflow: gather context with read tools (get_activities,
get_athlete_profile), reason, then act (update_*, create_*, save_*).
Never guess data - call a tool or say "I don't know yet".

## Memory Mandate (Critical)

EVERY time the athlete shares a fact, persist it BEFORE composing
your reply. The athlete journal is the single source of truth.

| Fact type | Tool |
|---|---|
| Name, sports, training days, max session, VO2max numbers | `update_profile(field=..., value=...)` |
| Goal commitment (event + date + time + course facts) | `update_goal(...)` (ALWAYS pass reasoning + source) |
| Identity, lifestyle, history, preferences | `update_journal_section` or `append_to_journal` |
| Pain, injury, anything to follow up next session | `append_to_journal(section="Open Threads", ...)` |
| Per-session note ("knee pain on this run") | `annotate_activity(activity_id, note)` |
| Performance data | `update_profile(fitness.*)` PLUS `append_to_journal(section="What I know about my training", ...)` |

Derive fitness metrics (VO2max from race times, FTP from tests) using
established formulas (Jack Daniels VDOT, etc.) - a rough estimate
beats null.

## Proactive Research

When the athlete mentions a specific race, event, methodology, or
external fact you cannot verify from memory:

```
spawn_subagent(task="Research <X>: date, course, elevation, ...")
```

The subagent has `web_search` + `web_fetch`, runs its own loop, returns
the synthesis. Do this BEFORE responding so your reply has real facts.
Then confirm with the athlete ("Meinst du den XYZ am DD.MM?") or fold
verified facts into your plan.

NEVER fabricate dates, distances, elevations from "likely" knowledge.

## Self-Persistence Pattern

`save_macrocycle()` and `save_plan()` persist the LAST draft from
their respective `create_*` call. Call them with no args after the
athlete approves - the draft is cached. After
  evaluate_plan returns acceptable=true, ALWAYS call save_plan() with no
  args.
- If any save_* tool returns an error: REPORT THE ERROR TO THE USER in
  the next response. Do not silently move on. The athlete must know if
  their plan was not persisted.

## Plan-Generation Pattern

For a training plan: gather context (profile, activities, daily metrics,
macrocycle), then create_training_plan -> evaluate_plan -> save_plan.
If score < 70, regenerate with `feedback=<eval issues>`. Macrocycle:
the same flow with create_macrocycle_plan + save_macrocycle.

## Critical Rules

**Language:** mirror the athlete's language exactly. German in -> German
out. NEVER mid-response code-switch.

**Honesty:**
- <5 sessions of data -> do NOT claim trends.
- No data -> NEVER reference sessions, paces, or metrics.
- Single data point = observation, not conclusion.
- Say "I don't know" when you genuinely don't.

**Athlete Welfare:**
- Youth (<18): minimum 2 rest days/week. Fatigue + low food + high load
  -> PRIORITY response, recommend parents/sports-medicine for RED-S.
- You are a coach, NOT a doctor. Persistent pain, return-from-injury,
  disordered-eating signs -> recommend professional evaluation.
- 6+ training days/week -> recommend at least one rest day.
- Multi-sport: account for TOTAL load across all sports.

**Error Handling:**
- Read tool errors carefully, try a different approach.
- If save_* fails: TELL THE USER. Never silently move on.
- After 3 attempts on the same problem, ask the athlete for help.
- NEVER expose raw error strings to the user in the reply.

**Context Discipline:**
After 8+ consecutive tool calls without responding, PAUSE and summarize
internally. Decide if you have enough to answer. Don't bloat context
with ever-deeper tool chains.

## Pre-Response Check

Before each reply, internally verify:
1. Language matches the athlete's
2. Only data I actually retrieved is referenced
3. Health concerns acknowledged + addressed
4. ALL new facts persisted via memory tools BEFORE composing the reply
"""



# ---------------------------------------------------------------------------
# 2. RUNTIME CONTEXT -- per-request, injected as first user message
# ---------------------------------------------------------------------------

ONBOARDING_MODE_INSTRUCTIONS = """\
# ONBOARDING MODE (Active)

You are in **onboarding mode**. Your job is to learn about this new athlete through
a warm, natural conversation -- NOT a form. Follow these rules:

## Conversation Style
- Start with a warm greeting and introduce yourself as Athletly
- Ask 2-3 questions per message -- never more
- Be conversational and enthusiastic -- this is the athlete's first impression
- Mirror the athlete's language and energy level
- If they share multiple pieces of info in one message, acknowledge ALL of them

## Information to Gather (minimum)
1. **Name** -- usually comes naturally in greeting
2. **Sport(s)** -- extract from free text ("Ich laufe und fahre Rad" -> running, cycling)
3. **Goal** -- what they want to achieve (event, general fitness, weight loss, etc.)
4. **Training days per week** -- how many days they can train
5. **Max session duration** -- how long each session can be (in minutes)

## Extraction Rules
- Extract and save information IMMEDIATELY as the athlete shares it
- Call `update_profile()` / `update_journal_section()` / `append_to_journal()` / `update_goal()` for EVERY piece of info -- do NOT wait
- Derive fitness metrics from any performance data mentioned
- If they mention injuries, constraints, or preferences -- save those too

## Completion Sequence
Once ALL 5 minimum items are gathered, execute this sequence:
1. `define_session_schema` -- for each sport mentioned
2. `define_metric` -- sport-specific metrics (pace, power, HR zones, etc.)
3. `define_eval_criteria` -- plan quality criteria
4. `define_periodization` -- multi-phase training structure
5. `define_trigger_rule` -- proactive alert rules (missed sessions, high fatigue, etc.)
6. If goal has a target date 8+ weeks away: `create_macrocycle_plan` -> `save_macrocycle`
7. `create_training_plan` (with `macrocycle_week` if macrocycle exists) -- generate their first plan
8. `evaluate_plan` -- quality check the plan
9. `save_plan` -- persist the approved plan
10. `recommend_products` -- suggest 3-4 relevant gear/equipment for their sport
11. `complete_onboarding` -- mark onboarding as done

After first health data sync:
12. `get_health_inventory()` -- discover available health metrics
13. Based on available data, define health-aware trigger rules

## Important
- Do NOT ask for all 5 items at once -- be natural
- Do NOT skip the setup sequence -- the athlete needs configs before their first plan
- Do NOT complete onboarding without at least one sport and one goal
- If the athlete asks coaching questions during onboarding, answer them AND continue gathering info
"""


def build_runtime_context(
    user_model,
    date: str | None = None,
    startup_context: str | None = None,
    context: str = "coach",
) -> str:
    """Build the runtime context block injected as the first user message.

    This contains all data that varies per user or per request:
    current date, athlete profile, active beliefs, plan summary,
    onboarding state, and any startup context pre-loaded by the CLI.

    Args:
        user_model: The UserModel instance for the current athlete.
        date: ISO date string for today. Defaults to date.today().isoformat().
        startup_context: Optional pre-computed context string from CLI
            (startup optimization). Contains athlete summary, recent activity
            stats, import results, plan compliance.
        context: Session context -- ``"coach"`` (default) or ``"onboarding"``.
            When ``"onboarding"``, appends onboarding-mode instructions.

    Returns:
        A formatted string to be injected as the first user-role message.
    """
    today = date or _date_cls.today().isoformat()
    weekday = _date_cls.fromisoformat(today).strftime("%A")

    profile = user_model.project_profile()
    athlete_name = profile.get("name") or "Unknown"
    sports = profile.get("sports") or []
    sports_str = ", ".join(sports) if sports else "Not yet known"

    # Optional sub-sections -- only emit if data is present
    sections: list[str] = []

    # --- Date ---
    sections.append(f"# Current Date\nToday is {today} ({weekday}).")

    # --- Athlete Profile (CLAUDE.md-style stable identity) ---
    # Pulls structured profile + beliefs + active macrocycle phase + recent
    # training summary + free-form athlete notes into one block. Injected
    # every turn so the coach has consistent self-context across sessions.
    try:
        from src.agent.athlete_md import build_athlete_md
        _uid_md = getattr(user_model, "user_id", None)
        if _uid_md:
            _md = build_athlete_md(_uid_md).strip()
            if _md:
                sections.append(_md)
    except Exception:
        pass  # Non-critical -- never break context build

    # --- Available Skills (Tier 3 playbooks) ---
    # The agent sees a short list of declarative workflows it can open
    # via invoke_skill(name=...). The body of each skill is only loaded
    # on demand to keep this turn cheap.
    try:
        from src.agent.skills import list_skills as _list_skills
        skills = _list_skills()
        if skills:
            lines = ["# Available Skills"]
            for s in skills:
                desc = s.description.strip().replace("\n", " ")
                if len(desc) > 200:
                    desc = desc[:197] + "..."
                lines.append(f"- {s.name}: {desc}")
            lines.append("")
            lines.append(
                "Invoke any of the above with "
                "`invoke_skill(name=\"<skill_name>\")` to get its full "
                "playbook. Skills are workflows - they orchestrate "
                "multiple tool calls. Atomic actions go through tools "
                "directly."
            )
            sections.append("\n".join(lines))
    except Exception:
        pass  # Non-critical

    # --- Output Style Preference ---
    # Athlete-controlled rendering preference. The coach reads this to
    # decide brevity vs detail for its replies.
    try:
        prefs = profile.get("preferences") or {}
        style = (prefs.get("output_style") or "coach").lower()
        style_guide = {
            "concise": (
                "OUTPUT STYLE: concise. Reply in 2-4 sentences plus a short "
                "table or bullet list if structured data helps. No long "
                "reasoning, no preamble. The athlete prefers density."
            ),
            "detailed": (
                "OUTPUT STYLE: detailed. Explain your reasoning, cite the "
                "data you used, and lay out alternatives. The athlete wants "
                "to understand the why."
            ),
            "coach": (
                "OUTPUT STYLE: coach. Default voice - direct, supportive, "
                "specific. Lead with the answer, then a short explanation, "
                "then next step. No fluff, no over-explanation."
            ),
        }.get(style, "")
        if style_guide:
            sections.append("# Output Style\n" + style_guide)
    except Exception:
        pass

    # --- Athlete Profile ---
    profile_lines = [
        f"# Current Athlete",
        f"Name: {athlete_name}",
        f"Sports: {sports_str}",
    ]

    goal_event = profile.get("goal", {}).get("event") if isinstance(profile.get("goal"), dict) else None
    goal_date = profile.get("goal", {}).get("target_date") if isinstance(profile.get("goal"), dict) else None
    if goal_event:
        profile_lines.append(f"Goal: {goal_event}" + (f" on {goal_date}" if goal_date else ""))

    constraints = profile.get("constraints") or {}
    if isinstance(constraints, dict):
        train_days = constraints.get("training_days_per_week")
        max_minutes = constraints.get("max_session_minutes")
        if train_days is not None:
            profile_lines.append(f"Training days per week: {train_days}")
        if max_minutes is not None:
            profile_lines.append(f"Max session duration: {max_minutes} min")

    fitness = profile.get("fitness") or {}
    if isinstance(fitness, dict):
        vo2max = fitness.get("estimated_vo2max")
        threshold_pace = fitness.get("threshold_pace_min_km")
        if vo2max is not None:
            profile_lines.append(f"Estimated VO2max: {vo2max}")
        if threshold_pace is not None:
            profile_lines.append(f"Threshold pace: {threshold_pace} min/km")

    sections.append("\n".join(profile_lines))

    # --- Beliefs block removed: athlete journal (rendered via
    # build_athlete_md above) is now the single source of truth for
    # identity, preferences, open threads, etc. ---

    # --- Training Plan Summary ---
    try:
        plan_summary = user_model.get_active_plan_summary()
    except Exception:
        plan_summary = None

    if plan_summary:
        sections.append(f"# Active Training Plan\n{plan_summary}")

    # --- Multi-Sport Load Summary (All Sources) ---
    try:
        from src.config import get_settings
        _settings = get_settings()
        _uid = getattr(user_model, "user_id", None) or _settings.agenticsports_user_id
        if _settings.use_supabase and _uid:
            from src.db.health_data_db import get_cross_source_load_summary
            load_summary = get_cross_source_load_summary(_uid, days=7)
            if load_summary["total_sessions"] > 0:
                load_sports_str = ", ".join(load_summary["sports_seen"])
                sources_str = ", ".join(
                    f"{src}: {count}"
                    for src, count in load_summary["sessions_by_source"].items()
                )
                load_header = (
                    f"# This Week's Training Load (All Sources)\n"
                    f"Sessions: {load_summary['total_sessions']} "
                    f"({load_sports_str})\n"
                    f"Duration: {load_summary['total_minutes']}min | "
                    f"TRIMP: {load_summary['total_trimp']}\n"
                    f"Data sources: {sources_str}"
                )
                # Per-sport breakdown -- only for multi-sport athletes
                by_sport = load_summary["sessions_by_sport"]
                if len(by_sport) > 1:
                    sport_lines = [
                        f"  {sport}: {count} sessions"
                        for sport, count in by_sport.items()
                    ]
                    load_header += (
                        "\n\n## Per-Sport Breakdown\n"
                        + "\n".join(sport_lines)
                    )
                sections.append(load_header)
    except Exception:
        pass  # Non-critical -- do not crash context building

    # --- Current Recovery Status ---
    try:
        from src.config import get_settings as _get_settings_recovery
        _rs = _get_settings_recovery()
        _uid_r = getattr(user_model, "user_id", None) or _rs.agenticsports_user_id
        if _rs.use_supabase and _uid_r:
            from src.services.health_context import (
                build_health_summary,
                format_recovery_context_block,
            )
            health_summary = build_health_summary(_uid_r, days=7)
            if health_summary and health_summary["data_available"]:
                sections.append(format_recovery_context_block(health_summary))
    except Exception:
        pass  # Non-critical -- do not crash context building

    # --- Onboarding State ---
    onboarding_missing = _onboarding_missing(profile)
    if onboarding_missing:
        missing_str = ", ".join(onboarding_missing)
        sections.append(
            f"# Onboarding State\n"
            f"This athlete is still being onboarded. Missing: {missing_str}.\n"
            f"Gather these naturally in conversation and save them with update_profile()."
        )

    # --- Startup Context (pre-loaded by CLI) ---
    if startup_context:
        sections.append(
            f"# Pre-Loaded Session Context\n"
            f"{startup_context}\n"
            f"Use this context to inform your greeting and coaching.\n"
            f"You SHOULD still call update_profile() / update_journal_section() / append_to_journal() / update_goal() for any NEW information\n"
            f"the athlete shares -- this context only saves you from calling data-retrieval\n"
            f"tools like get_activities() or get_athlete_profile() at session start."
        )

    # --- Onboarding Mode Instructions ---
    if context == "onboarding":
        sections.append(ONBOARDING_MODE_INSTRUCTIONS)

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# 3. STATIC PROMPT ACCESSOR -- returns ONLY the static prompt for caching
# ---------------------------------------------------------------------------

def build_system_prompt(
    user_model=None,
    startup_context: str | None = None,
    context: str = "coach",
) -> str:
    """Return the static system prompt for LLM provider caching.

    This function returns ONLY the static prompt. Runtime context is
    injected separately as a user-role message by the agent loop.

    Args are accepted for backward compatibility but ignored -- the
    system prompt is always identical regardless of user or context.

    Returns:
        The static system prompt string (identical for all users/requests).
    """
    return STATIC_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _onboarding_missing(profile: dict) -> list[str]:
    """Return a list of onboarding fields that are still missing."""
    missing = []
    if not profile.get("name"):
        missing.append("name")
    if not profile.get("sports"):
        missing.append("sport(s)")
    goal = profile.get("goal") or {}
    if isinstance(goal, dict) and not goal.get("event"):
        missing.append("goal/event")
    constraints = profile.get("constraints") or {}
    if isinstance(constraints, dict):
        if constraints.get("training_days_per_week") is None:
            missing.append("training days per week")
        if constraints.get("max_session_minutes") is None:
            missing.append("max session duration")
    return missing
