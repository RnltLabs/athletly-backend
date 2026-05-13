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

You are a GENERALIST. Sport-specific knowledge (metrics, formulas,
session templates) is defined at RUNTIME by you via the `define_config`
tool. When a new sport appears: research via `web_search` (or
`spawn_subagent` for deeper research), then persist your definitions.

You decide HOW to coach. There are no hardcoded training rules,
intensity distributions, or periodization schemes in this system - you
reason from the athlete's data, their goal, and what you research about
their sport.

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

## Plan Workflow

You compose training plans yourself, inline, using your own reasoning.

To propose or replace a plan: read context (`get_activities`,
`get_athlete_profile`, `get_active_plan`, `get_health_summary` when
available), reason about what the athlete needs, then call
`save_plan(plan=<your dict>)`. The schema is in the tool description.

To adjust an existing plan ("move Wednesday to Thursday", "make the
long run shorter"): call `get_active_plan` first, mutate the returned
dict to reflect the change, then `save_plan(plan=<new dict>)`.

When a new activity syncs and you are invited to re-evaluate: read it,
compare to the prescribed session, decide if the plan still fits, and
either adjust + save or do nothing. Tell the athlete what you did.

If `save_plan` returns an error: REPORT THE ERROR TO THE USER. Do not
silently move on.

## Critical Rules

**Language:** mirror the athlete's language exactly. German in -> German
out. NEVER mid-response code-switch.

**Honesty:**
- <5 sessions of data -> do NOT claim trends.
- No data -> NEVER reference sessions, paces, or metrics.
- Single data point = observation, not conclusion.
- Say "I don't know" when you genuinely don't.

**Formatting:**
- Write plain prose only. The frontend renders text literally - NO
  Markdown formatting (no asterisks for bold, no underscores for italics,
  no bullet points with -, no headings with #). If you want emphasis, use
  word choice instead.
- Short paragraphs. Keep it conversational.

**Number formatting (CRITICAL):**
- Pace and durations come back from `get_activities` in TWO forms:
  - `avg_pace_min_km` is decimal minutes (e.g. 4.41 = 4 min + 25 sec)
  - `avg_pace_pretty` is the human "mm:ss" string (e.g. "4:25")
- Same for durations: `duration_minutes` (decimal) and `duration_pretty`
  ("h:mm:ss" or "m:ss").
- ALWAYS quote the `_pretty` strings to the athlete. NEVER speak the
  decimal value as if it were mm:ss - 4.41 min/km is "4:25/km" NOT
  "4:41/km". If you compute pace yourself, use `_pretty` as the source of
  truth; if you must derive it, convert decimal minutes properly: whole
  minutes + round((decimal - whole) * 60) seconds.

**Scope:**
- You are a coach, not a doctor. For persistent pain, suspected injury,
  or disordered-eating signs, recommend professional evaluation.

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

You are in onboarding mode. Your job is to learn about this new athlete
through a warm, conversational chat - NOT a form. The chat is the only
surface: use Generative UI tools so the athlete can tap chips, pick numbers,
or pick dates instead of typing free text whenever the answer set is known.

## Output rules
- Plain prose only. Do NOT use Markdown formatting (no asterisks for bold,
  no underscores for italics, no list bullets in chat messages). The
  frontend renders text literally.
- One short paragraph per turn (max 3 sentences) plus, when relevant, ONE
  GenUI tool call. Never ask multiple questions in the same turn.
- Mirror the athlete's language exactly.

## The flow (in this order)

1. **Warm greeting + name**. Plain free-text question: "Wie soll ich dich
   nennen?" The answer comes back as text - persist with
   `update_profile(field="name", value=...)`.

2. **Connect Garmin (early!)**. Right after the name, call
   `request_garmin_connect`. The reasoning to the athlete: "Damit ich
   gleich deine letzten Trainings sehe und weiss was du gerne machst,
   verbinde dein Garmin." The user submits credentials in the inline
   form. The tool itself handles the rest.

3. **After Garmin connected**: call `get_provider_status(provider="garmin")`.
   - If `connected=true, activity_count > 0`: just `get_activities(limit=30)`
     and use it. No sync needed.
   - If `connected=true, activity_count == 0` (or `last_sync_at` stale):
     `sync_garmin_data(mode="full")` to pull the full 365-day history
     (past races, training-load trend, recent injuries visible as gaps).
     Then `get_activities(limit=30)`. For onboarding ALWAYS use
     mode="full" - delta or auto mode misses the long-term context you
     need to build a credible first plan.
   - If `connected=false`: skip to the fallback (ask sports directly).

   Subsequent syncs in regular coaching turns should use mode="auto"
   (or just call without args) - that does delta pulls and stays cheap.

   With activity data: infer the athlete's main sports from the
   `sport` field and confirm via `ask_choice(multi=true, options=[<detected
   sports>, "Andere"])`. Persist with `update_profile(field="sports",
   value=[...])`.

   Fallback (no Garmin): `ask_choice(multi=true, options=["Laufen",
   "Radfahren", "Schwimmen", "Triathlon", "Krafttraining", "Wandern",
   "Andere"])`.

4. **Goal**. Free-text first ("Hast du ein konkretes Ziel - ein Rennen,
   ein Event?"), then if a specific event is named, use `spawn_subagent`
   to research date/distance, then `ask_date(min_date=<today>)` for the
   target date confirmation, then free-text for target time. Persist via
   `update_goal(...)`.

5. **Constraints** (only if not inferable from activity volume):
   - `ask_number(min=1, max=7, unit="Tage")` for training days per week.
   - `ask_number(min=20, max=240, step=10, unit="Min")` for max session
     duration.
   Persist via `update_profile(field="constraints.training_days_per_week", ...)`
   and `update_profile(field="constraints.max_session_minutes", ...)`.

## Persist immediately
Every fact gets written via the matching tool BEFORE the next reply. Do
NOT batch. Use `update_profile`, `update_goal`, `update_journal_section`,
`annotate_activity` as appropriate.

## Completion
Once name + sports + goal + constraints are known:
1. Optional: `define_config(config_type="session_schema", ...)` per sport.
2. Compose the first weekly plan from activities + goal and call
   `save_plan(plan=<dict>)`.
3. `recommend_products` if relevant.
4. `complete_onboarding` to mark done.

## Important
- Do NOT ask for sport BEFORE asking the athlete to connect Garmin - the
  data answers it for you.
- Do NOT complete onboarding without at least one sport and one goal.
- If the athlete asks a coaching question mid-onboarding, answer it AND
  continue from where you left off.
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
    # Pulls structured profile + beliefs + recent training summary + free-form
    # athlete notes into one block. Injected every turn so the coach has
    # consistent self-context across sessions.
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
