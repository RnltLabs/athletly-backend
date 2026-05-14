"""Plan-and-Execute pipeline for high-stakes training plan generation.

This module implements Feature 4: a two-stage planning pipeline for
multi-week training plans on the Pro tier. The flow:

    1. Planner LLM call (Sonnet by default): produces a structured
       OUTLINE with phases, weekly volume, intensity distribution, and a
       weekday template. NO per-day session detail.
    2. Executor LLM call per week (Haiku by default): fills in the actual
       sessions for that week, respecting the outline slot constraints.
    3. Assembly: stitches per-week sessions into a single plan dict that
       conforms to the `save_plan` tool schema.
    4. Validation: checks the assembled plan against athlete constraints
       (`max_session_minutes`, `training_days_per_week`,
       `available_sports`, date ranges) before persistence.

The entry point is `generate_training_plan(request, profile, ...)`.

Heuristic for invoking this pipeline lives in `should_use_plan_and_execute`.
The Free tier and short plans bypass this module entirely and use the
existing inline path inside `save_plan`.

Backward compatibility: this module is invoked ONLY from inside
`save_plan` when the heuristic fires. The agent's tool interface is
unchanged.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from datetime import date as _date_cls, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from src.agent.json_utils import extract_json
from src.agent.llm import chat_completion

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Heuristic threshold: session_count = duration_weeks * training_days_per_week.
# At >= 14 sessions (~2-3 weeks) the one-shot generation starts to lose
# coherence and Plan-and-Execute pays off (see RESEARCH.md section 4).
_COMPLEXITY_SESSION_THRESHOLD = 14

# Explicit-intent threshold: when the agent declares a horizon of at
# least this many weeks the planner ALWAYS fires, regardless of
# training-days math. Matches the "DO NOT inline beyond 2 weeks" rule
# enforced in the save_plan tool description; we set this slightly
# above 2 to leave room for short adjustments without escalating.
_LONG_HORIZON_WEEKS_THRESHOLD = 4

# Keyword patterns that indicate a long-horizon (multi-week build)
# plan even when the agent forgot to set duration_weeks. Used as a
# defensive backstop. German + English. Lowercased before match.
_LONG_HORIZON_KEYWORDS: frozenset[str] = frozenset({
    "marathon",
    "ironman",
    "70.3",
    "triathlon",
    "half marathon",
    "halbmarathon",
    "roth",
    "kona",
    "build phase",
    "race build",
    "rennaufbau",
    "trainingsblock",
    "build block",
})

# Numeric horizon pattern: matches "8 weeks", "16-week", "12 Wochen",
# "8 Woche" (sloppy singular). Used by the keyword backstop.
_LONG_HORIZON_NUMERIC_RE = re.compile(
    r"\b(\d{1,2})\s*[-\s]?\s*(weeks?|wochen?)\b",
    re.IGNORECASE,
)

# Default models. Env overrides:
#   ATHLETLY_PLANNER_MODEL, ATHLETLY_EXECUTOR_MODEL
_DEFAULT_PLANNER_MODEL = "anthropic/claude-sonnet-4-5-20250929"
_DEFAULT_EXECUTOR_MODEL = "anthropic/claude-haiku-4-5-20251001"

# Retry budget per stage.
_PLANNER_MAX_ATTEMPTS = 2
_EXECUTOR_MAX_ATTEMPTS = 2

# Acceptable weekly_template values.
_VALID_SLOT_TYPES: frozenset[str] = frozenset({
    "easy", "quality", "long", "long_or_quality", "cross", "rest",
})

_WEEKDAYS: tuple[str, ...] = (
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
)

_WEEKDAY_TO_OFFSET: dict[str, int] = {d: i for i, d in enumerate(_WEEKDAYS)}

# Bounded plan length (Pro tier safety net). Anything over this is a
# misconfiguration; cap to a sane value.
_MAX_PLAN_WEEKS = 32

# ---------------------------------------------------------------------------
# Multi-sport floor heuristic (Sprint H sport-mix fix)
# ---------------------------------------------------------------------------

# Minimum number of sessions per discipline, per week, by triathlon
# distance label. These are FLOORS, not targets. See RESEARCH.md.
# Sport keys are normalised to the lower-case strings the agent uses
# across the codebase (running / cycling / swimming).
_MULTI_SPORT_FLOORS: dict[str, dict[str, int]] = {
    "long_course": {"swimming": 1, "cycling": 2, "running": 2},
    "middle_distance": {"swimming": 1, "cycling": 2, "running": 2},
    "short_course": {"swimming": 1, "cycling": 1, "running": 1},
    "unknown_triathlon": {"swimming": 1, "cycling": 1, "running": 1},
}

# Substring keyword -> discipline label. Matched in order; first hit wins.
# Longer / more-specific tokens come first so e.g. "ironman 70.3" maps to
# middle_distance even though "ironman" alone is long_course.
_DISCIPLINE_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("70.3", "middle_distance"),
    ("halbdistanz", "middle_distance"),
    ("mitteldistanz", "middle_distance"),
    ("half ironman", "middle_distance"),
    ("half-ironman", "middle_distance"),
    ("halfironman", "middle_distance"),
    ("langdistanz", "long_course"),
    ("140.6", "long_course"),
    ("full distance", "long_course"),
    ("full-distance", "long_course"),
    ("ironman", "long_course"),
    ("iron man", "long_course"),
    ("kona", "long_course"),
    ("roth", "long_course"),
    ("sprint", "short_course"),
    ("kurzdistanz", "short_course"),
    ("olympische", "short_course"),
    ("olympic", "short_course"),
)


def _detect_triathlon_discipline(request: dict, outline: dict | None) -> str | None:
    """Return the discipline label from `goal_event` / `focus`, or None.

    None means "not a triathlon plan" (or distance not stated). When the
    profile is multi-sport but the discipline is None we fall through to
    `unknown_triathlon` floors so we still enforce min one session per
    sport per week.
    """
    haystack_parts: list[str] = []
    for key in ("goal_event", "focus", "name"):
        val = request.get(key) if isinstance(request, dict) else None
        if isinstance(val, str):
            haystack_parts.append(val)
    if outline:
        for key in ("goal_event", "focus"):
            val = outline.get(key) if isinstance(outline, dict) else None
            if isinstance(val, str):
                haystack_parts.append(val)
    if not haystack_parts:
        return None
    haystack = " ".join(haystack_parts).lower()
    for kw, label in _DISCIPLINE_KEYWORDS:
        if kw in haystack:
            return label
    return None


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlannerResult:
    """The outcome of a Plan-and-Execute run."""

    plan: dict
    """Final plan dict matching the `save_plan` tool schema."""

    mode: str
    """One of: 'plan_and_execute', 'inline_fallback'."""

    meta: dict = field(default_factory=dict)
    """Telemetry: model names, week count, retry counts, validation notes."""


# ---------------------------------------------------------------------------
# Heuristic
# ---------------------------------------------------------------------------

def should_use_plan_and_execute(
    plan_request: dict,
    profile: dict,
) -> bool:
    """Return True when the Plan-and-Execute pipeline should run.

    Three independent gates, in order. The first matching gate wins:

    1. Explicit intent gate. If the agent set ``duration_weeks`` or
       ``weeks`` to >= ``_LONG_HORIZON_WEEKS_THRESHOLD`` (4) the
       pipeline runs regardless of session math. This is the canonical
       trigger: the agent declares the horizon, the runtime obliges.

    2. Keyword backstop. If ``focus``, ``goal_event``, or any other
       free-text field matches a long-horizon keyword (marathon,
       ironman, "8 Wochen", etc.) the pipeline runs. Defensive layer
       for the agent's failure mode of writing a build-phase label
       without setting duration_weeks.

    3. Legacy session-count gate. If the plan dict already carries
       enough inlined sessions to clear ``_COMPLEXITY_SESSION_THRESHOLD``
       (duration_weeks * training_days_per_week >= 14), escalate. Kept
       for backward compat; the agent should never reach this branch
       after the prompt updates.

    Returns False on missing/unparseable input.
    """
    if not isinstance(plan_request, dict):
        return False

    # Gate 1: explicit intent.
    explicit_weeks = _parse_explicit_weeks(plan_request)
    if explicit_weeks is not None and explicit_weeks >= _LONG_HORIZON_WEEKS_THRESHOLD:
        return True

    # Gate 2: keyword backstop.
    if _looks_like_long_horizon(plan_request):
        return True

    # Gate 3: legacy session-count.
    weeks = _infer_duration_weeks(plan_request)
    if not weeks:
        return False

    constraints = profile.get("constraints") or {}
    days = constraints.get("training_days_per_week") or 5
    try:
        days = int(days)
    except (TypeError, ValueError):
        days = 5

    return weeks * days >= _COMPLEXITY_SESSION_THRESHOLD


def _parse_explicit_weeks(plan_request: dict) -> int | None:
    """Return the agent's declared horizon in weeks, or None.

    Only reads ``duration_weeks`` and ``weeks`` directly; does NOT
    fall through to session-list inference. The point of this helper
    is to detect AGENT INTENT, not to infer from an artifact.
    """
    raw = plan_request.get("duration_weeks")
    if raw is None:
        raw = plan_request.get("weeks")
    if isinstance(raw, bool):
        # bool is an int subclass; treat True/False as missing.
        return None
    if isinstance(raw, int) and raw > 0:
        return min(raw, _MAX_PLAN_WEEKS)
    if isinstance(raw, str):
        try:
            n = int(raw.strip())
            if n > 0:
                return min(n, _MAX_PLAN_WEEKS)
        except ValueError:
            pass
    return None


def _looks_like_long_horizon(plan_request: dict) -> bool:
    """Return True if any free-text field signals a multi-week build.

    Scans ``focus``, ``goal_event``, ``goal_date``, and ``meta`` for
    a long-horizon keyword OR a numeric "N week(s) / N Wochen" pattern
    with N >= ``_LONG_HORIZON_WEEKS_THRESHOLD``.
    """
    haystack_parts: list[str] = []
    for key in ("focus", "goal_event", "name"):
        val = plan_request.get(key)
        if isinstance(val, str):
            haystack_parts.append(val)
    # Some agent shapes nest a `goal` dict (project_profile style).
    goal = plan_request.get("goal")
    if isinstance(goal, dict):
        for k in ("event", "target_event"):
            v = goal.get(k)
            if isinstance(v, str):
                haystack_parts.append(v)
    if not haystack_parts:
        return False

    haystack = " ".join(haystack_parts).lower()

    # Keyword match: any keyword as substring (lowercased haystack).
    for kw in _LONG_HORIZON_KEYWORDS:
        if kw in haystack:
            return True

    # Numeric horizon pattern.
    match = _LONG_HORIZON_NUMERIC_RE.search(haystack)
    if match:
        try:
            n = int(match.group(1))
            if n >= _LONG_HORIZON_WEEKS_THRESHOLD:
                return True
        except ValueError:
            pass

    return False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate_training_plan(
    request: dict,
    profile: dict,
    *,
    on_progress: Callable[[str, str], None] | None = None,
    planner_model: str | None = None,
    executor_model: str | None = None,
    parallel: bool = False,
) -> PlannerResult:
    """Run the Plan-and-Execute pipeline.

    Args:
        request: The agent's intent. May contain `duration_weeks`,
            `start_date`, `goal_event`, `goal_date`, `language`, and any
            free-text hints. Treated as immutable (we never mutate).
        profile: The athlete's `project_profile()` dict (see
            `UserModel.project_profile`). Provides `constraints`,
            `goal`, `sports`, `fitness`.
        on_progress: Optional callback(event_type, detail) for streaming
            progress events to the caller. Event types:
                "planner_start" | "planner_done" | "week_done" |
                "validation_failed" | "fallback".
        planner_model: Override the planner model. Env-resolvable via
            `ATHLETLY_PLANNER_MODEL`. Defaults to Sonnet.
        executor_model: Override the executor model. Env-resolvable via
            `ATHLETLY_EXECUTOR_MODEL`. Defaults to Haiku.
        parallel: Reserved for follow-up work. Currently sequential
            execution only.

    Returns:
        PlannerResult with the assembled plan, mode tag, and meta.

    Notes:
        On any unrecoverable failure inside Plan-and-Execute, returns a
        PlannerResult with `mode="inline_fallback"` and the original
        request as the plan. Callers should detect this and fall back to
        the one-shot path. We never raise from this function (the
        save_plan tool already has its own error envelope).
    """
    planner_model_resolved = _resolve_model("planner", planner_model)
    executor_model_resolved = _resolve_model("executor", executor_model)

    meta: dict[str, Any] = {
        "planner_model": planner_model_resolved,
        "executor_model": executor_model_resolved,
        "weeks": None,
        "planner_attempts": 0,
        "executor_retries": 0,
        "validation_notes": [],
        "sport_mix_retries": 0,
    }

    # Up to two attempts: the second one carries an extra annotation
    # telling the planner which sports were missing. If the SECOND
    # attempt still fails the multi-sport floor we fall back to inline
    # rather than ship a broken plan.
    sport_mix_hint: str | None = None
    for sport_attempt in range(2):
        attempt_result = _attempt_plan_generation(
            request=request,
            profile=profile,
            on_progress=on_progress,
            planner_model_resolved=planner_model_resolved,
            executor_model_resolved=executor_model_resolved,
            meta=meta,
            sport_mix_hint=sport_mix_hint,
        )
        if attempt_result.mode == "inline_fallback":
            return attempt_result
        sport_errors = _validate_sport_distribution(
            attempt_result.plan, profile, attempt_result.plan.get("outline") or {},
            request,
        )
        if not sport_errors:
            return attempt_result
        # Violation: log, record, and either retry or fall back.
        meta["validation_notes"].extend(sport_errors)
        logger.warning(
            "Sport-mix floor violation (attempt %d): %s",
            sport_attempt + 1, "; ".join(sport_errors),
        )
        _emit_progress(on_progress, "validation_failed", "; ".join(sport_errors))
        if sport_attempt == 0:
            # Build a short, machine-readable hint for the planner retry.
            meta["sport_mix_retries"] = 1
            sport_mix_hint = (
                "SPORT-MIX VIOLATION on previous attempt:\n"
                + "\n".join(f"  - {e}" for e in sport_errors)
                + "\nRe-emit the outline with a valid sport_per_day so EVERY "
                  "week respects the multi-sport floors documented in the "
                  "system prompt."
            )
            continue
        # Second attempt also failed -> fall back closed.
        return PlannerResult(
            plan=dict(request),
            mode="inline_fallback",
            meta={**meta, "fallback_reason": "sport_mix_floor_violation"},
        )

    # Defensive: the loop above always returns. If we ever fall through,
    # treat as inline fallback.
    return PlannerResult(
        plan=dict(request),
        mode="inline_fallback",
        meta={**meta, "fallback_reason": "exhausted_sport_mix_retries"},
    )


def _attempt_plan_generation(
    *,
    request: dict,
    profile: dict,
    on_progress: Callable[[str, str], None] | None,
    planner_model_resolved: str,
    executor_model_resolved: str,
    meta: dict[str, Any],
    sport_mix_hint: str | None,
) -> PlannerResult:
    """Run one full planner -> executors -> assemble cycle.

    Encapsulates everything that used to live inline in
    ``generate_training_plan``. Returns a PlannerResult that may be
    ``inline_fallback`` (irrecoverable failure) or ``plan_and_execute``.
    Sport-mix validation is handled by the OUTER loop in
    ``generate_training_plan``.
    """
    _emit_progress(on_progress, "planner_start", planner_model_resolved)

    try:
        outline = _run_planner(
            request=request,
            profile=profile,
            model=planner_model_resolved,
            meta=meta,
            extra_user_hint=sport_mix_hint,
        )
    except _PlannerError as exc:
        logger.warning("Planner failed: %s. Falling back to inline.", exc)
        _emit_progress(on_progress, "fallback", str(exc))
        return PlannerResult(
            plan=dict(request),
            mode="inline_fallback",
            meta={**meta, "fallback_reason": str(exc)},
        )

    duration_weeks = outline["duration_weeks"]
    meta["weeks"] = duration_weeks
    _emit_progress(on_progress, "planner_done", f"{duration_weeks} weeks")

    # Executor: one call per week, sequential for ship.
    all_sessions: list[dict] = []
    for week_index in range(1, duration_weeks + 1):
        try:
            week_sessions = _run_executor_for_week(
                outline=outline,
                week_index=week_index,
                profile=profile,
                language=_resolve_language(request, profile),
                model=executor_model_resolved,
                meta=meta,
            )
        except _ExecutorError as exc:
            logger.warning(
                "Executor failed on week %d: %s. Falling back to inline.",
                week_index, exc,
            )
            _emit_progress(on_progress, "fallback", f"week {week_index}: {exc}")
            return PlannerResult(
                plan=dict(request),
                mode="inline_fallback",
                meta={**meta, "fallback_reason": f"executor_week_{week_index}: {exc}"},
            )
        all_sessions.extend(week_sessions)
        _emit_progress(on_progress, "week_done", f"week {week_index}")

    # Assembly.
    assembled = _assemble_plan(
        request=request,
        outline=outline,
        sessions=all_sessions,
        meta=meta,
    )

    # Validation against athlete constraints.
    validation_errors = _validate_against_constraints(assembled, profile, outline)
    if validation_errors:
        meta["validation_notes"].extend(validation_errors)
        logger.warning(
            "Plan failed athlete-constraint validation: %s. Falling back to inline.",
            "; ".join(validation_errors),
        )
        _emit_progress(on_progress, "validation_failed", "; ".join(validation_errors))
        return PlannerResult(
            plan=dict(request),
            mode="inline_fallback",
            meta={**meta, "fallback_reason": "athlete_constraints"},
        )

    return PlannerResult(plan=assembled, mode="plan_and_execute", meta=meta)


# ---------------------------------------------------------------------------
# Planner stage
# ---------------------------------------------------------------------------

class _PlannerError(Exception):
    """Internal: unrecoverable planner failure."""


class _ExecutorError(Exception):
    """Internal: unrecoverable executor failure."""


def _run_planner(
    request: dict,
    profile: dict,
    model: str,
    meta: dict,
    extra_user_hint: str | None = None,
) -> dict:
    """Invoke the planner LLM and return a validated outline.

    ``extra_user_hint`` is appended to the user message ON EVERY ATTEMPT
    of this planner cycle. Used by the outer sport-mix retry loop to
    tell the planner exactly which sports were missing on a previous
    cycle. None means "no hint" (first attempt).
    """
    system_prompt = _load_prompt("planner_system.md")
    user_message = _build_planner_user_message(request, profile)
    if extra_user_hint:
        user_message = user_message + "\n\n" + extra_user_hint

    # Audit log: this call burns Sonnet (or whatever planner model the
    # operator configured). Emit a structured line so premium spend is
    # visible in container logs even though `chat_completion(model=...)`
    # bypasses the model_router's own log line.
    _log_planner_invocation(stage="planner", model=model, request=request)

    last_error: str | None = None
    for attempt in range(1, _PLANNER_MAX_ATTEMPTS + 1):
        meta["planner_attempts"] = attempt
        messages: list[dict] = [{"role": "user", "content": user_message}]
        if last_error:
            messages.append({
                "role": "user",
                "content": (
                    "Your previous response was invalid: "
                    f"{last_error}\n\nFix the error and return ONLY the JSON."
                ),
            })

        try:
            response = chat_completion(
                messages=messages,
                system_prompt=system_prompt,
                temperature=0.3,
                model=model,
            )
        except Exception as exc:
            logger.warning("Planner LLM call failed (attempt %d): %s", attempt, exc)
            last_error = f"LLM call failed: {exc}"
            continue

        content = response.choices[0].message.content if response.choices else ""
        try:
            payload = extract_json(content or "")
        except ValueError as exc:
            last_error = f"could not parse JSON: {exc}"
            continue

        outline_validation = _validate_outline(payload, request, profile)
        if outline_validation:
            last_error = outline_validation
            continue

        return payload["outline"]

    raise _PlannerError(f"planner exhausted retries: {last_error or 'unknown error'}")


def _build_planner_user_message(request: dict, profile: dict) -> str:
    """Build the user-role message that frames the planning task."""
    today = _date_cls.today().isoformat()
    constraints = profile.get("constraints") or {}
    goal = profile.get("goal") or {}
    fitness = profile.get("fitness") or {}
    sports = profile.get("sports") or []

    parts: list[str] = ["# Planning Request", f"Today: {today}"]

    duration_weeks = _infer_duration_weeks(request) or 12
    parts.append(f"duration_weeks: {duration_weeks}")

    start_date = _normalize_start_date(request.get("start_date"))
    parts.append(f"start_date: {start_date}")

    if request.get("goal_event"):
        parts.append(f"goal_event (requested): {request['goal_event']}")
    elif goal.get("event"):
        parts.append(f"goal_event (profile): {goal['event']}")

    if request.get("goal_date"):
        parts.append(f"goal_date (requested): {request['goal_date']}")
    elif goal.get("target_date"):
        parts.append(f"goal_date (profile): {goal['target_date']}")

    if goal.get("target_time"):
        parts.append(f"goal_target_time: {goal['target_time']}")

    parts.append("")
    parts.append("# Athlete Constraints (echo these in constraints_acknowledged)")
    parts.append(
        f"max_session_minutes: {constraints.get('max_session_minutes', 90)}"
    )
    parts.append(
        f"training_days_per_week: {constraints.get('training_days_per_week', 5)}"
    )
    available_sports = (
        constraints.get("available_sports")
        or sports
        or ["running"]
    )
    parts.append(f"available_sports: {available_sports}")

    if fitness:
        from src.utils.pace_format import decimal_min_to_mmss
        parts.append("")
        parts.append("# Athlete Fitness Snapshot")
        vo2 = fitness.get("estimated_vo2max")
        thresh = fitness.get("threshold_pace_min_km")
        thresh_pretty = decimal_min_to_mmss(thresh)
        vol = fitness.get("weekly_volume_km")
        if vo2 is not None:
            parts.append(f"estimated_vo2max: {vo2}")
        if thresh_pretty is not None:
            # Pre-formatted mm:ss so the planner LLM cannot misread
            # 4.50 as "four fifty per km" (Sprint C fix).
            parts.append(f"threshold_pace: {thresh_pretty} /km")
        if vol is not None:
            parts.append(f"recent_weekly_volume_km: {vol}")

    parts.append("")
    parts.append("Return JSON only, matching the schema in the system prompt.")
    return "\n".join(parts)


def _validate_outline(payload: dict, request: dict, profile: dict) -> str | None:
    """Return None if the outline is valid, else an error string."""
    if not isinstance(payload, dict):
        return "top-level payload is not a dict"
    outline = payload.get("outline")
    if not isinstance(outline, dict):
        return "missing 'outline' object"

    duration_weeks = outline.get("duration_weeks")
    if not isinstance(duration_weeks, int) or not (1 <= duration_weeks <= _MAX_PLAN_WEEKS):
        return f"duration_weeks must be int in [1, {_MAX_PLAN_WEEKS}]"

    start_date = outline.get("start_date")
    if not isinstance(start_date, str) or not _is_iso_date(start_date):
        return "start_date must be YYYY-MM-DD"
    try:
        sd = _date_cls.fromisoformat(start_date)
        if sd.weekday() != 0:
            return f"start_date {start_date} is not a Monday"
    except ValueError:
        return f"start_date {start_date} is not a valid date"

    phases = outline.get("phases")
    if not isinstance(phases, list) or not phases:
        return "phases must be a non-empty list"

    seen_weeks: set[int] = set()
    for idx, phase in enumerate(phases):
        if not isinstance(phase, dict):
            return f"phases[{idx}] is not a dict"
        weeks = phase.get("weeks")
        if not isinstance(weeks, list) or not weeks:
            return f"phases[{idx}].weeks must be a non-empty list"
        for w in weeks:
            if not isinstance(w, int) or w < 1 or w > duration_weeks:
                return f"phases[{idx}].weeks contains invalid week {w!r}"
            if w in seen_weeks:
                return f"phases[{idx}].weeks repeats week {w}"
            seen_weeks.add(w)
        if not isinstance(phase.get("phase"), str) or not phase["phase"]:
            return f"phases[{idx}].phase must be a non-empty string"

    if seen_weeks != set(range(1, duration_weeks + 1)):
        missing = sorted(set(range(1, duration_weeks + 1)) - seen_weeks)
        return f"phases do not cover weeks {missing}"

    template = outline.get("weekly_template")
    if not isinstance(template, dict):
        return "weekly_template must be a dict"
    for day in _WEEKDAYS:
        slot = template.get(day)
        if not isinstance(slot, str) or slot not in _VALID_SLOT_TYPES:
            return f"weekly_template.{day} must be one of {sorted(_VALID_SLOT_TYPES)}"

    constraints = profile.get("constraints") or {}
    expected_days = constraints.get("training_days_per_week") or 5
    non_rest = sum(1 for d in _WEEKDAYS if template[d] != "rest")
    if non_rest != expected_days:
        return (
            f"weekly_template has {non_rest} training days; "
            f"athlete expects {expected_days}"
        )

    # sport_per_day validation. Only ENFORCE shape when the profile is
    # multi-sport; for single-sport profiles a missing sport_per_day is
    # tolerated and the executor falls back to the lone sport. The
    # planner prompt still asks for it, but we do not block on it for
    # single-sport (backward-compat).
    available_sports = (
        constraints.get("available_sports")
        or profile.get("sports")
        or ["running"]
    )
    available_lower = [s.lower() for s in available_sports]
    multi_sport = len(set(available_lower)) > 1
    sport_per_day = outline.get("sport_per_day")
    if multi_sport:
        if not isinstance(sport_per_day, dict):
            return (
                "sport_per_day must be a dict when available_sports has "
                ">1 entry (got %r)" % type(sport_per_day).__name__
            )
        for day in _WEEKDAYS:
            sport = sport_per_day.get(day)
            if not isinstance(sport, str):
                return f"sport_per_day.{day} must be a string"
            sport_lc = sport.lower()
            if template[day] == "rest":
                if sport_lc != "rest":
                    return (
                        f"sport_per_day.{day} must be 'rest' when "
                        f"weekly_template.{day} is rest"
                    )
            else:
                if sport_lc == "rest":
                    return (
                        f"sport_per_day.{day} cannot be 'rest' when "
                        f"weekly_template.{day} is {template[day]!r}"
                    )
                if sport_lc not in available_lower:
                    return (
                        f"sport_per_day.{day}={sport_lc!r} not in "
                        f"available_sports {available_lower}"
                    )
    elif sport_per_day is not None and not isinstance(sport_per_day, dict):
        return "sport_per_day must be a dict when present"

    ack = payload.get("constraints_acknowledged")
    if not isinstance(ack, dict):
        return "constraints_acknowledged must be a dict"

    return None


# ---------------------------------------------------------------------------
# Executor stage
# ---------------------------------------------------------------------------

def _run_executor_for_week(
    outline: dict,
    week_index: int,
    profile: dict,
    language: str,
    model: str,
    meta: dict,
) -> list[dict]:
    """Invoke the executor LLM for one week. Returns session dicts."""
    system_prompt = _load_prompt("executor_system.md")
    week_slot = _build_week_slot(outline, week_index, profile)
    user_message = _build_executor_user_message(week_slot, language)

    # Audit log: typically Haiku (cheap) but operators may override.
    _log_planner_invocation(
        stage=f"executor_week_{week_index}",
        model=model,
        request={"week": week_index, "phase": week_slot.get("phase")},
    )

    last_error: str | None = None
    for attempt in range(1, _EXECUTOR_MAX_ATTEMPTS + 1):
        if attempt > 1:
            meta["executor_retries"] += 1

        messages: list[dict] = [{"role": "user", "content": user_message}]
        if last_error:
            messages.append({
                "role": "user",
                "content": (
                    "Your previous response was invalid: "
                    f"{last_error}\n\nFix the error and return ONLY the JSON."
                ),
            })

        try:
            response = chat_completion(
                messages=messages,
                system_prompt=system_prompt,
                temperature=0.5,
                model=model,
            )
        except Exception as exc:
            last_error = f"LLM call failed: {exc}"
            continue

        content = response.choices[0].message.content if response.choices else ""
        try:
            payload = extract_json(content or "")
        except ValueError as exc:
            last_error = f"could not parse JSON: {exc}"
            continue

        sessions = payload.get("sessions") if isinstance(payload, dict) else None
        if not isinstance(sessions, list) or not sessions:
            last_error = "missing or empty 'sessions' list"
            continue

        sanitized, err = _sanitize_executor_sessions(
            sessions, week_slot, profile,
        )
        if err:
            last_error = err
            continue

        return sanitized

    raise _ExecutorError(f"executor exhausted retries: {last_error or 'unknown error'}")


def _build_week_slot(outline: dict, week_index: int, profile: dict) -> dict:
    """Build the executor's input slot for a given week."""
    phase = _find_phase_for_week(outline["phases"], week_index)
    start_date = _date_cls.fromisoformat(outline["start_date"])
    week_start = start_date + timedelta(days=(week_index - 1) * 7)
    constraints = profile.get("constraints") or {}

    slot: dict = {
        "week_index": week_index,
        "week_start_date": week_start.isoformat(),
        "phase": phase["phase"],
        "weekly_volume_km": phase.get("weekly_volume_km"),
        "intensity_distribution": phase.get("intensity_distribution"),
        "phase_focus": phase.get("focus"),
        "weekday_template": dict(outline["weekly_template"]),
        "constraints": {
            "max_session_minutes": constraints.get("max_session_minutes", 90),
            "training_days_per_week": constraints.get("training_days_per_week", 5),
            "available_sports": constraints.get("available_sports")
                or profile.get("sports")
                or ["running"],
        },
    }

    # Sprint H: pass through the per-day sport assignment so the executor
    # cannot drift from the planner's multi-sport distribution decision.
    sport_per_day = outline.get("sport_per_day")
    if isinstance(sport_per_day, dict):
        slot["sport_per_day"] = {
            d: (sport_per_day.get(d) or "rest") for d in _WEEKDAYS
        }
    return slot


def _build_executor_user_message(week_slot: dict, language: str) -> str:
    """Build the user-role message for the executor."""
    import json as _json
    return (
        f"language={language}\n\n"
        f"# Week Slot\n```json\n{_json.dumps(week_slot, indent=2)}\n```\n\n"
        "Return a JSON object with a 'sessions' array containing exactly "
        "one entry per non-rest weekday in weekday_template."
    )


def _sanitize_executor_sessions(
    sessions: list,
    week_slot: dict,
    profile: dict,
) -> tuple[list[dict], str | None]:
    """Validate and lightly fix executor-produced sessions.

    Returns (sanitized_sessions, error). On error returns ([], error_str).
    """
    template: dict[str, str] = week_slot["weekday_template"]
    expected_days = {d for d in _WEEKDAYS if template[d] != "rest"}
    week_start = _date_cls.fromisoformat(week_slot["week_start_date"])
    max_minutes = int(week_slot["constraints"]["max_session_minutes"])
    available_sports = week_slot["constraints"]["available_sports"] or ["running"]
    available_lower = [s.lower() for s in available_sports]
    # Sprint H: the planner's outline assigns sport per day. If present,
    # the executor's per-session `sport` field is overridden to whatever
    # the planner decided so the multi-sport distribution stays correct.
    sport_per_day: dict[str, str] = week_slot.get("sport_per_day") or {}

    sanitized: list[dict] = []
    seen_days: set[str] = set()

    for entry in sessions:
        if not isinstance(entry, dict):
            return [], "session entry is not a dict"
        day = (entry.get("day") or "").lower().strip()
        if day not in _WEEKDAYS:
            return [], f"invalid weekday '{day}'"
        if template[day] == "rest":
            # Skip rest-day sessions the executor accidentally produced.
            continue
        if day in seen_days:
            return [], f"duplicate session for {day}"
        seen_days.add(day)

        offset = _WEEKDAY_TO_OFFSET[day]
        date_str = (week_start + timedelta(days=offset)).isoformat()

        duration = entry.get("duration_minutes")
        try:
            duration_int = int(duration)
        except (TypeError, ValueError):
            return [], f"{day}: duration_minutes is not numeric"
        if duration_int <= 0:
            return [], f"{day}: duration_minutes must be positive"
        if duration_int > max_minutes:
            return [], (
                f"{day}: duration_minutes {duration_int} exceeds "
                f"max_session_minutes {max_minutes}"
            )

        # Sprint H: prefer the planner's per-day sport assignment over the
        # executor's free-text guess. This eliminates the failure mode
        # where Haiku defaults every day to running.
        assigned = (sport_per_day.get(day) or "").lower()
        if assigned and assigned != "rest" and assigned in available_lower:
            sport = assigned
        else:
            sport = (entry.get("sport") or available_sports[0] or "running").lower()
            if available_lower and sport not in available_lower:
                sport = available_lower[0]

        intensity = (entry.get("intensity") or "low").lower()
        if intensity not in {"low", "moderate", "high"}:
            intensity = "low"

        clean: dict = {
            "day": day,
            "date": date_str,
            "sport": sport,
            "name": entry.get("name") or _default_name(template[day]),
            "description": entry.get("description") or "",
            "duration_minutes": duration_int,
            "intensity": intensity,
        }
        if isinstance(entry.get("steps"), list):
            clean["steps"] = entry["steps"]
        if entry.get("notes"):
            clean["notes"] = entry["notes"]
        sanitized.append(clean)

    if seen_days != expected_days:
        missing = sorted(expected_days - seen_days)
        return [], f"missing sessions for: {missing}"

    # Stable order Mon..Sun.
    sanitized.sort(key=lambda s: _WEEKDAY_TO_OFFSET[s["day"]])
    return sanitized, None


def _default_name(slot_type: str) -> str:
    return {
        "easy": "Easy Run",
        "quality": "Quality Session",
        "long": "Long Run",
        "long_or_quality": "Key Session",
        "cross": "Cross-Training",
        "rest": "Rest",
    }.get(slot_type, "Training")


# ---------------------------------------------------------------------------
# Assembly + validation
# ---------------------------------------------------------------------------

def _assemble_plan(
    request: dict,
    outline: dict,
    sessions: list[dict],
    meta: dict,
) -> dict:
    """Stitch executor output into the final save_plan-compatible dict."""
    focus = _summarize_focus(outline, request)

    return {
        "start_date": outline["start_date"],
        "focus": focus,
        "sessions": sessions,
        "outline": outline,
        "_generation_meta": {
            "mode": "plan_and_execute",
            "planner_model": meta["planner_model"],
            "executor_model": meta["executor_model"],
            "weeks": meta["weeks"],
            "session_count": len(sessions),
            "planner_attempts": meta["planner_attempts"],
            "executor_retries": meta["executor_retries"],
            "generated_at": datetime.now().isoformat(),
        },
    }


def _summarize_focus(outline: dict, request: dict) -> str:
    """Compact human-readable plan focus label."""
    if isinstance(request.get("focus"), str) and request["focus"].strip():
        return request["focus"].strip()
    goal = outline.get("goal_event") or request.get("goal_event")
    weeks = outline.get("duration_weeks")
    if goal and weeks:
        return f"{weeks}-week build to {goal}"
    if weeks:
        return f"{weeks}-week training block"
    return "Training plan"


def _validate_sport_distribution(
    plan: dict,
    profile: dict,
    outline: dict,
    request: dict,
) -> list[str]:
    """Per-week multi-sport floor check (Sprint H).

    Returns an empty list when the plan is OK, otherwise a list of human
    readable error strings. The caller decides whether to retry the
    planner or fail closed.

    Single-sport profiles ALWAYS pass (no constraint to violate).

    Multi-sport profiles are validated against ``_MULTI_SPORT_FLOORS``
    keyed by triathlon discipline. Unknown discipline -> default
    ``unknown_triathlon`` floors (>=1 per declared sport).
    """
    constraints = profile.get("constraints") or {}
    available_sports = [
        s.lower()
        for s in (constraints.get("available_sports")
                  or profile.get("sports") or [])
    ]
    if len(set(available_sports)) < 2:
        return []  # single-sport: nothing to validate.

    discipline = _detect_triathlon_discipline(request, outline) or "unknown_triathlon"
    floors = _MULTI_SPORT_FLOORS.get(discipline, _MULTI_SPORT_FLOORS["unknown_triathlon"])

    # Only enforce floors for sports the athlete ACTUALLY declared.
    enforced = {
        sport: floor for sport, floor in floors.items() if sport in available_sports
    }
    if not enforced:
        return []

    sessions = plan.get("sessions") or []
    try:
        start_date = _date_cls.fromisoformat(plan["start_date"])
    except (KeyError, TypeError, ValueError):
        return ["plan missing valid start_date"]
    duration_weeks = outline.get("duration_weeks") or 1

    # week_index (1-based) -> sport -> count
    per_week: dict[int, dict[str, int]] = {
        wk: {sport: 0 for sport in enforced}
        for wk in range(1, duration_weeks + 1)
    }
    for s in sessions:
        try:
            d = _date_cls.fromisoformat(s.get("date"))
        except (TypeError, ValueError):
            continue
        wk = (d - start_date).days // 7 + 1
        if wk not in per_week:
            continue
        sport = (s.get("sport") or "").lower()
        if sport in enforced:
            per_week[wk][sport] = per_week[wk].get(sport, 0) + 1

    errors: list[str] = []
    for wk in sorted(per_week.keys()):
        counts = per_week[wk]
        deficits: list[str] = []
        for sport, floor in enforced.items():
            got = counts.get(sport, 0)
            if got < floor:
                deficits.append(f"{sport} {got}/{floor}")
        if deficits:
            errors.append(f"week {wk}: " + ", ".join(deficits))
    if errors:
        errors.insert(
            0,
            f"sport-mix floor violation (discipline={discipline}, "
            f"floors={enforced}):",
        )
    return errors


def _validate_against_constraints(
    plan: dict,
    profile: dict,
    outline: dict,
) -> list[str]:
    """Return a list of validation errors (empty means valid)."""
    errors: list[str] = []
    constraints = profile.get("constraints") or {}
    max_minutes = constraints.get("max_session_minutes") or 90
    training_days = constraints.get("training_days_per_week") or 5
    available_sports = {
        s.lower()
        for s in (constraints.get("available_sports")
                  or profile.get("sports") or [])
    }

    sessions = plan.get("sessions") or []
    start_date = _date_cls.fromisoformat(plan["start_date"])
    duration_weeks = outline["duration_weeks"]
    end_date = start_date + timedelta(days=duration_weeks * 7)

    # Per-session checks.
    for s in sessions:
        if s.get("duration_minutes", 0) > max_minutes:
            errors.append(
                f"session {s.get('date')} duration "
                f"{s.get('duration_minutes')}min exceeds cap {max_minutes}"
            )
        if available_sports:
            sport = (s.get("sport") or "").lower()
            if sport and sport not in available_sports:
                errors.append(
                    f"session {s.get('date')} sport '{sport}' "
                    f"not in available_sports {sorted(available_sports)}"
                )
        try:
            d = _date_cls.fromisoformat(s.get("date"))
        except (TypeError, ValueError):
            errors.append(f"session has invalid date {s.get('date')!r}")
            continue
        if not (start_date <= d < end_date):
            errors.append(
                f"session date {d} outside plan range "
                f"[{start_date}, {end_date})"
            )

    # Per-week count check.
    week_counts: dict[int, int] = {}
    for s in sessions:
        try:
            d = _date_cls.fromisoformat(s.get("date"))
        except (TypeError, ValueError):
            continue
        wk = (d - start_date).days // 7 + 1
        if 1 <= wk <= duration_weeks:
            week_counts[wk] = week_counts.get(wk, 0) + 1

    for wk in range(1, duration_weeks + 1):
        got = week_counts.get(wk, 0)
        if got != training_days:
            errors.append(
                f"week {wk} has {got} sessions; expected {training_days}"
            )

    return errors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _infer_duration_weeks(plan_request: dict) -> int | None:
    """Return duration in weeks from any of: duration_weeks, weeks, sessions count."""
    if not isinstance(plan_request, dict):
        return None
    raw = plan_request.get("duration_weeks") or plan_request.get("weeks")
    if isinstance(raw, int) and raw > 0:
        return min(raw, _MAX_PLAN_WEEKS)
    if isinstance(raw, str):
        try:
            n = int(raw)
            if n > 0:
                return min(n, _MAX_PLAN_WEEKS)
        except ValueError:
            pass
    sessions = plan_request.get("sessions")
    if isinstance(sessions, list) and sessions:
        # Each session has a `date`; span the range, round up.
        dates = []
        for s in sessions:
            if isinstance(s, dict) and isinstance(s.get("date"), str):
                try:
                    dates.append(_date_cls.fromisoformat(s["date"]))
                except ValueError:
                    pass
        if dates:
            span_days = (max(dates) - min(dates)).days + 1
            return max(1, (span_days + 6) // 7)
    return None


def _normalize_start_date(raw: str | None) -> str:
    """Return a Monday in YYYY-MM-DD format. Defaults to next Monday."""
    if isinstance(raw, str) and _is_iso_date(raw):
        try:
            d = _date_cls.fromisoformat(raw)
            if d.weekday() == 0:
                return d.isoformat()
            # Snap to the next Monday.
            delta = (7 - d.weekday()) % 7
            return (d + timedelta(days=delta or 7)).isoformat()
        except ValueError:
            pass
    today = _date_cls.today()
    delta = (7 - today.weekday()) % 7
    return (today + timedelta(days=delta or 7)).isoformat()


def _is_iso_date(s: str) -> bool:
    return bool(re.match(r"^\d{4}-\d{2}-\d{2}$", s))


def _find_phase_for_week(phases: list[dict], week_index: int) -> dict:
    for phase in phases:
        if week_index in phase.get("weeks", []):
            return phase
    # Should not happen if outline validation passed, but guard anyway.
    raise _ExecutorError(f"no phase covers week {week_index}")


def _resolve_language(request: dict, profile: dict) -> str:
    raw = request.get("language") or profile.get("language") or "de"
    raw = str(raw).lower()
    return "en" if raw.startswith("en") else "de"


def _resolve_model(role: str, explicit: str | None) -> str:
    """Resolve the model for a stage.

    Resolution order:
      1. Explicit argument.
      2. Optional model_router (Feature 1) if available.
      3. ATHLETLY_<ROLE>_MODEL env var.
      4. Hardcoded default.
    """
    if explicit:
        return explicit

    # Optional model_router lookup. Soft-imported so we do not couple to
    # a feature that may not be merged yet.
    try:
        from src.agent.model_router import route as _route  # type: ignore[import-not-found]
        routed = _route("planning" if role == "planner" else "executor")
        if isinstance(routed, str) and routed:
            return routed
    except Exception:
        pass

    env_key = "ATHLETLY_PLANNER_MODEL" if role == "planner" else "ATHLETLY_EXECUTOR_MODEL"
    env_val = os.environ.get(env_key)
    if env_val:
        return env_val

    return _DEFAULT_PLANNER_MODEL if role == "planner" else _DEFAULT_EXECUTOR_MODEL


def _load_prompt(filename: str) -> str:
    """Read a prompt file from src/agent/prompts/."""
    path = Path(__file__).parent / "prompts" / filename
    return path.read_text(encoding="utf-8")


def _emit_progress(
    cb: Callable[[str, str], None] | None,
    event_type: str,
    detail: str,
) -> None:
    if cb is None:
        return
    try:
        cb(event_type, detail)
    except Exception:
        # Progress callback must never break the pipeline.
        logger.debug("Progress callback raised; ignoring.", exc_info=True)


def _log_planner_invocation(stage: str, model: str, request: dict) -> None:
    """Emit a structured audit line for a planner/executor LLM call.

    Mirrors the format of ``model_router.log_choice`` so premium spend
    can be greppable from a single log query: a router call shows up
    as ``model_router decision tier=complex model=... premium=True``;
    a planner call shows up as ``planner_invocation stage=... model=...
    premium=...``. The premium flag is computed locally because the
    planner sets ``model=`` directly and bypasses the router.
    """
    model_lower = (model or "").lower()
    is_premium = "sonnet" in model_lower or "opus" in model_lower
    logger.info(
        "planner_invocation stage=%s model=%s premium=%s reason=long_horizon_plan",
        stage,
        model,
        is_premium,
    )
