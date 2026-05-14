# Feature 4: Plan-and-Execute Pattern - Design

## Overview

Plan-and-Execute kicks in conditionally inside `save_plan`. The agent's tool surface stays identical: it still calls `save_plan(plan=<dict>)`. The new behaviour lives behind the tool boundary, transparent to the model.

Two paths inside `save_plan`:

1. **Direct path** (default, unchanged): the agent supplied a complete plan dict. We persist it as-is. Free tier and short Pro plans land here.
2. **Plan-and-Execute path** (new, opt-in via heuristic): the agent supplied a SKINNY plan or asked for a long horizon AND the user is Pro AND the heuristic threshold is crossed. We invoke `planner.generate_plan(...)` which runs Sonnet planner + Haiku executor + validator + assembly, then persists the result.

The agent can also call the path explicitly by adding a `generate=True` flag (or by passing a partial plan that only contains `duration_weeks` + `goal`). For backward compatibility we keep `generate` optional and default-off.

## Trigger (complexity heuristic)

```
def should_use_plan_and_execute(plan_request: dict, profile: dict, tier: str) -> bool:
    if tier != "pro":
        return False
    weeks = plan_request.get("duration_weeks") or _infer_weeks(plan_request)
    days = profile.get("constraints", {}).get("training_days_per_week") or 5
    session_count = (weeks or 0) * days
    return session_count >= 14
```

- Threshold: `session_count >= 14`. ~2 to 3 weeks at typical day counts.
- Free tier never triggers (clamped to 1 week elsewhere).
- Tier is read from `user_model.project_profile().get("tier")` with default `"free"`. (Tier infrastructure lives outside this feature; we add a single read with a safe default. Future Feature 1 model_router can override.)

## Planner output schema (structured intermediate)

```python
{
    "outline": {
        "duration_weeks": 16,
        "start_date": "2026-06-01",
        "goal_event": "Berlin Marathon",
        "goal_date": "2026-09-27",
        "phases": [
            {
                "weeks": [1, 2, 3, 4],
                "phase": "Base",
                "weekly_volume_km": 60,
                "intensity_distribution": "80/20",
                "focus": "aerobic capacity, easy mileage, drills"
            },
            {
                "weeks": [5, 6, 7, 8],
                "phase": "Build",
                "weekly_volume_km": 70,
                "intensity_distribution": "75/25",
                "focus": "threshold + tempo, long-run depth"
            },
            {
                "weeks": [9, 10, 11, 12],
                "phase": "Peak",
                "weekly_volume_km": 75,
                "intensity_distribution": "70/30",
                "focus": "race-pace specificity, VO2max intervals"
            },
            {
                "weeks": [13, 14, 15],
                "phase": "Race-Prep",
                "weekly_volume_km": 60,
                "intensity_distribution": "75/25",
                "focus": "marathon-pace blocks, dress rehearsal long run"
            },
            {
                "weeks": [16],
                "phase": "Taper",
                "weekly_volume_km": 35,
                "intensity_distribution": "85/15",
                "focus": "freshness, race-week sharpening, openers"
            }
        ],
        "weekly_template": {
            "monday": "easy",
            "tuesday": "quality",
            "wednesday": "easy",
            "thursday": "long_or_quality",
            "friday": "rest",
            "saturday": "long",
            "sunday": "easy"
        }
    },
    "constraints_acknowledged": {
        "max_session_minutes": 90,
        "training_days_per_week": 5,
        "available_sports": ["running"]
    }
}
```

Key invariants the planner MUST satisfy (validated before executor runs):
- `phases[*].weeks` covers `1..duration_weeks` exactly once, no gaps, no overlaps.
- `weekly_template` has the 7 weekday keys.
- Number of non-rest weekdays in `weekly_template` matches `constraints_acknowledged.training_days_per_week`.
- `constraints_acknowledged` echoes the athlete's actual constraints (the planner is REQUIRED to read them in).

If the planner output fails this validation, we retry once with the validation error fed back, then fall back to the direct (one-shot) path with a logged warning.

## Executor

Input per call: outline (full) + a single `WeekSlot`:

```python
WeekSlot = {
    "week_index": 1,
    "week_start_date": "2026-06-01",
    "phase": "Base",
    "weekly_volume_km": 60,
    "intensity_distribution": "80/20",
    "weekday_template": {"monday": "easy", ..., "sunday": "easy"},
    "constraints": {"max_session_minutes": 90, "training_days_per_week": 5},
}
```

Output per call: list of session dicts for that week, conforming to `save_plan`'s `sessions[]` schema. The executor system prompt is terse: "Fill this week. Do not invent new days. Respect max_session_minutes. Match the slot type for each weekday."

We run executor calls **sequentially** for Phase 1 ship (predictable order, single retry loop, easy progress events). A `parallel=True` flag is plumbed but defaults False, ready for follow-up work.

## Assembly

The assembler stitches per-week session lists into the final `save_plan` payload:

```python
{
    "start_date": outline["start_date"],
    "focus": _summarize_focus(outline),
    "sessions": flat_list_of_all_session_dicts,
    "outline": outline,                 # preserved for transparency / replan
    "_generation_meta": {
        "mode": "plan_and_execute",
        "planner_model": "...",
        "executor_model": "...",
        "weeks": duration_weeks,
        "session_count": len(sessions),
    },
}
```

`start_date` is computed: prefer `outline.start_date`, else "next Monday from today". Each session is stamped with `date` = `start_date + (week_index - 1) * 7 + weekday_offset`.

## Validation against athlete constraints

Pre-`save_plan` final check (raises if violated):
- For every session: `duration_minutes <= constraints.max_session_minutes`.
- For every week: count of sessions with non-zero `duration_minutes` and `intensity != "rest"` equals `constraints.training_days_per_week`.
- `sport` of every session is in `constraints.available_sports` (when set).
- Every `date` is a valid YYYY-MM-DD and lies inside `[start_date, start_date + duration_weeks * 7)`.

On failure: try one self-repair pass (ask executor to fix the specific week), then fall back to inline mode with a clear error in the result.

## Integration with `save_plan`

The existing tool keeps its signature. Internally:

```python
def save_plan(plan: dict, generate: bool | None = None) -> dict:
    if _should_invoke_planner(plan, user_model, settings, generate):
        plan = generate_training_plan(
            request=plan,
            user_model=user_model,
            settings=settings,
            on_progress=_progress_relay,
        )
    # existing persistence path follows unchanged
    ...
```

`generate` flag is documented in the tool schema as optional. The agent doesn't need to set it: the heuristic looks at `plan["duration_weeks"]` (if present) or the session count (if sessions[] is supplied but skinny). Backward compatible: existing callers that pass a full week with `sessions[]` always land in the direct path.

## Model selection

- Planner: Sonnet via `model_router.route("planning")` if module exists, else `anthropic/claude-sonnet-4-5-20250929` (env override: `ATHLETLY_PLANNER_MODEL`).
- Executor: Haiku via `model_router.route("executor")` if module exists, else `anthropic/claude-haiku-4-5-20251001` (env override: `ATHLETLY_EXECUTOR_MODEL`).

Resolution is lazy and wrapped: if a model_router import fails, fall back to env / hardcoded defaults. No import-time coupling.

## Failure modes

| Failure | Response |
|---|---|
| Planner returns invalid JSON | Retry once with parser error fed back. If second attempt fails: fall back to inline. |
| Planner output fails outline validation | Retry once with validation message. Then fall back to inline. |
| Executor returns malformed session | Retry just that week once. Then patch with a "rest" filler for missing slots and log a warning. |
| Assembled plan fails athlete-constraint validation | Retry the offending week. Then fall back to inline. |
| LLM call raises (network, rate limit) | Bubble up. The existing `save_plan` error path handles the user message. |

Every failure logs a structured event so we can audit fallback frequency.

## Files

- `src/agent/planner.py` (NEW): `generate_training_plan`, `PlannerOutline`, `WeekSlot`, validators, retry logic. ~400 lines.
- `src/agent/prompts/planner_system.md` (NEW): Sonnet planner system prompt.
- `src/agent/prompts/executor_system.md` (NEW): Haiku executor system prompt, terse.
- `src/agent/tools/planning_tools.py` (modified): `save_plan` consults the heuristic and may invoke `generate_training_plan` before persisting.
- `tests/test_planner.py` (NEW): heuristic threshold, planner output validation, executor output validation, assembly correctness, constraint validation, fallback paths, free-tier 1-week clamp.

## Out of scope (deferred)

- Parallel executor calls. Plumbed via flag, default off, ship sequentially.
- Re-plan flow ("athlete missed week 3, regenerate from week 4"). The outline is preserved in `_generation_meta` so this is easy follow-up.
- Adaptive plan selection (different planner prompt per sport). v1 ships one generalist planner prompt; sport-specific knowledge already lives in the static system prompt.
