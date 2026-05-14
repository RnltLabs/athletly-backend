# Sprint B - Fix Design

## Goals

1. The agent KNOWS when to ask for the long-horizon path (clear tool
   description + clear system-prompt nudge).
2. The trigger fires BEFORE the agent commits to writing a full plan
   inline.
3. Premium-model routing is auditable: every Sonnet call emits a log
   line with the reason.
4. On Sonnet failure, surface a clear error. Do NOT silently downgrade
   to inline Haiku output.

## Design

### 1. `should_use_plan_and_execute` becomes intent-driven

Today the heuristic only fires when `duration_weeks * training_days
>= 14`. After the fix it ALSO fires when:

- `plan_request.get("duration_weeks")` (explicit) >= 4, OR
- `plan_request.get("weeks")` (legacy) >= 4, OR
- `_looks_like_long_horizon(plan_request)` matches (free-text keyword
  backstop: "marathon", "ironman", "triathlon", "half marathon",
  "build phase", "race build", and the numeric pattern `\d{1,2}\s*(week|wochen|woche)` in `focus` or `goal_event`).

The existing session-count gate stays as a final fallback for plans
the agent did inline.

Implementation: keep the current logic, ADD two early-return branches:

```python
weeks_explicit = plan_request.get("duration_weeks") or plan_request.get("weeks")
if isinstance(weeks_explicit, int) and weeks_explicit >= 4:
    return True
if _looks_like_long_horizon(plan_request):
    return True
# ... existing logic ...
```

Plus a single private helper `_looks_like_long_horizon` that scans
`focus`, `goal_event`, and `goal` for the patterns above.

### 2. Tool description teaches the skinny path FIRST

`_SAVE_PLAN_DESCRIPTION` in planning_tools.py gets reordered: the
skinny multi-week shape goes FIRST with a strong "DO NOT inline" anchor
and an explicit threshold ("plans of >= 2 weeks"). Inline schema stays
for adjustments and short plans.

Anthropic cookbook recommendation: explicit number + rationale +
negative anchor. The new top of the description reads:

> "Persist a training plan. For plans of more than two weeks
> (multi-week builds, marathon/race buildups, anything >= 4 weeks),
> DO NOT compose sessions inline - pass a SKINNY request:
>    { "duration_weeks": N, "start_date": "YYYY-MM-DD",
>      "goal_event": "...", "goal_date": "YYYY-MM-DD",
>      "focus": "optional label" }
> and the Plan-and-Execute pipeline (Sonnet planner + Haiku per-week
> executor) builds the full plan for you. Inlining beyond 2 weeks
> loses coherence and is forbidden."

Then the inline schema is moved below under "For adjustments and
short (<=2 week) plans".

### 3. System prompt adds a long-horizon plan strategy block

Add a short subsection inside `STATIC_SYSTEM_PROMPT` `## Plan Workflow`
right after the existing inline guidance. It reads:

> "Long-horizon plans (>= 4 weeks): pass a SKINNY save_plan request
> with `duration_weeks`, `start_date`, `goal_event`, `goal_date`,
> and optional `focus`. Never inline more than 2 weeks of sessions -
> the per-week reasoning is delegated to the planner pipeline."

The existing STRICT rule at lines 237-241 is amended to add: "...
unless the plan is >= 3 weeks, in which case use the skinny
duration_weeks request and let save_plan expand it."

### 4. Auditable premium-model logging

Inside `_run_planner` (planner.py:296) and `_run_executor_for_week`
(planner.py:489), emit a structured log line BEFORE the
`chat_completion` call:

```
logger.info(
    "planner_invocation stage=%s model=%s premium=%s reason=%s",
    "planner" or "executor",
    model,
    "anthropic" in model and "sonnet" in model,
    "long_horizon_plan",
)
```

This makes premium spend visible in container logs even though the
planner sets `model=` directly and the router log line never fires.

### 5. Sonnet failure surfaces to the user

`save_plan` today (planning_tools.py:191-200) silently re-uses the
inline path when `result.mode == "inline_fallback"`:

```
if result.mode == "plan_and_execute":
    plan = result.plan
else:
    logger.warning(...)
```

Change: when the heuristic fired AND `result.mode == "inline_fallback"`,
return an explicit `error` dict from save_plan (without persisting
anything). The agent's existing system-prompt rule "If save_plan
returns an error: REPORT THE ERROR TO THE USER" then takes over.

```
if should_use_plan_and_execute(plan, profile):
    result = generate_training_plan(...)
    if result.mode == "plan_and_execute":
        plan = result.plan
    else:
        return {
            "error": (
                "Long-horizon plan generation failed: "
                f"{result.meta.get('fallback_reason', 'unknown')}. "
                "Tell the athlete you couldn't build the multi-week plan "
                "right now and offer to retry or build a 1-2 week block."
            ),
            "mode": "planner_failed",
        }
```

### 6. Backwards compatibility

- Adjustments (small plans, < 14 session slots, no `duration_weeks`):
  unchanged. Still go through the inline path.
- Existing saved plans without `_generation_meta` or `outline`: render
  the same in the UI (the `_coerce_sessions` helper doesn't care).
- The agent's tool surface is unchanged (still one `save_plan(plan=dict)`).

## Test cases (codified in `tests/test_planner.py`)

| Test | Input | Expected |
|---|---|---|
| `test_trigger_fires_on_explicit_duration_weeks` | `{duration_weeks: 16, ...}` + 5 training days | True |
| `test_trigger_fires_on_focus_keywords` | `{focus: "16-week marathon build", sessions: []}` | True |
| `test_trigger_fires_on_goal_event_keyword` | `{goal_event: "Marathon", duration_weeks: 8}` | True |
| `test_trigger_does_not_fire_on_adjustment` | `{sessions: [3 entries one week], start_date: ...}` | False |
| `test_trigger_does_not_fire_on_short_plan` | `{duration_weeks: 1}` | False |
| `test_save_plan_routes_to_planner_when_skinny_request` | `save_plan({duration_weeks: 16, ...})` -> mock `generate_training_plan` returns plan_and_execute mode | persisted plan is the planner output, `_generation_meta.mode == "plan_and_execute"` |
| `test_save_plan_returns_error_on_planner_failure` | mock planner returns `inline_fallback` | save_plan returns dict with `"error"` key, no row persisted |
| `test_inline_short_plan_still_works` | `{sessions: [3 sessions], start_date}` | persisted as inline, no error, no planner invocation |
| `test_long_horizon_detection_german_keywords` | `{focus: "8 Wochen bis Roth"}` | True |
| `test_planner_invocation_log_emitted` | Capture logs around `_run_planner` | log contains `"planner_invocation stage=planner ... premium=True"` |

## Files touched

- `src/agent/planner.py` - heuristic update + audit log lines
- `src/agent/tools/planning_tools.py` - tool description reorder + error envelope on planner failure
- `src/agent/system_prompt.py` - long-horizon plan subsection in plan workflow + amended STRICT rule
- `tests/test_planner.py` - NEW file (the old one was deleted), covers all the cases above
