# Sprint B - Root Cause

## TL;DR

The Plan-and-Execute heuristic did NOT fire for Lisa because two
independent gates each failed:

1. The agent passed a fully inlined plan dict to `save_plan`. The dict
   has no `duration_weeks` and no `weeks` key, just `sessions[]`. The
   heuristic falls back to date-span inference. The agent only inlined
   2 weeks of sessions, so the inferred duration is 2 weeks.
2. With `training_days_per_week = 6` and `duration_weeks = 2`,
   `2 * 6 = 12`, which is BELOW the `_COMPLEXITY_SESSION_THRESHOLD =
   14`. `should_use_plan_and_execute` returns False (planner.py:127).

So `save_plan` never invoked `generate_training_plan`, never called
Sonnet, and silently persisted the 2-week inline Haiku output.

The root cause is therefore an architectural one: **the agent never
gets a chance to ask for the long-horizon path.** The trigger fires
only AFTER the agent has already committed to writing a full
multi-week plan inline, and only if the agent happened to write enough
of it. The agent has no incentive to write 16 weeks of sessions
inline; it always shortens to 1-2 weeks and the heuristic always
under-fires.

## The exact failure chain (with line numbers)

`src/agent/tools/planning_tools.py:179-188`:

```
from src.agent.planner import (
    generate_training_plan,
    should_use_plan_and_execute,
)
profile = user_model.project_profile()

if should_use_plan_and_execute(plan, profile):
    logger.info(
        "save_plan: invoking Plan-and-Execute (weeks=%s)",
        plan.get("duration_weeks"),
    )
```

`should_use_plan_and_execute` is called with the agent's plan dict.
For Lisa's saved plan that dict was:

```
{
  "start_date": "2026-05-18",
  "focus": "Build 2: Langdistanz-Spezifik (8 Wochen bis Roth)",
  "sessions": [ ... 16 entries spanning 2026-05-18 to 2026-05-31 ... ]
}
```

No `duration_weeks`, no `weeks`. `_infer_duration_weeks`
(planner.py:764) walks the chain:

- `raw = plan.get("duration_weeks") or plan.get("weeks")` -> None
- Both isinstance branches skipped
- Falls to the sessions[] branch: `dates` length 14, span 14 days,
  `(14 + 6) // 7 = 2` -> returns 2

`should_use_plan_and_execute` (planner.py:99-127):

```
weeks = _infer_duration_weeks(plan_request)   # 2
days  = constraints['training_days_per_week'] # 6 for Lisa
return 2 * 6 >= 14                            # 12 >= 14 -> False
```

False. No planner. Inline path persists the 16-session dict as is.

## Why the agent never sets `duration_weeks`

The `save_plan` tool description in
`src/agent/tools/planning_tools.py:33-66` does include a SKINNY-request
hint near the bottom:

> Long horizons (Pro tier, multi-week builds): you may pass a SKINNY
> request with just `duration_weeks`, `start_date`, `goal_event`,
> `goal_date`, and optional `focus`.

But:

1. The hint is buried at the end, after the full inline schema. The
   model sees the inline schema first and primarily emits inline
   plans.
2. The `STATIC_SYSTEM_PROMPT` `## Plan Workflow` section
   (`system_prompt.py:89-107`) ONLY teaches the agent the inline
   pattern: "compose training plans yourself, inline, using your own
   reasoning". There is no nudge toward the skinny shape for
   multi-week builds.
3. `STRICT` rule at `system_prompt.py:237-241` says: "whenever you
   describe a training plan in chat (multiple weeks, structured
   sessions), you MUST persist it via save_plan(plan=<dict>)". This
   actively trains the agent to inline the multi-week plan into the
   dict, because the rule conflates "describing in chat" with "saving
   inline".

Net effect: the agent reads the system prompt, sees "compose inline",
sees "STRICT: any multi-week plan MUST be persisted via save_plan(...)",
and writes a multi-week plan inline. Because per-week composition is
token-expensive on the main loop, it usually cuts off after 1-2 weeks.
The heuristic then refuses to fire because the truncated inline plan
has too few sessions to cross the threshold.

## Secondary issue: heuristic depends on a post-hoc inference

`_infer_duration_weeks` falls back to `sessions[]` date-span inference.
This is fundamentally broken for the trigger use case: by the time
sessions exist, the agent has already paid the token cost of inlining
the plan. The heuristic is being asked "should I have escalated?" after
escalation is moot.

For a robust trigger the heuristic must read the AGENT'S INTENT
(`duration_weeks` from a skinny request) rather than INFER from a
post-hoc artifact.

## Tertiary issue: model_router log was silent

There were NO `model_router decision tier=... model=...` log lines in
the production container during Lisa's session. Confirmation that the
Sonnet path was never entered.

Note: the planner code path in `_run_planner` at planner.py:296 calls
`chat_completion(..., model=planner_model_resolved)`. Because `model=`
is set, `chat_completion` BYPASSES the router (llm.py:199 `if model:`)
and never emits a `model_router decision` line. So even if the planner
HAD fired we would not see a router log line for it. That makes the
premium routing un-auditable from container logs alone. A separate
log inside `_run_planner` would be needed; today there's only `logger.info("save_plan: invoking Plan-and-Execute ...")` which would NOT have fired in Lisa's case either.

## Fallback path safety

If `generate_training_plan` HAD fired and Sonnet had errored, the
existing fallback at planning_tools.py:194-200 is:

```
if result.mode == "plan_and_execute":
    plan = result.plan
else:
    logger.warning(
        "Plan-and-Execute fell back to inline: %s",
        result.meta.get("fallback_reason"),
    )
```

It silently re-uses the agent's incoming (skinny) plan dict and persists
that. The athlete would have received a near-empty plan with the user
NEVER being told a multi-week build failed. This is a separate bug
that masks the failure mode; the spec asks us to surface failures, so
the fix design should change this too.

## Summary of identified bugs

| # | Where | What | Severity |
|---|---|---|---|
| 1 | system_prompt.py:89-107 + 237-241 | No prompt nudge for skinny multi-week requests | HIGH (root cause) |
| 2 | planning_tools.py:33-66 | Skinny path documented but buried | MEDIUM |
| 3 | planner.py:99-127 | Heuristic relies on post-hoc inference; no explicit "long horizon" gate | HIGH |
| 4 | planning_tools.py:194-200 | Silent fall back to inline on Sonnet failure | HIGH |
| 5 | planner.py:296,489 | Planner uses `model=...`, bypasses router log line; premium calls are not auditable in critic-stats | MEDIUM |
