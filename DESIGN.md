# DESIGN: Sprint H Fixes

Scope: two bugs that hit Lisa's iteration-2 re-test plus a documentation
gap on distance terminology.

## A. Sport-mix constraint for multi-sport profiles

### Data model change

`outline.weekly_template` keeps the existing intensity slot vocabulary
and gains an OPTIONAL sister mapping `sport_per_day`:

```json
"weekly_template": {
  "monday":    "easy",
  "tuesday":   "quality",
  ...
},
"sport_per_day": {
  "monday":    "swimming",
  "tuesday":   "cycling",
  "wednesday": "running",
  "thursday":  "cycling",
  "friday":    "rest",
  "saturday":  "running",
  "sunday":    "long"   // or a sport
}
```

Backward compatibility: when `profile.sports` has a single sport,
`sport_per_day` MAY be omitted. The legacy single-sport path is
unchanged. When `profile.sports` has more than one sport,
`sport_per_day` is REQUIRED.

### Floor heuristic (per recognised discipline)

Hard-coded in `planner.py` as `_MULTI_SPORT_FLOORS`:

| Discipline label | swim | bike | run |
|---|---|---|---|
| long_course (Langdistanz / Ironman / IM / 140.6) | 1 | 2 | 2 |
| middle_distance (Mitteldistanz / 70.3 / Halbdistanz) | 1 | 2 | 2 |
| short_course (Sprint / Olympic / Kurz) | 1 | 1 | 1 |
| unknown_triathlon | 1 | 1 | 1 |

Per WEEK, not per plan. So for an 8-week long-course plan we expect
AT LEAST `8 * 1 = 8 swim sessions`, `8 * 2 = 16 bike sessions`, and
`8 * 2 = 16 run sessions`.

Discipline detection is done by lowercasing `goal_event + focus` and
matching against keywords (`langdistanz` / `ironman` / `140.6` /
`full distance` -> long_course, etc.). Default = `unknown_triathlon`.

### Validation flow

After `_assemble_plan` and BEFORE returning the PlannerResult, run a
new helper `_validate_sport_distribution(plan, profile, outline)`:

- if `len(profile.sports) <= 1`: pass.
- count sessions-per-sport per week.
- determine `discipline` from outline + request.
- assert every week meets the floor; collect missing sports per week.
- on violation: return an error list.

On non-empty errors, the entry point `generate_training_plan` REGENERATES
the plan ONCE with the planner using an explicit user-message annotation:

```
SPORT-MIX VIOLATION on previous attempt:
- week 2: 0 swim, 0 cycling (Langdistanz needs >=1 swim, >=2 cycling).
- week 5: 1 cycling (Langdistanz needs >=2 cycling).
Re-emit the outline with sport_per_day so EVERY week respects the floors.
```

If the SECOND attempt still violates the floors, the run falls back to
`inline_fallback` (caller surfaces an error to the athlete - we do NOT
silently ship a broken plan).

### Prompts

- `planner_system.md`: new section "Sport-mix for multi-sport athletes"
  documents `sport_per_day`, lists the floor heuristic, and forbids
  emitting an outline that violates the floors.
- `executor_system.md`: new rule "If the slot carries a `sport` field,
  use it verbatim. Do NOT substitute another sport from
  available_sports." A consistent sport selection from the OUTLINE is
  what guarantees the post-assembly count is correct.

### Executor / sanitizer changes

`_build_week_slot` now copies the per-day sport assignment from the
outline into each day's slot. `_sanitize_executor_sessions` looks up
the expected sport for each day and overrides the executor's choice if
it disagrees. (Cheaper than asking Haiku to retry.)

## B. Tier propagation - actual cause

Confirmed in production logs: every "complex" call from agent_loop hit
Anthropic with `temperature=0.7` + `thinking=enabled`, which Anthropic
rejects with a 400. The agent_loop's existing fallback path then
retries with `tier="routine"` (Haiku, no thinking) which succeeds. So
every premium-eligible turn was billed as Haiku.

### Fix

In `src/agent/llm.py`, when `is_anthropic and thinking_budget > 0`:

1. Force `kwargs["temperature"] = 1.0` (override the caller-supplied
   temperature). Log a warning at DEBUG level so the override is
   visible in tests.
2. Keep the existing `thinking` block.

Side effect: callers that requested a specific temperature lose it for
thinking-enabled calls. This is fine - extended thinking adds its own
sampling diversity inside the thinking block, and Anthropic's docs are
explicit that t=1 is required.

### Test plan

- Add `tests/test_llm_thinking_temperature.py` that monkeypatches
  `litellm.completion` and asserts that when `tier="complex"` is passed
  through chat_completion AND the resolved model is Anthropic, the
  outgoing kwargs include `temperature == 1.0` regardless of the
  caller-supplied temperature.
- Add a regression to `tests/test_complexity_detector.py` for Lisa's
  exact turn-1 phrasing.

## C. Distance terminology in system prompt

`src/agent/system_prompt.py` STATIC_SYSTEM_PROMPT gains a short section
under "Plan Workflow":

```
## Triathlon distance vocabulary

When the athlete mentions a triathlon distance, use these EXACT
definitions and NEVER swap them:

- Sprint: 0.75 km swim / 20 km bike / 5 km run
- Olympische / Kurzdistanz: 1.5 km / 40 km / 10 km
- Mitteldistanz / 70.3 / Halbdistanz: 1.9 km / 90 km / 21.1 km
- Langdistanz / Ironman / IM / 140.6: 3.8 km / 180 km / 42.2 km

Specific events:
- Challenge Roth = Langdistanz (full Ironman distance, 3.8/180/42.2).
- Ironman 70.3 (any city) = Mitteldistanz.

If the athlete says "Langdistanz" never call it "70.3". If the athlete
says "70.3" never call it Langdistanz. When in doubt, ASK.
```

## Out of scope (deferred)

- Brick session detection (sessions tagged as "bike+run combo"). Today
  every brick reads as a single bike OR single run; the count is correct
  per-session but humans may want a brick called out. Punt to next sprint.
- Per-week volume rebalancing across sports. The current planner already
  emits `weekly_volume_km` per phase; we do not redistribute volume
  between sports yet. Punt.
- Multi-session days (doubles). Sprint H stays on `training_days_per_week`
  = sessions/week. Doubles require an explicit consent flow.
