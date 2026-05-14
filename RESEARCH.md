# Sprint C Research: Pre-format vs. teach-the-LLM

## The failure mode

Profile field `fitness.threshold_pace_min_km: 4.50` is stored as a decimal float. The agent gets it as `4.50` (truthy float) in the runtime-context block:

```
Threshold pace: 4.50 min/km
```

The LLM, even under STRICT rules, reads `4.50` as "four minutes fifty seconds" and answers "deine Schwellen-Pace liegt bei 4:50/km". When Elena corrects ("nein, 4.50 sind decimal minutes = 4:30"), the model doubles down because the only source-of-truth it has IS the decimal.

The existing STRICT rule in `system_prompt.py` (lines 127 to 137) covers the `_pretty` vs. decimal split for `get_activities` results. It does not cover any other surface where a raw decimal escapes.

## Q2 2026 robust pattern: pre-format the input, not the output

The lesson from the Athletly cache work and from the `pace_to_mmss` / `minutes_to_hms` helpers in `src/agent/tools/format_helpers.py`:

> Do not rely on the LLM to do unit conversion. Pre-format every number it sees, leave the conversion code out of its hands entirely.

For the existing `get_activities` tool this is already done: `avg_pace_min_km` is paired with `avg_pace_pretty`. The fix for the profile surface is the same shape but applied to the runtime-context build step.

Defence in depth:

1. PRE-FORMAT at the runtime_context build (Phase 3): the LLM never sees `4.50 min/km`, it sees `4:30 /km`.
2. PRE-FORMAT at tool result level (Phase 1 audit, item 2): `get_athlete_profile` returns `threshold_pace_pretty` next to the decimal.
3. EXPLICIT STRICT RULE (Phase 4): tell the agent the pretty version is the truth, never compute from the decimal.
4. DETERMINISTIC POST-CHECK (Phase 5): a regex catches the failure if the LLM still emits a decimal-pace pattern (`\b[3-9]\.\d{2}\b /km`). Regex cannot be argued past.

## Helper module placement

`src/agent/tools/format_helpers.py` already owns `pace_to_mmss` and `minutes_to_hms`. The task spec calls for `src/utils/pace_format.py`. Decision: add `src/utils/pace_format.py` as the central, framework-agnostic home (pure functions, no agent dependency) and re-export from `format_helpers.py` to keep existing imports stable. This keeps the helpers usable from anywhere (planner, memory, system_prompt, services) without pulling in the tools layer.

## Regex shape for the post-check

The decimal-pace failure pattern is unambiguous in context:

- `4.50/km`, `4.50 /km`, `4.50 min/km`, `4.50min/km`

The minute digit is in `[3..9]` because:
- below 3 min/km is faster than world-record marathon pace (irrelevant)
- a `2.xx` decimal is more often a duration ("2.5 hours") than a pace
- above 9 (10, 11) is unusual; we still want to catch `10.50 min/km`, so widen to `\d{1,2}\.\d{2}` followed by `/km` or `min/km`. The qualifier `/km` makes the context unambiguous regardless of the minute digit.

Final regex: `\b\d{1,2}\.\d{2}\s*(?:min\s*)?/\s*km\b` (case-insensitive). This catches `4.50/km`, `4.50 min/km`, `4.50min/km`, `10.50 /km`. Does not catch `4:50/km` (which is correct format) or `4.50 hours` (not pace).

## Why a deterministic post-check beats an LLM critic

The existing critic (`src/agent/critic.py`) is a Haiku LLM call that fail-opens on errors. It cannot reliably catch this failure because (a) Haiku might also misread `4.50/km` as correct, and (b) under load the critic fail-opens. A regex is microsecond-cheap, never fail-opens, and the failure mode it catches is unambiguous.

Trade-off: regex can false-positive on prose like "the runner went from 5.50/km to 4.30/km" if the speaker really meant decimal minutes. That sentence is itself a bug per our coaching contract: the coach should write "from 5:30/km to 4:18/km". So a false positive there is actually a true positive.
