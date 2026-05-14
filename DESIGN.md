# Iter 2 Sprint I: Design - three fixes

## Bug 1: annotate_activity UUID hallucination

### Symptom

Elena and Lisa Iter 1 transcripts: when the response gate forces
`annotate_activity`, the model invents an id like
`today-easy-run-0515`. The Supabase update runs against that id, the
row does not exist, the call returns no error, and the agent thinks it
saved a note.

### Root cause

Two compounding issues:

1. The tool description says "UUID of the row" but does not say
   "look it up first". Under gate pressure the model picks the path of
   least resistance and synthesises a plausible-looking string.
2. The handler accepts any string and writes to Supabase without
   verifying the row exists. Supabase `.update().eq("id", non_uuid)`
   silently affects zero rows.

### Fix

`src/agent/tools/journal_tools.py`:

- Add UUID-format detection in the handler. Bad format -> return
  `{"error": "...", "hint": "call get_activities(limit=5) first ..."}`
  before touching the DB.
- Rewrite the tool description to lead with a precondition: "Required
  first step: call get_activities(limit=5) to find the real
  activity_id. NEVER pass a synthetic id like 'today-easy-run-0515'."

The validation is intentionally cheap (regex) and runs before the DB
call. The error dict carries the recovery hint so the next turn can
self-correct.

## Bug 2: CSS pace "1:63/100m"

### Symptom

Lisa's persona has `swim_css_min_per_100m: 1.63` (decimal minutes,
= 1 min 38 sec). The journal section gets rendered as
`Schwimm-CSS: 1.63 min/100m`. The model reads that and surfaces it as
`1:63/100m` (which is structurally invalid - 63 sec is impossible).

### Root cause

`tools/persona_test/seed.py` line 219:
`f"- Schwimm-CSS: {persona.swim_css_min_per_100m:.2f} min/100m"`. The
seed writes a decimal-form string. Sprint C fixed this class of bug
for run pace (`threshold_pace_min_km` -> `_pretty` via
`decimal_min_to_mmss`) but CSS was missed.

### Fix

Two surfaces:

1. `tools/persona_test/seed.py`: use `decimal_min_to_mmss(css)` and
   render `Schwimm-CSS: 1:38 /100m`. Decimal stays unprintable; the
   model only sees mm:ss.
2. `src/utils/pace_format.py`: pace_format.py is already correct
   (`decimal_min_to_mmss(1.63) == "1:38"` is what the existing
   `decimal - whole * 60` math returns). Add an explicit test pinning
   the CSS case so the conversion can never regress.

Audit other CSS surfaces: a grep shows only seed.py renders CSS as
a string outside the compute tool. The compute tool's `css_paces`
already uses `_format_mmss`, which is correct.

### Future hardening

Add a tiny utility `swim_pace_to_pretty` that wraps
`decimal_min_to_mmss` with the canonical suffix `/100m` so other
callers cannot drop the suffix or get the units wrong.

## Bug 3: VDOT/threshold reasoning inversion

### Symptom

Elena Iter 1 Turn 2: "VDOT 52 -> Marathon-Pace 4:27, du brauchst 4:59,
also nicht fit genug". 4:27 is FASTER than 4:59 so the verdict is
backwards.

### Root cause

The model retrieved both paces correctly (the lookup tool worked) but
inverted the comparison. Pace strings look like time and "smaller =
faster" is a load-bearing inversion the model misses under reasoning
pressure.

### Fix

Three layers, defence in depth:

1. **Deterministic helper** in `compute_tools.py`. New formula
   `compare_paces(predicted_pace, target_pace)`. Inputs can be mm:ss
   strings or decimal-minute floats; outputs include
   `delta_seconds`, `direction` ("faster" / "slower" / "equal"),
   `magnitude_label`, and a load-bearing `interpretation` the model
   quotes verbatim.

2. **System prompt** STRICT block extension. The existing SPORT MATH
   DISCIPLINE rule expands to:

   "When you discuss VDOT, threshold pace, or compare predicted-vs-target
   paces, you MUST call compute_sport_math(formula='paces_from_vdot')
   and compute_sport_math(formula='compare_paces'). NEVER reason about
   pace comparisons inline. Lower mm:ss values mean FASTER pace.
   Compare seconds-per-km, not raw strings."

3. **Constitutional critic** rule `pace_comparison_directional`. LLM
   judges whether a response compares two paces and gets the direction
   right. Flagged responses regenerate once.

### compare_paces behaviour

Inputs accepted:

- `"4:27"` (mm:ss string)
- `4.45` (decimal minutes, 4:27)
- `267` (seconds, when explicit)

Always returns the same dict shape. Direction interpretation is in
plain language because the LLM mirrors it.

Sample output for `compare_paces(predicted="4:27", target="4:59")`:

```
{
    "status": "success",
    "predicted_pace": "4:27",
    "target_pace": "4:59",
    "predicted_seconds_per_km": 267.0,
    "target_seconds_per_km": 299.0,
    "delta_seconds": -32.0,
    "direction": "faster",
    "magnitude_label": "32 sec/km faster",
    "verdict": "predicted is FASTER than target by 32 sec/km",
    "interpretation": "Vorhergesagte Pace 4:27/km ist 32 sec/km
        SCHNELLER als die Zielpace 4:59/km. Die Athletin liegt
        auf oder vor Zielniveau."
}
```

## Hard rules audit

- No em-dashes anywhere in the new code or strings.
- German umlauts in user-facing strings (interpretation field).
- Pure functions in compute_tools.py - no I/O, no LLM.
- Tool descriptions: short, directive, one example per failure mode.
- All math goes via compute_sport_math, never inline.

## Test plan

- `tests/test_compute_tools.py`: 4 tests for compare_paces (mmss
  strings, decimal minutes, equal paces, error on bad input).
- `tests/test_pace_format.py`: 1 test pinning CSS 1.63 -> "1:38".
- `tests/test_journal_tools.py` (new): 3 tests for annotate_activity
  validation (UUID accepted, slug rejected with hint, empty rejected).
- `tests/test_critic.py`: extend rule-count and id set tests for
  the new `pace_comparison_directional` rule.

## Out of scope

- Re-running Iter 1 persona transcripts. The fixes are unit-tested
  here; live verification happens in Iter 2 staging.
- Backfilling existing Lisa journal entries on hetzner. The next
  seed run rewrites them.
