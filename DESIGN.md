# Sprint C Design: Pace format robustness

## Architecture overview

```
RAW DECIMAL (DB, never changes)
        |
        v
project_profile() -> dict with fitness.threshold_pace_min_km = 4.5
        |
        +-> get_athlete_profile() tool handler ----+
        |        (adds threshold_pace_pretty)      |
        |                                          v
        +-> build_runtime_context() text block --> LLM context
                 (decimal_min_to_mmss applied      (sees "4:30 /km" ONLY)
                  before formatting)
                                                   |
                                                   v
                                            LLM response
                                                   |
                                                   v
                                  scan_response() regex post-check
                                  catches "X.YZ/km" decimal-leak
```

## Module 1: `src/utils/pace_format.py` (NEW)

Pure helpers, no I/O, no agent imports.

```python
def decimal_min_to_mmss(value: float | None) -> str | None: ...
def decimal_min_to_hms(value: float | None) -> str | None: ...
def watts_to_pretty(value: int | float | None) -> str | None: ...
def hr_to_pretty(value: int | float | None) -> str | None: ...
```

Edge cases handled:
- `None` returns `None`.
- Non-numeric / TypeError returns `None`.
- Carry-over: `4.99 min` -> `5:00` (sec rounds to 60, bump minute).
- Sub-minute: `0.5 min` -> `0:30`. `0.0 min` -> `0:00`.
- Negative input: treated as None (paces are non-negative; defensive).
- Hours: `decimal_min_to_hms(75.5)` -> `1:15:30`. `decimal_min_to_hms(7.5)` -> `7:30`.

Re-exports: `src/agent/tools/format_helpers.py` keeps its existing public API (`pace_to_mmss`, `minutes_to_hms`, `minutes_to_hm`) and gains a thin shim that delegates to `pace_format`. Existing call-sites stay untouched.

## Module 2: `build_runtime_context` patch

```python
fitness = profile.get("fitness") or {}
if isinstance(fitness, dict):
    vo2max = fitness.get("estimated_vo2max")
    threshold_decimal = fitness.get("threshold_pace_min_km")
    threshold_pretty = decimal_min_to_mmss(threshold_decimal)
    if vo2max is not None:
        profile_lines.append(f"Estimated VO2max: {vo2max}")
    if threshold_pretty is not None:
        profile_lines.append(f"Threshold pace: {threshold_pretty} /km")
```

Before:

```
Threshold pace: 4.5 min/km
```

After (for Elena, `threshold_pace_min_km=4.50`):

```
Threshold pace: 4:30 /km
```

The decimal NEVER leaves this function in the runtime context body.

## Module 3: `get_athlete_profile` tool handler patch

```python
def get_athlete_profile() -> dict:
    profile = user_model.project_profile()
    # Pre-format fitness fields so the LLM cannot misread decimals.
    fitness = profile.get("fitness")
    if isinstance(fitness, dict):
        pretty = decimal_min_to_mmss(fitness.get("threshold_pace_min_km"))
        if pretty is not None:
            fitness["threshold_pace_pretty"] = pretty
    profile["_has_activities"] = ...
    profile["_onboarding_complete"] = ...
    return profile
```

Keep the decimal for backwards compat (DB shape unchanged); the agent reads the `_pretty` field per the STRICT rule.

## Module 4: STRICT rule extension

Append to the existing "Number formatting (CRITICAL)" block in `STATIC_SYSTEM_PROMPT`:

```
- Profile fitness fields in runtime_context are ALREADY pre-formatted
  as mm:ss strings (e.g. "Threshold pace: 4:30 /km"). Use them
  verbatim. Never reinterpret. Never compute paces from raw decimals.
  If get_athlete_profile returns both threshold_pace_min_km (decimal)
  and threshold_pace_pretty (string), quote the _pretty value to the
  athlete; the decimal exists only for internal math.
```

## Module 5: Critic rule `pace_format_correct`

New rule id `pace_format_correct` added to `RULE_IDS` in `src/services/critic_metrics.py`. The critic's LLM-side description in `src/agent/critic.py`:

```
"pace_format_correct: pace values in response_text MUST be in mm:ss "
"notation with a colon, e.g. '4:30/km' or '4:30 /km'. A decimal-form "
"pace like '4.50/km' or '4.50 min/km' is a STRICT violation. The "
"profile decimals are pre-converted; the model MUST quote the pretty "
"form verbatim."
```

This brings the count from 8 to 9 rules. The defensive assertion in `critic.py` and all `RULE_IDS`-iterating tests/code stay in sync.

## Module 6: Deterministic regex detector in `prompt_metrics.py`

New regex `_DECIMAL_PACE_RE` and rule id `decimal_pace_leak`:

```python
_DECIMAL_PACE_RE = re.compile(
    r"\b\d{1,2}\.\d{2}\s*(?:min\s*)?/\s*km\b",
    re.IGNORECASE,
)
```

Wired into `scan_response`:

```python
violations.extend(
    _scan_pattern(text, _DECIMAL_PACE_RE, "decimal_pace_leak", "strict")
)
```

Cost: one regex pass over the response text. Microseconds.

## Module 7: Tests

- `tests/test_pace_format.py` (NEW): unit tests for `pace_format` helpers. ~14 tests covering happy path, edge cases, carry-over, None, negatives, non-numeric.
- `tests/test_critic.py`: extend `test_rule_ids_count` and `test_rule_ids_match_design_spec` to include `pace_format_correct`. The existing `RULE_IDS` assertion (in `critic.py`) is what catches drift.
- `tests/test_prompt_metrics.py`: add tests for `_DECIMAL_PACE_RE`:
  - `4.50/km` -> flagged
  - `4.50 min/km` -> flagged
  - `4:50/km` -> NOT flagged (correct format)
  - `4.50 hours` -> NOT flagged (no /km)
  - English context still flags (regex is language-neutral)

## Failure modes considered

- `threshold_pace_min_km` is missing or `None` -> `decimal_min_to_mmss` returns `None`, the line is skipped (existing behaviour).
- `threshold_pace_min_km = 0.0` -> currently treated as falsy and skipped. The patched code uses `is not None` so 0.0 would yield `0:00`. We keep `is not None` for consistency with the existing `vo2max is not None` check.
- The regex catches a coach quote like "Vorher 5.20/km" if the coach is German-speaking and means decimal. That is itself a bug (the coach should write `5:12/km`), so flagging is correct.
- The regex does NOT catch `5.20km` (no slash) or `5.20 per km` (no slash followed by km). That is acceptable; the failure mode is unambiguous when `/km` follows.

## Out of scope

- Frontend widget (`identity_widgets.py`) still emits the decimal as a stat. That is a separate fix-it tracked in AUDIT.md row 8.
- FTP, weekly_volume_km, easy_pace, long_run_pace fields do not exist in `user_model.fitness` today. The helper supports them for future use (`watts_to_pretty`).
- DB columns keep the decimal. Migration is not in scope.
