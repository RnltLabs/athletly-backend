# Design: Whole-Athlete Coaching Update

## Goal

Make the coach treat training as one input among many (sleep, HRV,
stress, body battery, recovery, RHR, life load). Surface red flags
PROACTIVELY in the runtime context so the agent's reply weaves them in
naturally instead of waiting to be asked.

## Architecture

Two layers, separated by responsibility:

1. **Detection layer** (deterministic, pure Python). Lives in
   `src/services/recovery_alerts.py`. Reads `health_daily_metrics`,
   applies fixed thresholds, returns a list of `RecoveryAlert` dataclasses.
   No LLM, no judgement calls, easy to unit-test.

2. **Synthesis layer** (LLM). Lives in the system prompt. A new STRICT
   block tells the agent how to read the `# Coach Alerts` section and
   weave acknowledgements into its reply. The decision of HOW to phrase
   the alert is the LLM's, the decision of WHETHER it triggers is the
   detection layer's.

This split makes the feature easy to tune (one constants block), easy
to test (deterministic), and impossible to "miss an alert" because the
agent silently decided it was not relevant.

## Pattern detection rules

All thresholds live as module-level constants in
`src/services/recovery_alerts.py`:

```
SLEEP_LOW_MINUTES = 360          # 6h
SLEEP_CRITICAL_MINUTES = 240     # 4h
SLEEP_LOW_CONSECUTIVE_DAYS = 3
SLEEP_CRITICAL_DAYS_IN_7 = 5
HRV_BASELINE_DAYS = 30
HRV_RECENT_DAYS = 7
HRV_DROP_PCT = 0.15              # 15 percent
RHR_BASELINE_DAYS = 14
RHR_ELEVATION_BPM = 5
RHR_ELEVATION_DAYS = 3
BB_CHRONIC_THRESHOLD = 30
BB_CHRONIC_DAYS = 3
STRESS_CHRONIC_THRESHOLD = 60
STRESS_CHRONIC_DAYS = 5
RECOVERY_LOW_THRESHOLD = 30
RECOVERY_LOW_DAYS = 2
RECOVERY_CRITICAL_THRESHOLD = 20
LOAD_SPIKE_RATIO = 1.5
```

Each pattern is one function `_check_<pattern_id>(metrics)` returning
`RecoveryAlert | None`. The top-level `detect_alerts(user_id)` runs all
checks, collects non-None results, and returns the list. Failures
inside any single check are caught and logged (DEBUG, not WARN, to
avoid log spam) and never block the others.

## Where to integrate

Three integration surfaces:

1. **Always emit in runtime context.** In `build_runtime_context`,
   after the existing recovery-status block, call
   `detect_alerts(user_id)` and append a `# Coach Alerts` section if
   the list is non-empty. Single import, single try-block, fully
   backwards compatible: when no alerts, nothing is appended and
   existing turns behave unchanged.

2. **STATIC system prompt teaches the pattern.** A new STRICT block in
   `STATIC_SYSTEM_PROMPT` named "WHOLE-ATHLETE COACHING" explains the
   contract: when `# Coach Alerts` is present, the agent MUST
   acknowledge or address the relevant alerts in its reply.

3. **On-demand tool.** A new tool `get_recovery_alerts()` exposes the
   same detection layer to the agent for cases where the alert was not
   triggered at session start (athlete mentions feeling off mid-conv).
   Registered in `health_tools.py`, deferred-loading-safe, added to
   `CORE_TOOL_NAMES` because the system prompt names it.

## Existing data sources we leverage

- `health_daily_metrics` table (single source of truth for sleep, HRV,
  stress, body battery, recovery score, RHR).
- `get_merged_daily_metrics(user_id, days)` from
  `src/db/health_data_db.py` (already used by `build_health_summary`).
- `get_cross_source_load_summary(user_id, days)` for the
  training-load-spike check (last 7 days vs prior 7 days).
- `format_recovery_context_block` continues to do its job; the new
  alerts section is appended AFTER it so the agent first sees latest
  values, then the deterministic flags.

## STATIC system prompt update

Insert a new STRICT block immediately after the existing
"## Strict Behaviour Rules" section ends. The block (full text in
implementation) teaches:

- The contract: `# Coach Alerts` is non-empty -> MUST acknowledge.
- Two main triggers for HOW to integrate:
  - Athlete reports performance issues ("schwer", "schlapp", "konnte
    nicht"): check alerts FIRST, attribute the bad feeling to a
    recovery deficit if one exists, before reaching for fitness as the
    explanation.
  - Athlete plans hard training: if body battery / recovery is low,
    surface this and propose an adjustment.
- The tone: one sentence empathy, one suggestion or question, never
  lecture.
- The escape hatch: when the alert is genuinely irrelevant to the
  current turn (e.g. athlete is asking about gear), acknowledge briefly
  and move on. Do NOT force-fit every alert into every reply.

## Tool: get_recovery_alerts

```
get_recovery_alerts() -> {
  "count": int,
  "alerts": [
    {
      "severity": "info" | "warn" | "critical",
      "pattern": str,
      "message_de": str,
      "evidence": dict,
    },
    ...
  ]
}
```

No parameters. Resolves user_id from settings the same way the other
health tools do. Returns the same shape that the runtime-context block
is built from, so the agent can call it mid-conversation and get
fresh detection without rebuilding the whole context.

Registered with description that explicitly tells the agent to use it
when:
- The athlete mentions feeling off mid-conversation and no alert is in
  the current context.
- The athlete proposes hard training and the agent wants a deterministic
  recovery check before agreeing.

## Failure mode discipline

The whole feature is best-effort. Any failure in
`detect_alerts` (DB unreachable, bad row, missing columns) is caught at
the runtime-context boundary and the section is simply omitted. The
existing recovery block stays. Behaviour with no alerts is identical
to behaviour with alert-detection errored, which is identical to
behaviour with feature disabled. This is the right safety property for
a piece of system that adds value when it works but must never break
the coach turn.

## Test plan

`tests/test_recovery_alerts.py` covers:

- One unit test per pattern, with a hand-crafted `metrics` list that
  pushes exactly one threshold and asserts the alert is emitted with
  the right severity, pattern id, and German message.
- One negative test per pattern, where the data is just below the
  threshold and the alert must NOT fire.
- Edge cases:
  - Empty `metrics` list (no data) returns empty alerts.
  - Sparse data (one row only) does not raise; HRV-drop and
    training-load-spike return None.
  - Baseline period shorter than 14 days: RHR check returns None.
  - All-green scenario: no alerts.
- The system_prompt integration path: with patched
  `get_merged_daily_metrics` returning a "3 consecutive low sleep
  nights" fixture, `build_runtime_context` emits a `# Coach Alerts`
  section that contains the expected message.
- Failure mode: `detect_alerts` raising propagates as an empty alerts
  list, never as a context-build failure.

## German message style

All `message_de` strings:
- Use real umlauts (ä, ö, ü, ß), never the ASCII transliteration.
- No em-dash, no en-dash. Hyphens only.
- One sentence. Names the observation, gives the number.
- No commands; the agent will translate into a suggestion.
- Example: "Drei Nächte hintereinander unter 6 Stunden Schlaf (Schnitt 5,4 h)."
