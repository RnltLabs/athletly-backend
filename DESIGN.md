# Sprint D: Design - Deterministic Response-Gate Layer

This document specifies `src/agent/response_gates.py`, the deterministic
policy layer that runs after the LLM produced a draft response and
before the SSE `message` event is emitted.

Companion: see `RESEARCH.md` for prior art and design rationale.

---

## 1. Goals and Non-Goals

**Goals:**

- Catch the three reproduced bug families:
  1. Tool-collapse mid-conversation (coach stops calling tools).
  2. Proactive-sync rule fails ("war heute laufen" without
     `sync_garmin_data`).
  3. Persistence misses on injury (knee pain not annotated /
     journaled).
- Catch two adjacent bug families that are also deterministic:
  4. Stats-grounding (numbers without a recent read tool).
  5. Holistic-alert acknowledgement (critical alerts in context but
     not mentioned in response).
- Catch the existing language-mirror rule with a deterministic check
  (the LLM critic in Sprint A handles it via judgement; we make it
  hard via regex).

**Non-Goals:**

- Replacing the LLM critic. Sprint A's constitutional critic still
  runs AFTER gates pass for the soft / semantic rules.
- Solving every failure mode. We pick rules that are
  (a) deterministically checkable and (b) high impact in the persona
  test suite.
- Mutating tool decisions. Gates inspect the draft; they do not call
  tools themselves. The regenerate cycle triggers the model to call
  tools.

---

## 2. Module Layout

```
src/agent/response_gates.py        # Gate definitions + runner (new, ~300 lines)
src/services/gates_metrics.py      # In-memory metrics buffer (new, mirrors critic_metrics)
src/agent/agent_loop.py            # Hook gates into AsyncAgentLoop.process_message_sse
src/api/routers/admin.py           # /admin/gates-stats endpoint (extension)
src/config.py                      # Feature flag fields
tests/test_response_gates.py       # Unit tests for all gates + edge cases
```

---

## 3. Type Shapes

```python
# src/agent/response_gates.py

@dataclass(frozen=True)
class GateResult:
    passed: bool                 # True: gate ok, False: gate would block
    gate_id: str                 # short id matching GATE_IDS
    reason: str | None           # short, human-readable reason for the fail
    required_action: str | None  # regenerate instruction (None on pass)

@dataclass(frozen=True)
class GateContext:
    user_message: str
    response_text: str
    tools_called_this_turn: tuple[str, ...]
    tools_called_recent: tuple[str, ...]   # last 3 user-turn windows
    runtime_context_alerts: tuple[str, ...]  # active critical/warn pattern ids

@dataclass(frozen=True)
class GateBatchResult:
    passed: bool
    failures: tuple[GateResult, ...]
    combined_action: str | None  # concatenated regenerate prompt
```

The context is built ONCE per turn by the agent loop and passed into
every gate. Gates are pure functions: `Callable[[GateContext], GateResult]`.

---

## 4. The Five Gates

### Gate 1: temporal_freshness

**Trigger:** `user_message` matches BOTH a temporal regex AND a sport
noun.

- Temporal regex: `\b(eben|gerade|heute|just got|gerade fertig|war heute|gerade zurueck|just finished|just done|just back|today|heute morgen)\b`
- Sport noun regex: `\b(laufen|gelaufen|run|running|ran|ride|rad|gefahren|swim|schwimmen|swam|workout|session|einheit|training|trainiert|race|rennen|track|interval|tempo|long run|lang(?:e[rn]?|en)?)\b`

**Required:** `tools_called_this_turn` contains AT LEAST `sync_garmin_data`
AND `get_activities`.

**Regenerate prompt:**

```
SYSTEM CHECK: The athlete mentioned a just-completed activity.
You MUST call sync_garmin_data, get_provider_status, then
get_activities BEFORE responding. Use the real data. Rewrite your
response after gathering the activity.
```

### Gate 2: injury_persistence

**Trigger:** `user_message` matches injury regex.

- Injury regex: `\b(schmerz|schmerzen|weh|wehtut|wehgetan|zwick|zwickt|zwicken|zwicke|ziehen|zieht|verletz|knie|wadenheber|wade|spannung|sehne|gelenk|muskelkater|muskel|hueft|hueftbeuger|achilles|fersen|fuss|fuessen|fersensporn|ruecken|nacken|schulter|hurt|pain|sore|tight|injury|injured)\b`

**Required:** AT LEAST ONE of `{annotate_activity, append_to_journal,
update_journal_section}` in `tools_called_this_turn`.

We allow `update_journal_section` too because the agent might update
an existing "Open Threads" section instead of appending. Both satisfy
the persistence invariant.

**Regenerate prompt:**

```
SYSTEM CHECK: The athlete reported a body issue (pain, tightness, or
injury). You MUST persist this via annotate_activity (latest run) AND
append_to_journal(section="Open Threads", ...) BEFORE responding. The
next session must remember this. Rewrite your response after persisting.
```

### Gate 3: stats_grounding

**Trigger:** `response_text` contains specific numbers in athlete context.

Patterns (any one trips):

- HR mention: `\bHR\s*\d{2,3}\b` or `\bHerzfrequenz\s*\d{2,3}\b` or `\bHRV\s*\d{1,3}\s*(ms)?\b`
- Pace: `\b\d:\d{2}\s*/\s*km\b` or `\b\d\.\d{1,2}\s*min/km\b`
- Distance: `\b\d{1,3}(?:[\.,]\d{1,2})?\s*km\b` (5km, 21,1km, 42 km)
- Duration: `\b\d{1,2}h\s*\d{1,2}\s*min\b` or `\b\d{1,3}\s*min\b` (followed by sport context)
- Recovery / body battery score: `\b(recovery|body battery)\s*(score)?\s*\d{1,3}\b`
- TRIMP: `\btrimp\s*\d{1,4}\b`
- VO2max: `\bVO2[mM]?ax?\s*\d{2,3}\b`
- FTP: `\bFTP\s*\d{2,4}\b`

**Required:** AT LEAST ONE of `{get_activities, get_activity_details,
get_health_summary, get_recovery_alerts}` in `tools_called_recent` (last
3 user-turn windows, not just this turn). This lets the model carry
data from a recent tool call without re-calling each turn.

**Regenerate prompt:**

```
SYSTEM CHECK: Your response referenced specific numbers but no recent
read-tool grounds them. Call get_activities, get_activity_details, or
get_health_summary FIRST, then respond with real data only. Do not
fabricate stats.
```

### Gate 4: holistic_alert_acknowledged

**Trigger:** `runtime_context_alerts` is non-empty AND contains at least
one alert of severity `critical` or `warn`.

**Required:** `response_text` mentions at least one of the recovery
domain keywords:

- Domain regex: `\b(schlaf|sleep|hrv|recovery|body battery|stress|ruhepuls|rhr|erholung|regeneration|m[uue]de|tired|erschoepf|erschoeft|fatigue)\b`

**Regenerate prompt:**

```
SYSTEM CHECK: Active recovery alerts exist in your runtime context.
You MUST acknowledge the relevant one (sleep, HRV, recovery, body
battery, stress, or resting heart rate). Rewrite to address the alert.
```

### Gate 5: language_mirror_hard

**Trigger:** Detect language of `user_message`:

- German: at least 3 of `{ich, und, der, die, das, wie, mit, von, zu,
  ist, auf, ein, eine, war, habe, hast, gerade, heute, fuer, fur, hat,
  hab, nicht, kein, schon, noch, mal, doch, denn}`
- Otherwise: English / unknown.

If German, response_text should contain at least 2 German function
words. If English, response should not contain >3 German function words.

**Regenerate prompt:**

```
SYSTEM CHECK: Mirror the athlete's language exactly. Reply in German
if they wrote German, in English if they wrote English. Do not
code-switch mid-response. Rewrite your response in the athlete's
language.
```

---

## 5. Execution Flow

```python
def run_gates(context: GateContext) -> GateBatchResult:
    failures: list[GateResult] = []
    for gate in GATES:
        if not _is_gate_enabled(gate.gate_id):
            continue
        try:
            result = gate.check(context)
        except Exception:
            # FAIL-OPEN: never break the loop on a gate bug.
            logger.warning("Gate %s raised, failing open", gate.gate_id, exc_info=True)
            metrics.record(gate.gate_id, "gate_error")
            continue
        metrics.record(gate.gate_id, "pass" if result.passed else "fail")
        if not result.passed:
            failures.append(result)
    if not failures:
        return GateBatchResult(passed=True, failures=(), combined_action=None)
    combined = "\n\n".join(f.required_action for f in failures if f.required_action)
    return GateBatchResult(passed=False, failures=tuple(failures), combined_action=combined)
```

All gates always run (even after one fails). This lets metrics see
every fire. Only ONE regenerate is triggered downstream, with the
concatenated instruction string.

---

## 6. Integration with AsyncAgentLoop

Hook point: `AsyncAgentLoop._run_critique_pass` is extended to run
gates FIRST. The current logic becomes:

```python
async def _run_critique_pass(...) -> str:
    # Phase D: deterministic gates.
    if get_settings().response_gates_enabled:
        gate_result = run_gates(GateContext(
            user_message=user_message,
            response_text=response_text,
            tools_called_this_turn=tuple(tool_names),
            tools_called_recent=tuple(self._recent_tools_window),
            runtime_context_alerts=tuple(self._active_alert_ids),
        ))
        if not gate_result.passed and not self._gates_regenerated_this_turn:
            self._gates_regenerated_this_turn = True
            rewritten = await asyncio.to_thread(
                self.regenerate_after_gates,
                response_text,
                gate_result,
            )
            # Re-check gates ONCE on the rewrite. On second-pass fail,
            # annotate via critic_review SSE.
            second = run_gates(replace(context, response_text=rewritten))
            if not second.passed:
                try:
                    await emit_fn("critic_review", _gate_event_payload(second))
                except Exception:
                    logger.warning("emit_fn raised on critic_review (gates)", exc_info=True)
            response_text = rewritten
    # ... existing Sprint A critic logic unchanged below ...
```

**Tool history tracking.** The `AgentLoop` maintains
`_recent_tools_window: deque[set[str]]` (max 3) with the union of tool
names called per user-turn. After each turn's tools, we push the
current turn's set. The gate context flattens this into a tuple of
unique names. Concretely:

```python
def __init__(self, ...):
    ...
    self._recent_tools_window: deque[set[str]] = deque(maxlen=3)
    self._gates_regenerated_this_turn: bool = False
```

We populate the window at the end of `process_message`, before
returning.

**Active alert ids.** `build_runtime_context` already computes alerts
via `detect_alerts(user_id)`. We add a thin wrapper that also exposes
the list of `(severity, pattern)` pairs back to the agent loop, stored
in `self._active_alert_ids`. The wrapper is best-effort; failure
silently produces an empty list.

---

## 7. Configuration (src/config.py)

New settings:

```python
# -- Deterministic response gates (Sprint D) ------------------------------
# Master switch. When False, the entire gate layer is skipped (zero overhead).
response_gates_enabled: bool = True
# Per-gate switches, all default true. Flip individually for incremental rollback.
response_gate_temporal_freshness: bool = True
response_gate_injury_persistence: bool = True
response_gate_stats_grounding: bool = True
response_gate_holistic_alert: bool = True
response_gate_language_mirror: bool = True
```

Each per-gate flag is read at gate-run time so a deploy is not needed
to roll back one gate.

---

## 8. Metrics (src/services/gates_metrics.py)

Mirrors `critic_metrics.py`. Records every gate fire (pass / fail /
error / regenerate_triggered / regenerate_failed). Buckets:

```python
GATE_IDS = (
    "temporal_freshness",
    "injury_persistence",
    "stats_grounding",
    "holistic_alert",
    "language_mirror",
)

@dataclass(frozen=True)
class GateRecord:
    ts: float
    gate_id: str
    outcome: str  # pass | fail | error | regenerate_triggered | regenerate_failed

class GatesMetrics:
    def record(self, gate_id: str, outcome: str) -> None: ...
    def summary(self) -> dict: ...
    def reset(self) -> None: ...
```

The admin endpoint `/admin/gates-stats` returns the metrics summary.

---

## 9. Cost Model

- Each gate is pure regex over <8KB of text. < 1 ms per gate.
- Five gates per turn: < 5 ms total. Negligible.
- One regenerate per turn at most: one extra LLM completion (Haiku-class
  via `chat_completion`). This is the same cost cap as Sprint A's
  critic regenerate.
- Combined upper bound: gates regenerate + critic regenerate = 2 extra
  LLM calls in worst case. To keep within the original "max ONE
  regenerate per turn" rule from the task spec, the gate layer takes
  priority: if gates regenerate, the critic uses the rewritten text
  and does NOT regenerate again on that turn. Sprint A's existing
  one-pass cap already enforces this.

---

## 10. Failure Modes

| Failure | Behaviour |
|---|---|
| A gate raises an exception | Log warn, record `gate_error`, skip that gate, continue. |
| `run_gates` raises | Log warn, return `GateBatchResult.passed=True`. Never break the loop. |
| Regenerate LLM call fails | Keep the original draft response. Log warn. |
| Second-pass gates still fail | Emit `critic_review` SSE annotation. Ship the rewritten response. |
| Feature flag off | Gates are skipped entirely. Zero overhead. |
| Per-gate flag off | That gate alone is skipped. Others still run. |

---

## 11. Test Plan

`tests/test_response_gates.py` covers:

- Each gate triggers on positive examples (German + English).
- Each gate passes on negative examples (no false positives).
- Each gate's regenerate prompt is non-empty German-or-English text.
- `run_gates` aggregates correctly (multiple fails -> combined action).
- Per-gate flags toggle individual gates off.
- Master switch turns the whole layer off.
- Exception in one gate fails open (other gates still run).
- Edge cases:
  - Empty response text passes all gates.
  - "ich war heute spazieren" does NOT trigger temporal_freshness
    (no sport noun).
  - "knie hat gehalten" DOES trigger injury_persistence (knie is a
    body part).
  - Number in plain text without context ("9. November") does NOT
    trigger stats_grounding.
  - Recovery alerts empty -> holistic_alert always passes.

The test file aims for ~30 tests, full coverage of the regex
patterns and the orchestration logic.

---

## 12. Persona Impact (expected)

The three reproduced personas:

**Marco (turns 3-7, tool collapse mid-conversation):**

- Turn 3 says "ich war eben laufen, war ganz okay" -> Gate 1 fires
  because `sync_garmin_data` + `get_activities` missing. Regenerate
  forces fresh sync.
- Turn 5 if Marco mentions HR or pace from memory, Gate 3 fires
  (`stats_grounding`) because no recent read tool. Regenerate forces
  the model to call `get_activities` again or stop quoting numbers.

**Lisa (turns 5-6, knee pain not persisted):**

- Turn 5 "knee zwickt nach dem long run" -> Gate 2 fires
  (injury_persistence). Regenerate forces `append_to_journal` and
  `annotate_activity` BEFORE the soothing reply.

**The "war heute laufen" case from the bug report:**

- Gate 1 fires immediately on turn 1. Regenerate forces the proactive
  Garmin sync the STRICT rule was supposed to mandate.

These are the three high-value catches the deterministic gates buy us
without an LLM call.

---

## 13. Rollout

1. Land code with master switch `response_gates_enabled=True` and all
   per-gate flags `True`. Tests pass.
2. Watch `/admin/gates-stats` for 24-48 hours. If any gate's
   regenerate-rate exceeds 5% of turns, flip THAT gate's per-gate
   flag to False, file an issue with the false-positive samples.
3. Re-tune the regex of the offending gate based on samples, land a
   patch, flip flag back on.

This is the same playbook used by the LLM critic (Sprint A) in its
own rollout, applied to a deterministic layer.
