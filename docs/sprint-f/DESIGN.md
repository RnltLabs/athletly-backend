# Sprint F Phase 3: Design - Tool-Forcing Regenerate

## 1. Gate taxonomy

Each registered gate is labelled `needs_tools: bool` on the `Gate`
dataclass. The label captures whether the gate's pass condition can be
satisfied by rewriting text alone or requires the presence of a tool
call in the turn.

| gate_id            | needs_tools | required tool union                                                   |
|--------------------|-------------|-----------------------------------------------------------------------|
| temporal_freshness | True        | `sync_garmin_data`, `get_provider_status`, `get_activities`           |
| injury_persistence | True        | `annotate_activity` AND (`append_to_journal` OR `update_journal_section`) |
| stats_grounding    | True        | one of: `get_activities`, `get_activity_details`, `get_health_summary`, `get_recovery_alerts` |
| holistic_alert     | False       | n/a (text-only acknowledgement)                                       |
| language_mirror    | False       | n/a (text-only rewrite)                                               |

The required tool union is encoded as a per-gate constant in
`response_gates.py` so the agent loop can compose a precise instruction
without re-parsing the regex.

## 2. Regenerate paths

Two paths, selected by the failure set:

### Path A: text-only regenerate (current behaviour, unchanged)

Triggered when ALL failing gates have `needs_tools=False`. The agent
loop calls `regenerate_after_gates(original, combined_action)` which
runs ONE text-only chat completion with the existing prompt that
forbids tool calls. Cost: one completion. Backwards compatible.

### Path B: tool-forcing regenerate (new)

Triggered when AT LEAST ONE failing gate has `needs_tools=True`.

Steps:

1. Compose a synthetic user message:

   ```
   SYSTEM REMINDER: Your previous response failed deterministic policy
   gates that require tool calls before answering.

   <per-gate required-action block, one per failing tool-required gate>

   REQUIRED ACTIONS THIS TURN:
   - Call <tool A>
   - Call <tool B>
   - Then re-answer the athlete using the real results.

   Do not skip tools. Do not invent stats. Do not apologise. The
   athlete is still waiting; gather what you need and respond.
   ```

   In German turns the imperative core is bilingual (German prefix +
   English tool names which are stable across locales). Real umlauts
   only, no ASCII transliteration in user-facing copy. Tool names stay
   English because they are API identifiers, not user-facing text.

2. Append the synthetic message to `self._messages` as a `user` turn so
   the tool loop sees it as the next instruction.

3. Re-enter the tool loop for AT MOST `_FORCED_REGEN_MAX_ROUNDS = 3`
   rounds with `tools=openai_tools` enabled. This is enough for one
   tool-call wave plus the model's text answer (round 1: tool calls,
   round 2: more tool calls if the first wave triggers follow-ups,
   round 3: final text answer). Three rounds is a hard ceiling.

4. If a `tool_choice` plumb is available on the resolved provider AND
   the failing set contains exactly one tool-required gate whose union
   maps cleanly to a single tool, we pass
   `tool_choice={"type": "tool", "name": "..."}` for round 1 of the
   forced retry to guarantee the first call shape. Otherwise we pass
   `tool_choice={"type": "any"}` which guarantees SOME tool call. If
   `tool_choice` plumbing fails for any provider reason, we fall back
   to prompt-only steering. The combination keeps the implementation
   fail-safe.

5. After the forced rounds complete, take the LAST assistant text as
   the new candidate response. Re-run `run_gates` on a fresh
   `GateContext` whose `tools_called_this_turn` is the union of the
   pre-forced and the forced-pass tool names (so the second-pass gate
   sees the tools we just invoked).

6. If the second pass returns `passed=True`: record
   `regenerate_success` and `tool_forced_regenerate_success`, ship the
   new text.

7. If the second pass still fails: record `regenerate_failed`, emit a
   `critic_review` SSE event with `degraded=true` and `source=
   "response_gates_tool_forced"`, ship the new text anyway (do not
   block the user, consistent with fail-open).

Cost: at most THREE completions plus the original tool round (vs ONE
extra completion for path A). Bounded by `_FORCED_REGEN_MAX_ROUNDS`
and by the existing `_gates_regenerated_this_turn` flag which prevents
a second forced retry in the same turn.

## 3. Per-gate system reminder composition

Computed by a small helper `_compose_tool_force_instruction(failures)`
in `response_gates.py`. Output is a string assembled from constant
fragments, one per failing tool-required gate. Constants live in the
gate module so future tweaks land in one place.

### temporal_freshness

```
SYSTEM REMINDER: The athlete just told you about a completed activity.
REQUIRED: call sync_garmin_data, get_provider_status, and
get_activities (in that order) before re-answering. Use the real
result.
```

### injury_persistence

```
SYSTEM REMINDER: The athlete reported a body issue (Schmerz, Zwicken,
Knie, Wadenheber, oder aehnlich). REQUIRED: call annotate_activity on
the latest run AND append_to_journal(section="Open Threads", ...) so
the next session remembers. Then re-answer with empathy and a concrete
adjustment.
```

(Note: in agent-facing system text we DO use real umlauts. The literal
"aehnlich" here is acceptable as agent-internal text. User-facing
strings continue to use real ae oe ue per project policy. To be safe
the implementation uses real umlauts throughout.)

### stats_grounding

```
SYSTEM REMINDER: Your previous answer quoted specific numbers (HR,
pace, distance, duration, recovery) with no read-tool to ground them.
REQUIRED: call get_activities (or get_activity_details /
get_health_summary if the question is about health) before answering.
Quote only numbers that come back from the tool. If the tool returns
nothing relevant, say so plainly instead of inventing data.
```

Multiple failing gates concatenate their fragments separated by a
blank line, followed by a single closing imperative.

## 4. Failure mode: forced retry still fails

Distinct from path A's failure mode because we have new degradation
information:

- the tools fired correctly but the gate still says no (e.g. the model
  called `get_activities` but the response still contains a number
  not present in the result) -> ship + `critic_review degraded=true,
  reason="gate_persistent_after_tool_force"`.
- the tools did NOT fire even with the forced prompt (rare, model
  bug) -> ship + `critic_review degraded=true, reason=
  "tool_force_ignored"`.
- the forced completion itself raised (rate limit, provider error) ->
  ship the ORIGINAL pre-regenerate response + log warning. No
  `critic_review` because the gate verdict is undefined.

The fail-safe invariant holds: under no error path does the user see a
blank or an error. They always see SOME response.

## 5. Telemetry

Two new outcome values added to `gates_metrics._VALID_OUTCOMES`:

- `tool_forced_regenerate_success`: tool-forcing retry produced a
  response that passes all gates on second-pass.
- `tool_forced_regenerate_failed`: tool-forcing retry ran but the
  response still failed at least one gate.

The existing `regenerate_success` / `regenerate_failed` counters keep
their meaning for path A. The new counters are surfaced via
`/admin/gates-stats` so ops can see the lift directly.

## 6. Backwards compatibility

- All five existing gates keep their `gate_id` and behaviour.
- `Gate` dataclass gains `needs_tools: bool = False` with a default so
  any third-party gate registration code keeps working.
- `regenerate_after_gates` keeps its signature and behaviour. Path B
  is a NEW method (`_handle_tool_required_gate_failure`) called only
  when at least one failure has `needs_tools=True`.
- Master switch and per-gate flags are unchanged.

## 7. Test plan

Unit (in `tests/test_response_gates.py`):

- gate registry has `needs_tools=True` on the three tool-required
  gates and `False` on the other two.
- `_compose_tool_force_instruction` produces a string containing the
  exact tool names for each failure type.
- Multiple failing tool-required gates concatenate cleanly.
- Mixed failures (one text-only, one tool-required) route to path B
  and include both instructions.

Integration (in `tests/test_response_gates.py` or a new
`tests/test_agent_loop_gates_tool_force.py`):

- Stub `chat_completion` so the first call returns a draft that fails
  `stats_grounding`. The second call (the forced retry) returns a
  `tool_calls=[get_activities(...)]` response. The third call returns
  a clean text answer that passes the gate. Assert
  `tool_forced_regenerate_success` is recorded.
- Stub the forced retry to return a still-fabricating text. Assert
  `tool_forced_regenerate_failed` is recorded AND the original
  response ships with `critic_review degraded=true`.
- Stub the forced retry to raise. Assert the original pre-regen
  response ships untouched and no metric is recorded for the forced
  path beyond `regenerate_failed`.

## 8. Expected impact

Empirical baseline (LIVE_DIAGNOSIS.md):

- tool-required regenerate success: 1 / 19 = 5%
- text-only regenerate success: 3 / 5 = 60%

Target (consistent with literature on tool_choice forcing reliability
plus the explicit-instruction baseline we already measure):

- tool-required forced regenerate success: ~70-85%
- text-only regenerate (unchanged): ~60%

Aggregate fail rate impact on the rolling 1000-record window:

- `stats_grounding` fail-rate today 48.8% with ~50% of failures
  surviving the regenerate. Lifting forced regenerate to 75% drops the
  net "still-failing-after-regen" rate from ~24% of all calls to ~12%
  of all calls (a 2x reduction at the same overall fail-detection
  rate).
- `temporal_freshness` and `injury_persistence` together account for
  fewer fires (~13 fails of ~86 runs combined), but currently 100% of
  those leak through unchanged. After the fix they collapse to ~25%
  leak, which is the dominant qualitative win (these gates flag the
  worst UX failure modes: hallucinated activities and unrecorded
  injuries).
