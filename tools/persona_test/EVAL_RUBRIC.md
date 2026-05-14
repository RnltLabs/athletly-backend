# Coach Response Eval Rubric

Used by Claude Code (the orchestrating persona test driver) to score every coach turn during a persona session. Each dimension is scored 1 to 5 against the rubric below. Total per turn is out of 30.

- Below 20 = bad turn (regression candidate, file it).
- 20-24 = OK.
- 25-29 = good.
- 30 = excellent.

## Dimensions

### 1. Accuracy (1-5)

Did the coach use real data fetched via tools, or did it hallucinate numbers / events / dates?

- 5: every fact in the response is grounded in a tool result or the journal.
- 4: one minor fact paraphrased loosely but still consistent with the data.
- 3: one number rounded or approximated where exactness was reasonable to skip.
- 2: at least one specific number or event that does not appear in tool output (suspected hallucination).
- 1: multiple hallucinated specifics OR coach made up a plan / pace / date.

### 2. Completeness (1-5)

Did it answer what the persona actually asked, including the implicit ask?

- 5: addressed every explicit point + obvious follow-up the persona expects.
- 4: addressed everything explicit, missed an obvious follow-up.
- 3: addressed the main point, partially missed a secondary.
- 2: only the surface question, ignored sub-asks.
- 1: did not answer the question (changed topic, deflected, asked unrelated clarifier).

### 3. Tool usage (1-5)

Did it follow the STRICT rules of the coach (sync before fetch, get_provider_status when relevant, get_activity_details for specifics, save_plan for plan requests, annotate_activity for context like injury, append_to_journal for open threads)?

- 5: tool calls match the STRICT pattern exactly, no missing, no superfluous.
- 4: pattern correct, one redundant call.
- 3: missed one expected tool but answer still coherent (e.g., didn't sync when persona mentioned a fresh activity).
- 2: skipped multiple expected tools (e.g., made up paces instead of calling get_activity_details).
- 1: no tools called when tools were clearly required.

### 4. Tone (1-5)

Did the response mirror the persona's voice / register?

Reference each persona's "Personality" section in their .md file.

- 5: voice matches (Elena: respectful, asks back; Marco: short and direct; Lisa: technical detail).
- 4: voice mostly matches, occasional drift.
- 3: neutral but coherent.
- 2: clearly off-register (formal with Marco, hand-holdy with Lisa).
- 1: tone clash that would feel wrong to a real user (lecturing, dismissive, or robotic).

### 5. Hallucinations (1-5)

Specifically about made-up facts. Pure subset of Accuracy but graded separately to surface it.

- 5: zero hallucinations.
- 4: one borderline fact that's plausible but unverifiable.
- 3: one minor invented detail (e.g., a pace estimate that wasn't measured).
- 2: one clear hallucination (date, event, performance).
- 1: multiple hallucinations.

### 6. Holistic awareness (1-5)

When relevant, did the coach cross-reference recovery (HRV, sleep, RHR, Body Battery) and the journal?

- 5: persona's question intersected with recovery and the coach correctly pulled health metrics (e.g., persona reports tired -> coach checks HRV).
- 4: noticed the intersection but did not fetch the data.
- 3: question did not need cross-reference; coach correctly stayed focused.
- 2: missed a clear opportunity (persona reports bad sleep, coach gives a hard session without acknowledging).
- 1: actively contradicted available data (e.g., recommended hard intervals on a day where HRV is in red).

## How to use it

After each coach turn, Claude Code writes a small JSON-shaped block in the transcript:

```
EVAL turn 3:
  accuracy: 5
  completeness: 4
  tool_usage: 5
  tone: 4
  hallucinations: 5
  holistic: 4
  total: 27
  notes: "Coach correctly called sync_garmin_data + get_activities and surfaced
   the actual HR average. Missed one open thread (knee twinge from Tue) - would
   have been a 30 with that."
```

At session end, Claude Code aggregates:
- Average total across turns.
- Lowest-scoring turn and why.
- Any turns below 20 (regressions to file).
- Patterns (e.g., "tool_usage consistently 5, tone consistently 3 - coach feels formal with Marco").

## Persona-specific calibration tips

- Elena: should get challenging "why" questions back. A coach that just gives a number without reasoning loses 1 on Completeness.
- Marco: short response valued. Verbose explanations lose 1 on Tone.
- Lisa: expects technical detail and acknowledgement of injury history. Skipping the knee context loses 2 on Holistic.
