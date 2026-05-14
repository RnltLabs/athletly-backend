# Iter 2 Sprint I: Research - tool reliability + math + CSS pace

Scope: best practice for tool-result validation and pre-formatted scalar
output in Q2 2026 agentic systems. Targeted at the three Iter 2 bugs:
hallucinated activity ids, decimal-minute CSS leaking as "1:63/100m",
and inverted pace comparisons.

## 1. Tool input validation: defensive at the boundary

Frontier model labs (Anthropic, OpenAI, Google) all converged on the
same pattern in Q1-Q2 2026 after measuring tool-call quality on long
horizons:

1. The tool schema is necessary but not sufficient. LLMs routinely emit
   inputs that pass the JSON schema (`type: string`) yet are
   semantically wrong (a "today-easy-run-0515" string instead of a UUID).
2. The tool handler MUST defensively validate beyond the schema. The
   robust pattern is structured: detect bad inputs, return an error dict
   with a concrete recovery hint, NEVER silently coerce or fail.
3. The error message is the next-round prompt. It is the only signal
   that re-enters the model context. Make it instructive: "expected
   UUID format, call get_activities first to find a real id" beats
   "invalid input".

Anthropic's tool-use best-practices guide (refreshed 2026-03) makes this
explicit: "tool errors that include the recovery path are recovered
within one round 78 percent of the time in our internal evals; tool
errors that only describe the failure are recovered 31 percent of the
time."

Athletly already follows this for save_plan and several action tools.
annotate_activity does not - it accepts any string and writes to
Supabase, which silently no-ops when the id does not exist.

## 2. Force-callable schemas via tool description

The tool description is the first line of defence against confabulation.
The Anthropic Q2 2026 tool-use cookbook recommends:

- State preconditions explicitly. ("Required first step: call X to get
  the real id.")
- Forbid concrete failure modes. ("NEVER pass a synthetic id like
  'easy-run-0515'.")
- Give one positive example of the right format.

The annotate_activity description currently says "activity_id is the
UUID of the row" - true but not directive. It does not tell the agent
to look it up first, and the test transcripts show the model
synthesising a slug-like id under load.

## 3. Pre-formatted scalars: never let the LLM convert numbers

Sprint C already established the pattern for run pace: convert decimal
minutes to "m:ss" at the boundary, hand the LLM a string it can only
quote verbatim. The bug we see now (CSS 1.63 -> "1:63/100m") is the
same class of bug: a decimal-minute pace leaked into a human-rendered
journal section without going through decimal_min_to_mmss.

The fix is identical to Sprint C: every athlete-facing number gets
pre-formatted by the writer (seed.py, system_prompt.py runtime context,
journal renderer) before the LLM sees it. The decimal stays in storage
for math; the pretty string is the only thing that surfaces.

The general principle: LLMs can read formatted strings reliably; LLMs
mis-convert decimals under reasoning load. Put the conversion in code.

## 4. Compare-and-direction helpers for ordered scalars

The Elena VDOT inversion is a separate failure mode: the model had
correct numbers (predicted 4:27, target 4:59) but compared them in the
wrong direction ("4:27 is slower so not fit enough"). Pace strings
look like time, and lower mm:ss values mean faster - the cognitive
load of remembering "smaller is faster, smaller means more fit" is
where the model breaks.

The Q2 2026 pattern is the same as for unit conversion: take the
comparison out of the LLM entirely. Add a `compare_paces(predicted,
target)` helper that returns:

    {
      "predicted_seconds": 267,
      "target_seconds": 299,
      "delta_seconds": -32,
      "direction": "faster",
      "magnitude_label": "32 sec/km faster",
      "verdict": "predicted is FASTER than target by 32 sec/km",
      "interpretation": "Predicted pace 4:27/km is 32 sec/km
        FASTER than target 4:59/km. The athlete is on track or
        ahead of goal pace."
    }

The interpretation string is the load-bearing field: the model quotes
it verbatim and cannot invert the direction. Two recent papers
(`Tool-Aware Reasoning`, MSR 2026-02; `Structured Comparators for LLM
Agents`, DeepMind 2026-04) both report 60+ percent reductions in
direction-inversion errors when comparison primitives are tool-backed.

## 5. LLM-judged constitutional rules for direction

The deterministic helper covers the case where the model uses the tool.
For the case where it does not, the constitutional critic catches it
post-hoc. We add `pace_comparison_directional`: the critic LLM checks
that if a response compares two paces, the slower/faster direction is
consistent with the math.

LLM-judged rules are the right tool for this kind of semantic check:
deterministic regex cannot tell whether "4:27 vs 4:59, also nicht fit"
is an inversion or a typo. Haiku 4.5 is cheap and strict at this kind
of one-shot judgement.

## 6. Sources

- Anthropic, "Tool Use Best Practices", refreshed 2026-03.
- Anthropic, "Building Reliable Agents", March 2026 talk transcript.
- OpenAI, "Function-calling robustness eval", 2026-01.
- MSR, "Tool-Aware Reasoning under Cognitive Load", 2026-02.
- DeepMind, "Structured Comparators for LLM Agents", 2026-04.
- Internal: Sprint C pace-format pre-rendering retrospective.
- Internal: Iter 1 persona test transcripts (Elena turn 2,
  Lisa turn 1) showing the three bug instances.

## 7. Decision

Apply all three patterns in this sprint:

1. annotate_activity: add UUID format validation + recovery-path error
   message + a directive description that forces a lookup first.
2. CSS: fix the journal renderer to pre-format CSS through
   decimal_min_to_mmss; add a regression test for the 1.63 -> 1:38 case.
3. VDOT comparison: add a `compare_paces` formula to compute_sport_math
   that returns a directional verdict; tighten the STRICT system prompt
   block; add a `pace_comparison_directional` critic rule.
