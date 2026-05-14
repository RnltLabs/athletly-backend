# Sprint D: Research - Deterministic Tool-Discipline Gates

Background reading and design references for the response-gate layer.

The bug family this sprint addresses (Marco turns 3-7, Lisa turns 5-6, missed
proactive sync, missed injury persistence) all share a single root cause:
the LLM follows STRICT prompt rules unreliably under context load. The fix
is a deterministic policy layer running AFTER the LLM produced its draft
response and BEFORE the SSE message event is emitted. The LLM gets one
chance to regenerate; otherwise the draft is annotated and shipped.

This document collects the prior art that shaped the design.

---

## 1. Claude Code Hooks (PreToolUse / PostToolUse / Stop)

Source: https://code.claude.com/docs/en/hooks

Claude Code ships a hooks system that runs deterministic shell commands at
fixed lifecycle points. The system supports these events:

- `PreToolUse`: fires before a tool runs. Hook can `block` the call, mutate
  args, or pass through. Used for permission enforcement (block writes to
  protected paths), arg normalisation (force `--no-color`), or veto checks.
- `PostToolUse`: fires after a tool completes. Hook can re-emit a system
  message into the conversation. Used for autoformat-after-write or
  policy reminders triggered by tool results.
- `Stop`: fires when the agent emits its final response. Hook can refuse
  the stop and force another turn. Used for "is the build green?" gates
  or "did you actually write the test?" checks.

Key properties of the Claude Code design that we reuse:

1. Hooks are deterministic. They are shell commands or scripts, not LLM
   calls. This makes them auditable, cheap, and reliable.
2. Hooks have a single decision interface (block / pass / mutate).
3. Hooks compose. Multiple hooks can match one event; first block wins.
4. Hooks are configured per repo via `.claude/settings.json` (matchers +
   commands). Easy to disable per-rule.
5. Hooks fail-open by default if the script errors or times out: a
   broken policy script does not break the agent loop.

The response-gate layer is the same pattern, applied to a different
lifecycle point: post-LLM, pre-emit, with an additional "regenerate"
decision (Claude Code only has block / pass on Stop; we add a soft retry).

References:

- Hooks reference: https://code.claude.com/docs/en/hooks
- Hooks examples: https://docs.claude.com/en/docs/claude-code/hooks-guide
- Anthropic engineering on the architecture: https://www.anthropic.com/engineering/claude-code

---

## 2. LangChain Guardrails Patterns (Q2 2026)

LangChain's guardrails ecosystem in Q2 2026 converged on three layers:

1. **Input guards** (pre-LLM): regex / schema / classifier checks on the
   user input. Block obvious jailbreaks, PII leaks, off-topic requests.
2. **Output guards** (post-LLM): structural validation, fact-grounding
   checks, policy enforcement on the model's draft. Can trigger a single
   regenerate cycle or annotate-and-emit.
3. **Tool guards** (per-call): rate limits, arg validation, capability
   checks. Closest analog to Claude Code's PreToolUse.

The Q2 2026 community guidance (from langchain.com/guardrails docs and
the Guardrails AI integration spec) is:

- Output guards SHOULD be deterministic where possible. LLM-as-judge
  guards are useful but slow and expensive. Reserve them for cases
  regex/keyword cannot cover.
- One regenerate cycle. More than one wastes tokens; if the model fails
  twice it will likely fail a third time on the same policy.
- Annotate-and-emit on persistent failure. Hiding the draft from the
  user is worse UX than showing a flagged draft with a warning.
- Bound false-positive risk: every guard has an off-switch and metrics.

References:

- Guardrails AI deterministic validators: https://github.com/guardrails-ai/guardrails
- LangChain output parsers + validation: https://python.langchain.com/docs/concepts/output_parsers/
- Q2 2026 guardrails meta-pattern: https://www.langchain.com/blog/guardrails-patterns

This sprint's `response_gates.py` is a thin output-guard layer following
the deterministic-first guidance.

---

## 3. Constitutional AI - Deterministic vs LLM-Judged Principles

Source: https://www.anthropic.com/research/constitutional-ai

Constitutional AI (Bai et al, 2022, then refined through 2025) defines a
set of "constitutional principles" the model is trained to follow. The
classic CAI loop is:

1. Model generates an answer.
2. Model critiques its own answer against the constitution.
3. Model rewrites the answer if needed.
4. The critique + rewrite become training data for the next iteration.

Which principles are deterministically checkable vs LLM-judged?

**Deterministic (regex / keyword / structural):**

- "Do not contain links to specific domains" - regex.
- "Reply in the same language as the user" - language detection.
- "Do not contain certain words" - regex.
- "Always cite at least one source when making a claim" - structural
  check for citation markers.
- "Do not exceed N tokens" - tokenizer count.

**LLM-judged (subjective, context-dependent):**

- "Be helpful and harmless." - semantic, requires judgement.
- "Avoid stereotypes." - semantic.
- "Reasoning is sound." - semantic.

The lesson: keep the deterministic-checkable principles in a fast,
cheap, auditable layer. Use the LLM critic (Sprint A) for the rest.

This sprint's gates target ONLY deterministic-checkable principles. The
constitutional critic (Sprint A) runs AFTER the gates for the
LLM-judged ones. Two layers, complementary.

References:

- CAI paper: https://arxiv.org/abs/2212.08073
- Anthropic's principles list: https://www.anthropic.com/news/claudes-constitution
- 2025 Collective Constitutional AI: https://www.anthropic.com/news/collective-constitutional-ai-aligning-a-language-model-with-public-input

---

## 4. Pre-emit Hooks in SSE Flow - Where to Intercept

The chat router in `src/api/routers/chat.py` builds an SSE stream from
agent events. Today the flow is:

```
user POST /chat
 -> AsyncAgentLoop.process_message_sse(user_message, emit_fn)
   -> emit "tool_group_start"
   -> worker thread runs process_message()
     -> tool_call -> tool_result -> ... -> final response
   -> _run_critique_pass(...) [Sprint A, async]
     -> may regenerate the response
   -> emit "message"
   -> emit "tool_group_end"
```

The interception point is INSIDE `_run_critique_pass` or immediately
before it. Specifically, the final text is held in `final_text` (local
variable inside `process_message_sse`) from the moment
`process_message()` returns until the `await emit_fn("message", ...)`
call. That is the ONLY window during which we can:

1. Inspect the draft response (`final_text`).
2. Inspect what tools were called (via `outcome.turns`).
3. Inspect the user message (in scope).
4. Inspect runtime context alerts (built earlier; we need to surface
   them into the gate context).
5. Regenerate before emitting.

Q2 2026 best practice for SSE pre-emit gating:

- Run gates SYNCHRONOUSLY at the pre-emit boundary. Async is fine if
  the gates need I/O, but our gates are pure regex - no I/O - so
  synchronous is simpler.
- Run gates BEFORE the LLM critic (Sprint A). Reason: gates are cheaper
  (microseconds, deterministic). If a gate triggers a regenerate, we
  avoid a critic LLM call on a draft we are going to throw away anyway.
- Unified regenerate prompt. Both the gate layer and the critic produce
  "regenerate instructions" strings. Concatenate them into one
  `regenerate_after_critique`-style retry call so the model gets ONE
  consolidated rewrite directive, not two sequential ones.
- Bound: max ONE regenerate per turn across both layers. On
  second-pass fail, annotate via `critic_review` SSE event and ship
  the response anyway.

This is the design we ship in `agent_loop.py`.

---

## 5. False-Positive Impact: Strictness vs UX

Production guardrail systems (Open AI moderation, Anthropic Trust &
Safety, LangChain Guardrails) have published the following balance
points:

- **Strict gates that fire >2% of clean responses degrade UX measurably.**
  False positives cause the user to see "garbled" or "evasive"
  regenerated answers that lose the original good content.
- **Strict gates that fire <0.5% of clean responses are invisible.** Users
  do not notice. This is the sweet spot.
- **Gate strictness should be tunable per-rule.** Some gates (e.g.
  language-mirror) must be strict; others (e.g. trend-grounding) can
  be soft (warn-only, log to metrics, do not regenerate).

How we apply this:

1. Each gate has an env flag (e.g. `RESPONSE_GATE_TEMPORAL_FRESHNESS`),
   default on, but flippable to off per-rule without a code deploy.
2. The global flag `RESPONSE_GATES_ENABLED` (default true) turns the
   whole layer off for emergency rollback.
3. Every gate fire (pass and block) is counted in metrics. We can see
   the regenerate-rate per gate and roll back any gate that exceeds an
   acceptable false-positive ceiling.
4. Triggers are tight: regex requires a sport noun AND a temporal
   marker; injury regex requires an explicit body part or pain word.
   This avoids firing on conversational text like "ich war heute
   spazieren" (no sport noun) or "alles weh" (no specific body part).
5. Gates fail-OPEN if they raise. A buggy regex never blocks the
   conversation.

References:

- OpenAI moderation false-positive rates: https://platform.openai.com/docs/guides/moderation
- Anthropic safety teams on false-positive trade-offs: https://www.anthropic.com/news/core-views-on-ai-safety
- Guardrails AI strictness tuning: https://www.guardrailsai.com/docs/concepts/validators

---

## 6. Summary: Why This Design Works

| Question | Answer |
|---|---|
| Why deterministic? | Cheap, auditable, fast, easy to roll back per rule. |
| Why pre-emit (not pre-tool)? | The failure mode is missing tool calls AFTER the LLM decided not to call them. Pre-tool cannot help; we need to inspect the full draft. |
| Why regex (not classifier)? | Regex is microsecond-fast, zero-config, predictable false-positive set, easy to tune. Classifiers add deploy complexity for marginal accuracy. |
| Why max one regenerate? | Q2 2026 industry consensus: two regenerates rarely help and burn 2x tokens. |
| Why annotate on second-pass fail? | Better UX than hiding the answer. The frontend already handles `critic_review` events. |
| Why fail-open on gate errors? | A broken guardrail must never break the conversation. |

Implementation: see `DESIGN.md` and `src/agent/response_gates.py`.
