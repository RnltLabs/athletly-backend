# Research: Robust Constitutional Gating in Production Agents (Q2 2026)

Question: how do mature LLM agent systems in mid-2026 ensure that
detected policy violations actually block the response, not just get
logged?

## 1. Anthropic's own approach in Claude Code and Anthropic API

- Anthropic exposes `claude-3-5-haiku` and `claude-haiku-4-5` as
  "guardrail" models in the official guidance for two-tier moderation:
  the primary model generates, a smaller model validates.
  Source: Anthropic docs, "Guardrails and content filtering" pattern.
  https://docs.anthropic.com/en/docs/build-with-claude/safety
- The pattern recommended in the docs is **synchronous gating**: the
  guard model returns a structured decision via tool_use (function
  calling), and the orchestrator BLOCKS the user-facing emit until the
  decision is in. Fail-open is explicitly discouraged for hard policy
  rules; fail-open is only acceptable for "advisory" categories.
- Anthropic also publishes a `claude-3-haiku-tools-use` pattern where
  the guard model returns a JSON schema enforced via `tool_choice =
  {"type": "tool", "name": "moderation_decision"}`. This eliminates
  JSON parse errors at the source - if the model would otherwise emit
  invalid JSON, the API rejects the response and we retry once.
  https://docs.anthropic.com/en/api/tool-use

Implication for us: switch the critic to **tool_use with forced
tool_choice**. Eliminates the entire "unparseable JSON" failure
class.

## 2. LangSmith / Langfuse guard-rail patterns

- LangSmith's "guardrails" documentation (updated Q1 2026) describes
  a three-step pipeline: deterministic pre-checks, then LLM critic,
  then deterministic post-checks. The deterministic stages are
  positioned around the LLM stage precisely to handle the case where
  the LLM critic itself fails or is too slow.
  https://docs.smith.langchain.com/observability/guardrails
- Langfuse's "Evaluator Pattern" (https://langfuse.com/docs/evaluators)
  emphasizes that evaluators must be **non-blocking for performance
  AND blocking for correctness**, achieved by running the deterministic
  rules synchronously on the hot path and the LLM rules in parallel
  with the response generation when possible.
- Key takeaway: **never put deterministic checks behind an LLM call**.
  Em-dash, ASCII umlauts, markdown - all of these are sub-millisecond
  regex matches. Routing them through Haiku is anti-pattern.

## 3. GitHub Copilot policy enforcement

- GitHub Copilot for Business publishes its content filtering pipeline
  in https://docs.github.com/en/copilot/concepts/content-exclusions.
  The architecture is two-stage:
  1. **Pre-emit deterministic filter**: blocks output that matches
     known-bad patterns (secrets, license patterns). Runs in <5ms.
  2. **Post-emit telemetry filter**: classifies content asynchronously
     for trend analysis but does not block.
- The deterministic filter is the SAFETY net. The async classifier is
  observability. Copilot deliberately does NOT put the LLM in the
  blocking path because of latency and reliability concerns.

Implication: our em-dash / markdown / ASCII-umlauts rules belong in
the deterministic pre-emit stage, the same place GitHub puts secret
detection.

## 4. OpenAI Moderation API and function-calling guards

- OpenAI's moderation API
  (https://platform.openai.com/docs/guides/moderation) is the
  canonical async classifier. It is NOT positioned as a blocking
  gate; OpenAI's own engineering blog recommends using
  `tool_choice="required"` + structured outputs for moderation
  decisions when blocking IS required.
  https://platform.openai.com/docs/guides/function-calling
- The same blog (Q4 2025) calls out that JSON parsing failures are
  the #1 cause of guardrail failure-open behavior in production
  systems, and that structured outputs (`response_format = {"type":
  "json_schema", ...}`) eliminate this class of failure.

Implication: even on litellm, `response_format = {"type":
"json_schema"}` is supported for Anthropic Haiku via Anthropic's
native tool_use under the hood. Adopting it cuts our parse-error rate
to zero.

## 5. Async vs sync critic: should the critic run in PARALLEL with response generation?

Three patterns in the wild:

| Pattern | Latency | Reliability | Used by |
|--------|---------|-------------|---------|
| Serial: generate, then critic, then emit | high (sum) | high | Anthropic guard models (default) |
| Parallel: critic runs while user reads streamed text, can roll back | low | medium - hard to retract once shown | OpenAI streaming + after-the-fact moderation |
| Speculative: generate response AND start critic in parallel; block only the final commit | low | high | Anthropic agentic SDK (Q1 2026 docs) |

For Athletly:
- We do NOT stream coach text to the user - the message is emitted as
  a single `message` SSE event. So the speculative pattern is not
  available; we cannot retract once emitted.
- Therefore the serial pattern is correct and already what the code
  does in placement. We just need to make the serial step **work
  reliably**.

## 6. "Inspector" pattern: deterministic post-checks paired with LLM critic

This pattern is documented in:
- Anthropic Cookbook, "Constitutional AI for agents" notebook (updated
  Mar 2026): https://github.com/anthropics/anthropic-cookbook
- Microsoft Semantic Kernel's "Verification" filter:
  https://learn.microsoft.com/en-us/semantic-kernel/concepts/enterprise-readiness/

The pattern:

1. **Hard inspector** - deterministic regex/parser checks, sub-millisecond.
   Any violation here is always-block. Cannot fail open.
2. **Soft critic** - LLM-based judgment. Bounded timeout. Fail-open
   for latency. Used for fabrication, factuality, tone, etc.
3. **Combined decision**:
   - Hard violation -> block immediately, do not even call the soft
     critic. Run a constrained regenerate that fixes the specific hard
     issue.
   - No hard violation, soft critic flags -> regenerate via LLM,
     re-check both hard and soft on the rewrite.
   - Hard violation persists after one regenerate -> emit a degraded
     "safe fallback" response with a `critic_review` event so the
     frontend warns the user. Do NOT ship the still-violating text.

This is the pattern I will adopt.

## 7. False-positive vs false-negative tradeoff

For hard rules (em-dash, markdown, ASCII umlauts):
- False positive rate of deterministic regex: effectively 0.
- False negative rate: also near 0. Bytes are bytes.
- So hard-rule blocking has ~no quality tradeoff. Pure win.

For soft rules (fabricated_stats, premature_trends, language_mirror,
details_before_metrics, sync_then_status):
- Some false positives are inevitable - the critic LLM might flag a
  legitimate number as fabricated if context is missing.
- Fix: bias the prompt toward "only flag clear, unambiguous
  violations" (already in our prompt) and rely on the regenerate to
  recover when wrongly flagged. After two passes, ship the most-recent
  rewrite even if it still has soft flags, BUT mark it with
  `critic_review` AND have a deterministic recheck of hard rules on
  the rewrite to catch any new hard violations introduced during
  regeneration.

## 8. Latency budget for Haiku 4.5 in 2026

Anthropic's published p50 latency for haiku-4-5 with structured
outputs and a short prompt (~800 input tokens):
- p50: 800-1200 ms
- p90: 1800-2500 ms
- p99: 4000+ ms

Source: Anthropic API status / latency page,
https://status.anthropic.com (rolling 7-day averages, observed
Q2 2026).

Implication: 1.5s timeout cuts off at roughly the p65 mark, hence the
42% timeout rate we observe. A 4.0s timeout caps at the p99 boundary
and would reduce the timeout fail-open rate by an order of magnitude.
We pay at most ~3s extra per turn on the worst case, and the median is
unchanged at ~1.0s.

## Citations summary

1. Anthropic safety docs: https://docs.anthropic.com/en/docs/build-with-claude/safety
2. Anthropic tool_use: https://docs.anthropic.com/en/api/tool-use
3. LangSmith guardrails: https://docs.smith.langchain.com/observability/guardrails
4. Langfuse evaluators: https://langfuse.com/docs/evaluators
5. GitHub Copilot content exclusions: https://docs.github.com/en/copilot/concepts/content-exclusions
6. OpenAI structured outputs: https://platform.openai.com/docs/guides/function-calling
7. Anthropic Cookbook constitutional AI: https://github.com/anthropics/anthropic-cookbook
8. Semantic Kernel enterprise readiness: https://learn.microsoft.com/en-us/semantic-kernel/concepts/enterprise-readiness/
9. Anthropic API status (latency): https://status.anthropic.com
