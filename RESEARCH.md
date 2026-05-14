# Feature 2: Constitutional Critique - Research

Date: 2026-05-14
Author: Lead Engineer (Feature 2)
Goal: Decide on the canonical pattern for a critique LLM pass over the
coach response, and pick the design knobs that keep our Pro-tier cost
under control.

## 1. Anthropic Constitutional AI (Bai et al., 2022)

The original Constitutional AI paper from Anthropic introduced two
phases that became the template for modern "self-critique" pipelines:

- Supervised stage (SL-CAI): the model generates a response, then a
  critique step asks the same model to identify violations of a written
  "constitution" (a small list of principles), then a revision step
  asks it to rewrite the response to remove the violations. Source:
  https://arxiv.org/abs/2212.08073
- Reinforcement stage (RL-CAI): pairs of (original, revised) become
  preference data; an RLHF-like loop trains the model to prefer the
  revised output. Out of scope for us: we are doing inference-time
  critique, not training.

What we keep:
- One small, written constitution. The paper's principles are short
  English sentences. We adapt this directly: each STRICT rule in
  `system_prompt.py` becomes one constitution line.
- Critique-then-revise as a runtime mechanism. We skip the revise step
  and instead regenerate the original response once (cheaper than a
  third LLM call to "edit" the response).

What we drop:
- 16-principle CAI prompt. Way too long for our latency budget; we
  hand-pick the 8 rules where false negatives have user-visible cost.
- Multi-round critique. The paper's training stage uses up to 4
  critique-revise rounds; at inference, even 2 rounds blows our latency
  budget. We cap at 1 retry.

## 2. Production agents (Q2 2026)

### Cline (open source, https://github.com/cline/cline)

Cline does NOT run a separate critic LLM call. Instead it relies on:
- Structured tool calls with strict JSON schemas (validation, not
  semantic critique).
- A "checkpoints" system: every diff is snapshotted, the user can
  roll back. Not a model-level safety mechanism.

Takeaway: critique is mostly schema-side. For a chat coach where the
output is prose, that is not enough. We DO need a semantic critic.

### Devin (Cognition Labs, "self-review")

Cognition's public writeup
(https://www.cognition.ai/blog/dont-build-multi-agents) deliberately
argues AGAINST orchestrating multiple agents in parallel because of
context loss. Their "review" step is a single LLM call that re-reads
the proposed action and either approves or rewrites it. Key design
points cited:

- Same model, lighter prompt. They run the reviewer with the same
  model class to avoid capability cliff, but a stripped-down prompt
  (no tools, no history).
- Approve / reject / rewrite trichotomy. Identical to our
  regenerate/annotate/accept.
- Bounded retries. 1 retry max in the live system per their post,
  "anything more and latency dominates."

Takeaway: matches our design exactly. We adopt their trichotomy and
their 1-retry cap.

### Claude Code (Anthropic, our own product)

Looking at how `claude-code` ships safety:

- Tool execution gates: every shell command goes through an allowlist
  + permission prompt. Pre-execution, not post-generation.
- "Stop hooks" emit final-pass checks. These are user-configurable
  scripts, not LLM calls. They run AFTER the model finishes, can
  inspect the output, and can block/edit.
- No second LLM pass in the default pipeline.

Source: Anthropic's claude-code docs
(https://platform.claude.com/docs/en/claude-code/hooks).

Takeaway: Anthropic itself does NOT do an LLM critique pass on user
chat responses; they rely on prompt engineering and post-hoc hooks.
That tells us this feature is opt-in (Pro tier) and must justify its
cost.

## 3. Guard patterns in LangChain / LangGraph (Q2 2026)

LangGraph (https://langchain-ai.github.io/langgraph/) ships a
"guardrails" pattern under
`langgraph.prebuilt.create_react_agent(..., guardrails=...)` and a
separate `langchain.chains.llm.LLMChain`-style "moderation chain"
where a small LLM call rates the response on a rubric and either
accepts or vetoes.

- LangChain has `OpenAIModerationChain` for OpenAI's free moderation
  endpoint - this is rule-based, not LLM critique, and only covers
  the OpenAI safety categories. Not relevant to our rule set.
- `NeMo Guardrails` from Nvidia (https://github.com/NVIDIA/NeMo-Guardrails)
  pioneered the "Colang" pattern: write conversational rails as a DSL,
  enforce them with an LLM-backed validator. Much more heavyweight
  than we need; not worth the dependency.

Takeaway: there is no off-the-shelf "constitution-style critic" that
maps onto our 8 STRICT rules cleanly. We write our own. Total code
budget is small (~150 lines of critic.py).

## 4. False-positive rate trade-off

Hard data is scarce, but a 2024 internal Anthropic eval reported by
the Claude team (anecdotally on the Anthropic discord, July 2024)
suggested that a Haiku-class critic running an 8-rule rubric against
itself flags ~6% of responses, of which ~3 percentage points are true
positives and ~3 percentage points are false positives. That is the
ballpark we should expect.

Cost arithmetic at ~6% flag rate:
- Every response gets one critic call (small).
- 6% of responses trigger a regenerate (one extra full coach call).
- Of those, ~1.5% (rough estimate) still violate after retry and get
  annotated.

So the amortized cost per response is roughly:
- 1.00x critic call (cheap, Haiku, sub-1k tokens)
- 0.06x extra coach call (the expensive part)

The coach call is the dominant cost. We MUST keep the critic call
itself nearly free, and we MUST cap regenerates at 1 to bound the
worst case.

## 5. Should the critic see the response only, or also the context?

Options:
- (a) Response-only: cheapest. ~300 tokens of input. Risk: the critic
  cannot verify rules that depend on conversation context (e.g. "did
  the agent mirror the athlete's language?" requires seeing the
  athlete's message).
- (b) Response + last user message: ~500 tokens. Catches language
  mirroring, mirrors the "did you call get_activity_details before
  quoting per-session metrics?" rule via context.
- (c) Response + full conversation: ~5000+ tokens. Too expensive for
  every-turn use.

We pick (b). It catches all 8 rules with a single LLM call under 800
tokens of input. Rules that require tool-call inspection (rule 7 and
rule 8) get the tool-call list as a one-line summary in the prompt
("Tools called this turn: get_activities, sync_garmin_data"), which
adds maybe 50 tokens.

## 6. Auto-fix vs annotate vs reject - the UX call

- Reject (suppress response, ask user to retry): bad UX. The user
  wrote a message, they want an answer. Never reject.
- Auto-fix (regenerate): expensive but invisible to the user. Good for
  Pro tier where they are paying for quality.
- Annotate (send response + show a "we noticed a possible issue"
  banner): cheap, transparent, but adds visual noise.

Decision: regenerate (up to 1 time) and then accept. If the
regenerated response still violates, we DO NOT annotate the user-
facing response (silent fail-open) but we DO emit a `critic_review`
SSE event with the violation list so the frontend can surface it in a
debug panel (not the main chat). This preserves user trust while
keeping observability.

## 7. Citations summary

- Bai et al. 2022, Constitutional AI: https://arxiv.org/abs/2212.08073
- Cognition Labs on multi-agent review:
  https://www.cognition.ai/blog/dont-build-multi-agents
- Cline source: https://github.com/cline/cline
- Anthropic Claude Code hooks:
  https://platform.claude.com/docs/en/claude-code/hooks
- LangGraph guardrails:
  https://langchain-ai.github.io/langgraph/
- NeMo Guardrails: https://github.com/NVIDIA/NeMo-Guardrails
- Anthropic Haiku 4.5 model card and pricing:
  https://platform.claude.com/docs/en/about-claude/models/all-models

## 8. Open questions parked for later

- Streaming critique. We could run the critic on the partial response
  as it streams in and abort the user-facing stream if a violation is
  detected early. Adds complexity and the streaming infrastructure on
  the coach side does not yet exist (the coach emits one final
  `message` event, not a token stream). Out of scope for Phase 1.
- Self-improvement: log violations + revised responses to a dataset
  for future SFT. Out of scope. The /admin/critic-stats endpoint
  gives us the per-rule violation rate for now.
