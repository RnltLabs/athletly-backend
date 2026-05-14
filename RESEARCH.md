# Feature 6 Research: Prompt A/B Metrics and STRICT Rule Telemetry

State: Q2 2026. Lead engineer notes; cite-or-die.

## 1. Prompt-level observability tools (Q2 2026)

| Tool | What it tracks | OSS-friendliness | Notes |
|---|---|---|---|
| LangSmith | Traces (LLM calls, tool calls, latencies), datasets, evaluators (LLM-as-judge), prompt versions, A/B compare runs, token costs | Proprietary, paid SaaS. SDK is open. | Owned by LangChain. Strong "evals" tab, supports custom evaluators + regex graders. https://docs.smith.langchain.com/observability and https://docs.smith.langchain.com/evaluation |
| Helicone | LLM proxy logs (request/response, cost, latency), prompt versioning, experiments, user feedback signals | OSS-friendly (helicone/helicone, MIT). Self-host supported. | Proxy-based. Lower-friction integration via base_url swap. https://docs.helicone.ai/features/prompts and https://github.com/Helicone/helicone |
| Langfuse | Traces, scores, datasets, prompt management, evaluations, sessions, users. Score = numerical/categorical quality tag. | OSS (langfuse/langfuse, MIT). Self-host first. | Built around "scores" - first-class concept for tracking quality signals tied to traces. https://langfuse.com/docs/scores/overview and https://github.com/langfuse/langfuse |
| Weights and Biases Weave | Traces, datasets, evaluations, models. Tight Python decorator API (`@weave.op`). | Proprietary SaaS, free tier. SDK is OSS. | Best for ML-heavy teams already on W&B. https://wandb.github.io/weave/ |
| Arize Phoenix | OpenTelemetry-native tracing for LLMs. Self-host first. | OSS (Arize-ai/phoenix, Elastic v2). | OTel standard. Pairs with Arize SaaS for prod. https://docs.arize.com/phoenix |

Common OSS-friendly pattern: **traces + scores**. A trace is one LLM call (or full agent run). A score is a key-value tag attached to a trace ("rule_violation:em_dash=1" or "quality:0.7"). All five tools converged on this. The scoring API is what Feature 6 needs.

What we don't need: full distributed-trace infra. A counter buffer + Supabase table covers 95% of value at 0% cost.

## 2. Anthropic's recommended approach

Key Anthropic engineering posts read for this feature:

- "Building effective agents" (Anthropic, 2024-12-19) - https://www.anthropic.com/engineering/building-effective-agents - "Measure performance and iterate. Add complexity only when simpler solutions fall short." Implies: ship the metric first, then change the prompt.
- "How we built our multi-agent research system" (Anthropic, 2025-06-13) - https://www.anthropic.com/engineering/built-multi-agent-research-system - emphasizes LLM-as-judge for hard-to-rubric outputs (correctness, completeness), regex/heuristic for objectively-checkable rules.
- "Effective context engineering for AI agents" (Anthropic, 2025-09-29) - https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents - sub-agents and offline evaluation of context strategies. Recommends sampling production traces to a side eval set rather than scoring every request.
- "Claude's tool use guide" (Anthropic docs) - https://docs.claude.com/en/docs/build-with-claude/tool-use - tool errors are a quality signal: a high tool-error rate often means the prompt is misleading the model.

Anthropic's pattern in short: 
1. Define rubrics (binary, objective where possible).
2. Score in two tiers: cheap heuristics in prod, LLM-as-judge offline on samples.
3. Iterate prompt against the rubric, not against feels.
4. Use prompt caching: stable prompts = better cache hits = lower cost + faster. A prompt that drifts a lot under "fixes" is a smell.

## 3. How do production agents detect rule violations post-hoc?

Three patterns, in increasing cost/sophistication:

**(a) Pure regex / pattern matching** (cheap, deterministic, used here for the easy rules).
- Em-dash present? `re.search(r'—', text)`.
- Markdown header? `re.search(r'^#{1,6}\s', text, re.MULTILINE)`.
- ASCII transliteration in German? `re.search(r'\b(ueber|fuer|Praeferenz|Maerz|Laeufer|Groesse|ueberfaellig)', text)`.
- Strengths: zero cost, deterministic, runs in microseconds in prod.
- Weakness: false positives (a code snippet mentioning `*foo*` is not a Markdown italic).

**(b) Heuristic state checks** (cheap, deferred from this PR).
- "Trends claimed with <5 data points": needs to inspect tool-call history of the turn, not just the final response. Requires tool-trace integration.
- "Stats fabricated without get_activity_details": same, needs tool history.

**(c) Secondary LLM call ("LLM-as-judge")** (expensive, defer for now).
- Sample a small fraction of conversations (1% to 5%) offline.
- Run Haiku with a checklist prompt: "Did the assistant follow these N rules? Return JSON: {rule_id: bool}."
- Cost: ~$0.001 per scored conversation at Haiku 4.5 prices. Cheap if sampled, prohibitive if every turn.
- Reference: https://www.anthropic.com/engineering/built-multi-agent-research-system describes this exact pattern for their research agent.

Industry references:
- LangSmith "Online Evaluators": https://docs.smith.langchain.com/evaluation/concepts#online-evaluators
- Langfuse "Model-based Evaluations": https://langfuse.com/docs/scores/model-based-evaluations
- OpenAI Evals: https://github.com/openai/evals

## 4. A/B testing prompt variants: what metrics matter?

User-perceived quality is hard to measure directly (no thumbs-up rate yet). Use proxies, ranked by signal strength:

| Metric | Signal direction | Why it works |
|---|---|---|
| Rule-violation rate per 1k turns | Lower = better | Direct measure of prompt-rule adherence. Strict, objective. |
| Tool-error rate (% of tool calls that errored) | Lower = better | Bad prompts cause the model to invent tool args or call the wrong tool. |
| Avg tool calls per turn | Stable = better | Sudden jumps mean the model is fishing for context. Drops to near-zero may mean it stopped grounding claims. |
| Avg conversation length to resolution | Lower = better | Bad prompts force users to clarify or correct. |
| Self-correction rate | Lower = better | If the model emits a rule violation and then walks it back, the prompt is unclear. |
| Cache hit rate per variant | Higher = better (PROXY for stability) | A prompt that drifts under iteration loses cache hits. See section 5. |
| User correction rate ("nein, ich meinte...", "das war falsch") | Lower = better | Heuristic: detect German/English correction phrases in next user turn. Future work. |
| Response length distribution | Stable within style = better | Sudden lengthening on `OUTPUT_STYLE=concise` means the prompt is being ignored. |

What does NOT work as a single metric:
- Pure latency: bad proxy for quality. A fast wrong answer is still wrong.
- Single tool-call count: a complex question legitimately needs many tools.
- Response length alone: depends on the question.

Industry references:
- Microsoft "Evaluating generative AI applications": https://learn.microsoft.com/en-us/azure/ai-studio/concepts/evaluation-approach-gen-ai
- LangSmith pairwise eval: https://docs.smith.langchain.com/evaluation/how_to_guides/evaluation/evaluate_pairwise

## 5. Can prompt-cache hit rate per variant be a quality signal?

**Yes, but as a stability proxy, not a quality proxy directly.**

Mechanism (Anthropic prompt caching, https://docs.claude.com/en/docs/build-with-claude/prompt-caching):
- Anthropic caches the prefix of the system prompt up to a cache breakpoint. Identical prefix = cache hit on the next call. The cache TTL is 5 minutes (ephemeral) or 1 hour (extended).
- Hit rate goes down when:
  - The static prompt itself changes between deployments (expected, transient).
  - The "static" prompt is accidentally taking runtime data (a bug; our `STATIC_SYSTEM_PROMPT` is constant, so this is not us).
  - Traffic falls below the rate needed to keep the cache warm.
- For A/B testing: variant A and variant B have different prefixes, so they DO NOT share cache. Each variant needs enough traffic to sustain its own cache.

The quality signal:
- A variant we keep "fixing" with small edits will show a sawtooth cache hit rate (drop on every redeploy). That sawtooth is a quality smell: the prompt is unstable.
- A variant that holds steady at high cache hit rate AND has low violation rate is the keeper.

Treat cache hit rate as a confounded but useful tertiary signal:
1. Primary: violation rate.
2. Secondary: tool-error rate.
3. Tertiary: cache hit rate (stability indicator).

We already have `cache_telemetry` capturing cache hit rate per model. To split per variant we will tag each LLM call with the variant id and read it back per group. That is a Phase 2+ extension; the scaffold goes in this PR but the variant-aware cache split is deferred.

## Decisions for Feature 6

1. Storage: in-memory ring buffer (live) + optional Supabase table (history). Both ship behind a feature flag. Default to in-memory only; flip to "+supabase" when the table exists and `use_supabase=true`.
2. Detection: regex-only in the production path. No LLM calls. Defer LLM-as-judge to a follow-up offline worker.
3. A/B scaffold: `PromptVariant` enum with one value (`DEFAULT`) plus a placeholder routing function. No actual variants exist yet. Telemetry already tags every record with the variant, so adding variant B is a one-line change.
4. Alerting: structured `logger.warning` when violation rate over the last 60s exceeds 5% of recorded responses. No external alerting hooks; rely on existing log aggregation.
5. Per-rule cost ceiling: each detector must run in under 1ms on a 2KB response. Verified in tests with `time.perf_counter`.

## URLs cited

- https://www.anthropic.com/engineering/building-effective-agents
- https://www.anthropic.com/engineering/built-multi-agent-research-system
- https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
- https://docs.claude.com/en/docs/build-with-claude/tool-use
- https://docs.claude.com/en/docs/build-with-claude/prompt-caching
- https://docs.smith.langchain.com/observability
- https://docs.smith.langchain.com/evaluation
- https://docs.smith.langchain.com/evaluation/concepts#online-evaluators
- https://docs.smith.langchain.com/evaluation/how_to_guides/evaluation/evaluate_pairwise
- https://docs.helicone.ai/features/prompts
- https://github.com/Helicone/helicone
- https://langfuse.com/docs/scores/overview
- https://langfuse.com/docs/scores/model-based-evaluations
- https://github.com/langfuse/langfuse
- https://wandb.github.io/weave/
- https://docs.arize.com/phoenix
- https://learn.microsoft.com/en-us/azure/ai-studio/concepts/evaluation-approach-gen-ai
- https://github.com/openai/evals
