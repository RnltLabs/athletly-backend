# Sprint F Phase 2: Research

Production patterns for "policy-enforced retry with tool requirements"
as of Q2 2026.

## 1. Anthropic native: `tool_choice` flavours

Anthropic exposes four `tool_choice` modes ([Tool use overview](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview),
[Tool choice cookbook](https://platform.claude.com/cookbook/tool-use-tool-choice)):

| mode                                | meaning                                                          |
|-------------------------------------|------------------------------------------------------------------|
| `{"type": "auto"}`                  | model decides (default when tools present)                       |
| `{"type": "any"}`                   | model MUST call SOME tool, choice is free                        |
| `{"type": "tool", "name": "X"}`     | model MUST call tool X exactly                                   |
| `{"type": "none"}`                  | tools hidden, free-text only                                     |

Key behaviour: when `tool_choice` is `any` or `tool`, the API prefills
the assistant message so the model emits NO natural-language preamble,
only `tool_use` blocks. This is exactly what we want for a forced
gather step.

Anthropic also offers `disable_parallel_tool_use`. With
`tool_choice={"type":"any","disable_parallel_tool_use": true}` the
model is constrained to "use exactly one tool" per call. For our
multi-tool gates (e.g. temporal_freshness needs `sync_garmin_data` then
`get_provider_status` then `get_activities`) we deliberately leave
parallel use ENABLED so all three can fire in one round.

Caveat from the wild: Claude Opus 4.6 / Sonnet 4.6 currently regress on
parallel tool calls in Batch API ([anthropic-sdk-typescript#956](https://github.com/anthropics/anthropic-sdk-typescript/issues/956)),
producing one call per response even when the prompt asks for many.
Non-Batch streaming and live request paths still parallelise. We are on
live request path, so we expect parallelisation to work; the
implementation tolerates either by re-entering the loop until the gate
is satisfied or the per-turn cap is hit.

Sources: [Tool use overview](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview),
[Anthropic cookbook tool_choice](https://github.com/anthropics/anthropic-cookbook/blob/main/tool_use/tool_choice.ipynb),
[Implement tool use](https://docs.claude.com/en/docs/agents-and-tools/tool-use/implement-tool-use).

## 2. LangChain / LangGraph: forced tool calling

LangChain has standardised `tool_choice` across providers
([Standardized forced tool calling in LangChain](https://changelog.langchain.com/announcements/standardized-forced-tool-calling-in-langchain)).
Accepted values:

- `auto`: default
- `any` or `required`: must call some tool (OpenAI uses `required`,
  Anthropic uses `any`, the wrapper normalises)
- `<tool name>`: must call that specific tool

LangGraph v1 `create_agent` deprecated passing bound models;
`tool_choice` must now be set per node ([Forcing tool calls in
Langchain/Langgraph v1 create_agent](https://forum.langchain.com/t/forcing-tool-calls-in-langchain-langgraph-v1-create-agent/1898)).

The pattern that matters for us:

  1. Run a free agent step.
  2. Observe the output against a validator.
  3. If the validator fails because a required side effect is missing,
     route to a dedicated node whose ONLY job is to call the missing
     tool(s) with `tool_choice="required"`.
  4. After the forced call, re-run the validator.

This is precisely the Tool-Forcing Regenerate pattern we adopt.

Sources: [How to force tool calling behavior](https://js.langchain.com/docs/how_to/tool_choice/),
[How to force models to call a tool](https://python.langchain.com/v0.2/docs/how_to/tool_choice/),
[Forcing tool calls in Langgraph v1](https://forum.langchain.com/t/forcing-tool-calls-in-langchain-langgraph-v1-create-agent/1898).

## 3. LiteLLM and pass-through

We use LiteLLM as a multi-provider wrapper. LiteLLM accepts the OpenAI
`tool_choice` parameter and forwards it to Anthropic's `tool_choice`
([LiteLLM Anthropic provider](https://docs.litellm.ai/docs/providers/anthropic)).
Mapping is automatic for `auto` and `none`; for `required` LiteLLM
translates to Anthropic's `any`.

For our codebase the immediate decision is: do we pass `tool_choice`
through `chat_completion` for the regenerate call, or do we steer with
prompt-engineering alone?

Trade-off:

- `tool_choice` is a hard guarantee at the API level: the model
  CANNOT respond with plain text on that call. Pro: deterministic. Con:
  requires plumbing through `chat_completion(...)` which today does
  not surface `tool_choice` as a parameter.
- prompt-only steering is soft: relies on the model obeying an
  explicit instruction. Pro: zero plumbing, fail-safe (if the model
  ignores us we still get a text fallback). Con: not deterministic.

Decision (see DESIGN.md): **combine both**. Plumb a minimal
`tool_choice` parameter through `chat_completion` AND craft an explicit
imperative "REQUIRED: call X then Y" system reminder. Belt and braces.
Fallback if `tool_choice` is unsupported by the resolved provider:
prompt-only steer still works at ~80% reliability based on the
language_mirror baseline.

Sources: [LiteLLM Anthropic provider](https://docs.litellm.ai/docs/providers/anthropic),
[LiteLLM Anthropic programmatic tool calling](https://docs.litellm.ai/docs/providers/anthropic_programmatic_tool_calling).

## 4. Avoiding infinite loops: budget caps

Production retry guidance ([buildmvpfast 2026](https://www.buildmvpfast.com/blog/idempotent-ai-agent-retry-safe-patterns-production-workflow-2026),
[Fastio retry patterns](https://fast.io/resources/ai-agent-retry-patterns/)):

- Cap retries at 3-5 attempts, exponential backoff (1s, 2s, 4s).
- Idempotent side effects: tool calls during retry must not double-bill.
- Distinguish transient (rate limit, 5xx) from logical (validator says
  no) failures. Logical failures should NOT retry the same call shape;
  they should escalate or degrade.

Apache Airflow's `AgentOperator(durable=True)` ([buildmvpfast 2026](https://www.buildmvpfast.com/blog/idempotent-ai-agent-retry-safe-patterns-production-workflow-2026))
caches LLM and tool outputs across retries so a retry never re-bills
the same call.

For us the per-turn cap is already 1 (set by
`_gates_regenerated_this_turn`). We keep that cap. We do NOT loop the
forced-tool regenerate beyond a single extra round. If the forced
round still fails the gate the response ships with
`critic_review degraded=true`.

## 5. UX impact: when even the forced retry fails

Three documented outcomes from production retry literature ([Braintrust
2026](https://www.braintrust.dev/articles/agent-observability-complete-guide-2026)):

1. retry succeeds: ship the rewritten response, increment a success
   counter, no user-visible artefact.
2. retry attempt itself errors (timeout, rate limit): ship the original
   response with a soft annotation. Do not block the user.
3. retry runs to completion but the validator still fails: emit a
   structured "degraded" signal alongside the response. User sees the
   answer; observability sees the degradation; ops can act.

We use exactly this taxonomy. The third case maps to our existing
`critic_review` SSE event; we add `degraded=true` so the dashboard can
distinguish "gate failed, regen failed" from "gate passed". The
response still ships rather than blocking the user, consistent with
the project's existing fail-open policy.

## 6. Why the rewrite-only prompt was structurally wrong

LLMs are unreliable at meta-instructions that contradict their own
action space ([Databricks agent design patterns](https://docs.databricks.com/aws/en/generative-ai/guide/agent-system-design-patterns)):
when you tell a model "you must say X but you may not do the action
that lets you know X", you create an impossible task. The current
`regenerate_after_gates` prompt does exactly that for tool-required
gates: it forbids tool calls AND demands fixing a constraint whose
only fix is a tool call. The model's best legal response is to remove
the offending content (drop stats, dodge the freshness question, etc.),
which then fails the SECOND gate pass because the gate looks at the
TOOL set, not just the response text. This is the structural bug.

The fix is to put the model in a position where the easiest path to
success is the one the gate wants: re-enter the tool loop with an
explicit instruction. LLMs are reliable INSTRUCTION-followers (the
exact framing the briefing uses) when the instruction maps cleanly to
their action space.

## Sources

- [Tool use with Claude](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview)
- [Anthropic cookbook tool_choice](https://github.com/anthropics/anthropic-cookbook/blob/main/tool_use/tool_choice.ipynb)
- [Implement tool use (Anthropic)](https://docs.claude.com/en/docs/agents-and-tools/tool-use/implement-tool-use)
- [Claude tool choice cookbook](https://platform.claude.com/cookbook/tool-use-tool-choice)
- [LangChain how-to force tool calling (JS)](https://js.langchain.com/docs/how_to/tool_choice/)
- [LangChain how-to force tool calling (Python)](https://python.langchain.com/v0.2/docs/how_to/tool_choice/)
- [LangChain changelog: standardized forced tool calling](https://changelog.langchain.com/announcements/standardized-forced-tool-calling-in-langchain)
- [LangGraph forum: force tool calls in v1](https://forum.langchain.com/t/forcing-tool-calls-in-langchain-langgraph-v1-create-agent/1898)
- [LiteLLM Anthropic provider](https://docs.litellm.ai/docs/providers/anthropic)
- [LiteLLM programmatic tool calling](https://docs.litellm.ai/docs/providers/anthropic_programmatic_tool_calling)
- [Idempotent AI agents (BuildMVPFast 2026)](https://www.buildmvpfast.com/blog/idempotent-ai-agent-retry-safe-patterns-production-workflow-2026)
- [AI agent retry patterns (Fastio 2026)](https://fast.io/resources/ai-agent-retry-patterns/)
- [Agent observability complete guide (Braintrust 2026)](https://www.braintrust.dev/articles/agent-observability-complete-guide-2026)
- [Agent system design patterns (Databricks)](https://docs.databricks.com/aws/en/generative-ai/guide/agent-system-design-patterns)
- [9 components production agent harness (MindStudio)](https://www.mindstudio.ai/blog/9-components-production-agent-harness)
- [Agents at work 2026 playbook (PromptEngineering.org)](https://promptengineering.org/agents-at-work-the-2026-playbook-for-building-reliable-agentic-workflows/)
