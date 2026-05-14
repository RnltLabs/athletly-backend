# Research: Hybrid Model Router + Extended Thinking (Q2 2026)

Research conducted on 2026-05-14 for Feature 1 of the athletly-backend
hybrid model routing initiative.

## 1. Does Claude Haiku 4.5 support Extended Thinking?

**Yes.** Confirmed by the official Anthropic model overview page
([models/whats-new-claude-4-5][1], [models/overview][2]).

The latest-models comparison table lists Extended Thinking support as:

- Claude Opus 4.7: No (uses Adaptive Thinking instead)
- Claude Sonnet 4.6: Yes
- Claude Haiku 4.5: Yes

Haiku 4.5 is in fact the first Haiku model to support Extended
Thinking. It does not support Adaptive Thinking (Sonnet/Opus only).

Confirmed identifiers:
- `claude-haiku-4-5-20251001` (or alias `claude-haiku-4-5`)
- `claude-sonnet-4-6` (dateless format, pinned snapshot)

## 2. Pricing impact of thinking tokens

**Thinking tokens are billed as output tokens at the standard model rate.**
([extended-thinking docs][3])

Pricing per million tokens (MTok), confirmed against
[pricing docs][4]:

| Model      | Input | Cache write 5m | Cache read | Output |
|------------|-------|----------------|------------|--------|
| Haiku 4.5  | $1.00 | $1.25          | $0.10      | $5.00  |
| Sonnet 4.6 | $3.00 | $3.75          | $0.30      | $15.00 |
| Opus 4.7   | $5.00 | $6.25          | $0.50      | $25.00 |

Key facts:
- Thinking tokens count toward the per-request `max_tokens`. The
  `budget_tokens` ceiling must be less than `max_tokens` (unless using
  interleaved thinking).
- "Summary" thinking display still bills the full thinking token count.
  `display: "omitted"` lowers latency but not cost.
- Thinking parameter changes (enabled/disabled or different
  `budget_tokens`) invalidate message-level cache breakpoints, NOT the
  system block cache. Static thinking settings keep caching healthy.
- Sonnet 4.6 and Haiku 4.5 both cap output at 64k tokens; thinking
  budget counts against that.

Cost example: a typical Sonnet call with 4k thinking + 500 visible
output = 4500 output tokens. At $15/MTok that is $0.0675 just for the
output. Add 8k cached input + 2k fresh input: $0.0024 + $0.006 = $0.009
input. Total roughly $0.077 per Sonnet reasoning call. Compare to an
all-Haiku version with 4k thinking + 500 output: 4500 * $5/MTok =
$0.0225 plus input ~$0.0028 = ~$0.025. Sonnet is 3x the cost of Haiku
on equivalent thinking workloads.

## 3. Hybrid routing pattern in production (Q2 2026)

**Industry standard is task-complexity routing with the cheap model as
the default and the strong model as the explicit escalation.**

What production agents do (from the comparisons summarized in
[claudelab][5], [padiso routing tree][6], [tech-insider][7]):

- **Claude Code** uses Sonnet as the workhorse and Haiku for the
  sub-agent / "low-stakes worker" tier. Opus is rare. Token
  efficiency wins because the orchestrator never asks Sonnet to do
  classification work.
- **Cursor**: explicit user-facing model selector (auto/Sonnet/Opus
  /GPT-x). The "auto" mode classifies the request with a cheap
  router model first.
- **Cline**: provider-agnostic; routes by user-set rule rather than
  by automatic complexity scoring. Heavily uses prompt caching to
  amortize tool definitions.
- **Aider**: lets the user pin a "main" and a "weak" model. Weak
  runs the diff/commit-message + edit-classifier; main runs the
  actual reasoning.

The shared pattern: **two-tier routing**, default cheap, escalate on
explicit signal (complexity score OR caller-declared tier). Three-tier
(adding Opus) is reserved for premium tiers in B2C apps.

Padiso's 2026 decision tree explicitly identifies the rule:
> Haiku 4.5 serves as the router, classifying incoming requests and
> handling simple ones directly. Sonnet 4.6 processes the bulk of
> medium-complexity tasks.

## 4. Heuristics used to decide Haiku vs Sonnet per call

Synthesizing the LiteLLM auto-routing scoring dimensions and what
production agents publish:

The seven-dimension auto-routing score (from [LiteLLM][8]) covers:
1. token count of the prompt
2. presence of code blocks
3. reasoning markers ("step by step", "analyze", "why")
4. technical-term density
5. simple-indicator markers ("hi", "thanks", short)
6. multi-step patterns (enumeration, plans)
7. question complexity (single vs nested questions)

Production agents do not run all seven. The dominant practical
heuristics are:

- **Caller-declared tier** ("complex", "routine", "compression"). This
  is what Claude Code and Aider rely on. Cheapest and most reliable:
  the calling code knows what it is asking for.
- **Tool count** > N triggers escalation. Sonnet handles multi-tool
  coordination materially better than Haiku.
- **Prompt size** above some threshold (e.g., 30k input tokens) is a
  weak escalation signal: large input correlates with synthesis tasks.
- **Subagent / leaf-task** -> always cheap. Subagents have one job and
  do not need frontier reasoning.

We will adopt: **caller-declared tier as primary, with tool-count
and "is this an orchestrator turn" as automatic escalations.**
We explicitly do NOT run a separate LLM-based classifier on every
turn (that would cost more than it saves at our volumes).

## 5. LiteLLM recommendation for multi-model routing (Q2 2026)

LiteLLM publishes two patterns ([LiteLLM Router docs][8],
[Auto Routing docs][9]):

1. **Router class** with declarative model_list + routing_strategy
   ("simple-shuffle" is default and recommended). Supports
   per-deployment rpm/tpm caps, fallbacks, weighted load balancing.
   Used for HA / scaling, not capability routing.
2. **Auto Routing**: LLM-based routing where a small classifier picks
   between deployments. Higher latency, requires API key in the
   classifier loop, expensive at our scale.

**For our use case (capability routing, not load balancing) LiteLLM's
recommendation is the simple Router pattern with a thin classifier in
front, OR direct deployment selection in application code.** Their
docs flag that auto-routing adds latency and a classifier API call;
they push folks to declarative routing for capability tiers.

We will NOT pull in the LiteLLM Router class. The current codebase
already calls `litellm.completion(...)` directly with a `model=` kwarg.
A single dispatcher function that picks the model string and calls the
existing chat_completion is simpler, has zero new dependencies, and
keeps cache behavior identical (Router can reorder calls in ways that
hurt the 5-minute TTL).

## 6. Gotchas with prompt caching across models

**Caches are model-bound.** Confirmed via [prompt-caching docs][10].

Implications for our design:

- A Haiku cache write is NOT readable on a Sonnet call. Switching
  models mid-conversation re-bills the entire prefix as fresh input.
  This is the single biggest cost gotcha and dictates our cache
  strategy: do NOT bounce between models inside a single user turn.
- Cache isolation is also per-workspace as of 2026-02-05 (Claude API
  + Claude Platform on AWS + Microsoft Foundry beta). Bedrock and
  Vertex AI remain org-level.
- Changing `thinking` parameters invalidates the message cache but
  not the system cache. So thinking must stay constant across all
  rounds of one user turn.
- Cache prefix order is `tools -> system -> messages`. A toggle on
  tools (e.g., introducing or removing tool_search) invalidates
  everything below.

Routing strategy that respects this:

1. **Decide model ONCE at the start of the user turn.** No mid-turn
   switching. The router picks Haiku-or-Sonnet for the entire tool
   loop of one user message.
2. **Cache key space is intrinsically separated by model**; we do not
   need to add model name to our cache_control keys. Anthropic does
   that for us.
3. **Sonnet "complex" turns will pay a cold cache write on round 1.**
   At a 3x cost multiplier and a 1.25x cache-write multiplier, the
   single cache-write penalty on Sonnet system prompt + tools is the
   most expensive single moment in our pipeline. Worth budgeting.
4. **Compression (truncation.py) keeps using Haiku** regardless of the
   parent call's tier. Compression is a leaf task with its own message
   structure that has nothing in common with the agent loop prefix.

---

## Sources

[1]: Anthropic, "What's new in Claude 4.5":
     https://platform.claude.com/docs/en/about-claude/models/whats-new-claude-4-5
[2]: Anthropic, "Models overview" (latest-models comparison table):
     https://platform.claude.com/docs/en/about-claude/models/overview
[3]: Anthropic, "Building with extended thinking":
     https://platform.claude.com/docs/en/build-with-claude/extended-thinking
[4]: Anthropic, "Pricing":
     https://platform.claude.com/docs/en/about-claude/pricing
[5]: Claude Lab, "Sonnet 4.6 vs Haiku 4.5 selection guide":
     https://claudelab.net/en/articles/claude-ai/claude-sonnet-46-vs-haiku-45-model-selection-guide
[6]: Padiso Blog, "Sonnet 4.6 vs Haiku 4.5 routing decision tree":
     https://www.padiso.co/blog/claude-sonnet-4-6-vs-haiku-4-5-model-routing-decision-tree/
[7]: Tech Insider, "Claude Opus 4.6 vs Sonnet 4.6 vs Haiku 4.5":
     https://tech-insider.org/claude-opus-vs-sonnet-vs-haiku-2026/
[8]: LiteLLM, "Router - Load Balancing":
     https://docs.litellm.ai/docs/routing
[9]: LiteLLM, "Auto Routing":
     https://docs.litellm.ai/docs/proxy/auto_routing
[10]: Anthropic, "Prompt caching":
      https://platform.claude.com/docs/en/build-with-claude/prompt-caching
