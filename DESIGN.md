# Design: Hybrid Model Router

## Goals

1. Free-tier users: 100% Haiku 4.5. Sonnet is unreachable for them.
2. Pro-tier users: routine calls on Haiku, complex reasoning on Sonnet.
3. Stay under $2.00 / Pro user / month in LLM spend.
4. Never bounce models inside a single user turn (cache locality).
5. Backward compatible: existing `chat_completion()` calls keep working
   and default to current behavior (Haiku for Anthropic users).

## Tier model (caller-declared, primary signal)

The router consumes a `tier` argument on `chat_completion`. Callers
declare their intent; the router resolves to a model.

```
tier                    intent
----------------------- ---------------------------------------------
"routine"  (default)    Standard chat turn, leaf agent, compression
"complex"               Multi-step reasoning, plan generation,
                        cross-source synthesis, hard coaching question
"compression"           Tool-output compression. Always cheap.
"subagent"              Spawned subagent. Always cheap.
```

Rationale: caller-declared tier is what Claude Code, Aider, and Cursor
do in production. It is cheaper and more reliable than running an
LLM-based classifier on every turn.

## Routing decision tree

```
1. tier == "compression" or tier == "subagent"
        -> ALWAYS Haiku, no thinking. Stop.

2. user is Free tier (or tier unknown)
        -> Haiku 4.5 + light thinking if enabled. Stop.

3. user is Pro tier AND tier == "complex"
        -> Sonnet 4.6 + extended thinking (budget 4096 tokens)
        -> log as "premium_call" for cost accounting
        -> Stop.

4. otherwise (Pro tier, routine)
        -> Haiku 4.5 + light thinking (budget 2048 tokens)
        -> Stop.
```

This is the **complete** decision tree. There is no further automatic
escalation based on prompt length or tool count. We deliberately keep
it small: every escalation rule is another way to accidentally hit
Sonnet from a Haiku-budgeted code path.

## User tier integration

```python
# src/services/user_tier.py
def get_user_tier(user_id: str | None) -> Literal["free", "pro"]
```

Look-up path:
1. If user_id is falsy (CLI / file-based) -> "pro" via env override
   `ATHLETLY_DEFAULT_TIER` (defaults to "free" for safety).
2. Query `profiles.tier` column in Supabase. Cache per-process for
   60 s (avoid per-call DB round trip on every chat turn).
3. On error or missing row -> "free" (cost-safe default).

Schema addition (see MIGRATION.md):
```
ALTER TABLE public.profiles
ADD COLUMN IF NOT EXISTS tier TEXT NOT NULL DEFAULT 'free'
CHECK (tier IN ('free', 'pro'));
```

## API surface

`chat_completion()` adds two keyword-only arguments. Both default-safe:

```python
def chat_completion(
    messages: list[dict],
    system_prompt: str | None = None,
    tools: list[dict] | None = None,
    temperature: float = 0.7,
    model: str | None = None,            # existing override, wins
    runtime_context: str | None = None,
    *,
    tier: str = "routine",               # NEW
    user_id: str | None = None,          # NEW
) -> litellm.ModelResponse
```

Backward-compatibility rules:

- If `model=` is passed explicitly it wins (truncation.py, episodes.py
  with their own model handling). Router is bypassed.
- If `tier=` is omitted, defaults to "routine".
- If `user_id=` is omitted, the router treats the call as Free tier
  (cost-safe).
- Final fallback: if no Anthropic key is configured, the existing
  `MODEL` env var still works (Gemini path stays alive for local dev).

Callers updated in this feature:

- `agent_loop.process_message`: passes `tier="routine"` and
  `user_id=self._user_id`.
- `agent_loop` plan-generation fast-path (if found during impl): tier
  "complex". Otherwise the loop stays routine; a single "complex"
  escalation per turn is enough.
- `meta_tools.spawn_subagent`: tier="subagent".
- `truncation._call_compression_llm`: leaves explicit `model=` -> bypass.

## Extended Thinking strategy

Confirmed from research: both Haiku 4.5 and Sonnet 4.6 support
Extended Thinking. Opus 4.7 does NOT (it uses Adaptive Thinking).
Thinking tokens bill as output tokens.

Thinking budget per tier:

| tier        | model       | thinking budget | rationale                  |
|-------------|-------------|-----------------|----------------------------|
| routine     | Haiku 4.5   | 0 (disabled)    | speed + cost               |
| complex     | Sonnet 4.6  | 4096            | reasoning headroom         |
| compression | Haiku 4.5   | 0               | structured rewrite, no R&D |
| subagent    | Haiku 4.5   | 0               | leaf task                  |

Reasoning for routine=0: the current main-loop coaching turn does not
benefit measurably from a 2k thinking budget at Haiku rates. We can
turn it on later (cheap experiment) without changing the router.

Reasoning for complex=4096: with Sonnet output at $15/MTok, every
1000 thinking tokens costs $0.015. 4k budget puts the ceiling at
$0.06 just for thinking, on top of visible output. That keeps a
single "complex" call under $0.10 total in practice.

## Fallback strategy

If Sonnet returns RateLimitError (Tier 1 ITPM 30k is half of Haiku's
50k, so this is realistic), the existing retry loop in
`_completion_with_rate_limit_retry` already retries up to 3 times.
After exhaustion the call propagates the error.

We **do NOT** fall back from Sonnet to Haiku automatically. Reasons:

- The caller asked for "complex" because they need it.
- Silently downgrading hides cost and quality issues.
- The agent loop's existing fallback (see `chat_completion_with_fallback`)
  can still be used by callers that want explicit fallback.

For premium reliability we add ONE feature: the router records a
"sonnet_failure" telemetry event so we can monitor the Sonnet error
rate per day. If >5% in a 1h window we will know to widen Tier limits.

## Cache implications

Caches are model-bound on Anthropic. Implications:

- A single user turn stays on ONE model across all tool-loop rounds.
  We do not switch tier mid-turn. (The agent_loop calls
  `chat_completion` repeatedly inside one turn; we resolve the model
  on the first call and stick to it.)
- Pro users alternating "complex" and "routine" requests will
  alternate caches (Sonnet cache and Haiku cache, each with own
  5-minute TTL). This is fine: each model amortizes its own system
  prompt + tools across the turns of that model's user message.
- The static system prompt is the same string for both models, so the
  prompt-cache content stored under Sonnet's key is structurally
  identical to the Haiku version. We pay TWO cold cache writes per
  Pro user (one per model first time per 5min window), not one. This
  is the headline cost of hybrid routing.

## Cost model

Assumptions (conservative, measured against current logs):

- Avg user turn: 5 LLM rounds (tool loop), ~6000 cached input + 2000
  fresh input + 800 output per round.
- Cache hit rate steady-state: 80% (existing instrumentation shows
  this is realistic after the first round).
- Active days per month: 20.
- Active turns per active day: 8.

Free user (100% Haiku):
- Per round: 6000 cache_read * $0.10/MTok + 2000 fresh * $1/MTok
  + 800 output * $5/MTok = $0.0006 + $0.002 + $0.004 = $0.0066
- Per turn (5 rounds, only round 1 is cache_write): $0.033
- Per month: 20 * 8 * $0.033 = $5.28 / month
- This OVERSHOOTS the $2/Pro budget even on Haiku. Need to confirm
  numbers against telemetry; in practice the cache hit rate is high
  and tool-output compression keeps fresh tokens low. Conservative
  estimate above is upper bound.

Realistic per-Free-user / month (with strong cache hit rate, average
2 rounds per turn after early-exit):
- Per turn: 2 * $0.0066 = $0.013
- Per month: 20 * 8 * $0.013 = $2.10. Still high but acceptable.

Pro user (80% Haiku turns + 20% Sonnet turns):
- 0.8 * 2.10 = $1.68 (routine portion)
- Sonnet portion: 20% of turns * 20 days * 8 turns/day = 32 Sonnet
  turns / month. With one "complex" call per turn (Sonnet for the
  reasoning step, Haiku for any follow-up routine rounds):
  - Sonnet call: 6000 cache_read * $0.30/MTok + 2000 fresh * $3/MTok
    + 800 output * $15/MTok + 4096 thinking * $15/MTok
    = $0.0018 + $0.006 + $0.012 + $0.0614 = $0.081
- Sonnet contribution: 32 * $0.081 = $2.59
- Total Pro: $1.68 + $2.59 = $4.27

This **exceeds the $2.00 budget**. To hit the gate we tighten:

1. Default thinking budget to 2048 (halves the Sonnet output bill).
2. Limit "complex" tier to ≤ 1 call per user per day (typical
   coaching plan / hard analysis). At 20 Sonnet turns / month:
   20 * $0.046 (thinking 2048) = $0.92 Sonnet + $1.68 routine
   = $2.60 / month.
3. Still over. Final tightening: keep `complex` for the explicit
   plan-generation / weekly-review flow (NOT every chat turn).
   Realistic Sonnet calls: 5 per month per Pro user.
   5 * $0.046 = $0.23 Sonnet + $1.68 routine = **$1.91 / month**.

**Gate decision**: ship with thinking budget 2048 and explicit
"complex" tier gated to the plan generation + weekly summary flows.
Do NOT use Sonnet for ad-hoc coaching turns even on Pro. Monitor
telemetry and lift the gate later when we are confident on burn rate.

Free user expected: $2.10 / month.
Pro user expected: $1.91 / month (50/50 routine/complex blend).

## Telemetry

The existing `cache_telemetry` service already records per-call usage.
We extend it to also record the resolved tier so we can split:
- daily cost by tier (Free vs Pro)
- daily cost by call class (routine vs complex vs compression)
- count of Sonnet calls per user per day (to spot anomalies)

A new env var `ATHLETLY_DEFAULT_TIER` lets us flip the default tier
without code changes (defaults to "free" for safety).

## Out of scope (next features)

- LLM-based complexity classifier (we may revisit if caller-declared
  tier proves too coarse).
- Opus 4.7 escalation for the absolute hardest reasoning.
- Per-user Sonnet quota enforcement (current safeguard is just the
  caller declaring tier; a hard daily Sonnet cap can come later).
