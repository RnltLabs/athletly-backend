# Feature 2: Constitutional Critique - Design

Date: 2026-05-14
Status: Final, ready to implement.

## 1. The constitution (8 STRICT rules)

Extracted verbatim from `src/agent/system_prompt.py`. Each rule has a
short id we use in the critic JSON output.

| id | Rule (compact form for the critic prompt) |
|----|-------------------------------------------|
| no_em_dash | The response MUST NOT contain em-dash (U+2014) or en-dash (U+2013). Only hyphen-minus is allowed. |
| no_markdown | The response MUST NOT contain Markdown formatting: no `**bold**`, no `__bold__`, no `*italic*`, no `# heading`. |
| umlauts | When the response is in German, real umlauts (ae, oe, ue, ss) MUST be used. ASCII transliteration (ae, oe, ue, ss for ae, oe, ue, ss) is forbidden. |
| no_fabricated_stats | The response MUST NOT cite numeric stats (pace, distance, HR, power, VO2max) for an activity unless `get_activities` or `get_activity_details` was called this turn. |
| no_premature_trends | The response MUST NOT claim a trend, improvement, or decline based on fewer than 5 data points / sessions. |
| language_mirror | The response language MUST match the athlete's last message language. No mid-response code-switching. |
| details_before_metrics | If the response discusses per-session VO2max / threshold / FTP / training load, `get_activity_details` MUST have been called this turn. |
| sync_then_status | If `sync_garmin_data` was called this turn, `get_provider_status` MUST also have been called this turn. |

We did NOT add a 9th rule for `save_plan` (the system prompt's
"persist any multi-week plan" rule). Reason: detecting "is this a
plan?" reliably requires another LLM call. Leave it for a follow-up.

## 2. Critic prompt template

The critic gets a single user message. System prompt is short and
static (cacheable). User content carries the variable bits.

System prompt (cached, ~250 tokens):

```
You are a STRICT rule-checker. Score a coach response against 8
rules. Respond ONLY with valid JSON. No prose, no Markdown.

Rules:
1. no_em_dash: no U+2014 or U+2013 in response_text
2. no_markdown: no **bold**, __bold__, *italic*, or # headings
3. umlauts: if response language is German, MUST use ae oe ue ss
   not ae oe ue ss
4. no_fabricated_stats: numeric stats for an activity are allowed
   ONLY if tools_called includes get_activities or
   get_activity_details
5. no_premature_trends: trend claims need 5+ sessions of data
6. language_mirror: response language must match athlete language
7. details_before_metrics: per-session VO2max/threshold/FTP/load
   requires get_activity_details in tools_called
8. sync_then_status: if tools_called includes sync_garmin_data, it
   MUST also include get_provider_status

Output schema:
{
  "violations": [
    {"rule": "<rule_id>", "reason": "<one short sentence>"}
  ],
  "action": "accept" | "regenerate"
}

action=regenerate when len(violations) > 0, else accept.
Be conservative: only flag clear, unambiguous violations.
```

User prompt (per call, ~300-500 tokens):

```
ATHLETE LAST MESSAGE (language reference):
<user_message text, max 500 chars>

TOOLS CALLED THIS TURN:
<comma-separated tool names, or "none">

COACH RESPONSE TO REVIEW:
<response_text>

Return JSON only.
```

Total budget: well under 800 tokens of input + ~150 tokens of output
(JSON is compact). On Haiku 4.5 at $1/MTok input and $5/MTok output
(approximate Q2 2026 prices, see model card), one critic call is
roughly:

- 800 input tokens * $1 / 1M = $0.0008
- 150 output tokens * $5 / 1M = $0.00075
- Total: ~$0.0016 per call, i.e. 0.16 cents

System prompt portion (~250 tokens) is cacheable on Haiku 4.5 - but
falls below the 4096-token minimum cache threshold (see llm.py
`_MIN_CACHE_CHARS = 16400`). So no caching for the critic; budget
above assumes full uncached pricing. Still cheap enough.

## 3. Decision logic

```
critic_result = critic.review(response, user_msg, tools_called)

if critic_result.action == "accept":
    emit("message", response)
    return

# action == "regenerate"
if retry_count == 0:
    retry_count = 1
    response = agent.regenerate_response()
    # Run critic ONCE more on the regenerated response.
    critic_result = critic.review(response, user_msg, tools_called)

# Whatever we have now, send it. If still violating, emit a
# critic_review SSE event with the violations so the frontend can log
# them, but DO NOT block the user-facing message.
emit("message", response)
if critic_result.action == "regenerate":
    emit("critic_review", {
        "violations": critic_result.violations,
        "annotated": True,
    })
```

Why no second-retry: latency budget. Each regenerate is a full coach
LLM call (Sonnet, expensive). 1 retry caps worst-case latency at
~2x normal, and the marginal value of retry #2 is empirically tiny
(per Devin's published numbers).

## 4. Parallelization and latency budget

Latency target: critique adds < 1 second to user-perceived response
time.

Mechanism:
- The coach call finishes and produces `response_text`.
- We invoke the critic via `asyncio.to_thread(critic.review, ...)` in
  parallel with EMITTING the `message` SSE event. The user starts
  seeing the message immediately; if the critic flags a violation,
  the next event the client gets is `critic_review`.

Important nuance: today our coach response is emitted as a SINGLE
`message` event (no token streaming). That means we cannot truly hide
the critique latency from the user when we regenerate, because
regeneration is a fresh LLM call that must complete before we emit
ANY message. We accept this: regenerate is on the happy path only
~6% of responses (per RESEARCH.md ballpark). For the other 94% the
critic runs AFTER the message is already on the wire and adds zero
user-perceived latency.

Optimization for the regenerate path: when the first critic call
returns `regenerate`, we drop the original response, run the coach
once more, and emit ONLY the regenerated response. The user never
sees the bad first draft. They just see a slightly later message.

Measured target overhead (Haiku 4.5, single-region):
- Critic call alone: 250-500 ms (median ~350 ms).
- Critic call parallel with message emit: ~0 ms user-perceived on
  the accept path.
- Regenerate path: +1 coach call (~3-6 s), but capped at ~6% of
  turns, so amortized ~+200 ms per turn.

## 5. Failure mode: critic API down

Fail-open. If `critic.review()` raises (network, rate limit, JSON
parse error), we log a warning, increment a `critic_errors` counter
in metrics, and accept the response as-is. The user must NEVER see
an error caused by the critic. The coach response is the user's
answer; the critic is a quality net.

## 6. Pro-tier gating

The critic is the kind of feature that pays for itself on Pro and is
unaffordable on Free. We gate the critic call behind:

1. Settings flag `critic_enabled: bool` (default False). Operators
   can turn the whole feature off via env var.
2. Per-user check `is_pro_tier(user_model)` exposed in
   `src/agent/critic.py`. Today this is a stub that returns False
   (no Pro infrastructure yet); Feature 5/6 will wire it to the real
   subscription store. For local dev we honor env var
   `CRITIC_FORCE_PRO=1` so we can demo.

The check happens in `agent_loop.process_message_sse()` BEFORE the
critic call. Free-tier users get zero overhead.

## 7. Files we touch

- `src/agent/critic.py` (NEW, ~180 lines):
  - `class CriticResult` (dataclass, frozen)
  - `class Critic` with `review(response, user_msg, tools_called)`
  - `is_pro_tier(user_model)` helper
  - Uses `litellm.completion` with model="anthropic/claude-haiku-4-5"
  - Hard timeout (1.5 s) on the API call to keep us inside the
    latency budget.
- `src/agent/agent_loop.py`:
  - Add ~25 lines at the end of `process_message_sse` to run the
    critic, handle regenerate, emit `critic_review` event when
    needed.
- `src/services/critic_metrics.py` (NEW, ~120 lines, mirrors
  cache_telemetry.py shape):
  - Per-rule violation counter.
  - Per-action counter (accept / regenerate / regenerate_failed).
  - Singleton + lock.
- `src/api/routers/admin.py`:
  - Extend with `/admin/critic-stats` endpoint.
- `src/api/routers/chat.py`:
  - Handle `critic_review` event in `_make_sse_event`.
- `tests/test_critic.py` (NEW, ~250 lines).

## 8. Sample critique JSON shapes

Accept path (no violation):

```json
{
  "violations": [],
  "action": "accept"
}
```

Regenerate path (markdown leak + em-dash):

```json
{
  "violations": [
    {"rule": "no_em_dash", "reason": "Response contains U+2014 em-dash in 'Du bist auf einem guten Weg - das zeigt sich' position 42."},
    {"rule": "no_markdown", "reason": "Response contains **bold** wrapper around 'wichtig'."}
  ],
  "action": "regenerate"
}
```

Annotated path (violations remain after retry, emitted as SSE event):

```
event: critic_review
data: {"violations": [{"rule": "no_em_dash", "reason": "..."}], "annotated": true}
```

## 9. Metrics shape (`/admin/critic-stats`)

```json
{
  "window_calls": 100,
  "accept_rate": 0.94,
  "regenerate_rate": 0.06,
  "regenerate_failed_rate": 0.015,
  "critic_error_rate": 0.002,
  "by_rule": {
    "no_em_dash": 12,
    "no_markdown": 8,
    "umlauts": 3,
    "no_fabricated_stats": 1,
    "no_premature_trends": 0,
    "language_mirror": 2,
    "details_before_metrics": 0,
    "sync_then_status": 0
  },
  "avg_critic_latency_ms": 340
}
```

## 10. Test plan

`tests/test_critic.py` covers (unit, no real LLM calls):

- Constitution rules are exactly 8, ids match spec.
- Critic prompt builder produces deterministic output for fixed
  inputs.
- `CriticResult.from_llm_json()` parses both accept and regenerate
  shapes.
- `CriticResult.from_llm_json()` raises on malformed JSON.
- Fail-open path: when `chat_completion` raises, `Critic.review`
  returns `accept` with `error=True`.
- Timeout: when the LLM call exceeds the deadline, we fail-open.
- Pro-tier check: returns False by default, returns True when
  `CRITIC_FORCE_PRO=1`.
- Metrics: every action increments the right counter.
- SSE handling: a `critic_review` event is correctly produced by
  `_make_sse_event`.

We do NOT test the regenerate end-to-end (that requires the full
agent loop with mocked LLM). The regeneration hook is small enough
(10 lines) that integration coverage from the existing
`test_phase8_integration.py` is sufficient.
