# Root Cause Analysis: Critic Detects But Does Not Block

After reading `src/agent/critic.py`, `src/agent/agent_loop.py` (lines
1013-1361), `src/agent/llm.py` (chat_completion path), and
`src/services/critic_metrics.py`, and correlating with the live
diagnostics in `LIVE_DIAGNOSIS.md`, there are **four** root causes.
Multiple of them combine to produce the observed behavior.

## Root Cause 1: Timeout budget is too tight (PRIMARY)

**Location:** `src/config.py:105`, `src/agent/critic.py:209,276`.

```
critic_timeout_s: float = 1.5
```

Live data shows `avg_critic_latency_ms = 2476` and 9 of 19 calls (47%)
hit the timeout. The timeout was chosen to cap user-perceived latency,
but it is below the typical Haiku 4.5 cold-start + JSON-generation
budget. Result: half the critic calls fail open silently, the response
ships unvalidated.

**Why this matches hypothesis 1:** confirmed by `Critic call timed out
after 1.5 s, failing open` log lines.

## Root Cause 2: Regenerate-failed path ships the rewritten-but-still-bad response (PRIMARY)

**Location:** `src/agent/agent_loop.py:1350-1361`.

```python
# Still violating after retry. Send the rewritten response (best
# we have) and emit a critic_review event for observability.
metrics.record(
    action="regenerate_failed",
    violations=second.violation_ids(),
    latency_ms=first.latency_ms + second.latency_ms,
)
try:
    await emit_fn("critic_review", second.to_event_payload())
except Exception:
    logger.warning("emit_fn raised on critic_review", exc_info=True)
return rewritten
```

When the second pass still flags violations, the code:
1. records `regenerate_failed`
2. emits `critic_review` for observability
3. **returns the still-violating rewrite as the final user-facing text**

So even when the critic DOES catch a problem, even after a regenerate,
the bad text ships. This is exactly the "Marco: flagged in turns 1,4,5,7,
all shipped" pattern. The `regenerate_failed_rate = 0.42` in metrics
proves the path is hit frequently.

**Why this matches hypothesis 2:** confirmed by reading the code. The
"best we have" comment in the source is the explicit anti-feature.

## Root Cause 3: Hard guardrails (em-dash, ASCII umlauts, markdown) are routed through the LLM critic instead of a deterministic post-check (SECONDARY)

**Location:** `src/agent/critic.py:45-72` (all 8 rules treated the same).

Three of the 8 rules are deterministic and cheap to detect with a
regex:
- `no_em_dash`: substring search for `—` or `–`
- `no_markdown`: regex for `**`, `__`, `*`, `# ` at line start, etc.
- `umlauts`: language detection + regex for ASCII transliterations
  `ae|oe|ue|ss` in German context

Currently we pay the full LLM round-trip cost (and timeout risk) for
detecting these. The Haiku critic also sometimes misses them - live
data shows `no_em_dash: 0` and `no_markdown: 0` in the by_rule
counters across 19 calls, which is statistically improbable if the
coach is the source of the persona-test violations. This is consistent
with the critic missing obvious cases under time pressure or model
noise.

If we ran these three rules as deterministic Python checks BEFORE the
LLM call, we would:
- Always catch them (no false negatives on the hard cases)
- Never time out on them
- Free up the LLM critic budget for the soft, judgment-based rules

## Root Cause 4: critic_error path silently accepts (CONTRIBUTING)

**Location:** `src/agent/agent_loop.py:1320-1326`.

```python
if first.error or first.action == "accept":
    metrics.record(
        action="critic_error" if first.error else "accept",
        ...
    )
    return response_text
```

When `first.error` is True (timeout / unparseable JSON / network
error), we record `critic_error` AND return the unvalidated response.
With a 42% critic_error_rate in production, this fail-open path is
the dominant ship-the-bad-response source.

Fail-open is documented as intentional ("the critic must NEVER add
more than ~1.5 s of latency [...] Fail-open"). But the spec contradicts
itself: the feature exists precisely to prevent fabricated stats from
shipping. A fail-open default that triggers 42% of the time defeats the
purpose. We need a two-tier policy:
- Hard rules (deterministic): always-block, never fail-open.
- Soft rules (LLM judgment): fail-open is acceptable for latency, but
  we should at least try once with a sane timeout.

## Hypotheses ruled out

- **H3 (JSON parse failure dominates):** not the primary cause. We see
  9 timeouts and presumably a handful of parse errors, but the timeout
  log message dominates. The fail-open path is the same either way, so
  fixing H1/H2 also covers H3.
- **H4 (critic prompt is wrong):** partially right but not the root
  cause. The prompt could be tightened (especially for fabricated_stats),
  but the metrics show the critic IS detecting violations when it
  runs. The deeper bug is that detections do not BLOCK.
- **H5 (hook in wrong place, runs after message emit):** NOT the case.
  `_run_critique_pass` is called at line 1258, and
  `await emit_fn("message", ...)` is at line 1273. The hook IS before
  the emit. The problem is what `_run_critique_pass` returns, not when
  it runs.

## Summary

The critic IS positioned correctly in the loop. The bug is that:
1. **It usually times out** because the budget is too tight (1.5s vs
   2.5s actual).
2. **When it does work and finds violations after regenerate, it still
   ships the bad response** because of the "best we have" policy.
3. **Hard, deterministic rules are routed through the slow, error-prone
   LLM** instead of being checked locally.

Fix must address all three. See `DESIGN.md`.
