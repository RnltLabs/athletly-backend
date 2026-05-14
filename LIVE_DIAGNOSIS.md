# Live Diagnosis: Constitutional Critic Regenerate Path

Date: 2026-05-14
Host: hetzner (production)
Container: athletly-api-1

## 1. /admin/critic-stats Snapshot

Raw response from `curl -s http://127.0.0.1:8000/admin/critic-stats`:

```json
{
  "window_calls": 19,
  "accept_rate": 0.0526,
  "regenerate_rate": 0.1053,
  "regenerate_failed_rate": 0.4211,
  "critic_error_rate": 0.4211,
  "by_rule": {
    "no_em_dash": 0,
    "no_markdown": 0,
    "umlauts": 1,
    "no_fabricated_stats": 6,
    "no_premature_trends": 0,
    "language_mirror": 0,
    "details_before_metrics": 3,
    "sync_then_status": 1
  },
  "avg_critic_latency_ms": 2476
}
```

### Interpretation

- 19 critic invocations in the window.
- `critic_error_rate` is **42.1%** - nearly half of all critic calls fail
  outright (timeout or unparseable JSON). Every failure is a fail-open:
  the bad coach response ships to the user.
- `regenerate_failed_rate` is **42.1%** - of the calls that did succeed,
  nearly half went through one regenerate attempt and STILL had violations
  on the second pass. The current code ships the rewritten-but-still-bad
  text anyway (see `agent_loop.py:1361`).
- `regenerate_rate` (clean regenerate, second pass accepted) is only
  **10.5%**.
- `accept_rate` is **5.3%** - this is the tail of clean first-pass
  responses. The vast majority of coach output triggers some violation.
- `avg_critic_latency_ms` is **2476 ms**. The configured timeout is
  **1500 ms** (`settings.critic_timeout_s = 1.5`). Average latency
  exceeds the deadline by 65%. This is the proximate cause of the
  42% `critic_error_rate`.

### By-rule distribution

Of the violations the critic actually managed to flag:

- `no_fabricated_stats`: 6 hits (the dominant failure mode, matches the
  Marco persona report).
- `details_before_metrics`: 3 hits (matches the Lisa persona report).
- `sync_then_status`: 1 hit (matches Lisa).
- `umlauts`: 1 hit.
- All others: 0 hits in this window. Note that the critic having `0`
  hits for `no_em_dash` and `no_markdown` is NOT proof the coach is
  clean on those rules. It means the critic LLM either did not see those
  cases or did not flag them.

## 2. Container log evidence

`docker logs --since 2h athletly-api-1 2>&1 | grep -iE 'critic'`:

```
Critic call timed out after 1.5 s, failing open
Critic call timed out after 1.5 s, failing open
Critic call timed out after 1.5 s, failing open
Critic call timed out after 1.5 s, failing open
Critic call timed out after 1.5 s, failing open
Critic call timed out after 1.5 s, failing open
Critic call timed out after 1.5 s, failing open
Critic call timed out after 1.5 s, failing open
Critic call timed out after 1.5 s, failing open
```

9 timeout entries in 2 hours. Every one of those is a fail-open path
where the bad coach response went unblocked.

## 3. Environment

`docker exec athletly-api-1 sh -c 'env | grep -i critic'`:

```
CRITIC_ENABLED=true
```

- `CRITIC_ENABLED` is set, so the critic IS running, the flag is not
  the bug.
- No `CRITIC_MODEL` override: defaults to
  `anthropic/claude-haiku-4-5-20251001` (correct, Haiku-class).
- No `CRITIC_TIMEOUT_S` override: defaults to 1.5s (too tight given
  the measured 2.5s average).

## 4. Cross-reference to persona reports

Marco persona run:
- Critic flagged `no_fabricated_stats` in turns 1, 4, 5, 7. All four
  responses still shipped. The metrics show 6 `no_fabricated_stats`
  hits but `regenerate_failed_rate` was 42%, meaning the regenerate
  attempt produced a rewrite that still violated, and we shipped the
  rewrite per the current "best we have" policy.

Lisa persona run:
- Same pattern: `details_before_metrics`, `no_fabricated_stats`,
  `sync_then_status` all detected, all shipped.

Elena persona run:
- Coach quoted fabricated HRV numbers. Critic did NOT catch it,
  most likely because the critic call timed out and we fail-opened.
  Matches the high `critic_error_rate`.

## 5. Failure modes seen in live data

| Failure mode | Evidence | Frequency |
|--------------|----------|-----------|
| Timeout, fail-open | 9 log entries / 19 calls, 42% error rate | Dominant |
| Regenerate produces still-bad text, shipped anyway | 8 of 19 calls in `regenerate_failed` bucket | Equal to dominant |
| Critic accepts a clean response | 1 of 19 calls | Minority |
| Critic catches and regenerate fixes it | 2 of 19 calls | Minority |

**Combined, 84% of critic calls in production ship a response that
either was never validated (timeout) or is known to still violate
(regenerate_failed). Only 16% of calls actually deliver the safety net
the feature is supposed to provide.**
