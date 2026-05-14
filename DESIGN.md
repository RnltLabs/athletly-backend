# Design: Two-Tier Constitutional Critic with Always-Block Hard Guardrails

## Goals

1. Detected violations MUST block. No more "logged but shipped".
2. Hard rules (em-dash, markdown, ASCII umlauts) MUST never fail open.
3. Soft rules (fabricated_stats, etc.) may fail open for latency but
   the failure rate must be visible in `/admin/critic-stats`.
4. Latency budget: at most +3s extra on the p99 turn. Median unchanged.

## Architecture

```
coach response text
        |
        v
[1] HARD INSPECTOR (deterministic, sub-ms)
    - regex for em-dash / en-dash
    - regex for markdown markers (**, __, *italic*, # heading)
    - language detection + regex for ASCII transliteration in DE
    |
    +-- violations found ---> [2a] regenerate (constrained prompt)
    |                              |
    |                              v
    |                         re-run HARD INSPECTOR
    |                              |
    |                              +-- still violating ---> [3] safe fallback
    |                              +-- clean -----------> soft critic
    |
    +-- clean --------------> [2b] SOFT CRITIC (LLM, Haiku 4.5, 4s timeout)
                                   |
                                   +-- accept ----> ship
                                   +-- regenerate -> rewrite, hard inspect, soft re-check
                                   +-- error (timeout/parse) -> ship + record critic_error
```

## Component-by-component design

### Hard Inspector (new code, `src/agent/critic.py`)

A deterministic, allocation-light function:

```python
def hard_inspect(response_text: str, user_message: str) -> tuple[Violation, ...]:
    """Run sub-millisecond deterministic rule checks.

    Returns a tuple of Violations. Empty tuple means clean.
    """
```

Rules checked:
- `no_em_dash`: substring search for U+2014 (em-dash) and U+2013 (en-dash).
- `no_markdown`: regex for `**bold**`, `__bold__`, single-asterisk
  italic (`\*\S[^*\n]*\S\*`), and headings (`^# ` at line start).
- `umlauts`: only fires when `user_message` is detected as German
  (cheap heuristic: presence of any of {ae, oe, ue, ich, ist, und,
  nicht, der, die, das}). Then scans `response_text` for ASCII
  transliterations that should be real umlauts: `\b(?:ae|oe|ue|ss)\b`
  context-sensitive. Conservative: only flags if no real umlauts
  (ae oe ue ss) are present in the same response (avoids false
  positives on words like "Status" or "Phase").

Output: tuple of `Violation` instances using the existing dataclass.

Why this is safe to always-block:
- Zero false positives by construction (we are searching for specific
  bytes).
- Sub-millisecond, no network, no LLM, cannot time out.

### Soft Critic (refactor of existing `Critic.review`)

Same shape as today, with these changes:

1. **Timeout raised from 1.5s to 4.0s** (config), based on the Anthropic
   p99 latency budget. Logged + visible in metrics.
2. **Move from free-form JSON to JSON schema via `response_format` /
   tool_use**. Eliminates parse-error fail-opens.
   - For litellm + Anthropic, this uses `response_format={"type":
     "json_object"}` plus a strict prompt; we keep the existing
     `from_llm_json` parser as a fallback for non-Anthropic models.
3. The soft critic ONLY checks the 5 soft rules:
   `no_fabricated_stats`, `no_premature_trends`, `language_mirror`,
   `details_before_metrics`, `sync_then_status`.

### Decision logic (refactor `_run_critique_pass` in `agent_loop.py`)

```python
async def _run_critique_pass(self, response_text, user_message, tool_names, emit_fn):
    # Stage 1: hard inspector (always-block, deterministic)
    hard_violations = hard_inspect(response_text, user_message)
    if hard_violations:
        rewritten = await asyncio.to_thread(
            self.regenerate_after_critique,
            response_text,
            [{"rule": v.rule, "reason": v.reason} for v in hard_violations],
        )
        # Re-check hard rules on the rewrite. If still violating,
        # ship a safe-fallback notice.
        rewrite_hard = hard_inspect(rewritten, user_message)
        if rewrite_hard:
            metrics.record(action="regenerate_failed",
                           violations=tuple(v.rule for v in rewrite_hard),
                           latency_ms=0)
            await _emit_safely(emit_fn, "critic_review",
                               {"violations": [...], "annotated": True,
                                "degraded": True})
            # Strip the obvious bad chars from the rewrite as last
            # resort; this gives the user something usable rather
            # than fabricated data, and never re-introduces hard
            # violations.
            return _sanitize_hard(rewritten)
        # Hard rules clean: continue to soft critic on the rewrite.
        response_text = rewritten

    # Stage 2: soft critic (LLM, bounded timeout, fail-open allowed)
    soft = await asyncio.to_thread(critic.review, response_text, user_message, tool_names)
    if soft.error:
        metrics.record(action="critic_error", violations=(), latency_ms=soft.latency_ms)
        return response_text
    if soft.action == "accept":
        metrics.record(action="accept", violations=(), latency_ms=soft.latency_ms)
        return response_text

    # Soft rules violated -> regenerate
    rewritten = await asyncio.to_thread(self.regenerate_after_critique, response_text,
                                        [{"rule": v.rule, "reason": v.reason} for v in soft.violations])
    # Re-check both hard AND soft on the rewrite.
    rewrite_hard = hard_inspect(rewritten, user_message)
    if rewrite_hard:
        # Regeneration introduced a hard violation. Sanitize and ship.
        metrics.record(action="regenerate_failed",
                       violations=soft.violation_ids() + tuple(v.rule for v in rewrite_hard),
                       latency_ms=soft.latency_ms)
        await _emit_safely(emit_fn, "critic_review",
                           {"violations": [...], "annotated": True, "degraded": True})
        return _sanitize_hard(rewritten)
    second = await asyncio.to_thread(critic.review, rewritten, user_message, tool_names)
    if second.error or second.action == "accept":
        metrics.record(action="regenerate", violations=soft.violation_ids(),
                       latency_ms=soft.latency_ms + second.latency_ms)
        return rewritten
    # Still violating after one regenerate. Emit critic_review with
    # degraded=True flag. The frontend shows a "this answer may contain
    # inaccuracies" banner. We DO ship the rewrite because soft
    # violations are judgment calls and the LLM may be wrong, BUT we
    # mark it.
    metrics.record(action="regenerate_failed", violations=second.violation_ids(),
                   latency_ms=soft.latency_ms + second.latency_ms)
    await _emit_safely(emit_fn, "critic_review",
                       {"violations": [{"rule": v.rule, "reason": v.reason} for v in second.violations],
                        "annotated": True, "degraded": True})
    return rewritten
```

### `_sanitize_hard` (last-resort fixer)

Applied only when a regenerate introduced a NEW hard violation. The
function:
- Replaces em-dash / en-dash with ` - ` (hyphen with spaces).
- Replaces `**X**` / `__X__` / `*X*` markdown with plain `X`.
- Replaces ASCII transliteration in known German words (limited list)
  with real umlauts.

It guarantees zero hard violations on the output. It is NOT a
substitute for the LLM regenerate; it is the last 10% of safety.

## Failure modes and how we handle them

| Failure mode | Detection | Response |
|---|---|---|
| Hard rule in first draft | hard_inspect (deterministic) | always block, regenerate |
| Hard rule in rewrite | hard_inspect on rewrite | `_sanitize_hard` strips it, ship sanitized + critic_review |
| Soft critic times out | concurrent.futures TimeoutError | log, record `critic_error`, ship original (fail-open, but soft rules only) |
| Soft critic unparseable | from_llm_json ValueError | log, record `critic_error`, ship original |
| Soft critic flags, regenerate fixes | second pass accept | record `regenerate`, ship rewrite |
| Soft critic flags, regenerate also flags | second pass regenerate | record `regenerate_failed`, ship rewrite with degraded flag |
| Both hard and soft clean | n/a | ship as-is |

## Latency budget impact

- Hard inspector: < 1 ms per call. Negligible.
- Soft critic timeout: raised from 1.5s to 4.0s.
- Worst-case turn now: response + 4.0s (soft critic) + ~1-2s (regen) +
  ~1-2s (second soft critic) = +6-8s on the absolute worst case (was
  +3-5s before because timeouts truncated to 1.5s and accepted).
- Median turn: +0.8-1.2s for soft critic (Haiku p50 latency) + 0 for
  hard inspector. Same as before for the common-case clean response.
- Median worst regression: ~250ms because we run the hard inspector
  always. The regex is cheap.

This is acceptable: we trade a small p50 increase and a larger p99
increase for the actual feature working.

## Test cases (regression coverage)

In `tests/test_critic.py`:

1. `test_hard_inspect_em_dash_blocked`: response with em-dash, expect
   Violation tuple containing `no_em_dash`.
2. `test_hard_inspect_markdown_blocked`: response with `**bold**`,
   expect `no_markdown`.
3. `test_hard_inspect_ascii_umlauts_german`: German user message,
   response with "fuer" instead of "fuer" (real "fuer"), expect
   `umlauts`.
4. `test_hard_inspect_ascii_umlauts_skipped_english`: English user
   message, response with "fuer" (which is just English noise),
   expect no violation.
5. `test_hard_inspect_clean_response`: passes through cleanly.
6. `test_run_critique_pass_hard_violation_regenerated_and_sanitized`:
   first draft has em-dash, regenerate ALSO returns em-dash, expect
   sanitized output AND a degraded `critic_review` event.
7. `test_run_critique_pass_soft_critic_timeout_records_error`:
   simulate timeout, expect `critic_error` metric and original shipped.
8. `test_run_critique_pass_soft_violation_regenerate_fix`: first soft
   pass flags fabricated stats, regenerate is clean, expect rewrite
   shipped + `regenerate` metric.
9. `test_sanitize_hard_replaces_em_dash`: deterministic sanitizer
   test.
10. `test_critic_timeout_default_is_four_seconds`: config regression
    so we do not silently revert.

## Hard rules vs soft rules (final taxonomy)

Hard (deterministic, always-block):
- no_em_dash
- no_markdown
- umlauts (German-only conditional)

Soft (LLM judgment, fail-open allowed):
- no_fabricated_stats
- no_premature_trends
- language_mirror
- details_before_metrics
- sync_then_status

Why the split:
- Hard rules are about bytes/characters. Regex is the right tool.
- Soft rules require semantic understanding (was THIS number
  fabricated? does THIS claim need more data points?). Only an LLM
  can answer those.
