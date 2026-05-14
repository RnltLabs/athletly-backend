# Sprint G - Design: hard_inspect ACTUALLY blocks

## Goal

Zero hard-rule violations in any SSE `message` event payload. Deterministic.
Always-on. No opt-out, no critic flag dependency.

## Decisions

### D1: Drop the "real-umlaut suppression" in _detect_ascii_umlaut_violation

Old behaviour (Bug 1): if the response contains ANY real umlaut, return
None. The intent was to avoid penalising mixed text (proper names etc).
The cost is letting `Naechte` ship when the same response also has `über`.

New behaviour: scan EVERY token in the response. Flag the first token that
looks like a German word containing `ae/oe/ue/ss` AND is not in the
allowlist. Real umlauts elsewhere in the response are irrelevant; each
ASCII-translit token is independently flagged.

To minimise false-positive risk, expand the allowlist to cover common
sports / English loan words (we already have `aerobic`, `process`, etc.)
and add a heuristic: skip tokens that contain NO other German indicator
characters (so `tissue` / `queue` still pass).

### D2: Replace sanitize_hard's tiny replacement table with a fuller mapping

Build a deterministic ASCII-to-umlaut mapping for common German letter
groups that appear in athlete-coach context. Heuristic ordering: longest
prefix first so `Naecht*` does not interfere with `Naechst*`.

Mappings cover:
- Naecht*, naecht* -> Nächt*, nächt*
- Koerper, koerper -> Körper, körper
- Koennt*, koennt* -> Könnt*, könnt*
- Uebermorgen, uebermorgen -> Übermorgen, übermorgen
- Uebertraining, uebertraining -> Übertraining, übertraining
- Schluess*, schluess* -> Schlüss*, schlüss*
- Aussenseite, aussenseite -> Außenseite, außenseite
- Plaetz*, plaetz* -> Plätz*, plätz*
- Saetz*, saetz* -> Sätz*, sätz*
- Hoeh*, hoeh* -> Höh*, höh*
- Erschoep*, erschoep* -> Erschöp*, erschöp*
- staerk*, Staerk* -> Stärk*, stärk* (but not "stars")
- verlaess*, Verlaess* -> verlläss*, Verlläss*
- taegl*, Taegl* -> Tägl*, Tägl*
- Maerz, maerz -> März, märz
- Erklaer*, erklaer* -> Erklär*, erklär*
- Saetze, saetze -> Sätze, sätze
- groesser, Groesser -> größer, Größer
- Geraet, geraet -> Gerät, gerät
- etc.

These are applied via ordered replacement (longest first to avoid
prefix collisions).

### D3: Em-dash replacement uses a single hyphen with collapse of adjacent spaces

Old: replace U+2014 with `" - "`. If original had `"a U+2014 b"` (space-padded
em-dash) it becomes `"a  -  b"` (double space).

New: replace U+2014 with `" - "`, THEN collapse runs of internal whitespace
adjacent to the hyphen to single spaces. Same for en-dash U+2013. Use a
single regex that matches the dash plus its surrounding whitespace:

```python
text = re.sub(r"\s*[U+2014 U+2013]\s*", " - ", text)
```

(literal U+2014 / U+2013 in the character class).

This collapses pre and post whitespace and inserts canonical " - ".

### D4: Hard-inspect MUST run on every SSE message event, independent of CRITIC_ENABLED

Add `_finalize_response(text, user_message)` to `agent_loop.py`. It runs:
1. `sanitize_hard(text, user_message)` - deterministic, fast, no LLM.
2. `hard_inspect(sanitized, user_message)` - residual check, logs an
   error if anything survives.

Hook it into `process_message_sse` immediately BEFORE the `emit_fn(
"message", ...)` call:

```python
if final_text:
    final_text = _finalize_response(final_text, user_message)
    await emit_fn("message", {"text": final_text})
```

This runs unconditionally. CRITIC_ENABLED toggles only the LLM critic;
the deterministic sanitization happens regardless.

Also persist the finalized text back into `outcome.response_text` so push
notifications and session history see the user-facing text.

### D5: Same finalize step applies to gate-driven and critic-driven paths

The existing two-tier (`_run_critique_pass`) and gates (`_run_gates_pass`)
keep their LLM-regenerate flow. But the final pre-emit `_finalize_response`
is the single source of truth for "what ships". Belt and suspenders.

### D6: Test the SSE-emit invariant end-to-end

Integration test: build a fake `AsyncAgentLoop`, stub `process_message`
(synchronous worker) to return an `AgentResult` with a known-bad
response text containing all three hard violations. Call
`process_message_sse(user_message, emit_fn)` with `emit_fn` collecting
events. Assert that the `message` event payload has:
- No em-dash (U+2014) or en-dash (U+2013).
- No `**bold**`, `__bold__`, `*italic*`, `# heading`.
- No `Naechte`, `Koerper`, `koennten`, `uebermorgen`, `Uebertraining`.

This test is the regression net: it goes through the full SSE flow and
asserts on the bytes that reach the SSE consumer.

## Expected metrics impact

`/admin/critic-stats` after deploy:
- `by_rule.no_em_dash` -> 0 (the finalize step always sanitizes).
- `by_rule.no_markdown` -> 0.
- `by_rule.umlauts` -> 0.

NOTE: this is the COUNT IN SHIPPED RESPONSES. The pre-emit metrics may
still increment when hard_inspect flags the draft pre-sanitize. That's
intentional: the metric measures "how often the LLM tries to ship a
violation", not "how often we ship a violation to the user". The new
behavioural guarantee is the LATTER stays at zero.

If we want to track the "drafts the LLM produces with violations" vs
"violations that reach the user", we can split the metric. Out of scope
for Sprint G; the immediate need is to stop shipping the violations.
