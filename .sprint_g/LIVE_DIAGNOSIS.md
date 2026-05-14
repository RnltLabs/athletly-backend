# Sprint G - Live Diagnosis: hard_inspect ships violations

## Question

The Sprint A two-tier critic flow is in place (hard_inspect + sanitize_hard +
_handle_hard_violations). The persona-test reports say violations are LOGGED
but coach text ships UNCHANGED to the user. Find out why.

## What we know from persona reports

Elena iter1 (tools/persona_test/reports/elena_test_iter1_20260514_135600.md):

- Turn 4 shipped: `"Drei Naechte unter 6 Stunden hintereinander  -  das ist
  dein Koerper... ueber drei Naechte..."`. The double-space-hyphen `"  -  "`
  is exactly what `sanitize_hard` emits when an em-dash was already
  space-surrounded (em-dash gets replaced with `" - "`). So the em-dash
  detector at least fired and SOMETHING ran.
- Turn 6 shipped: `"Anatomisch klingen sie unterschiedlich  -  ... Aber:
  Beide koennten vom gleichen Grund kommen (Uebertraining, Spannung,
  Tracking-Problem)..."`. The words `koennten`, `Uebertraining`, `Naechte`,
  `uebermorgen` all shipped despite the conversation being in clear German.

Marco iter1 (tools/persona_test/reports/marco_test_iter1_20260514_135618.md):

- Confirms `critic_metrics` had non-zero `no_em_dash` and `umlauts` counters
  in 28 calls. Hard inspect rules should drop those to 0 if working.

## Code-path trace

`src/api/routers/chat.py` -> `process_message_sse` (line 1208) of
`src/agent/agent_loop.py` runs the worker, then on completion:

1. `_run_gates_pass` (Sprint D, deterministic gates) - runs gates over
   tools / response, may regenerate ONCE.
2. `_run_critique_pass` (Sprint A, hard inspector + soft critic) -
   ONLY runs if `_should_run_critic_safe(self.user_model)` returns True
   (i.e. `CRITIC_ENABLED=true`). Inside:
   - `hard_inspect(response_text, user_message)` -> tuple of Violations.
   - If non-empty, `_handle_hard_violations` is called and its return value
     is assigned BACK to `response_text`. Good.
   - `_handle_hard_violations` calls `regenerate_after_critique` once, then
     `hard_inspect_fn(rewritten)`. If rewrite still has hard violations,
     calls `sanitize_hard(rewritten)`. Else ships rewrite as-is.
3. `await emit_fn("message", {"text": final_text})` is the SSE emit.

Then `src/api/routers/chat.py:_make_sse_event` routes the `message` event
through `_sanitize_assistant_text` which ALSO strips em/en-dashes (replaces
with `"-"`, single char) and Markdown markers. But it does NOT touch
ASCII-umlaut transliteration.

So the question is: how do `Naechte`, `Koerper`, `koennten` survive the
hard-inspect + regenerate cycle?

## Reproduce via direct unit call

`uv run python -c "..."` with the actual Elena turn 4 user message and an
ASCII-translit-heavy coach response showed:

- `_is_german_text(elena_user)` -> True. German detector works.
- For a PURE-ASCII coach response: `hard_inspect` correctly flags
  `umlauts`. But after `sanitize_hard` runs:
  - The replacement table only has `Naechst`, `Koennen`, `Muessen`, `Ueber`,
    etc. - it does NOT cover `Naecht*` (without an `s`), `Koerper`,
    `Schluess*`, `Aussen*`. So `Naechte` and `Koerper` SURVIVE the
    sanitizer.
  - The sanitizer adds `"über"` (replacement for `ueber`). Now the text
    has at least one real umlaut.
  - Re-running `hard_inspect(cleaned)` returns `[]` for the `umlauts`
    rule because of the short-circuit in `_detect_ascii_umlaut_violation`:
    `if _has_real_umlaut(response_text): return None`.
  - `_handle_hard_violations` therefore concludes "the rewrite is clean,
    ship it" - and ships text that still contains `Naechte` and `Koerper`.
- For a MIXED response (LLM regenerate produced ONE real umlaut + many
  ASCII translits): the `umlauts` rule never even fires on the FIRST
  inspect pass for the same reason. The em-dash IS caught and replaced
  but the regenerate's ASCII translits ship unchanged.

## What `/admin/critic-stats` should look like

If hard_inspect were truly enforcing:
- `by_rule.no_em_dash` -> 0 (caught and either regen-fixed or sanitized
  away).
- `by_rule.umlauts` -> 0 (same).
- `by_rule.no_markdown` -> 0 (same).

Persona report shows these are non-zero (Marco saw 7 em-dash, 6 umlauts in
28 calls). That matches the bug: the regenerate cycle "fixes" some but the
metrics record the FIRST inspect as a violation. The shipped text still
contains transliterations because of the short-circuit + partial sanitizer
table.

## Additional observation: chat.py double-sanitize

`src/api/routers/chat.py:_sanitize_assistant_text` ALWAYS runs (no
CRITIC_ENABLED gate) and strips em/en-dashes + markdown. So the em-dash
"escape" is partly mitigated downstream. But:
- It replaces em-dash with `"-"` (no spaces), which collides with the
  earlier `sanitize_hard` " - " replacement to produce `"  -  "` if the
  original em-dash had spaces around it.
- It does NOT touch ASCII umlauts, which is why `Naechte` ships.

## Verdict

Three independent defects compound to produce the user-visible bug:

1. `_detect_ascii_umlaut_violation` short-circuits on any real umlaut in
   the response, so mixed responses never trip the umlauts rule.
2. `sanitize_hard`'s umlaut replacement table is partial: it patches
   `Naechst*` but not `Naecht*`, patches `Koennen` but not `Koerper`,
   patches `Ueber*` but not `Schluess*` / `Aussen*` / `Hoehepunkt`, etc.
3. The final SSE emit is gated behind `CRITIC_ENABLED` (the Sprint A
   pass only runs when the critic is on). If someone toggles the critic
   off, hard violations ship via the chat.py downstream sanitizer which
   does NOT cover umlauts.

Action plan: see DESIGN.md.
