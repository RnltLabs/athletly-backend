# Sprint G - Root Cause

Three compounding bugs in `src/agent/critic.py`:

## Bug 1: umlauts detector short-circuits on any real umlaut

`_detect_ascii_umlaut_violation` (critic.py around line 404):

```
if _has_real_umlaut(response_text):
    return None
```

Intent: "if the author clearly knows how to type umlauts, treat remaining
`ae/oe/ue/ss` as intentional (e.g. proper names)". Reality: when the LLM
mixes real `ü` and ASCII `ue` in the same response (common, especially in
LLM regenerate output), the detector blinds itself and ALL ASCII translit
words ship.

Proof: response text `"Drei Nächte ... Koerper ... ueber drei Naechte"`
returns `[]` for the `umlauts` rule.

## Bug 2: sanitize_hard replacement table is partial

The replacement table (critic.py around line 523) covers a tiny vocabulary:

- `Fuer/fuer`, `Ueber/ueber`, `Moeglich/moeglich`, `Naechst/naechst`,
  `Waere/waere`, `Koennen/koennen`, `Muessen/muessen`, `Suess/suess`,
  `strasse/Strasse`, `grosse/Grosse`.

Missing common patterns seen in production:

- `Naecht` (Nächte, Nächten) - WITHOUT the `s` of Naechst*
- `Koerper`, `Koennt` (koennten), `koennt`
- `Schluess` (Schlüssel), `Aussen` (Außenseite), `Hoeh*` (Höhe),
  `Uebermorgen`, `Uebertraining`, `groesser`, `Erschoepf*`,
  `verlaess*`, `taegl*`, `Saetze`, `Vorraet*`, `Fluess*`, `Plaetz*`,
  `Maerz`, `Schwaeche`, `Verstaerk*`, `Erklaer*`, `geraet`, `staerker`.

The sanitizer is a "safety net" but it leaks like a sieve.

## Bug 3: hard-violation pipeline is conditional on CRITIC_ENABLED

In `process_message_sse` (agent_loop.py line 1366):

```
if final_text and _should_run_critic_safe(self.user_model):
    final_text = await self._run_critique_pass(...)
```

If `CRITIC_ENABLED=false`, the deterministic hard inspector does NOT run
at all. Markdown / em-dash / ASCII-umlaut text would ship raw (modulo
`_sanitize_assistant_text` in chat.py which is em-dash+markdown only).

The hard inspector is a deterministic GUARDRAIL. It is not "a critic
feature". Tying it to the critic feature flag means an ops-level toggle
to "disable LLM critic to save cost" simultaneously unblocks
ASCII-umlaut leakage. That's wrong.

## Bug 4 (minor): em-dash sanitize produces visible " - " with adjacent spaces

`sanitize_hard` replaces U+2014 with `" - "`. If the original em-dash was
already space-padded (e.g. `"foo U+2014 bar"`), the result is `"foo  -  bar"` -
the double-space is visually a dash glyph to most readers. The
downstream `_sanitize_assistant_text` in chat.py replaces em-dash with
`"-"` (no spaces). They both have legitimate use-cases; the issue is
they run in sequence and produce ugly output.

Spec note: the persona report flagged "  -  " explicitly as a user-visible
defect. Fix: have ONE canonical sanitization that handles dash + spaces in
a single pass, collapsing adjacent whitespace.

## Compound bug picture

A typical Elena turn:

1. LLM produces a draft with `"Nächte U+2014 Koerper, ueber drei Naechte"`.
2. `hard_inspect` flags `no_em_dash`. The `umlauts` rule does NOT
   fire because of Bug 1 (the `Nächte` umlaut blinds the detector).
3. `_handle_hard_violations` calls `regenerate_after_critique`. The LLM
   produces a similar mixed-umlaut rewrite (likely worse, fewer umlauts
   to start since temperature changes).
4. Post-regenerate `hard_inspect` may flag em-dash only. `sanitize_hard`
   runs. It replaces U+2014 with `" - "` and tries umlauts but most words
   (e.g. `Koerper`) are not in the table.
5. `hard_inspect(cleaned)` returns `[]` because Bug 1 short-circuits on
   the surviving `ü` from `über` (which the sanitizer DID patch).
6. Cleaned text ships. User sees `"Naechte ... Koerper ... über"` plus
   the `"  -  "` double-space dash.

Fix design in DESIGN.md.
