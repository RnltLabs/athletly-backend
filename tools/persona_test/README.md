# Athletly Persona Test Framework

This framework lets a Claude Code session autonomously test the Athletly coach by role-playing one of three predefined personas (Elena, Marco, Lisa). You (Claude Code, future session) read this doc and immediately know how to seed a persona, drive a real multi-turn conversation against the live `/chat` endpoint, evaluate each response against the rubric, and produce a structured report.

This is NOT a pytest-style automated harness. The LLM-in-the-loop is intentional: the orchestrating Claude Code IS the persona, and is also the judge.

---

## MANDATORY: ground every finding in production telemetry

This rule was added 2026-05-14 after a re-test agent reported "26 em-dashes
shipped across 6 turns" when production telemetry said `no_em_dash: 0`. See
[`FAILURE_MODE.md`](./FAILURE_MODE.md) for the full forensic.

Every persona-test agent MUST:

1. **Before claiming a violation count, query the three admin endpoints**
   (`/admin/critic-stats`, `/admin/gates-stats`, `/admin/prompt-metrics`) and
   compare the claim to the measured number. If they conflict, the agent
   reports the discrepancy, not its imagined count. The helper
   `tools/persona_test/verify_telemetry.py` pulls all three in one call and
   prints a "fact pack" that goes verbatim into the report's evidence section.
2. **Quote VERBATIM lines from the actual coach response** (no paraphrasing
   for evidence). The verbatim text must come from the
   `=== EVIDENCE TRACE ===` block emitted by `chat.py --print-evidence`.
3. **Tool calls reported must come from the SSE event stream** printed by
   `chat.py`, not from the agent's expectation of which tools should have
   fired.
4. **Eval scores 1-5 per dimension MUST cite the specific observable
   evidence**: a tool event from the SSE stream, a substring from the
   coach response (quoted with surrounding context), or a telemetry counter
   from the fact pack. No evidence, no score.

Operationally:

- Run `chat.py <persona> "<msg>" --print-evidence` for every turn. Do not
  pipe to `head` / `tail` / `grep` while the conversation is in progress;
  truncated output is how invented findings creep in.
- After the conversation ends, run `verify_telemetry.py` and paste the fact
  pack into the report.
- Per-turn evaluation block must reference the evidence trace block (e.g.,
  "response substring at line N of evidence trace" or "tool event
  `sync_garmin_data` in evidence trace turn 3").

---

## TL;DR for Claude Code

When Roman says "test the coach as Elena", do this:

1. Seed: `uv run python tools/persona_test/seed.py elena`
2. For each turn:
   - Compose a message in Elena's voice (see `personas/elena.md` `## Personality`)
   - Send: `uv run python tools/persona_test/chat.py elena "<message>"`
   - Read the coach response in your terminal output
   - Score it against `EVAL_RUBRIC.md` (six dimensions, 1-5 each, total 30)
   - Decide the next turn based on what the coach did (or missed)
3. At session end: produce an aggregated report (avg score, lowest turn, regressions).
4. To restart cleanly: `uv run python tools/persona_test/reset.py elena` then re-seed.

CRITICAL: every command MUST be prefixed with `uv run` so the project's
venv (with `httpx`, `supabase`, etc.) is used. Plain `python` or `python3`
without the venv will fail with `ModuleNotFoundError: No module named 'httpx'`.

Default API URL is `https://athletly.rnltlabs.de` (production Hetzner).
Override with `ATHLETLY_API_URL` env or `--api-url http://localhost:8000`
for local testing.

---

## Prerequisites

Before the first run:

- `.env` must contain `SUPABASE_URL`, `SUPABASE_ANON_KEY`, and `SUPABASE_SERVICE_ROLE_KEY`.
- The Athletly backend must be running and reachable at the configured API URL.
- Python 3.12+ and `uv` available (we use `uv run` for the right venv).

The first seed run will auto-create `.env.persona_test` with persona passwords (gitignored). Do not commit it.

---

## Available personas

Three personas ship with this framework. Each lives at `personas/<slug>.md` with YAML frontmatter and structured markdown sections.

| Slug    | Profile                                              |
|---------|------------------------------------------------------|
| `elena` | 32, healthy marathoner, Koeln 2026, sub-3:30 target |
| `marco` | 28, beginner, Berlin HM 2026, sub-2:15 target       |
| `lisa`  | 36, triathlete, Roth 2026, ITB-history, sub-11:30   |

Before each session, read the persona's `## Personality` section. That defines the voice you'll be writing in.

---

## Step 1: Seed

```bash
python tools/persona_test/seed.py elena
```

What this does:
- Ensures the auth user exists in Supabase via the service-role admin API
- Marks user metadata with `is_test=true` and `persona=elena`
- Upserts `profiles` row from frontmatter
- Upserts `athlete_journal` markdown built from persona sections
- Generates 12 weeks of realistic Garmin activities (deterministic IDs, idempotent)
- Generates 30 days of `health_daily_metrics`
- If persona has `has_active_plan: true`, creates a 24-week plan

Re-running is safe: all writes are idempotent upserts keyed on deterministic IDs.

Expected output:
```
Persona:    elena
User id:    <UUID>
Email:      elena@persona.test.athletly.local
Profile:    upserted
Journal:    upserted
Activities: 60
Metrics:    30
Plan:       none
```

---

## Step 2: Chat

```bash
python tools/persona_test/chat.py elena "war eben laufen, 10km easy bei 5:00/km HR 142"
```

Output (illustrative):
```
> Elena Vogel: war eben laufen, 10km easy bei 5:00/km HR 142
[session] 9f8c3...
[tool] sync_garmin_data(mode=auto)
[tool] get_activities(limit=10, days=3)
[tool] get_activity_details(activity_id=...)

< Coach: Sehe deinen 10er von heute frueh, 10.02 km in 50:14, avg HR 142, ...

[usage] input=1250 output=180 model=anthropic/claude-sonnet-4-6
```

Event types you'll see:
- `[session]` once per session start (id is auto-persisted)
- `[tool] name(args)` every tool call
- `[tool result] name: ...` every tool result (truncated)
- `[ui]` if the coach emits a generative UI card
- `[action]` if the coach proposes a pending action
- `< Coach: <text>` the final message
- `[usage]` token/model summary at end
- `[error]` on agent error (also non-zero exit code)

The chat session id is persisted at `.persona_test_session/<slug>.json` so the next `chat.py` call continues the same conversation.

Flags:
- `--new` start a fresh chat session
- `--api-url URL` point at a different backend
- `--show-thinking` also render `thinking` events
- `--print-evidence` emit a structured `=== EVIDENCE TRACE ===` block at the
  end with the verbatim coach response, tool-call list, and usage line.
  MANDATORY when running a persona-test conversation; the report's evidence
  must quote from this block.
- `-v` verbose logging

### Evidence trace block

`chat.py --print-evidence` prints a block of the following shape after the
normal SSE rendering. Copy this verbatim into the per-turn evidence section
of the report:

```
=== EVIDENCE TRACE ===
session_id: 4e701f74-6790-4ac0-8aea-fd3f87875d96
tools_called: [get_activities, sync_garmin_data, get_provider_status]
response_text_verbatim: <full text, exactly as the SSE message event delivered it>
tokens: input=850 cache_read=0 output=180 model=anthropic/claude-haiku-4-5
critic_review_event: null
=== END EVIDENCE ===
```

If `critic_review_event` is not `null`, paste the full event JSON into the
report. That is the only acceptable source for claims like "critic flagged
no_fabricated_stats" or "critic flagged no_em_dash".

---

## Step 2b: Pull production telemetry fact pack

Before writing scores, run:

```bash
uv run python tools/persona_test/verify_telemetry.py
```

This pulls `/admin/critic-stats`, `/admin/gates-stats`,
`/admin/prompt-metrics?window_minutes=60` and prints a compact fact pack:

```
=== TELEMETRY FACT PACK (pulled 2026-05-14T14:35:30Z) ===
critic_stats.window_calls: 12
critic_stats.accept_rate: 0.92
critic_stats.regenerate_rate: 0.08
critic_stats.by_rule.no_em_dash: 0
critic_stats.by_rule.no_fabricated_stats: 1
critic_stats.by_rule.umlauts: 0
...
gates_stats.window_records: 14
gates_stats.totals.fails: 2
gates_stats.totals.regenerate_success: 1
...
prompt_metrics.window_minutes: 60.0
prompt_metrics.total_responses_scanned: 12
prompt_metrics.total_violations: 1
prompt_metrics.violation_rate_pct: 8.33
...
=== END TELEMETRY FACT PACK ===
```

Paste the fact pack verbatim into the report's evidence section. Then for
every claim in the report, write `claim X = telemetry Y` next to it. If
they disagree, the report MUST headline the discrepancy.

Flags:
- `--api-url URL` point at a different backend (default uses
  `ATHLETLY_API_URL` env or the production URL)
- `--window-minutes N` window for `/admin/prompt-metrics` (default 60)
- `--json` print the raw merged JSON instead of the flat fact pack

---

## Step 3: Evaluate

After each `< Coach: ...` line, score the response against `EVAL_RUBRIC.md`. Every dimension score MUST cite specific observable evidence (tool event from the SSE stream, quoted substring from the evidence trace, or counter from the telemetry fact pack). Write a small block in your transcript:

```
EVAL turn 1:
  accuracy:       5  evidence: response_text_verbatim contains "10.02 km in 50:14, avg HR 142" which matches get_activity_details tool result
  completeness:   4  evidence: persona asked "is this good for marathon?" implicitly; response did not mention marathon pace target. quoted: "Sehe deinen 10er von heute frueh"
  tool_usage:     5  evidence: tools_called = [sync_garmin_data, get_activities, get_activity_details], matches STRICT proactive-sync pattern
  tone:           4  evidence: quoted "Sehe deinen 10er ..." mirrors Elena's "war eben laufen" register. Missed asking back, persona expects a "why" question.
  hallucinations: 5  evidence: no fabricated numbers; verify_telemetry shows critic_stats.by_rule.no_fabricated_stats = 0 for the window
  holistic:       4  evidence: response_text_verbatim does not mention HRV / sleep; no get_health_metrics call in tools_called
  total: 27
```

A score without an `evidence:` clause is invalid. A claim like "9 em-dashes
in the response" without a quoted substring containing U+2014 is invalid;
cross-reference `critic_stats.by_rule.no_em_dash` from the fact pack before
making the claim. See [`FAILURE_MODE.md`](./FAILURE_MODE.md) for the
canonical example of a confabulated finding.

Then write the next persona-voiced message that pushes the coach in a direction. The persona's `## Open Threads` section gives you ideas: pending questions Elena would naturally raise.

---

## Step 4: Reset and rerun

```bash
python tools/persona_test/reset.py elena         # soft: keeps auth user
python tools/persona_test/reset.py elena --hard  # also deletes auth user
```

What this clears (filtered on the persona's user_id, refuses to operate without `is_test=true`):
- `session_messages`, `sessions`, `episodes`
- `plans`, `pending_actions`
- `health_daily_metrics`, `activities`, `import_manifest`
- `athlete_journal`, `profiles`

Cached files cleared:
- `.persona_test_jwt/<slug>.json`
- `.persona_test_session/<slug>.json`

---

## Sample test scenarios

These are the three canonical scenarios. Use them as warm-up scripts; then let the conversation evolve.

### Scenario 1: Plan request (Elena)

Goal: verify the coach builds a plan and persists it.

```
Turn 1: "Ich starte gleich mit der Marathon-Vorbereitung fuer Koeln im Oktober. Kannst du mir einen 24-Wochen-Plan aufbauen?"

Expected coach behaviour:
- get_active_plan -> should return None
- read journal to confirm goal
- get_activities to anchor fitness baseline
- save_plan with weeks/sessions array
- Respond with plan summary

EVAL checklist for the coach response:
- Tool usage: did it call save_plan?
- Completeness: did it mention phases (base/build/peak/taper)?
- Tone: did it confirm Elena's specifics (4:58 marathon pace target)?
```

### Scenario 2: Injury report (Lisa)

Goal: verify the coach annotates the activity, adjusts the next session, appends to Open Threads.

```
Turn 1: "Habe gerade meinen Long Run absolviert, 22km bei 5:15/km. Auf den letzten 5km hatte ich wieder dieses Spannungsgefuehl im rechten Knie. Kein Schmerz, aber unangenehm."

Expected coach behaviour:
- get_activities -> identify the long run just done
- annotate_activity with "knee tension last 5km"
- read journal -> see ITB history from 2025
- append_to_journal Open Threads
- Optionally adjust the next planned session
- Respond with empathy + acknowledgement of knee history + concrete next-session adjustment

EVAL checklist:
- Holistic: did it explicitly reference the ITB history from the journal?
- Tool usage: annotate_activity + append_to_journal both fired?
- Accuracy: did it pull the actual long run, not invent one?
```

### Scenario 3: Proactive sync (any persona, esp. Elena)

Goal: verify the STRICT rule "if user reports a fresh activity not yet in get_activities, the coach must sync_garmin_data first".

```
Turn 1: "war eben laufen, 10km easy bei 5:00 HR 142"

Expected coach behaviour:
- get_activities first
- See that today's activity is NOT in the list
- sync_garmin_data(mode='auto') -> picks up the synthetic activity
- get_activities again -> now contains today
- get_activity_details for full numbers
- Respond with actual numbers from the sync, not made up

EVAL checklist:
- Tool usage 5: sync called before responding
- Hallucinations 5: numbers match what get_activity_details returned
- If coach responded with paces/HR without sync_garmin_data: this is a 1 on Tool usage and likely a 2 on Accuracy
```

---

## Reporting

At session end, produce a structured report. Template:

```markdown
# Persona test session: <slug>, <date>

## Setup
- Persona: <slug>
- API: <api_url>
- Turns: <N>
- Avg score: <X>/30

## Per-turn summary
| Turn | Topic                     | Accuracy | Compl | Tools | Tone | Hallu | Holistic | Total |
|------|---------------------------|----------|-------|-------|------|-------|----------|-------|
| 1    | activity report           | 5        | 4     | 5     | 4    | 5     | 4        | 27    |
| 2    | plan request              | 4        | 5     | 5     | 5    | 4     | 3        | 26    |
| ...  |                           |          |       |       |      |       |          |       |

## Regressions (turns scoring below 20)
- Turn N: <topic>. Reason: <why>. Suggested coach fix: <hypothesis>.

## Patterns
- <e.g. "Tool usage consistently 5; holistic consistently 3 - coach rarely cross-references HRV when persona reports tiredness">

## Recommendations
- <concrete changes to STRICT rules, system prompt, or tool descriptions>
```

Save this in `.planning/persona_tests/<slug>_<YYYY-MM-DD>.md` (create the directory if missing).

---

## Troubleshooting

| Symptom                                                | Fix                                                                   |
|--------------------------------------------------------|------------------------------------------------------------------------|
| `ERROR: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY...` | Add both to `.env`. Service role is mandatory.                         |
| `sign_in_with_password failed`                         | Run `seed.py <slug>` first to ensure the auth user exists.             |
| `HTTP 401` from chat.py                                | JWT expired between cache and request. The script retries once auto.   |
| `HTTP 429: Daily token budget exceeded`                | You've hit the daily budget. Wait or bump the budget in config.        |
| `stream ended without done event`                      | Likely a backend crash mid-stream. Check server logs.                  |
| Activities not visible                                 | Check that seed completed all 60+ activities; run seed.py again.       |
| Journal looks wrong                                    | `personas/<slug>.md` section headings must match canonical names.      |
| Coach gives stale data after reset                     | Run `reset.py <slug>` then `seed.py <slug>` to fully refresh.          |

---

## Files in this framework

| File              | Role                                                                  |
|-------------------|------------------------------------------------------------------------|
| `personas/*.md`   | Persona definitions (frontmatter + identity / goal / voice sections)   |
| `seed.py`         | End-to-end idempotent seeder                                           |
| `reset.py`        | Per-persona data wipe (safe-by-default, requires is_test marker)       |
| `chat.py`         | One-shot chat CLI with SSE parsing + `--print-evidence` block          |
| `verify_telemetry.py` | Pulls `/admin/critic-stats`, `/admin/gates-stats`, `/admin/prompt-metrics` into one fact pack |
| `fake_garmin.py`  | Synthetic activity generator (deterministic, importable)               |
| `fake_metrics.py` | Synthetic health_daily_metrics generator                               |
| `_persona.py`     | Markdown + frontmatter parser                                          |
| `_supabase.py`    | Auth admin + JWT sign-in helpers                                       |
| `RESEARCH.md`     | Q2 2026 SOTA on persona-driven eval (with citations)                   |
| `DESIGN.md`       | Architecture and design choices                                        |
| `EVAL_RUBRIC.md`  | Scoring rubric used by the orchestrating Claude Code                   |
| `FAILURE_MODE.md` | Forensic of the imagined-findings failure mode + mandate rationale     |
| `README.md`       | This file                                                              |

---

## What's NOT here (intentional)

- No Python "auto-judge" script: the LLM-in-the-loop (Claude Code) does the scoring.
- No pytest harness driving conversations: Roman wants the LLM to drive in real time.
- No mock backend: we hit the real `/chat` SSE endpoint to exercise the full stack.

If a future session wants to add a programmatic judge LLM, the place to add it is a new `judge.py` that takes a transcript and emits a structured JSON eval. But by default, the orchestrating Claude Code judges live, in-line.
