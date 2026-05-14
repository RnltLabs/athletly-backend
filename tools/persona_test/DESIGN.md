# Design: Athletly Persona Test Framework

## Goal

Give Roman a way to say "test the coach autonomously as persona Elena" and have Claude Code seed a realistic test user, drive a real chat conversation against the live `/chat` endpoint, and evaluate every coach response against a rubric. The Claude Code session is the persona role-player; the framework is the scaffolding.

## Non-goals

- Not a hands-off pytest run. The LLM-in-the-loop is intentional.
- Not a replacement for unit/integration tests.
- Not for testing the frontend. Pure backend conversation evaluation.

## Directory layout

```
tools/persona_test/
  personas/
    elena.md         # Marathon, healthy, 32, Karlsruhe
    marco.md         # Beginner, half-marathon, 28, Munich
    lisa.md          # Triathlete, injury history, 36, Hamburg
  seed.py            # Idempotent end-to-end seed for a persona
  reset.py           # Wipes a persona's test data back to clean state
  chat.py            # CLI: sends one chat message, prints streaming response
  fake_garmin.py     # Activity generator from persona profile (importable)
  fake_metrics.py    # health_daily_metrics generator (importable)
  _supabase.py       # Auth admin + JWT helpers shared by seed/reset/chat
  _persona.py        # Persona-markdown parser (frontmatter + sections)
  RESEARCH.md
  DESIGN.md
  README.md          # Usage doc - the most important file (written for Claude Code)
  EVAL_RUBRIC.md     # Scoring rubric for coach turns
```

## Persona file format

YAML frontmatter for structured fields. Markdown body for human-readable narrative and voice. Sections are addressable by H2 heading.

```markdown
---
slug: elena
name: Elena Vogel
email: elena@persona.test.athletly.local
password: <generated at seed-time, stored in .env.persona_test>
age: 32
location: Karlsruhe
sports: [running]
goal_event: Koeln Marathon 2026
goal_date: 2026-10-04
goal_target_time: "03:30:00"
goal_type: marathon
training_days_per_week: 5
max_session_minutes: 120
estimated_vo2max: 52
threshold_pace_min_km: 4.50
weekly_volume_km: 65
weeks_of_history: 12
---

## Identity
<free-form markdown>

## Goal
...

## Training history
<what the fake-garmin generator should produce>

## Recovery patterns
<what the fake-metrics generator should produce>

## Open Threads
<concerns the persona should bring up in conversation>

## Personality
<how Claude Code should role-play this persona's voice>
```

Parsing: `_persona.py` provides `load_persona(slug) -> Persona` (a dataclass with frontmatter fields plus `sections: dict[str, str]`).

## Seeding pipeline

`seed.py elena` performs these idempotent steps:

1. Resolve persona file at `personas/elena.md`.
2. Load or mint persona test password. Stored in `.env.persona_test` (gitignored) so subsequent runs use the same credentials.
3. Look up auth user by email via service-role admin API.
   - If absent: `auth.admin.create_user({email, password, email_confirm: True, user_metadata: {is_test: True, persona: slug}})`.
   - If present: refresh `user_metadata` to ensure `is_test=True`.
4. Upsert `profiles` row (name, sports, goal fields, vo2max, threshold pace, weekly volume).
5. Build full `athlete_journal` markdown from persona sections (Identity, Goal, Training history summary, Preferences, Open Threads). Upsert into `athlete_journal` table.
6. Seed 12 weeks of activities via `fake_garmin.generate(persona)`. Each activity uses a deterministic synthetic id `persona_<slug>_<date>_<sport>_<seq>` written to `garmin_activity_id`. Upsert on `(user_id, garmin_activity_id)` so re-runs are no-ops.
7. Seed 30 days of `health_daily_metrics` via `fake_metrics.generate(persona)`. Upsert on `(user_id, date, source='garmin')`.
8. If persona declares an active plan in frontmatter, build it via `plans_db.store_plan` (deactivating any prior plan for the test user).

The whole flow is wrapped in a `seed_persona(persona) -> SeedResult` function so other scripts (and tests) can call it.

### Idempotency invariants

- Re-running seed.py for the same persona produces identical row counts.
- Activity ids are deterministic from `(slug, date_iso, sport, seq)` so upserts hit the unique constraint and update in place.
- `health_daily_metrics` keys on `(user_id, date, source)` and is upserted with `on_conflict`.
- `athlete_journal` keys on `user_id` and uses `upsert`.

## Chat CLI

`chat.py elena "war eben laufen 10km easy bei 5:00 HR 142"` performs:

1. Load persona, look up auth user, fetch or refresh a JWT.
   - JWT cached at `.persona_test_jwt/<slug>.json` with expiry.
   - On expiry: re-sign-in with the persona password.
2. Read previous `session_id` from `.persona_test_session/<slug>.json` (or "new" if absent or `--new` flag).
3. POST to `{api_url}/chat` with body `{message, session_id, context: "coach"}` and `Authorization: Bearer <jwt>`.
4. Parse the SSE stream event-by-event using a small reader (no external SSE client dep needed; the protocol is line-based and well-defined here).
5. Render events to the terminal in a human-readable format:
   ```
   > Elena: <user message>
   [tool] get_activities()
   [tool] sync_garmin_data(mode='auto')
   < Coach: <text>
   [usage] input=850 cache_read=12500 output=180
   ```
6. Persist the resolved `session_id` (from the `session_start` event) so the next call continues the same chat.

API URL: configurable via `--api-url` flag or `ATHLETLY_API_URL` env, default `http://localhost:8000`.

### SSE parsing

Events the chat CLI handles:
- `session_start` - extract `session_id`, persist.
- `thinking` - show `[thinking] <text>` in dim style (optional, suppress by default).
- `tool_hint` / `tool_call` - show `[tool] <name>(<args>)`.
- `tool_group_start` / `tool_group_end` - separator hints (silent by default).
- `tool_result` / `tool_error` - show `[result]` / `[tool error]`.
- `message` - show `< Coach: <text>`.
- `ui_component` - show `[ui] <type>`.
- `action_request` - show `[action] <action_type>: <label>`.
- `usage` - show `[usage] input=N output=N model=X`.
- `error` - show `[error] <code>: <message>` in red.
- `done` - terminate the read loop.

## Reset

`reset.py elena` performs (all scoped to a single test user_id):

1. Resolve the auth user. If absent: noop, exit 0.
2. Delete from these tables in dependency order, all filtered by `user_id`: `session_messages`, `sessions`, `episodes`, `episode_consolidations`, `plans`, `pending_actions`, `health_daily_metrics`, `activities`, `import_manifest`, `coaching_lessons`, `prompt_violations`, `goal_history` (if present), `athlete_journal`, `profiles`.
3. Remove the auth user via `auth.admin.delete_user(user_id)` ONLY if `--hard` flag is passed; default keeps the auth user so the next `seed.py` skips the create_user round-trip.
4. Wipe `.persona_test_jwt/<slug>.json` and `.persona_test_session/<slug>.json`.

Safety: the deletes use `service_role` but every query filters on the resolved `user_id`. The script refuses to run if `user_metadata.is_test != True`.

## Eval rubric

Documented in `EVAL_RUBRIC.md`. Six dimensions, scored 1-5:
- Accuracy (did it use real data or hallucinate?)
- Completeness (did it answer the persona's actual concern?)
- Tools used (right tools per the STRICT rules?)
- Tone (matched persona register?)
- Hallucinations (any made-up facts?)
- Holistic awareness (recovery/sleep/HRV cross-reference where relevant?)

Total 30. Below 20 = bad turn; above 25 = excellent.

The orchestrating Claude Code does the scoring in the chat transcript using the rubric, then summarises across the session in a structured report at the end.

## Failure modes and guards

- Missing `SUPABASE_SERVICE_ROLE_KEY` -> seed/reset exit with code 2 and a clear message.
- Missing `.env.persona_test` -> auto-generated on first seed run.
- Persona file malformed (missing required frontmatter) -> validation error before any DB call.
- Chat CLI: HTTP 401 -> re-sign-in once, retry; on second failure exit with code 3.
- Chat CLI: HTTP 429 (budget) -> exit cleanly with a hint that we hit the daily token budget.
- Chat CLI: SSE stream cut mid-response -> show partial output, exit 4.

## What's intentionally NOT here

- No "auto-judge" Python harness. The judging is Claude Code reading the rubric and writing a structured assessment. This keeps the loop transparent and adaptable.
- No PyTorch / async test runner. Plain CLI tools that print to stdout.
- No persistent persona test database. Everything runs against the same Supabase project; only the `is_test=True` marker separates test users from real ones.
