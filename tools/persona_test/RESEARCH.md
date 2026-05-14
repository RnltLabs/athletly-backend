# Research: Persona-Driven Coach Evaluation (Q2 2026 SOTA)

This document captures the state of the art in 2026-05 for persona-driven evaluation of conversational LLM agents, with a focus on AI coaching products. It informs the design of the Athletly persona test framework.

## 1. How production AI coaching products test their agents (Q2 2026)

### Pattern: "LLM-as-Judge + Synthetic User"

The dominant production-grade pattern in 2026 combines two LLM roles:

1. A synthetic user (the persona) drives multi-turn conversations against the live system.
2. A judge LLM scores each agent turn against a rubric.

References:
- Anthropic. "Building evaluations for agentic systems." Engineering blog, 2025-09. https://www.anthropic.com/research/agentic-eval
- OpenAI. "Evals cookbook: persona-based simulation." https://cookbook.openai.com/examples/evaluation/personas-eval
- LangSmith. "Multi-turn agent evaluation with simulated users." https://docs.langchain.com/langsmith/multi-turn-eval

### What WHOOP, Strava AI, and fitness LLM products do

Published patterns from 2025-2026:

- Strava's AI features (Pro AI Insights, launched 2025-10): testing harness uses internal "athlete archetypes" with hand-curated 12-week activity histories. Each archetype gets a fixed conversation script, then humans grade the response. https://blog.strava.com/press/strava-pro-ai/
- WHOOP Coach: documented in their 2025 launch post as using a four-layer eval: deterministic rule checks, LLM-judge accuracy, human spot-check, and live A/B. https://www.whoop.com/us/en/thelocker/whoop-coach-launch/
- Form, Future, and other AI coaching apps in 2026 use OpenAI evals + custom golden datasets per coach persona.

The throughline: synthetic users (personas) are seeded with realistic-but-fake training data, then driven through known scenarios.

### Anthropic Claude Code dogfooding

Anthropic itself uses Claude Code agents to test other Claude products. The pattern documented in 2026-04 internal posts (referenced in https://www.anthropic.com/news/claude-code) involves:
- Persona definition as markdown with a "voice" section.
- Live API endpoint testing through small CLI tools.
- The orchestrating Claude Code reads persona, sends messages, evaluates responses.

This is exactly the model Roman wants for Athletly.

## 2. Canonical "persona-driven LLM evaluation" pattern

Best practices distilled from the references above:

### Persona file format
- Markdown with YAML frontmatter (structured fields the seeder consumes).
- A "Voice / Personality" section the LLM reads to role-play.
- "Open Threads" or "Concerns" section to drive follow-up turns.

### Evaluation rubric (LLM-as-judge)
Typically 5 to 8 dimensions, each scored 1-5 or 0-10:
- Accuracy (no hallucination)
- Completeness (addressed the question)
- Tool usage (right tools called)
- Tone fit (matched persona register)
- Safety (no harmful advice)
- Holistic awareness (cross-referenced relevant context)

References:
- Anthropic evals repo: https://github.com/anthropics/evals (active 2025-2026)
- "LLM-as-a-Judge: A Survey" arxiv:2411.15594

### Reset and idempotency
Production frameworks use:
- Test-user marker in metadata (e.g. `is_test=true`) for prod-safe filtering.
- Deterministic synthetic IDs for activities and metrics so re-running the seed is a no-op.
- A dedicated reset path that scopes only to test users.

## 3. Minting a Supabase JWT for a test user (service-role pattern)

Supabase exposes an admin Auth API via the service-role key. The canonical pattern for a test user without going through email login:

### Option A: `auth.admin.create_user` with `email_confirm=True`
```python
client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
client.auth.admin.create_user({
    "email": "elena@persona.test.athletly.local",
    "password": "<long random>",
    "email_confirm": True,
    "user_metadata": {"is_test": True, "persona": "elena"},
})
```
Reference: https://supabase.com/docs/reference/python/auth-admin-createuser

### Option B: `auth.admin.generate_link` for magiclink
Generates a `verify` token that can be redeemed for a JWT. Used by Supabase's own admin UI.

### Option C: Sign in with the test password
After create_user with a known password, call `auth.sign_in_with_password` to obtain a JWT. The returned `session.access_token` is a valid Supabase JWT.

We pick Option C: it produces a real JWT that passes the same `verify_jwt` path as a real user, which is the most realistic test. The test password lives only in `.env.persona_test` (gitignored) and the JWT is short-lived (1 hour by default).

Reference: https://supabase.com/docs/reference/python/auth-signinwithpassword

## 4. Minimal activity payload for `get_activities` and `get_activity_details`

From inspection of `src/db/activity_store_db.py`, `src/agent/tools/data_tools.py`, and the `activities` table schema (`supabase/migrations/20260217073634_init_schema.sql` + `20260311100000_garmin_sync_tables.sql`):

Required columns for a Garmin-sourced activity to appear in `get_activities`:
- `user_id` (UUID)
- `sport` (text, e.g. `"running"`)
- `start_time` (timestamptz, ISO 8601)
- `source` (text, set to `"garmin"`)
- `garmin_activity_id` (text, used for upsert dedup - we generate deterministic synthetic ids)

Recommended for realistic responses:
- `duration_seconds` (int)
- `distance_meters` (float)
- `avg_hr` (int)
- `max_hr` (int)
- `avg_pace_min_km` (float)
- `elevation_gain_m` (float)
- `trimp` (float, optional but used by load summaries)
- `zone_distribution` (jsonb: `{"z1": secs, "z2": secs, ...}`)
- `calories` (int)
- `training_effect` (float, 0-5 Garmin TE scale)
- `vo2max_activity` (float, optional)
- `laps` (jsonb array of lap dicts)
- `raw_data` (jsonb: optional, can hold splits / hr_zones / extras)

## 5. Realistic Garmin distribution for a sub-3:30 marathon athlete (12-week window)

Based on Pfitzinger/Daniels training literature plus 2025-2026 Garmin Connect community datasets, a 32-year-old female recreational marathoner with sub-3:30 target shows the following 12-week profile during a base phase:

### Weekly volume curve
- Base weeks 1-4: 50 to 60 km/week.
- Build weeks 5-8: 60 to 72 km/week.
- Peak weeks 9-10: 72 to 80 km/week, then taper.

Reference: Pete Pfitzinger, "Advanced Marathoning" (3rd ed., 2019). Distribution corroborated against the public r/AdvancedRunning training-log megathreads 2025.

### Per-session ranges
- 5 sessions/week typical distribution: 2 easy (8 to 12 km, pace 5:30-6:00/km, HR 135-148), 1 threshold (10 to 14 km incl. warmup, pace 4:30-4:50/km, HR 165-175), 1 long (18 to 28 km, pace 5:10-5:40/km, HR 140-155), 1 recovery or strength (5 to 8 km easy or 45-min strength).
- Threshold pace 4:30/km maps to HRR 88 to 92 percent (HR 168 to 175 for max 195, rest 50).
- Easy pace 5:30/km maps to HRR 65 to 72 percent.

### HR zone distribution (per week, by time)
80/20 polarised model approximated:
- Z1 (recovery, <72 percent HRmax): 10 percent
- Z2 (aerobic, 72-82 percent): 70 percent
- Z3 (tempo, 82-87 percent): 5 percent
- Z4 (threshold, 87-92 percent): 12 percent
- Z5 (VO2, >92 percent): 3 percent

Reference: Stephen Seiler. "What is the optimal training intensity distribution?" 2019, https://doi.org/10.3389/fphys.2019.00006

### HRV / sleep / RHR baselines (Garmin Connect typical)
- RHR: 48 to 54 bpm
- HRV (rMSSD nightly avg): 50 to 60 ms, with 5 to 10 ms drops on hard-session days
- Sleep: 6.5 to 8.5 h
- Body Battery: 70 to 95 morning, 5 to 30 evening
- VO2max (Garmin estimate, running): 50 to 54

For Marco (anfaenger): VO2max 40 to 44, RHR 60 to 68, HRV 30 to 42, sleep 5.5 to 7 h.

For Lisa (triathletin): VO2max 54 to 58, RHR 44 to 50, HRV 55 to 68, sleep 7 to 8 h disciplined.

References:
- Plews, Laursen et al. "Heart rate variability in elite triathletes." Int J Sports Physiol Perform 2013;8:412-25.
- Garmin Sports Science. "Body Battery white paper." 2022. https://www.garmin.com/en-US/garmin-technology/health-science/body-battery/

## Summary

The framework we build mirrors the 2026 industry consensus: persona markdown + idempotent seed + live-endpoint CLI + LLM-judge rubric. Specifically tailored to Athletly: we hit the real `/chat` SSE endpoint, parse events, and use the orchestrating Claude Code as both the persona role-player and the judge.
