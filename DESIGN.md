# Feature 3: Reflexion Loop - Design

Date: 2026-05-14
Author: Lead engineer (Feature 3)

Scope: per-user session-end self-reflection that extracts durable
lessons from a conversation, stores them with pgvector embeddings,
and injects the top-K relevant lessons into the next session's
runtime context.

## Trigger

Two paths, both running as background tasks so the SSE stream is
never blocked.

### Pro tier: after every session

Hook point: `src/api/routers/chat.py::_chat_event_generator`.

We already have a "new session detected" branch that triggers
`summarize_previous_session`. We reuse that exact branch: when a
client opens a new session (`body.session_id is None` or "new"),
the PREVIOUS session is considered closed and reflection is
scheduled via `asyncio.create_task`.

Why not on stream end? Because the user could just be pausing.
Treating "new session" as "previous session ended" is what the
existing summarizer already does. Defining a session boundary
twice would be wrong.

For the MVP we also gate every reflection on a simple per-day
idempotency check (see "Free-tier rate limit" below) so that
opening four sessions in a row will still only run reflection
once per session, BUT will never run more than (N for tier) per
user per day, no matter what.

Detection of "tier":
- Read from `profiles.meta->>subscription_tier` (existing JSONB).
  Values: `"pro"` or `"free"` (default: `"free"`).
- A tiny helper in `src/services/reflexion.py::_get_tier()` does
  the lookup once and caches per-call.

### Free tier: monthly batch

A daily check (piggy-backed on the same chat hook, gated by
"have we run a free-tier reflection in this calendar month?") runs
a batch reflection covering the latest unreflected session
summaries from the past 30 days. We process at most one batch per
user per month.

This is consistent with the existing `episode_consolidation`
service pattern: same calendar bucket, same idempotency table
shape (we add `(user_id, kind, period)` uniqueness on the new
`reflexion_runs` table; see Storage below).

### Failure mode

Reflection is fire-and-forget. Any exception is logged and the
chat flow continues. Reflection failures NEVER surface to the user.

## Input: what subset of conversation history feeds reflection

For Pro tier (per-session run):
- Load the messages of the previous session via
  `load_session_messages(session_id, max_messages=80)`.
- Filter to roles in `{user, model}` only (we strip
  `tool_call` and `system` rows for the reflection prompt; tool
  results add noise and we already store activity facts elsewhere).
- If `len(messages) < 4`, skip reflection (too short to learn from).
- Otherwise pass the trimmed transcript verbatim to the LLM, with
  hard upper bound of 50 turns to keep the Haiku call cheap.

For Free tier (monthly batch):
- Pull `compressed_summary` from each session in the past 30 days
  via `session_store_db.get_recent_sessions(user_id, limit=30)`,
  filter to sessions with non-null `compressed_summary`.
- Feed concatenated summaries (not raw messages) into the same
  reflection prompt with a "this is a monthly digest" framing.

## Output schema

Reflection LLM returns a JSON object:

```
{
  "lessons": [
    {
      "topic": "scheduling",
      "observation": "Athlete said the Thursday tempo run is 'never going to happen' twice in this session.",
      "lesson": "Thursday is a bad day for tempo work for this athlete. Move tempo to Wednesday by default.",
      "applicable_to": ["running", "plan-generation"],
      "evidence": "I told you already, Thursday tempo is never going to happen with my work schedule.",
      "confidence": 0.85
    }
  ]
}
```

Field semantics:

- `topic`: short slug, one of:
  `scheduling | volume | intensity | recovery | nutrition | injury | motivation | preference | constraint | other`.
  Used as a coarse filter on retrieval.
- `observation`: what the LLM saw in the conversation. Past tense.
- `lesson`: actionable rule the coach should remember next time.
- `applicable_to`: list of contexts where the lesson should fire
  (sport, planning, conversation tone, etc.). Used as an extra
  filter.
- `evidence`: at least one direct quote from the user. We
  post-validate that this substring really exists in the source
  transcript (case-insensitive, whitespace-normalised). Lessons
  with invalid evidence are dropped before storage.
- `confidence`: 0..1 prior set by the LLM. Decayed over time on
  read.

If the LLM finds nothing worth recording, it MUST return
`{"lessons": []}`. The prompt instructs this and the schema
validator enforces it.

## Storage: `coaching_lessons` table

New table, new migration:
`supabase/migrations/20260514000000_coaching_lessons.sql`.

Columns:

| column           | type         | notes                                              |
|------------------|--------------|----------------------------------------------------|
| id               | UUID PK      | `gen_random_uuid()`                                |
| user_id          | UUID FK      | `auth.users(id) on delete cascade`                 |
| topic            | TEXT         | constrained to the topic taxonomy above            |
| observation      | TEXT         |                                                    |
| lesson           | TEXT         | the actionable insight                             |
| applicable_to    | TEXT[]       | postgres text array                                |
| evidence         | TEXT         | user quote, validated against transcript           |
| confidence       | FLOAT        | 0..1, decays on read                               |
| embedding        | vector(768)  | Gemini `text-embedding-004` over topic+lesson+applicable_to |
| source_session_id| UUID         | nullable FK to `sessions(id)`                      |
| reinforced_count | INT          | times the lesson was retrieved & still valid       |
| last_retrieved_at| TIMESTAMPTZ  | NULL if never                                      |
| active           | BOOL         | soft-delete flag                                   |
| created_at       | TIMESTAMPTZ  | `now()`                                            |
| updated_at       | TIMESTAMPTZ  | bumped on supersede/reinforce                      |

Indexes:
- `(user_id, active)` partial WHERE active = true.
- `(user_id, topic, active)` partial.
- `ivfflat (embedding vector_cosine_ops)` with lists = 20, same
  as the beliefs table.

RLS: same pattern as `beliefs` (select/insert/update for own user,
service-role bypass for backend writes).

RPC: `match_coaching_lessons(p_user_id uuid, p_embedding
vector(768), p_match_count int default 5, p_min_confidence float
default 0.0)` returning `(id, topic, lesson, applicable_to,
confidence, similarity)` ordered by cosine distance ascending.
Pattern copied verbatim from `match_beliefs`.

Also a sibling `reflexion_runs` table for idempotency:

| column        | type   |
|---------------|--------|
| user_id       | UUID FK|
| run_kind      | TEXT   | `pro_session` or `free_monthly`              |
| period        | TEXT   | session_id for pro, `YYYY-MM` for free       |
| ran_at        | TIMESTAMPTZ |                                          |
| lessons_added | INT    |                                              |
| PRIMARY KEY (user_id, run_kind, period) |

This prevents double-running on retries or duplicate session
boundaries.

## Retrieval at next session

Hook: `src/agent/system_prompt.py::build_runtime_context`. We add a
new optional section "Lessons learned about this athlete".

Algorithm in `src/services/lesson_retrieval.py::fetch_relevant_lessons`:

1. Build a retrieval query string from the current runtime context:
   the athlete name, sports, goal event, and the recent activity
   summary. Reason: we want lessons that apply NOW, not abstract
   ones.
2. Generate a 768-dim embedding via the existing
   `UserModelDB._generate_embedding` helper. If embeddings are
   disabled (no Gemini key or `ATHLETLY_EMBEDDINGS_DISABLED=1`),
   fall back to the most-recent-N highest-confidence lessons.
3. RPC `match_coaching_lessons` with K=5 and min_confidence=0.4.
4. Apply read-time decay (see Decay) to the returned confidence
   values and re-rank by `effective_confidence * similarity`.
5. Drop anything whose effective confidence < 0.3.
6. Format up to top 5 surviving lessons into a short bullet block,
   capped at ~600 tokens of output.
7. Update `last_retrieved_at` and increment `reinforced_count` for
   the surfaced lessons (fire-and-forget background task).

Output block (verbatim format that goes into runtime context):

```
# Lessons learned about this athlete
- [scheduling] Thursday is a bad day for tempo work for this athlete. Move tempo to Wednesday by default.
- [volume] Athlete consistently undershoots prescribed long runs by 20+ minutes. Cap long runs at 90 min.
```

We intentionally drop `observation` and `evidence` from the prompt
to keep tokens low; those fields live in the DB for audit only.

## Decay

Two complementary mechanisms:

1. Read-time decay: when retrieving, multiply `confidence` by
   `exp(-age_days / 90)`. A 90-day-old lesson loses ~63% of its
   weight; a 180-day-old one keeps ~13%.
2. Reinforcement: every time a lesson is actually retrieved and
   not contradicted, bump `reinforced_count` by 1 and reset its
   age clock by updating `last_retrieved_at`. A reinforced lesson
   uses `max(created_at, last_retrieved_at)` as the age origin.

A nightly (or weekly) sweeper job is OUT OF SCOPE for the MVP; we
rely on read-time filtering and soft-delete on supersede.

Supersession: if a NEW lesson with the same `topic` and a cosine
similarity >= 0.9 against an existing lesson arrives, we mark the
old one `active = false` and set `superseded_by_id` (we add this
column as nullable FK to the same table). This is the same
pattern as `beliefs.superseded_by`.

## Hallucination guard (concrete checks in `reflexion.py`)

- Prompt says explicitly: "Do not invent facts the user did not
  state. Use direct quotes for evidence. If unsure, return an
  empty list."
- After JSON parse, for each candidate lesson:
  - Normalise the `evidence` quote (lowercase, collapse whitespace).
  - Normalise each `user` message content the same way.
  - Require the evidence substring to appear in at least one user
    message. If not, drop the lesson and log a counter.
- Reject lessons whose `topic` is not in the allowed taxonomy.
- Reject lessons longer than 500 chars (anti-bloat).
- Reject lessons mentioning blacklisted PII keywords (full names
  beyond the known athlete name, email patterns, phone-like digit
  runs). MVP regex-based, can be tightened later.

## Free-tier rate limit and Pro-tier cost guard

The `reflexion_runs` PRIMARY KEY enforces:
- One Pro run per (user_id, session_id).
- One Free run per (user_id, YYYY-MM).

The reflexion engine itself also enforces a higher-level rule:
"never run more than once per user per calendar day, regardless of
tier" (the user can spam-open sessions; we do not need 50
reflections in an hour). This is a SELECT-then-skip check at the
top of `run_reflexion()`.

## Touch points (final list)

- NEW `src/services/reflexion.py`: reflection LLM call, hallucination
  guard, idempotency check, storage delegation.
- NEW `src/services/lesson_retrieval.py`: embedding-based fetch +
  decay + bullet formatter.
- NEW `src/db/lessons_db.py`: CRUD on `coaching_lessons` and
  `reflexion_runs`, RPC wrapper.
- NEW `supabase/migrations/20260514000000_coaching_lessons.sql`:
  schema, indexes, RLS, RPC, idempotency table.
- EDIT `src/api/routers/chat.py`: add ~20 lines that schedule
  `asyncio.create_task(run_reflexion(user_id, prev_session_id))`
  in the existing "new session detected" branch.
- EDIT `src/agent/system_prompt.py::build_runtime_context`: add a
  small section that calls `lesson_retrieval.fetch_relevant_lessons`
  and concatenates the formatted block into the existing sections
  list.
- NEW `tests/test_reflexion.py`: unit + integration tests
  (LLM mocked, DB calls mocked via the standard supabase mock).

## Expected lesson store size per active user

Per Pro user: cap of ~3 new lessons per session (the prompt asks
for "1-3 specific lessons"). Realistic Pro user has ~5
sessions/week. After dedupe / supersede roughly half survive:
`~3 * 5 * 0.5 = ~7.5 lessons/week`. After read-time decay drops
old entries below the threshold the effective active store
plateaus around 100-150 rows after 4-6 months.

Per Free user: cap of ~3 new lessons per month. Active store
plateaus around 20-30 rows per active year.

Row size: ~1 KB (text fields) + 768 floats * 4 bytes = ~4 KB
embedding. Call it 5 KB per lesson. 150 lessons = ~750 KB per
Pro user. Supabase free tier handles this without issue
(500 MB DB equates to ~660 Pro users at full saturation, which is
way past free-tier moonshot anyway).

## Pro-tier monthly cost

Assumptions:
- 5 sessions/week per Pro user, 22 sessions/month.
- Average reflection input ~2000 tokens (trimmed transcript), output
  ~400 tokens (JSON with 1-3 lessons).
- Model: `anthropic/claude-haiku-4-5-20251001`.
- Pricing per `MODEL_PRICING`: $1.00 / 1M input, $5.00 / 1M output.

Per reflection:
- input: 2000 * $1.00 / 1_000_000 = $0.002
- output: 400 * $5.00 / 1_000_000 = $0.002
- total: ~$0.004

Per Pro user per month:
- 22 reflections * $0.004 = ~$0.088 (~9 cents).

Plus one embedding call per reflection (Gemini
`text-embedding-004` is effectively free under our existing usage),
and one embedding call per session-open (retrieval). Negligible.

For 1000 Pro users: ~$88/month. Fits the "cheap reflection"
directive.

Free tier (monthly batch): 1 call/month per user with a
larger input (~6000 tokens of summaries) and similar output.
Per Free user: ~1 * (6000 * 1e-6 + 400 * 5e-6) = ~$0.008/month.
Negligible.
