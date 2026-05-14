# Feature 5: Episode Replay - Design

Semantic memory retrieval injecting top-K relevant past episodes into the
agent's per-turn runtime context.

## 1. Episode schema

### Existing (from `episodes` table, migration 20260302000000)

```sql
CREATE TABLE public.episodes (
  id            UUID PRIMARY KEY,
  user_id       UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  episode_type  TEXT NOT NULL DEFAULT 'weekly_reflection',
  period_start  DATE,
  period_end    DATE,
  summary       TEXT NOT NULL DEFAULT '',
  insights      JSONB DEFAULT '[]'::jsonb,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

### Augmentation (new migration)

Add three columns:

```sql
ALTER TABLE public.episodes
  ADD COLUMN embedding   vector(768),
  ADD COLUMN utility     FLOAT NOT NULL DEFAULT 0.0,
  ADD COLUMN last_replayed_at TIMESTAMPTZ;
```

- `embedding` - 768-dim Gemini text-embedding-004 vector. Nullable so
  ingestion never blocks if embedding fails.
- `utility` - outcome-tracking score in [0, 1]. Bumped externally when an
  episode's insight leads to a confirmed lesson. Default 0.0 keeps existing
  episodes unchanged.
- `last_replayed_at` - retrieval telemetry. Useful for measuring how often
  retrieval actually fires and for a future "frequently replayed" UX.

HNSW index for cosine similarity:

```sql
CREATE INDEX idx_episodes_embedding_hnsw
  ON public.episodes
  USING hnsw (embedding vector_cosine_ops);
```

HNSW is Q2 2026 standard for pgvector (IVFFlat used for legacy `beliefs`
index pre-2026). No `WITH (m=..., ef_construction=...)` tuning needed at our
volume - the defaults (m=16, ef_construction=64) are well-tested.

### Episode text for embedding

We do not embed the raw `summary` field alone - too short and varies in
quality. The embedding source is a synthesized text:

```
{summary}

Insights:
- {insights[0]}
- {insights[1]}
...
```

Capped at 2000 chars before sending to the embedding API. This matches what
`user_model_db._generate_embedding` does for beliefs.

## 2. Embedding model selection

### Primary: Gemini `text-embedding-004`

- 768 dimensions (matches existing pgvector schema)
- Free tier: 1500 RPM, 100K TPM
- Already in use for beliefs / journal entries - reuse `get_client()` from
  `src/agent/llm.py`
- Latency: ~50 to 100 ms p50, ~200 ms p95

### Fallback: OpenAI `text-embedding-3-small`

- 1536 dimensions natively; can be truncated to 768 via OpenAI's
  `dimensions` parameter (officially supported via MRL training)
- Cost: $0.02 per 1M tokens (~$0.000004 per typical episode embedding)
- Only invoked when `EMBEDDING_MODEL=openai/text-embedding-3-small` is set
  explicitly or when `GEMINI_API_KEY` is missing
- Latency: ~30 to 80 ms p50

### Selection logic

```python
def get_embedding_model() -> EmbeddingModel:
    settings = get_settings()
    override = settings.embedding_model.lower()

    if override.startswith("openai/"):
        return OpenAIEmbedding()
    if override.startswith("gemini/") or override == "":
        if settings.gemini_api_key:
            return GeminiEmbedding()
        if settings.openai_api_key:
            return OpenAIEmbedding()

    raise EmbeddingProviderUnavailable()
```

The `EMBEDDING_MODEL` env var defaults to empty (auto-select). Possible
values:
- `""` or `gemini/text-embedding-004` - Gemini (default)
- `openai/text-embedding-3-small` - OpenAI

Both providers must produce **768-dim vectors** to match the pgvector
schema. OpenAI is asked for `dimensions=768`.

### Dimension compatibility

Both Gemini text-embedding-004 (native 768) and OpenAI 3-small (truncated to
768) target the same column. Mixed providers across episodes is acceptable -
cosine similarity is geometrically meaningful regardless of which provider
generated each side; in practice cross-provider similarity is degraded but
the same provider on both query and stored side is the common case.

## 3. Retrieval algorithm

### Inputs

- `user_id`: UUID, RLS scope
- `query_text`: live user message
- `top_k`: default 3, max 5
- `similarity_threshold`: default 0.55
- `episode_types`: filter list (default `["weekly_reflection", "monthly_review", "coaching_insight"]`)

### Algorithm

```
1. embedding = embed(query_text)
   - If fail -> return []  (graceful degradation, no replay this turn)

2. candidates = match_episodes_rpc(
     user_id, embedding,
     match_count = top_k * 4,   # over-fetch for reranking
     min_similarity = similarity_threshold,
     episode_types = filter,
   )
   - Returns: list of (episode, similarity)

3. rerank each candidate:
   days_since   = (today - period_end).days
   recency      = exp(-days_since / 90)
   utility      = episode.utility           # in [0, 1]
   final_score  = 0.7 * similarity
                + 0.2 * recency
                + 0.1 * utility

4. sort candidates by final_score desc, take top_k

5. for each retained episode:
   update episodes.last_replayed_at = now()  (best-effort, non-blocking)

6. return formatted block
```

### Cosine via pgvector

The RPC uses the existing pattern from `match_beliefs`:

```sql
SELECT
  e.id, e.episode_type, e.period_start, e.period_end,
  e.summary, e.insights, e.utility,
  1 - (e.embedding <=> p_embedding) AS similarity
FROM public.episodes e
WHERE e.user_id = p_user_id
  AND e.embedding IS NOT NULL
  AND (p_episode_types IS NULL OR e.episode_type = ANY(p_episode_types))
  AND 1 - (e.embedding <=> p_embedding) >= p_min_similarity
ORDER BY e.embedding <=> p_embedding
LIMIT p_match_count;
```

### Latency budget

- Query embedding: ~80 ms p50 (Gemini)
- pgvector lookup with HNSW: ~10 ms p50 at our volumes
- Reranking + formatting: ~1 ms
- **Total**: ~100 ms p50, ~250 ms p95

Acceptable because this runs once per turn before the LLM call (which is
~1-3 s anyway).

## 4. Context injection

### Where in `build_runtime_context`

Insert a new section between "Current Recovery Status" (line ~733) and
"Onboarding State" (line ~737). Rationale:

- After recovery and load summaries: those are facts of the current week
  the coach must always see first.
- Before onboarding state: onboarding mode appended last as instructions.
- Before startup context: startup context is a one-shot session-start dump
  and stays at the end as it logically follows freshly synthesized info.

The new section is **only emitted when at least one episode passes the
similarity threshold**. No episodes -> no section -> no token cost.

### Format

```
# Past Insights ({k} relevant episodes from your history)
- 2026-02-20: After a 5-day rest week, your first easy run paced 4:25/km (HRR 145). Pattern: 5+ days rest = clear paced bounce.
- 2026-01-15: Tempo run at 4:05/km gave shin pain. Adjusted to 4:15 for 3 weeks. No recurrence.
- 2025-12-01: Long run > 18 km in race week blew up the taper. Capped at 14 km in race-week.

Use these to inform your reasoning but do NOT claim you just observed them.
Reference them only if the athlete's current question relates.
```

Format rules:
- One bullet per episode
- Date prefix: `YYYY-MM-DD` from `period_end` (fallback `period_start`, then
  `created_at`)
- Body: first sentence of `summary`, then top insight if it fits the budget
- Total cap: 800 tokens hard limit (truncate bullets if necessary,
  trailing ellipsis on the truncated one)

### Token budget breakdown

For the post-feature runtime_context:

| Section | Tokens (typical) |
|---|---|
| Current Date | ~20 |
| Athlete journal (`build_athlete_md`) | 500 to 1500 |
| Available Skills | 100 to 300 |
| Output Style flag | ~10 |
| Current Athlete profile | 50 to 100 |
| Active Training Plan | 100 to 400 |
| Cross-source Training Load | 100 to 200 |
| Current Recovery Status | 100 to 200 |
| **Past Insights (NEW)** | **0 to 800** |
| Onboarding State (when applicable) | 50 to 100 |
| Startup Context (session start only) | 0 to 1000 |
| **Total per-turn** | **1000 to 4600** |

Past Insights occupies max 800 tokens, average ~400. Within Anthropic's
prompt-caching breakpoint envelope - the runtime_context block stays under
~5K tokens which is well below the cache-write efficiency threshold of
~10K characters.

## 5. Trigger

Episode replay runs on **every athlete user message** during the
`build_runtime_context` call. Specifically:

- Every call to `process_message()` triggers a single retrieval.
- The retrieval result is woven into the runtime_context once and remains
  byte-stable across all tool-rounds of that turn (so Anthropic caching
  works inside the turn).
- The next user turn re-runs retrieval with the new message.

We deliberately do **not**:
- Trigger on every tool round (would invalidate cache breakpoints)
- Trigger from heartbeat / proactive flows (no live query to retrieve
  against)
- Trigger when `user_message` is empty / whitespace
- Trigger when `ATHLETLY_EPISODE_REPLAY_DISABLED=1`

### Trigger gating

```python
def should_retrieve(user_message: str | None) -> bool:
    if os.environ.get("ATHLETLY_EPISODE_REPLAY_DISABLED") == "1":
        return False
    if not user_message or not user_message.strip():
        return False
    # Very short messages ("hi", "ok") rarely benefit
    if len(user_message.strip()) < 8:
        return False
    return True
```

The 8-char floor avoids embedding spend on greetings. Tunable via env if
needed.

## 6. Hallucination guard

Three layers stack:

### Layer 1 - Section header label

`# Past Insights (3 relevant episodes from your history)` - the word
"history" and the count both signal these are stored memories.

### Layer 2 - Date prefix on every bullet

`2026-02-20:` makes it impossible for the model to treat the line as a
fresh observation.

### Layer 3 - Footer instruction

```
Use these to inform your reasoning but do NOT claim you just observed them.
Reference them only if the athlete's current question relates.
```

Two directives: do not confabulate, only cite when relevant. Tested by us
internally on Claude Sonnet 4.6 and Gemini 2.5 Flash with positive results
(the model treats the section as memory, not data).

### Static system prompt addendum

We add a short directive to `STATIC_SYSTEM_PROMPT` so the model has
top-level context for the new section. Single sentence so cache doesn't
churn:

```
When you see a "# Past Insights" section in the runtime context, those
are retrieved past coaching episodes. Treat them as your memory, not as
new observations. Reference them only when relevant to the athlete's
current question.
```

## 7. Wiring touch points

| Layer | File | Change |
|---|---|---|
| Migration | `supabase/migrations/20260514000000_episode_embeddings.sql` | Add columns + HNSW index + `match_episodes` RPC |
| Service | `src/services/embeddings.py` (NEW) | `EmbeddingProvider` interface, Gemini + OpenAI implementations, model selection |
| Service | `src/services/episode_retrieval.py` (NEW) | `retrieve_relevant_episodes()`, formatting, scoring |
| DB | `src/db/episodes_db.py` | Extend `store_episode` to compute + persist embedding; add `match_episodes` wrapper |
| Memory | `src/memory/episodes.py` | Extend `store_episode` path (file-backed) to optionally embed |
| Tool | `src/agent/tools/memory_tools.py` | `store_episode` tool calls embedding pipeline |
| Prompt | `src/agent/system_prompt.py` | `build_runtime_context` accepts `user_message`, injects retrieved block. Add static directive |
| Loop | `src/agent/agent_loop.py` | Pass `user_message` to `build_runtime_context` |
| Config | `src/config.py` | Add `embedding_model`, `episode_replay_top_k`, `episode_replay_threshold` settings |
| Tests | `tests/test_episode_retrieval.py` (NEW) | Unit + integration tests |

### Embedding generation surface

A single `embed_text(text: str) -> list[float] | None` function exposed
from `src/services/embeddings.py`. Returns `None` on any failure - all call
sites must handle the None gracefully (store episode without embedding,
skip retrieval that turn). The function:

- Reads `EMBEDDING_MODEL` env at first call, caches choice
- Routes to Gemini or OpenAI implementation
- Implements retry with exponential backoff (max 2 retries)
- Returns 768-dim float list or None

## 8. Free-tier compatibility

All components fit the free tier:

| Component | Free? | Why |
|---|---|---|
| pgvector + HNSW | Yes | Supabase free plan |
| Gemini text-embedding-004 | Yes | 1500 RPM free tier, well above per-user usage |
| Storage | Yes | Embeddings are ~3 KB each (768 * 4 bytes). 100 episodes = 300 KB |
| Compute | Yes | Retrieval is ~10 ms in DB, no extra LLM call |
| OpenAI fallback | Yes | $0.02 / 1M tokens, ~$0.0001 per active user / month at typical episode volume |

## 9. Failure modes and graceful degradation

| Failure | Effect |
|---|---|
| Embedding provider unavailable | No retrieval this turn. Episode storage continues without embedding (back-filled later if env recovers) |
| Embedding returned wrong dim | Refuse to insert, log warning |
| pgvector RPC error | Return [] from retrieval, runtime_context omits the section |
| User has no episodes yet | RPC returns empty list, section omitted |
| Similarity below threshold for all candidates | Section omitted |
| Anthropic cache invalidation | Acceptable - retrieval is per-turn, designed to change |

## 10. Out of scope for v1

- Hybrid retrieval (BM25 + vector reciprocal rank fusion)
- Adaptive top-K based on similarity gap
- Cross-user / cohort-level pattern retrieval
- Importance scoring via LLM
- Backfill job for existing episodes without embeddings (provided via a
  one-shot script in `tools/` only)
- Memory decay / archival

Documented for v2.
