# Feature 5: Episode Replay - Research (Q2 2026)

Semantic memory retrieval for agent context injection. RAG-style replay of past
coaching episodes pulled from pgvector. Goal: agent sees fresh data plus
historically similar coaching moments.

## 1. RAG-for-agents - Production patterns Q2 2026

By Q2 2026 the dominant pattern for "agent memory" RAG is a two-tier model:

- **Recent / hot context**: last N user turns plus structured state injected
  verbatim into the system prompt. Cheap, deterministic.
- **Long-term episodic store**: every meaningful event ("episode") is embedded
  on write and looked up on read via vector similarity. The agent treats
  retrieved episodes as "what I remember", not "what I just observed".

The shared lessons from 2025-2026 production deployments (LangMem, Mem0,
LlamaIndex AgentMemory, Letta/MemGPT, OpenAI Memory):

1. Embed at **write time**, never at read time for the stored items. Only the
   live query gets embedded per turn.
2. Use **HNSW** indexes for pgvector in 2026. IVFFlat is now considered legacy
   for production (smaller ops, faster recall, less tuning).
3. Inject retrieved items as a **labelled context block**, not as fake user
   turns. The agent must be told "these are memories, not new observations".
4. Keep retrieval **synchronous and short** (under 50 ms p95) so it does not
   blow the latency budget for the first token.
5. Apply a **similarity threshold** (typical 0.55 to 0.70 cosine) to avoid
   stuffing the prompt with weakly relevant items.

Sources:
- https://blog.langchain.dev/langmem-sdk-launch/ (LangMem core architecture)
- https://www.llamaindex.ai/blog/agent-memory-introducing-llamaindex-memory
- https://platform.openai.com/docs/guides/agents-memory (general patterns)
- https://github.com/letta-ai/letta (MemGPT-derived agent memory)

## 2. Top-K - signal vs context bloat

Empirical findings reported across 2025-2026 production agents converge on:

- **K = 3** is the sweet spot for "background remember-style" injection. Adds
  ~300-800 tokens of context, signal dominated by the top hit.
- **K = 5** is the sane cap. Beyond 5, recall plateaus and the prompt starts
  to drown the live signal.
- **K adaptive on similarity gap** is slightly better than fixed-K but harder
  to debug. Rule: take everything above similarity threshold T, capped at 5.
  Several teams (Mem0, MemGPT, Pinecone) recommend this hybrid.

References:
- https://mem0.ai/blog/state-of-ai-agent-memory-2026 (Mem0 production K=3 to 5)
- https://www.pinecone.io/learn/retrieval-augmented-generation/ (K selection)
- https://research.trychroma.com/evaluating-chunking (chunking + K interaction)

Our choice: **K = 3 default, max 5, similarity threshold 0.55**. Cap context
contribution at 800 tokens hard limit.

## 3. Composing retrieved episodes into prompt context without confusing the agent

Three sub-decisions:

### Where in the prompt

Retrieved episodes belong in the **per-turn runtime context block**, not in
the static system prompt and not as a fake user message. They are scoped to
this turn, change with every user message, and must not poison the cached
prefix.

For Anthropic with prompt caching, the runtime context is its own
cache_control breakpoint - episodes change per turn so cache hit on this
sub-block is OK only within a turn's tool rounds, not across turns. Our
existing infrastructure already handles that.

### Format

A labelled section is what works in production:

```
# Past Insights (3 relevant episodes from your history)
- 2026-02-20: After a 5-day rest week, your first easy run paced 4:25/km...
- 2026-01-15: Tempo run at 4:05/km gave shin pain. We adjusted to 4:15...
- 2025-12-01: Long run > 18 km in week before race blew up the taper...

Use these to inform your reasoning but do NOT claim you just observed them.
Reference them only if the athlete's current question relates.
```

This format works because:
- Date prefix gives temporal grounding (no confabulation).
- One-line summary keeps tokens low.
- Bullet form is easy for the LLM to scan.
- Explicit anti-confabulation instructions at the end.

### Anti-confabulation guardrails

The single biggest failure mode is the model claiming an injected memory as a
current observation. Three mitigations stack:

1. Explicit label in the section header ("Past Insights from your history").
2. Date prefix on every bullet ("2026-02-20: ...").
3. Footer instruction ("Use these to inform your reasoning but do NOT claim
   you just observed them.").

We adopt all three.

References:
- https://arxiv.org/abs/2304.03442 (Generative Agents - relevance, recency,
  importance scoring)
- https://www.anthropic.com/news/prompt-caching (cache-control breakpoints)

## 4. Mem0 architecture analysis

Mem0 ships a managed service we cannot use (data sovereignty / cost), but
their public architecture is well-documented and worth porting:

### Mem0 core algorithm

1. **Fact extraction**: when a user message arrives, an LLM extracts atomic
   facts ("the athlete prefers morning runs"). Stored as separate memory
   records.
2. **Add/Update/Delete pipeline**: each new fact is compared against existing
   memories via vector similarity. If a near-duplicate exists with conflicting
   info, the old one is updated or marked stale.
3. **Retrieval**: on the next user message, embed the query, fetch top-K
   memories by cosine similarity, return them as context.
4. **Decay**: memories accumulate a `last_accessed` timestamp and slowly
   age out if never retrieved.

### What we adopt vs leave out

| Mem0 element | Adopt? | Why |
|---|---|---|
| pgvector cosine similarity retrieval | Yes | Free, fast, standard |
| K = 3 to 5 default | Yes | Production sweet spot |
| Embed at write time | Yes | Standard practice |
| Fact extraction at write | Partial | We already have episode reflection - reuse that, do not add another LLM call |
| Add/Update/Delete dedup | No (v1) | Episodes are already deduped by period; add later if needed |
| Decay scoring | Light version | Recency boost in final score, no hard expiry |

Source: https://mem0.ai/blog/state-of-ai-agent-memory-2026

## 5. Generative Agents (Stanford) - relevance scoring

Park et al. 2023 introduced the "memory stream" pattern that is now the
academic reference for agent memory:

```
score = a * recency + b * importance + c * relevance
```

Where:
- **recency**: exponential decay since last access (Mem0 calls this "freshness")
- **importance**: LLM-assigned score 1-10 at write time
- **relevance**: cosine similarity between query and memory embedding

Park et al. used `a = b = c = 1.0`. Later production deployments
(LangMem, Letta) typically weight relevance heaviest (0.5 to 0.7) because
recency is already captured by hot-context injection and importance is hard
to score reliably without manual labels.

### Our scoring

We adopt a simplified weighted blend:

```
score = 0.7 * similarity + 0.2 * recency + 0.1 * utility
```

- `similarity`: cosine in [0, 1] from pgvector
- `recency`: exp(-days_since_episode / 90) -> recent episodes worth more
- `utility`: episode-attached score from outcome tracking (matches existing
  `record_episode_outcome` field)

Threshold + ranking: filter by `similarity > 0.55`, then sort by combined
`score`, take top K=3. This is similar to the LangMem approach (similarity
filter first, then re-rank).

Source: https://arxiv.org/abs/2304.03442 (Park et al. 2023)

## 6. Hybrid retrieval - vector vs vector + keyword + recency

Pure vector similarity is fast but loses on rare proper nouns ("Berlin
Marathon", "Stryd") and very recent events. Q2 2026 production answer:

- **Pure vector** for semantic queries ("how do I taper", "shin pain")
- **Hybrid (BM25 + vector + reciprocal rank fusion)** for queries with
  named entities or numeric ranges
- **Recency boost** is universally applied as a post-rerank score adjustment

We are already importing `bm25s` in the project. The 2026 pattern is:

1. Run a vector top-N (e.g. N=20) against pgvector
2. Run BM25 over the same set's text on the candidates
3. Reciprocal Rank Fusion to combine
4. Apply recency boost
5. Trim to K

For v1 we keep it simple: **pure vector + recency boost** only. We do not
add BM25 yet because:
- Episodes are coach-generated text - low proper-noun density
- Our episode volume is small per user (under 100 typically)
- Adding BM25 in v1 doubles complexity for marginal recall

We leave a clean integration point in `episode_retrieval.py` so v2 can plug
in BM25 without touching the call site. The existing pattern from
`session_summarizer.py` uses `bm25s` for chat history - same pattern would
apply.

Sources:
- https://www.elastic.co/blog/improving-information-retrieval-elastic-stack-hybrid (hybrid retrieval RRF)
- https://qdrant.tech/articles/hybrid-search/ (hybrid search 2025)
- https://github.com/xhluca/bm25s (bm25s library we already use)

## 7. Privacy - PHI / sensitive content concerns

Sports coaching episodes contain semi-sensitive content:

- **Injury history** ("shin pain at 4:05/km")
- **Body metrics** (HR, VO2max, weight in some cases)
- **Mental state** ("felt unmotivated", "anxious about race")
- **Schedule** (training times, locations - rarely captured)

This is not PHI in the strict HIPAA sense (we are not a covered entity), but
it is sensitive enough that:

1. **All retrieval is user-scoped via RLS**. The `match_episodes()` RPC
   filters by `user_id` and never crosses users.
2. **Embeddings are derived from coach-generated text**, not raw athlete
   chat. Reduces the risk of embedding sensitive verbatim quotes.
3. **No external embedding service for sensitive text** - we use Gemini's
   `embed_content` which is part of the standard Gemini data path the
   athlete already implicitly consents to. OpenAI fallback only when Gemini
   key is missing, same data-handling posture as the chat path.
4. **Embeddings are deletable** - the same `ON DELETE CASCADE` on
   `episodes.user_id` removes embeddings when an account is deleted.
5. **Replay is opt-out-able** - we expose `ATHLETLY_EPISODE_REPLAY_DISABLED=1`
   for users / tests to disable the whole feature.

We do **not** add a content classifier or PII redactor for v1. The retrieval
target is the coach's own structured summary, not the raw athlete utterance,
so the surface area is already small.

References:
- https://supabase.com/docs/guides/auth/row-level-security (RLS reminder)
- https://ai.google.dev/gemini-api/docs/embeddings (Gemini embeddings privacy)
- https://platform.openai.com/docs/guides/embeddings (OpenAI embeddings - data
  not used for training when API-only)

## Summary - decisions feeding into DESIGN.md

| Decision | Value |
|---|---|
| Embedding model (primary) | Gemini `text-embedding-004` (768-dim) |
| Embedding model (fallback) | OpenAI `text-embedding-3-small` (truncate / pad to 768-dim or store 1536 in separate column) |
| Vector dim | 768 (matches existing pgvector schema for beliefs and 2.5 Gemini standard) |
| Index | HNSW with `vector_cosine_ops` (Q2 2026 standard) |
| K default | 3 |
| K max | 5 |
| Similarity threshold | 0.55 cosine |
| Scoring | `0.7 * similarity + 0.2 * recency + 0.1 * utility` |
| Recency decay | `exp(-days / 90)` |
| Context budget | 800 tokens hard cap |
| Position | Per-turn runtime context block, after recovery status, before onboarding |
| Anti-confabulation | Header label + date prefix + footer instruction |
| Trigger | Every turn the athlete sends a message (not heartbeat) |
| Privacy | RLS scoped, opt-out via env, no PII redactor v1 |
