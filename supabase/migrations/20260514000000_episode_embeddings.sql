-- Episode Replay: semantic retrieval over past coaching episodes.
--
-- Adds:
--   * embedding         vector(768) for cosine similarity search
--   * utility           outcome-tracked usefulness score in [0, 1]
--   * last_replayed_at  retrieval telemetry
--
-- Adds an HNSW index for cosine similarity (Q2 2026 standard for pgvector)
-- and a match_episodes() RPC mirroring the match_beliefs() pattern.
--
-- Backward compatible: every new column is nullable or has a default, and
-- the embedding is allowed to be NULL so episode creation never blocks on
-- embedding-provider failure.

BEGIN;

-- 1. New columns on the existing episodes table.

ALTER TABLE public.episodes
  ADD COLUMN IF NOT EXISTS embedding         vector(768),
  ADD COLUMN IF NOT EXISTS utility           FLOAT NOT NULL DEFAULT 0.0,
  ADD COLUMN IF NOT EXISTS last_replayed_at  TIMESTAMPTZ;

COMMENT ON COLUMN public.episodes.embedding IS
  '768-dim vector (Gemini text-embedding-004 or OpenAI text-embedding-3-small@768)';
COMMENT ON COLUMN public.episodes.utility IS
  'Outcome-tracked usefulness in [0, 1]; bumped when episode insights prove out';
COMMENT ON COLUMN public.episodes.last_replayed_at IS
  'Telemetry: most recent time this episode was injected into agent context';

-- 2. HNSW index for cosine similarity. HNSW is Q2 2026 standard for
-- pgvector and outperforms ivfflat on recall + maintenance.
-- Defaults (m=16, ef_construction=64) are appropriate for our volume.

CREATE INDEX IF NOT EXISTS idx_episodes_embedding_hnsw
  ON public.episodes
  USING hnsw (embedding vector_cosine_ops);

-- 3. Cosine similarity RPC, scoped by user_id (RLS-aware).
-- Mirrors match_beliefs() so the call site is consistent.

CREATE OR REPLACE FUNCTION public.match_episodes(
  p_user_id        UUID,
  p_embedding      vector(768),
  p_match_count    INT DEFAULT 20,
  p_min_similarity FLOAT DEFAULT 0.55,
  p_episode_types  TEXT[] DEFAULT NULL
)
RETURNS TABLE (
  id            UUID,
  episode_type  TEXT,
  period_start  DATE,
  period_end    DATE,
  summary       TEXT,
  insights      JSONB,
  utility       FLOAT,
  created_at    TIMESTAMPTZ,
  similarity    FLOAT
)
LANGUAGE plpgsql STABLE SECURITY DEFINER
AS $$
BEGIN
  RETURN QUERY
  SELECT
    e.id,
    e.episode_type,
    e.period_start,
    e.period_end,
    e.summary,
    e.insights,
    e.utility,
    e.created_at,
    1 - (e.embedding <=> p_embedding) AS similarity
  FROM public.episodes e
  WHERE e.user_id = p_user_id
    AND e.embedding IS NOT NULL
    AND (p_episode_types IS NULL OR e.episode_type = ANY(p_episode_types))
    AND 1 - (e.embedding <=> p_embedding) >= p_min_similarity
  ORDER BY e.embedding <=> p_embedding
  LIMIT p_match_count;
END;
$$;

COMMENT ON FUNCTION public.match_episodes IS
  'Cosine similarity search over past episodes; returns top-N with score';

COMMIT;
