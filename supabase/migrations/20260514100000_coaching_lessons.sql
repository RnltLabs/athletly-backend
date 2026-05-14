SET search_path TO public, extensions;

-- ============================================================================
-- Feature 3: Reflexion Loop
--
-- coaching_lessons:   per-user lessons extracted by the reflection LLM,
--                     with pgvector embeddings for semantic retrieval.
-- reflexion_runs:     idempotency table that prevents double-running a
--                     reflection for the same session or month.
-- match_coaching_lessons(): pgvector cosine-similarity RPC, sibling of
--                     match_beliefs().
--
-- Idempotent: all CREATEs are guarded by IF NOT EXISTS / OR REPLACE so a
-- rerun of the migration is safe.
-- ============================================================================

-- pgvector is already created by the agentic-sports schema migration, but
-- we re-assert it here so this migration is self-contained for new branches.
CREATE EXTENSION IF NOT EXISTS vector;


-- ============================================================================
-- 1. COACHING_LESSONS
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.coaching_lessons (
  id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id            UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,

  topic              TEXT NOT NULL CHECK (topic IN (
                       'scheduling', 'volume', 'intensity', 'recovery',
                       'nutrition', 'injury', 'motivation', 'preference',
                       'constraint', 'other'
                     )),
  observation        TEXT NOT NULL,
  lesson             TEXT NOT NULL,
  applicable_to      TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
  evidence           TEXT NOT NULL,

  confidence         FLOAT NOT NULL DEFAULT 0.7
                       CHECK (confidence >= 0.0 AND confidence <= 1.0),

  -- pgvector embedding (768-dim, matches Gemini text-embedding-004).
  embedding          vector(768),

  source_session_id  UUID REFERENCES public.sessions(id) ON DELETE SET NULL,

  -- Reinforcement / decay tracking
  reinforced_count   INT NOT NULL DEFAULT 0,
  last_retrieved_at  TIMESTAMPTZ,

  -- Lifecycle
  active             BOOLEAN NOT NULL DEFAULT true,
  archived_at        TIMESTAMPTZ,
  superseded_by_id   UUID REFERENCES public.coaching_lessons(id),

  created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE  public.coaching_lessons IS
  'Reflexion-style lessons distilled from coaching conversations.';
COMMENT ON COLUMN public.coaching_lessons.embedding IS
  '768-dim Gemini text-embedding-004 over topic + lesson + applicable_to';
COMMENT ON COLUMN public.coaching_lessons.evidence IS
  'Direct user quote that justifies the lesson, validated against transcript.';

-- Indexes
CREATE INDEX IF NOT EXISTS idx_coaching_lessons_user_active
  ON public.coaching_lessons(user_id, active)
  WHERE active = true;

CREATE INDEX IF NOT EXISTS idx_coaching_lessons_user_topic
  ON public.coaching_lessons(user_id, topic)
  WHERE active = true;

CREATE INDEX IF NOT EXISTS idx_coaching_lessons_embedding
  ON public.coaching_lessons
  USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 20);

-- RLS
ALTER TABLE public.coaching_lessons ENABLE ROW LEVEL SECURITY;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_policies
    WHERE tablename = 'coaching_lessons'
      AND policyname = 'coaching_lessons_select_own'
  ) THEN
    CREATE POLICY "coaching_lessons_select_own" ON public.coaching_lessons
      FOR SELECT TO authenticated USING (auth.uid() = user_id);
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM pg_policies
    WHERE tablename = 'coaching_lessons'
      AND policyname = 'coaching_lessons_insert_own'
  ) THEN
    CREATE POLICY "coaching_lessons_insert_own" ON public.coaching_lessons
      FOR INSERT TO authenticated WITH CHECK (auth.uid() = user_id);
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM pg_policies
    WHERE tablename = 'coaching_lessons'
      AND policyname = 'coaching_lessons_update_own'
  ) THEN
    CREATE POLICY "coaching_lessons_update_own" ON public.coaching_lessons
      FOR UPDATE TO authenticated USING (auth.uid() = user_id);
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM pg_policies
    WHERE tablename = 'coaching_lessons'
      AND policyname = 'coaching_lessons_service_all'
  ) THEN
    CREATE POLICY "coaching_lessons_service_all" ON public.coaching_lessons
      FOR ALL TO service_role USING (true) WITH CHECK (true);
  END IF;
END
$$;


-- ============================================================================
-- 2. match_coaching_lessons() RPC
-- ============================================================================

CREATE OR REPLACE FUNCTION public.match_coaching_lessons(
  p_user_id        UUID,
  p_embedding      vector(768),
  p_match_count    INT DEFAULT 5,
  p_min_confidence FLOAT DEFAULT 0.0
)
RETURNS TABLE (
  id            UUID,
  topic         TEXT,
  lesson        TEXT,
  applicable_to TEXT[],
  confidence    FLOAT,
  reinforced_count INT,
  last_retrieved_at TIMESTAMPTZ,
  created_at    TIMESTAMPTZ,
  similarity    FLOAT
)
LANGUAGE plpgsql STABLE SECURITY DEFINER
AS $$
BEGIN
  RETURN QUERY
  SELECT
    cl.id,
    cl.topic,
    cl.lesson,
    cl.applicable_to,
    cl.confidence,
    cl.reinforced_count,
    cl.last_retrieved_at,
    cl.created_at,
    1 - (cl.embedding <=> p_embedding) AS similarity
  FROM public.coaching_lessons cl
  WHERE cl.user_id = p_user_id
    AND cl.active = true
    AND cl.embedding IS NOT NULL
    AND cl.confidence >= p_min_confidence
  ORDER BY cl.embedding <=> p_embedding
  LIMIT p_match_count;
END;
$$;

COMMENT ON FUNCTION public.match_coaching_lessons IS
  'Cosine similarity search over active coaching_lessons.';


-- ============================================================================
-- 3. REFLEXION_RUNS (idempotency)
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.reflexion_runs (
  user_id        UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  run_kind       TEXT NOT NULL CHECK (run_kind IN ('pro_session', 'free_monthly')),
  period         TEXT NOT NULL,
  ran_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
  lessons_added  INT NOT NULL DEFAULT 0,
  PRIMARY KEY (user_id, run_kind, period)
);

COMMENT ON TABLE public.reflexion_runs IS
  'Idempotency log for the reflection loop. period = session_id (pro) or YYYY-MM (free).';

CREATE INDEX IF NOT EXISTS idx_reflexion_runs_user_kind_ran
  ON public.reflexion_runs(user_id, run_kind, ran_at DESC);

ALTER TABLE public.reflexion_runs ENABLE ROW LEVEL SECURITY;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_policies
    WHERE tablename = 'reflexion_runs'
      AND policyname = 'reflexion_runs_select_own'
  ) THEN
    CREATE POLICY "reflexion_runs_select_own" ON public.reflexion_runs
      FOR SELECT TO authenticated USING (auth.uid() = user_id);
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM pg_policies
    WHERE tablename = 'reflexion_runs'
      AND policyname = 'reflexion_runs_service_all'
  ) THEN
    CREATE POLICY "reflexion_runs_service_all" ON public.reflexion_runs
      FOR ALL TO service_role USING (true) WITH CHECK (true);
  END IF;
END
$$;

GRANT SELECT ON public.coaching_lessons TO authenticated;
GRANT ALL    ON public.coaching_lessons TO service_role;
GRANT SELECT ON public.reflexion_runs   TO authenticated;
GRANT ALL    ON public.reflexion_runs   TO service_role;
