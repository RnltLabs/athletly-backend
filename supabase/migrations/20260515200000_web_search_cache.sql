-- ============================================================================
-- Web search cache (shared, non user-specific)
--
-- The coach researches sport-specific knowledge (periodisation patterns,
-- unknown sports, equipment, methodologies) via the web_search tool.
-- The same query yields the same answer regardless of who asked, so the
-- cache key is a hash of (query, num_results, freshness) only.
--
-- TTL is 30 days, enforced at read time (lookups filter by
-- created_at >= now() - interval '30 days'). Expired rows are NOT deleted
-- by the migration; a future GC job can sweep them.
--
-- All access goes through the service role (backend agent loop).
-- ============================================================================

SET search_path TO public, extensions;

CREATE TABLE IF NOT EXISTS public.web_search_cache (
    cache_key     TEXT PRIMARY KEY,
    query         TEXT NOT NULL,
    num_results   INTEGER NOT NULL,
    freshness     TEXT NOT NULL,
    response      JSONB NOT NULL,
    hit_count     INTEGER NOT NULL DEFAULT 0,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_hit_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Hot path: lookup by primary key already indexed. Add an index on
-- created_at for the TTL sweep and on hit_count for top-queries reports.
CREATE INDEX IF NOT EXISTS idx_web_search_cache_created_at
    ON public.web_search_cache (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_web_search_cache_hit_count
    ON public.web_search_cache (hit_count DESC);

ALTER TABLE public.web_search_cache ENABLE ROW LEVEL SECURITY;

-- Service role gets full access (backend agent loop reads and writes).
CREATE POLICY "web_search_cache_service_all" ON public.web_search_cache
    FOR ALL TO service_role
    USING (true)
    WITH CHECK (true);

-- No client-side access. The cache is internal infrastructure, not
-- exposed via PostgREST to authenticated users.

GRANT ALL ON public.web_search_cache TO service_role;
