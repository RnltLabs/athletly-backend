-- Prompt rule violation telemetry (Feature 6).
--
-- Records every STRICT rule violation detected by the regex scanners
-- in src/services/prompt_metrics.py. The live ring buffer covers
-- short-horizon dashboarding; this table provides the historical
-- record for retrospective A/B comparisons and trend analysis.
--
-- All inserts go via the service role (backend agent loop). The table
-- is opt-in via the existing use_supabase feature flag: the agent
-- loop's prompt_metrics hook records to the in-memory buffer
-- unconditionally and only writes here if Supabase is reachable.

CREATE TABLE IF NOT EXISTS public.prompt_violations (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    ts TIMESTAMPTZ NOT NULL DEFAULT now(),
    user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL,
    model TEXT NOT NULL,
    variant TEXT NOT NULL DEFAULT 'default',
    rule_id TEXT NOT NULL,
    severity TEXT NOT NULL CHECK (severity IN ('strict', 'warn')),
    response_length INT NOT NULL DEFAULT 0,
    snippet TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_prompt_violations_ts
    ON public.prompt_violations (ts DESC);

CREATE INDEX IF NOT EXISTS idx_prompt_violations_rule_ts
    ON public.prompt_violations (rule_id, ts DESC);

CREATE INDEX IF NOT EXISTS idx_prompt_violations_variant_ts
    ON public.prompt_violations (variant, ts DESC);

ALTER TABLE public.prompt_violations ENABLE ROW LEVEL SECURITY;

-- Service role gets full access (backend agent loop writes here).
CREATE POLICY "prompt_violations_service_all" ON public.prompt_violations
    FOR ALL TO service_role
    USING (true)
    WITH CHECK (true);

-- No SELECT for authenticated users. This is an internal observability
-- table; admins read it via the /admin/prompt-metrics endpoint which
-- runs as the service role behind the Hetzner reverse-proxy gate.

GRANT ALL ON public.prompt_violations TO service_role;
