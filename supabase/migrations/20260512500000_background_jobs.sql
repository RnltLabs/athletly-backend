-- Background job queue for the Coach agent.
--
-- Pattern: Claude Code's run_in_background. The agent enqueues long
-- operations (subagent spawns, macrocycle generation, weekly plan
-- generation) and returns immediately. A worker pool processes the
-- jobs and updates this table. The agent polls or lists jobs on the
-- next turn to surface results back to the user.

CREATE TABLE IF NOT EXISTS public.background_jobs (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    job_type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued'
        CHECK (status IN ('queued', 'running', 'done', 'failed')),
    input_args JSONB NOT NULL DEFAULT '{}'::jsonb,
    result JSONB,
    error TEXT,
    session_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_background_jobs_user_status_created
    ON public.background_jobs (user_id, status, created_at DESC);

ALTER TABLE public.background_jobs ENABLE ROW LEVEL SECURITY;

-- Service role gets full access (backend worker uses service key).
CREATE POLICY "background_jobs_service_all" ON public.background_jobs
    FOR ALL TO service_role
    USING (true)
    WITH CHECK (true);

-- Authenticated users can SELECT their own jobs.
CREATE POLICY "background_jobs_select_own" ON public.background_jobs
    FOR SELECT TO authenticated
    USING (auth.uid() = user_id);

GRANT SELECT ON public.background_jobs TO authenticated;
GRANT ALL ON public.background_jobs TO service_role;
