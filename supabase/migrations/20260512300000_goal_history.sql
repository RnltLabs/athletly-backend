-- Goal change audit log.
--
-- Every time the athlete updates their target event / date / time we append
-- a row here with the old goal, the new goal, and the reasoning the agent
-- captured from the conversation. Useful for:
-- - Showing the athlete how their goals have evolved.
-- - Letting the agent reason about adherence vs. ambition history.
-- - Self-improvement: did adjusting the goal correlate with adherence?

CREATE TABLE IF NOT EXISTS public.goal_history (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id       UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  old_goal      JSONB,
  new_goal      JSONB NOT NULL,
  reasoning     TEXT,
  changed_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_goal_history_user_time
  ON public.goal_history (user_id, changed_at DESC);

ALTER TABLE public.goal_history ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Goal history readable by owner" ON public.goal_history
  FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Service role full access goal_history" ON public.goal_history
  FOR ALL USING (auth.role() = 'service_role');
