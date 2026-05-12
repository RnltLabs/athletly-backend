-- Simplification: drop multi-store identity layer, replace with single
-- athlete_journal markdown per user. Per Roman's pragmatic-MVP refactor:
-- we want ONE source of truth for who the athlete is and what they want.
--
-- Drops:
--   * beliefs              -- soft facts with confidence (overkill)
--   * goal_history         -- audit log (now lives in journal as a section)
--
-- Adds:
--   * athlete_journal      -- ONE markdown document per user holding all
--                            identity/goals/preferences/timeline/notes
--   * activities.notes     -- per-session annotation (e.g. "knee pain on
--                            this run") so the coach can refer back
--
-- No data migration: development environment, only test users exist.

BEGIN;

-- 1. Drop legacy tables. Cascade so dependent indexes/policies go too.
DROP TABLE IF EXISTS public.beliefs CASCADE;
DROP TABLE IF EXISTS public.goal_history CASCADE;

-- 2. The new single source of truth.
CREATE TABLE public.athlete_journal (
  user_id    UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  content    TEXT NOT NULL DEFAULT '',
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE public.athlete_journal IS
  'Single markdown document per athlete. Holds identity, goals, beliefs, '
  'preferences, goal timeline, open threads. Coach reads it each turn, '
  'writes via update_journal_section / append_to_journal tools.';

ALTER TABLE public.athlete_journal ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Athlete journal readable by owner"
  ON public.athlete_journal
  FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Athlete journal writable by owner"
  ON public.athlete_journal
  FOR ALL USING (auth.uid() = user_id);

CREATE POLICY "Service role full access athlete_journal"
  ON public.athlete_journal
  FOR ALL USING (auth.role() = 'service_role');

-- 3. Per-activity notes so the coach can annotate context (e.g. injury
-- during this specific run) and refer back in later turns.
ALTER TABLE public.activities
  ADD COLUMN IF NOT EXISTS notes TEXT;

COMMENT ON COLUMN public.activities.notes IS
  'Free-form annotation about THIS activity: pain reports, perceived '
  'effort, context notes. Set by annotate_activity tool.';

COMMIT;
