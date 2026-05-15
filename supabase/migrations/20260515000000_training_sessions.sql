SET search_path TO public, extensions;

-- ============================================================================
-- 14-day rolling window: training_sessions
--
-- Replaces the 24-week locked-plan model. Each row is one prescribed or
-- completed training session for the athlete. The coach plans the next
-- 14 days at a time and modulates as life shifts. No enums anywhere:
-- modality is a free string, payload and athlete_note are free-form.
--
-- Naming: the table is called ``training_sessions`` because the existing
-- ``sessions`` table is the agent-brain chat session log (turn tracking,
-- compressed summaries). The two have nothing in common except the noun.
-- Renaming the chat-session table would break the entire agent loop, so
-- this new table takes a more specific name.
--
-- The legacy ``plans`` table is intentionally NOT dropped here. It stays
-- around (unused by the agent) until a future cleanup migration so the
-- transition is reversible during the MVP rebuild.
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.training_sessions (
  id                   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id              UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  date                 DATE NOT NULL,

  -- FREE STRING. Coach picks whatever fits the athlete. No enum.
  -- Examples: 'run', 'bike', 'swim', 'gym', 'hyrox', 'skitour',
  -- 'yoga', 'mobility', 'climbing'. Validation: non-empty.
  modality             TEXT NOT NULL CHECK (length(trim(modality)) > 0),

  -- Lifecycle: planned -> completed | skipped | modulated.
  status               TEXT NOT NULL DEFAULT 'planned'
                         CHECK (status IN ('planned', 'completed', 'skipped', 'modulated')),

  -- FREE-FORM. Coach-defined per modality (intervals, RPE targets,
  -- distance, duration, gear cues, whatever the coach finds useful).
  payload              JSONB NOT NULL DEFAULT '{}'::jsonb,

  -- Coach-authored description of the session, visible to the athlete.
  notes                TEXT NOT NULL DEFAULT '',

  -- Free-form post-session reflection from the athlete.
  athlete_note         TEXT NOT NULL DEFAULT '',

  -- Link to the activity that fulfilled this session (Garmin import, etc).
  actual_activity_id   UUID REFERENCES public.activities(id) ON DELETE SET NULL,

  -- Frozen snapshot of the payload at the time the coach proposed the
  -- session, kept for the audit trail even when ``payload`` is later
  -- modulated.
  planned_payload      JSONB NOT NULL DEFAULT '{}'::jsonb,

  created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at           TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE  public.training_sessions IS
  '14-day rolling window of prescribed and completed training sessions.';
COMMENT ON COLUMN public.training_sessions.modality IS
  'Free-form sport identifier picked by the coach. No enum.';
COMMENT ON COLUMN public.training_sessions.payload IS
  'Free-form coach-defined per-modality details (intervals, targets, etc.).';
COMMENT ON COLUMN public.training_sessions.planned_payload IS
  'Frozen plan-time payload kept for audit when payload is later modulated.';
COMMENT ON COLUMN public.training_sessions.athlete_note IS
  'Free-form athlete reflection after the session is completed or skipped.';


-- ============================================================================
-- Indexes
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_training_sessions_user_date
  ON public.training_sessions(user_id, date);

CREATE INDEX IF NOT EXISTS idx_training_sessions_user_status
  ON public.training_sessions(user_id, status);


-- ============================================================================
-- updated_at trigger
-- ============================================================================

CREATE OR REPLACE FUNCTION public.training_sessions_set_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_training_sessions_updated_at
  ON public.training_sessions;

CREATE TRIGGER trg_training_sessions_updated_at
  BEFORE UPDATE ON public.training_sessions
  FOR EACH ROW
  EXECUTE FUNCTION public.training_sessions_set_updated_at();


-- ============================================================================
-- Row Level Security
-- ============================================================================

ALTER TABLE public.training_sessions ENABLE ROW LEVEL SECURITY;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_policies
    WHERE tablename = 'training_sessions'
      AND policyname = 'training_sessions_select_own'
  ) THEN
    CREATE POLICY "training_sessions_select_own" ON public.training_sessions
      FOR SELECT TO authenticated USING (auth.uid() = user_id);
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM pg_policies
    WHERE tablename = 'training_sessions'
      AND policyname = 'training_sessions_insert_own'
  ) THEN
    CREATE POLICY "training_sessions_insert_own" ON public.training_sessions
      FOR INSERT TO authenticated WITH CHECK (auth.uid() = user_id);
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM pg_policies
    WHERE tablename = 'training_sessions'
      AND policyname = 'training_sessions_update_own'
  ) THEN
    CREATE POLICY "training_sessions_update_own" ON public.training_sessions
      FOR UPDATE TO authenticated USING (auth.uid() = user_id);
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM pg_policies
    WHERE tablename = 'training_sessions'
      AND policyname = 'training_sessions_delete_own'
  ) THEN
    CREATE POLICY "training_sessions_delete_own" ON public.training_sessions
      FOR DELETE TO authenticated USING (auth.uid() = user_id);
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM pg_policies
    WHERE tablename = 'training_sessions'
      AND policyname = 'training_sessions_service_all'
  ) THEN
    CREATE POLICY "training_sessions_service_all" ON public.training_sessions
      FOR ALL TO service_role USING (true) WITH CHECK (true);
  END IF;
END
$$;


GRANT SELECT, INSERT, UPDATE, DELETE ON public.training_sessions TO authenticated;
GRANT ALL                            ON public.training_sessions TO service_role;
