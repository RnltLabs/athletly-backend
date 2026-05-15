-- Typed coach-maintained state: intents, programs, coach_notes.
--
-- Background: the athlete_journal is one Markdown document with sections.
-- That works for static identity (Name, Constraints, Goals, Preferences)
-- but breaks down for high-mutation coach state:
--
--   * Active Intents      - 1 to 5 commitments (event + capacity + habit).
--   * Active Programs     - persisting structures (gym split, run block).
--   * Coach Notes         - append-only learnings the coach writes over time.
--
-- The existing journal regex parser drifts on heading variants ("Coach Notes"
-- vs "Coach notes:" creates parallel sections silently). Concurrent
-- read-modify-write on the single markdown doc loses data. So we move
-- these three categories into typed tables with atomic CRUD, and render
-- them back to markdown only for system-prompt injection.
--
-- All three are append-or-update; no destructive deletes through tools.
-- coach_notes is strictly append-only: a new observation = a new row.
--
-- Idempotent (IF NOT EXISTS) so this migration is safe to rerun in dev.

BEGIN;

SET search_path TO public, extensions;


-- ============================================================================
-- 1. intents - active commitments (1 to 5 per athlete, frequently edited)
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.intents (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    type        TEXT NOT NULL,
    title       TEXT NOT NULL,
    priority    INTEGER NOT NULL DEFAULT 3,
    payload     JSONB NOT NULL DEFAULT '{}'::jsonb,
    status      TEXT NOT NULL DEFAULT 'active',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE  public.intents IS
    'Active athlete commitments. Coach maintains; types are free text but '
    'the prompt suggests event / capacity / habit / recovery.';
COMMENT ON COLUMN public.intents.type IS
    'Free text, hinted: event | capacity | habit | recovery.';
COMMENT ON COLUMN public.intents.payload IS
    'Type-specific data, e.g. {race_date, target_time} or {target_lift}.';
COMMENT ON COLUMN public.intents.status IS
    'active | paused | achieved | abandoned. Free text, not enforced.';

CREATE INDEX IF NOT EXISTS idx_intents_user_status
    ON public.intents(user_id, status);


-- ============================================================================
-- 2. programs - persisting training structures (gym split, run block)
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.programs (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    kind        TEXT NOT NULL,
    label       TEXT,
    payload     JSONB NOT NULL DEFAULT '{}'::jsonb,
    active      BOOLEAN NOT NULL DEFAULT TRUE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE  public.programs IS
    'Persisting program structures (gym split, run block, swim block). '
    'Coach defines, athlete reviews. Set active=false to retire; no delete.';
COMMENT ON COLUMN public.programs.kind IS
    'Free text, e.g. gym_split | run_block | swim_block.';

CREATE INDEX IF NOT EXISTS idx_programs_user_active
    ON public.programs(user_id, active);


-- ============================================================================
-- 3. coach_notes - append-only learnings
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.coach_notes (
    id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id            UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    content            TEXT NOT NULL,
    scope              TEXT,
    source_session_id  UUID REFERENCES public.training_sessions(id) ON DELETE SET NULL,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE  public.coach_notes IS
    'Append-only coach learnings about the athlete. No update operation. '
    'When something changes, the coach writes a new note.';
COMMENT ON COLUMN public.coach_notes.scope IS
    'Optional grouping, e.g. training_response | life_pattern | preference_learned.';
COMMENT ON COLUMN public.coach_notes.source_session_id IS
    'Optional reference to the session that prompted the note.';

CREATE INDEX IF NOT EXISTS idx_coach_notes_user_created
    ON public.coach_notes(user_id, created_at DESC);


-- ============================================================================
-- Row Level Security
-- ============================================================================

ALTER TABLE public.intents      ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.programs     ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.coach_notes  ENABLE ROW LEVEL SECURITY;

DO $$
BEGIN
    -- intents policies
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies
        WHERE tablename = 'intents' AND policyname = 'intents_select_own'
    ) THEN
        CREATE POLICY "intents_select_own" ON public.intents
            FOR SELECT TO authenticated USING (auth.uid() = user_id);
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_policies
        WHERE tablename = 'intents' AND policyname = 'intents_modify_own'
    ) THEN
        CREATE POLICY "intents_modify_own" ON public.intents
            FOR ALL TO authenticated USING (auth.uid() = user_id);
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_policies
        WHERE tablename = 'intents' AND policyname = 'intents_service_all'
    ) THEN
        CREATE POLICY "intents_service_all" ON public.intents
            FOR ALL TO service_role USING (true) WITH CHECK (true);
    END IF;

    -- programs policies
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies
        WHERE tablename = 'programs' AND policyname = 'programs_select_own'
    ) THEN
        CREATE POLICY "programs_select_own" ON public.programs
            FOR SELECT TO authenticated USING (auth.uid() = user_id);
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_policies
        WHERE tablename = 'programs' AND policyname = 'programs_modify_own'
    ) THEN
        CREATE POLICY "programs_modify_own" ON public.programs
            FOR ALL TO authenticated USING (auth.uid() = user_id);
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_policies
        WHERE tablename = 'programs' AND policyname = 'programs_service_all'
    ) THEN
        CREATE POLICY "programs_service_all" ON public.programs
            FOR ALL TO service_role USING (true) WITH CHECK (true);
    END IF;

    -- coach_notes policies (insert-only for the user; service role does
    -- bulk reads on behalf of the agent).
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies
        WHERE tablename = 'coach_notes' AND policyname = 'coach_notes_select_own'
    ) THEN
        CREATE POLICY "coach_notes_select_own" ON public.coach_notes
            FOR SELECT TO authenticated USING (auth.uid() = user_id);
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_policies
        WHERE tablename = 'coach_notes' AND policyname = 'coach_notes_insert_own'
    ) THEN
        CREATE POLICY "coach_notes_insert_own" ON public.coach_notes
            FOR INSERT TO authenticated WITH CHECK (auth.uid() = user_id);
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_policies
        WHERE tablename = 'coach_notes' AND policyname = 'coach_notes_service_all'
    ) THEN
        CREATE POLICY "coach_notes_service_all" ON public.coach_notes
            FOR ALL TO service_role USING (true) WITH CHECK (true);
    END IF;
END
$$;


-- ============================================================================
-- Grants
-- ============================================================================

GRANT SELECT, INSERT, UPDATE         ON public.intents     TO authenticated;
GRANT SELECT, INSERT, UPDATE         ON public.programs    TO authenticated;
GRANT SELECT, INSERT                 ON public.coach_notes TO authenticated;
GRANT ALL                            ON public.intents     TO service_role;
GRANT ALL                            ON public.programs    TO service_role;
GRANT ALL                            ON public.coach_notes TO service_role;

COMMIT;
