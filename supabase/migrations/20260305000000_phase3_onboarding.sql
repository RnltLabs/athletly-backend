-- Phase 3: Add context column to sessions table for onboarding vs coach mode.

ALTER TABLE public.sessions
  ADD COLUMN IF NOT EXISTS context TEXT NOT NULL DEFAULT 'coach'
  CHECK (context IN ('coach', 'onboarding'));

CREATE INDEX IF NOT EXISTS idx_sessions_context
  ON public.sessions(user_id, context);
