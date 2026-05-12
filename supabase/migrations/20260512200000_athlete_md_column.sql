-- Add free-form athlete memory column to profiles.
--
-- Holds a small markdown document with notes the user adds about themselves
-- (e.g. "I have a sensitive Achilles - no plyometrics"). The auto-generated
-- AthleteProfile.md context that we inject into every coach turn pulls from
-- structured columns + beliefs + this freeform note section.

ALTER TABLE public.profiles
  ADD COLUMN IF NOT EXISTS athlete_md TEXT;

COMMENT ON COLUMN public.profiles.athlete_md IS
  'Free-form athlete memory notes (markdown). Set by edit_athlete_memory tool.';
