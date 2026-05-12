-- Migration: Add Strava activity ID column for provider-agnostic activity sync.
--
-- Garmin activities are deduplicated by (user_id, garmin_activity_id).
-- Strava activities need their own dedup key. We add strava_activity_id
-- + a partial unique index so providers stay independent.

ALTER TABLE public.activities
  ADD COLUMN IF NOT EXISTS strava_activity_id TEXT;

-- Partial unique index: only enforced when strava_activity_id IS NOT NULL.
-- Garmin rows (with strava_activity_id = NULL) are not affected.
CREATE UNIQUE INDEX IF NOT EXISTS activities_user_strava_uniq
  ON public.activities (user_id, strava_activity_id)
  WHERE strava_activity_id IS NOT NULL;
