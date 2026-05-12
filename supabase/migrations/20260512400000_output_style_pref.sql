-- Free-form preferences blob for non-structured user preferences.
-- output_style: "concise" | "detailed" | "coach" (default).

ALTER TABLE public.profiles
  ADD COLUMN IF NOT EXISTS preferences JSONB DEFAULT '{}'::jsonb;
