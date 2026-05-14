-- Feature 1: hybrid Haiku / Sonnet model router.
-- Adds a tier flag to profiles. Free users are pinned to Haiku;
-- Pro users may receive Sonnet 4.6 for "complex" calls.
-- See MIGRATION.md and DESIGN.md in the repo root for context.

ALTER TABLE public.profiles
    ADD COLUMN IF NOT EXISTS tier TEXT NOT NULL DEFAULT 'free'
    CHECK (tier IN ('free', 'pro'));

COMMENT ON COLUMN public.profiles.tier IS
    'Billing tier. "free" = Haiku only. "pro" = Haiku + Sonnet for complex tier.';
