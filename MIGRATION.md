# Migration: Pro/Free Tier Flag on `profiles`

Feature 1 (hybrid Haiku/Sonnet routing) reads a `tier` column from the
`profiles` table to decide whether Sonnet 4.6 is reachable for a given
user. This migration adds the column.

## What changes

- Adds `tier TEXT NOT NULL DEFAULT 'free'` to `public.profiles`.
- Adds a CHECK constraint restricting values to `'free'` or `'pro'`.
- Index is intentionally NOT added: lookups are by `user_id` (already
  a primary key) and the column is read alongside the existing profile
  fetch.

## SQL

The migration file is at
`supabase/migrations/20260514000000_user_tier_column.sql` and contains:

```sql
ALTER TABLE public.profiles
ADD COLUMN IF NOT EXISTS tier TEXT NOT NULL DEFAULT 'free'
CHECK (tier IN ('free', 'pro'));
```

## Apply

```bash
# Local Supabase
supabase db push

# Production
supabase db push --linked
```

## Rollback

```sql
ALTER TABLE public.profiles DROP COLUMN IF EXISTS tier;
```

Rolling back is safe: the model router defaults to "free" on any
lookup error, so all callers downgrade to Haiku gracefully.

## Code paths affected

- `src/services/user_tier.py` reads `profiles.tier` (60s TTL cache).
- `src/agent/model_router.py` consumes the tier value.
- `src/agent/llm.py` calls the router from `chat_completion`.

## Manual tier flip (admin)

```sql
UPDATE public.profiles SET tier = 'pro' WHERE user_id = '<uuid>';
```

After flipping, the in-process cache may take up to 60s to pick up the
change. To force-evict immediately, restart the API process or expose
`src.services.user_tier.invalidate(user_id)` from your admin handler.

## Backwards compatibility

- Existing rows: default to `'free'`. No Sonnet access until manually
  promoted.
- Callers that omit `user_id` on `chat_completion`: treated as the
  `default_user_tier` env value (`free` by default, override via
  `ATHLETLY_DEFAULT_TIER=pro` for local dogfooding).
- Callers that pass `model=` explicitly: continue to bypass the router
  entirely (compression, fallback chain, episode consolidation).
