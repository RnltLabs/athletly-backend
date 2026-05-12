-- Push tokens for mobile push notifications via Expo.
--
-- The mobile app registers its Expo push token here on login. The agent
-- (and proactive triggers) use send_notification to push to the registered
-- token. One token per user - re-registration upserts.

CREATE TABLE IF NOT EXISTS public.push_tokens (
  id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id      UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  token        TEXT NOT NULL,
  platform     TEXT,                       -- 'ios' | 'android' | 'web'
  device_name  TEXT,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (user_id)
);

CREATE INDEX IF NOT EXISTS idx_push_tokens_user
  ON public.push_tokens (user_id);

ALTER TABLE public.push_tokens ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Push tokens readable by owner" ON public.push_tokens
  FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Push tokens writable by owner" ON public.push_tokens
  FOR ALL USING (auth.uid() = user_id);

CREATE POLICY "Service role full access push_tokens" ON public.push_tokens
  FOR ALL USING (auth.role() = 'service_role');
