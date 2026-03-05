-- Phase 8.6: Calculated Metrics, Push Notifications, Health Data
-- Also adds missing columns (tags, ended_at) to sessions table.

-- ============================================================================
-- 1. ALTER sessions — add tags and ended_at
-- ============================================================================

ALTER TABLE public.sessions
  ADD COLUMN IF NOT EXISTS tags TEXT[] DEFAULT '{}',
  ADD COLUMN IF NOT EXISTS ended_at TIMESTAMPTZ;

-- ============================================================================
-- 2. calculated_metrics
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.calculated_metrics (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id       UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  activity_id   UUID REFERENCES public.activities(id) ON DELETE SET NULL,
  metric_name   TEXT NOT NULL,
  value         DOUBLE PRECISION NOT NULL,
  unit          TEXT,
  formula_id    UUID REFERENCES public.metric_definitions(id) ON DELETE SET NULL,
  source        TEXT DEFAULT 'agent',
  calculated_at TIMESTAMPTZ DEFAULT now(),
  UNIQUE(activity_id, metric_name, source)
);

COMMENT ON TABLE public.calculated_metrics
  IS 'Agent-computed metrics per activity or standalone (e.g. CTL, ATL, TSB)';

-- Indexes
CREATE INDEX idx_calculated_metrics_user
  ON public.calculated_metrics(user_id);

CREATE INDEX idx_calculated_metrics_activity
  ON public.calculated_metrics(activity_id);

CREATE INDEX idx_calculated_metrics_name
  ON public.calculated_metrics(user_id, metric_name);

-- RLS
ALTER TABLE public.calculated_metrics ENABLE ROW LEVEL SECURITY;

CREATE POLICY "calculated_metrics_select_own" ON public.calculated_metrics
  FOR SELECT TO authenticated USING (auth.uid() = user_id);

CREATE POLICY "calculated_metrics_service_all" ON public.calculated_metrics
  FOR ALL TO service_role USING (true) WITH CHECK (true);

GRANT ALL ON public.calculated_metrics TO authenticated, service_role;

-- ============================================================================
-- 3. push_notifications
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.push_notifications (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id         UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  title           TEXT,
  body            TEXT NOT NULL,
  data            JSONB DEFAULT '{}'::jsonb,
  trigger         TEXT NOT NULL,
  status          TEXT DEFAULT 'sent',
  sent_at         TIMESTAMPTZ DEFAULT now(),
  expo_receipt_id TEXT
);

COMMENT ON TABLE public.push_notifications
  IS 'Expo push notification log with delivery tracking';

-- Indexes
CREATE INDEX idx_push_notifications_user
  ON public.push_notifications(user_id);

CREATE INDEX idx_push_notifications_user_sent
  ON public.push_notifications(user_id, sent_at DESC);

-- RLS
ALTER TABLE public.push_notifications ENABLE ROW LEVEL SECURITY;

CREATE POLICY "push_notifications_select_own" ON public.push_notifications
  FOR SELECT TO authenticated USING (auth.uid() = user_id);

CREATE POLICY "push_notifications_service_all" ON public.push_notifications
  FOR ALL TO service_role USING (true) WITH CHECK (true);

GRANT ALL ON public.push_notifications TO authenticated, service_role;

-- ============================================================================
-- 4. health_data (generic multi-provider table)
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.health_data (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id     UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  provider    TEXT NOT NULL,
  data_type   TEXT NOT NULL,
  value       JSONB NOT NULL,
  recorded_at TIMESTAMPTZ NOT NULL,
  synced_at   TIMESTAMPTZ DEFAULT now(),
  UNIQUE(user_id, provider, data_type, recorded_at)
);

COMMENT ON TABLE public.health_data
  IS 'Multi-provider health data (Garmin, Apple Health, etc.)';

-- Indexes
CREATE INDEX idx_health_data_user
  ON public.health_data(user_id);

CREATE INDEX idx_health_data_provider
  ON public.health_data(user_id, provider, data_type);

CREATE INDEX idx_health_data_recorded
  ON public.health_data(user_id, recorded_at DESC);

-- RLS
ALTER TABLE public.health_data ENABLE ROW LEVEL SECURITY;

CREATE POLICY "health_data_select_own" ON public.health_data
  FOR SELECT TO authenticated USING (auth.uid() = user_id);

CREATE POLICY "health_data_service_all" ON public.health_data
  FOR ALL TO service_role USING (true) WITH CHECK (true);

GRANT ALL ON public.health_data TO authenticated, service_role;

-- ============================================================================
-- MIGRATION COMPLETE
-- ============================================================================
-- Summary:
--   - Added tags TEXT[] and ended_at TIMESTAMPTZ to sessions
--   - Created calculated_metrics with RLS + indexes
--   - Created push_notifications with RLS + indexes
--   - Created health_data with RLS + indexes
