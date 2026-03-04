-- ============================================================================
-- Phase 4a: proactive_queue table
-- ============================================================================
-- Replaces the file-based proactive_queue.json used by the heartbeat worker.
-- Stores proactive notification messages that the agent has composed and
-- queued for delivery to a user.  The heartbeat service writes rows here;
-- the SSE layer reads pending rows and delivers them to the connected client.
--
-- Key design decisions:
--   - status CHECK constraint enforces the full lifecycle: pending → delivered
--     or pending → expired.
--   - Partial UNIQUE index on (user_id, trigger_type) WHERE status = 'pending'
--     prevents the heartbeat from enqueuing a second message of the same type
--     while one is still awaiting delivery.
--   - engagement_tracking JSONB bucket is reserved for future analytics
--     (open-rate, tap-rate, dismissal reason) without requiring schema changes.
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.proactive_queue (
  id                  UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id             UUID        NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  trigger_type        TEXT        NOT NULL,
  priority            REAL        NOT NULL DEFAULT 0.5,
  data                JSONB       NOT NULL DEFAULT '{}'::jsonb,
  message_text        TEXT        NOT NULL,
  status              TEXT        NOT NULL DEFAULT 'pending'
                        CHECK (status IN ('pending', 'delivered', 'expired')),
  created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
  delivered_at        TIMESTAMPTZ,
  engagement_tracking JSONB       NOT NULL DEFAULT '{}'::jsonb
);

COMMENT ON TABLE public.proactive_queue IS
  'Proactive notification messages composed by the heartbeat worker and '
  'queued for SSE delivery to the user. Replaces file-based proactive_queue.json.';

-- ============================================================================
-- INDEXES
-- ============================================================================

-- Primary read path: fetch all pending messages for a user ordered by priority.
CREATE INDEX IF NOT EXISTS idx_proactive_queue_user_status
  ON public.proactive_queue(user_id, status);

-- Secondary read path: retrieve a user's message history newest-first.
CREATE INDEX IF NOT EXISTS idx_proactive_queue_user_created
  ON public.proactive_queue(user_id, created_at DESC);

-- Prevents duplicate pending messages of the same trigger type per user.
-- Only one 'pending' row per (user_id, trigger_type) is allowed at a time;
-- delivered and expired rows are excluded so history is preserved.
CREATE UNIQUE INDEX IF NOT EXISTS idx_proactive_queue_unique_pending
  ON public.proactive_queue(user_id, trigger_type)
  WHERE status = 'pending';

-- ============================================================================
-- ROW LEVEL SECURITY
-- ============================================================================

ALTER TABLE public.proactive_queue ENABLE ROW LEVEL SECURITY;

-- Authenticated users may read their own queued messages (e.g. for an
-- in-app notification centre).
CREATE POLICY "proactive_queue_select_own" ON public.proactive_queue
  FOR SELECT TO authenticated
  USING (auth.uid() = user_id);

-- Service role bypasses RLS so the heartbeat worker can INSERT new messages
-- and UPDATE status for any user without impersonating them.
CREATE POLICY "proactive_queue_service_all" ON public.proactive_queue
  FOR ALL TO service_role
  USING (true)
  WITH CHECK (true);

-- ============================================================================
-- GRANTS
-- ============================================================================

GRANT ALL ON public.proactive_queue TO authenticated, service_role;

-- ============================================================================
-- MIGRATION COMPLETE
-- ============================================================================
-- Summary:
--
--   1 new table:
--     proactive_queue — proactive notification queue (replaces JSON file)
--
--   3 indexes:
--     idx_proactive_queue_user_status   — (user_id, status)
--     idx_proactive_queue_user_created  — (user_id, created_at DESC)
--     idx_proactive_queue_unique_pending — UNIQUE (user_id, trigger_type)
--                                          WHERE status = 'pending'
--
--   RLS policies:
--     proactive_queue_select_own  — authenticated users read own rows
--     proactive_queue_service_all — service_role full bypass
-- ============================================================================
