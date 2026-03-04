-- Phase 4b: Pending Actions for Adaptive Replanning
-- Stores proposed plan changes that need user confirmation before execution.

CREATE TABLE IF NOT EXISTS pending_actions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    session_id TEXT,
    action_type TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    preview JSONB NOT NULL DEFAULT '{}'::jsonb,
    checkpoint_type TEXT NOT NULL DEFAULT 'HARD' CHECK (checkpoint_type IN ('HARD', 'SOFT')),
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'confirmed', 'rejected', 'expired')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    resolved_at TIMESTAMPTZ
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_pending_actions_user_status
    ON pending_actions (user_id, status);
CREATE INDEX IF NOT EXISTS idx_pending_actions_created
    ON pending_actions (created_at);

-- Only one pending action per (user_id, action_type) at a time
CREATE UNIQUE INDEX IF NOT EXISTS idx_pending_actions_unique_pending
    ON pending_actions (user_id, action_type) WHERE status = 'pending';

-- RLS
ALTER TABLE pending_actions ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can read own pending_actions"
    ON pending_actions FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own pending_actions"
    ON pending_actions FOR INSERT
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own pending_actions"
    ON pending_actions FOR UPDATE
    USING (auth.uid() = user_id);

-- Service role bypass
CREATE POLICY "Service role full access pending_actions"
    ON pending_actions FOR ALL
    USING (auth.role() = 'service_role');
