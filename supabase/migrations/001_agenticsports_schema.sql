-- ============================================================================
-- AgenticSports Schema Migration
-- ============================================================================
-- Creates all tables required by the Python backend (AgenticSports) that do
-- NOT already exist in the athletly-app Supabase instance.
--
-- Pre-existing tables (from athletly-app migrations) that we DO NOT recreate:
--   users, garmin_activities, garmin_daily_stats, goals, training_plan,
--   activity_feedback, chat_messages, user_summaries, cycle_tracking,
--   user_consents, intent_routes, weekly_plans, pending_actions, llm_usage,
--   proactive_messages, user_sports, muscle_load_log, push_tokens,
--   health_providers, health_activities, health_daily_metrics,
--   user_metric_baselines, sport_research_cache, agent_status, rate_limits,
--   user_schedule_blocks
--
-- Tables created by this migration:
--   profiles            — agent brain's structured user model
--   beliefs             — belief system with pgvector embeddings
--   activities          — simplified activity store for the agent brain
--   import_manifest     — FIT file deduplication
--   sessions            — chat session lifecycle
--   session_messages    — per-session message history
--   episodes            — episodic memory (training reflections)
--   plans               — versioned training plans with evaluation
--   metric_definitions  — agent-defined computed metrics
--   eval_criteria       — agent-defined scoring criteria
--   session_schemas     — sport-specific session structure templates
--   periodization_models — multi-phase training model definitions
--   proactive_trigger_rules — conditions that wake the agent proactively
--
-- Also:
--   ALTER weekly_plans  — add coach_message and reasoning columns
--   match_beliefs()     — pgvector cosine similarity search function
-- ============================================================================

-- Ensure pgvector is available (idempotent).
CREATE EXTENSION IF NOT EXISTS vector;


-- ============================================================================
-- 1. PROFILES
-- ============================================================================
-- The agent brain's structured user model, separate from the athletly-app
-- "users" table.  One row per user.  Contains goal, fitness, constraints
-- data that the Python UserModelDB reads/writes.
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.profiles (
  user_id    UUID PRIMARY KEY REFERENCES public.users(id) ON DELETE CASCADE,
  name       TEXT,
  sports     JSONB DEFAULT '[]'::jsonb,

  -- Goal
  goal_event       TEXT,
  goal_target_date TEXT,
  goal_target_time TEXT,
  goal_type        TEXT,

  -- Fitness
  estimated_vo2max         FLOAT,
  threshold_pace_min_km    FLOAT,
  weekly_volume_km         FLOAT,
  fitness_trend            TEXT DEFAULT 'unknown',

  -- Constraints
  training_days_per_week   INT,
  max_session_minutes      INT,
  available_sports         JSONB DEFAULT '[]'::jsonb,

  -- Flags
  onboarding_complete BOOLEAN DEFAULT false,

  -- Flexible metadata bucket
  meta JSONB DEFAULT '{}'::jsonb,

  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE public.profiles IS 'Agent brain structured user model — one row per user';

-- updated_at trigger
CREATE TRIGGER update_profiles_updated_at
  BEFORE UPDATE ON public.profiles
  FOR EACH ROW EXECUTE FUNCTION public.update_updated_at();

-- RLS
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;

CREATE POLICY "profiles_select_own" ON public.profiles
  FOR SELECT TO authenticated USING (auth.uid() = user_id);

CREATE POLICY "profiles_insert_own" ON public.profiles
  FOR INSERT TO authenticated WITH CHECK (auth.uid() = user_id);

CREATE POLICY "profiles_update_own" ON public.profiles
  FOR UPDATE TO authenticated USING (auth.uid() = user_id);

-- Service role bypass for backend operations
CREATE POLICY "profiles_service_all" ON public.profiles
  FOR ALL TO service_role USING (true) WITH CHECK (true);


-- ============================================================================
-- 2. BELIEFS
-- ============================================================================
-- Belief system with pgvector embeddings for semantic similarity search.
-- Category values must match BELIEF_CATEGORIES in user_model_db.py.
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.beliefs (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id         UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,

  text            TEXT NOT NULL,
  category        TEXT NOT NULL CHECK (category IN (
                    'preference', 'constraint', 'history', 'motivation',
                    'physical', 'fitness', 'scheduling', 'personality', 'meta'
                  )),
  confidence      FLOAT NOT NULL DEFAULT 0.7 CHECK (confidence >= 0.0 AND confidence <= 1.0),
  stability       TEXT NOT NULL DEFAULT 'stable',
  durability      TEXT NOT NULL DEFAULT 'global',
  source          TEXT NOT NULL DEFAULT 'conversation',
  source_ref      TEXT,

  -- Temporal validity
  first_observed  TIMESTAMPTZ NOT NULL DEFAULT now(),
  last_confirmed  TIMESTAMPTZ NOT NULL DEFAULT now(),
  valid_from      DATE DEFAULT CURRENT_DATE,
  valid_until     DATE,

  -- Lifecycle
  active          BOOLEAN NOT NULL DEFAULT true,
  archived_at     TIMESTAMPTZ,
  superseded_by   UUID REFERENCES public.beliefs(id),

  -- pgvector embedding (768-dim for Gemini text-embedding-004)
  embedding       vector(768),

  -- Outcome tracking (P6: active memory)
  utility         FLOAT NOT NULL DEFAULT 0.0,
  outcome_count   INT NOT NULL DEFAULT 0,
  last_outcome    TEXT,
  outcome_history JSONB DEFAULT '[]'::jsonb,

  created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE  public.beliefs IS 'User beliefs with pgvector embeddings for semantic search';
COMMENT ON COLUMN public.beliefs.embedding IS '768-dimensional vector from Gemini text-embedding-004';

-- Indexes
CREATE INDEX idx_beliefs_user_active
  ON public.beliefs(user_id, active)
  WHERE active = true;

CREATE INDEX idx_beliefs_user_category
  ON public.beliefs(user_id, category)
  WHERE active = true;

CREATE INDEX idx_beliefs_embedding
  ON public.beliefs
  USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 20);

-- RLS
ALTER TABLE public.beliefs ENABLE ROW LEVEL SECURITY;

CREATE POLICY "beliefs_select_own" ON public.beliefs
  FOR SELECT TO authenticated USING (auth.uid() = user_id);

CREATE POLICY "beliefs_insert_own" ON public.beliefs
  FOR INSERT TO authenticated WITH CHECK (auth.uid() = user_id);

CREATE POLICY "beliefs_update_own" ON public.beliefs
  FOR UPDATE TO authenticated USING (auth.uid() = user_id);

CREATE POLICY "beliefs_service_all" ON public.beliefs
  FOR ALL TO service_role USING (true) WITH CHECK (true);


-- ============================================================================
-- 2a. match_beliefs() — pgvector similarity search RPC
-- ============================================================================

CREATE OR REPLACE FUNCTION public.match_beliefs(
  p_user_id       UUID,
  p_embedding      vector(768),
  p_match_count    INT DEFAULT 5,
  p_min_confidence FLOAT DEFAULT 0.0
)
RETURNS TABLE (
  id         UUID,
  text       TEXT,
  category   TEXT,
  confidence FLOAT,
  similarity FLOAT
)
LANGUAGE plpgsql STABLE SECURITY DEFINER
AS $$
BEGIN
  RETURN QUERY
  SELECT
    b.id,
    b.text,
    b.category,
    b.confidence,
    1 - (b.embedding <=> p_embedding) AS similarity
  FROM public.beliefs b
  WHERE b.user_id = p_user_id
    AND b.active = true
    AND b.embedding IS NOT NULL
    AND b.confidence >= p_min_confidence
  ORDER BY b.embedding <=> p_embedding
  LIMIT p_match_count;
END;
$$;

COMMENT ON FUNCTION public.match_beliefs IS 'Cosine similarity search over active beliefs using pgvector';


-- ============================================================================
-- 3. ACTIVITIES (agent brain simplified view)
-- ============================================================================
-- Separate from health_activities — this is the agent brain's own activity
-- store with a simpler schema.  The webhook router and activity_store_db.py
-- write directly to this table.
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.activities (
  id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id           UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,

  sport             TEXT NOT NULL DEFAULT 'running',
  start_time        TIMESTAMPTZ,
  duration_seconds  INT,
  distance_meters   FLOAT,
  avg_hr            INT,
  max_hr            INT,
  avg_pace_min_km   FLOAT,
  elevation_gain_m  FLOAT,
  trimp             FLOAT,

  zone_distribution JSONB DEFAULT '{}'::jsonb,
  laps              JSONB DEFAULT '[]'::jsonb,
  raw_data          JSONB DEFAULT '{}'::jsonb,

  source            TEXT DEFAULT 'manual',
  garmin_activity_id TEXT,

  -- Webhook-inserted activities may use these columns
  activity_type     TEXT,
  data              JSONB,

  created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE public.activities IS 'Agent brain simplified activity store';

-- Indexes
CREATE INDEX idx_activities_user_start
  ON public.activities(user_id, start_time DESC);

CREATE INDEX idx_activities_user_sport
  ON public.activities(user_id, sport);

CREATE INDEX idx_activities_garmin_id
  ON public.activities(garmin_activity_id)
  WHERE garmin_activity_id IS NOT NULL;

-- RLS
ALTER TABLE public.activities ENABLE ROW LEVEL SECURITY;

CREATE POLICY "activities_select_own" ON public.activities
  FOR SELECT TO authenticated USING (auth.uid() = user_id);

CREATE POLICY "activities_insert_own" ON public.activities
  FOR INSERT TO authenticated WITH CHECK (auth.uid() = user_id);

CREATE POLICY "activities_update_own" ON public.activities
  FOR UPDATE TO authenticated USING (auth.uid() = user_id);

CREATE POLICY "activities_delete_own" ON public.activities
  FOR DELETE TO authenticated USING (auth.uid() = user_id);

CREATE POLICY "activities_service_all" ON public.activities
  FOR ALL TO service_role USING (true) WITH CHECK (true);


-- ============================================================================
-- 4. IMPORT MANIFEST (FIT file deduplication)
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.import_manifest (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id     UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  file_hash   TEXT NOT NULL,
  file_name   TEXT NOT NULL,
  activity_id UUID NOT NULL,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),

  UNIQUE(user_id, file_hash)
);

COMMENT ON TABLE public.import_manifest IS 'FIT file deduplication — tracks imported file hashes';

-- Index
CREATE INDEX idx_import_manifest_user_hash
  ON public.import_manifest(user_id, file_hash);

-- RLS
ALTER TABLE public.import_manifest ENABLE ROW LEVEL SECURITY;

CREATE POLICY "import_manifest_select_own" ON public.import_manifest
  FOR SELECT TO authenticated USING (auth.uid() = user_id);

CREATE POLICY "import_manifest_insert_own" ON public.import_manifest
  FOR INSERT TO authenticated WITH CHECK (auth.uid() = user_id);

CREATE POLICY "import_manifest_service_all" ON public.import_manifest
  FOR ALL TO service_role USING (true) WITH CHECK (true);


-- ============================================================================
-- 5. SESSIONS (chat session lifecycle)
-- ============================================================================
-- The Python backend uses a dedicated sessions table with turn tracking and
-- compressed summaries, separate from the athletly-app chat_messages.session_id
-- column approach.
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.sessions (
  id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id             UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  started_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
  last_active         TIMESTAMPTZ NOT NULL DEFAULT now(),
  compressed_summary  TEXT,
  turn_count          INT NOT NULL DEFAULT 0,
  tool_calls_total    INT NOT NULL DEFAULT 0,
  created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE public.sessions IS 'Agent brain chat sessions with turn tracking';

-- Indexes
CREATE INDEX idx_sessions_user_started
  ON public.sessions(user_id, started_at DESC);

CREATE INDEX idx_sessions_user_last_active
  ON public.sessions(user_id, last_active DESC);

CREATE INDEX idx_sessions_last_active
  ON public.sessions(last_active DESC);

-- RLS
ALTER TABLE public.sessions ENABLE ROW LEVEL SECURITY;

CREATE POLICY "sessions_select_own" ON public.sessions
  FOR SELECT TO authenticated USING (auth.uid() = user_id);

CREATE POLICY "sessions_insert_own" ON public.sessions
  FOR INSERT TO authenticated WITH CHECK (auth.uid() = user_id);

CREATE POLICY "sessions_update_own" ON public.sessions
  FOR UPDATE TO authenticated USING (auth.uid() = user_id);

CREATE POLICY "sessions_service_all" ON public.sessions
  FOR ALL TO service_role USING (true) WITH CHECK (true);


-- ============================================================================
-- 6. SESSION MESSAGES
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.session_messages (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id  UUID NOT NULL REFERENCES public.sessions(id) ON DELETE CASCADE,
  user_id     UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  role        TEXT NOT NULL CHECK (role IN ('user', 'model', 'tool_call', 'system')),
  content     TEXT NOT NULL,
  meta        JSONB DEFAULT '{}'::jsonb,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE public.session_messages IS 'Individual messages within an agent brain session';

-- Indexes
CREATE INDEX idx_session_messages_session
  ON public.session_messages(session_id, id);

CREATE INDEX idx_session_messages_user
  ON public.session_messages(user_id, created_at DESC);

-- RLS
ALTER TABLE public.session_messages ENABLE ROW LEVEL SECURITY;

CREATE POLICY "session_messages_select_own" ON public.session_messages
  FOR SELECT TO authenticated USING (auth.uid() = user_id);

CREATE POLICY "session_messages_insert_own" ON public.session_messages
  FOR INSERT TO authenticated WITH CHECK (auth.uid() = user_id);

CREATE POLICY "session_messages_service_all" ON public.session_messages
  FOR ALL TO service_role USING (true) WITH CHECK (true);


-- ============================================================================
-- 7. EPISODES (episodic memory — training reflections)
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.episodes (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id       UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  episode_type  TEXT NOT NULL DEFAULT 'weekly_reflection',
  period_start  DATE,
  period_end    DATE,
  summary       TEXT NOT NULL DEFAULT '',
  insights      JSONB DEFAULT '[]'::jsonb,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE public.episodes IS 'Episodic memory — training block reflections and reviews';

-- Indexes
CREATE INDEX idx_episodes_user_period
  ON public.episodes(user_id, period_end DESC);

CREATE INDEX idx_episodes_user_type
  ON public.episodes(user_id, episode_type, period_end DESC);

-- RLS
ALTER TABLE public.episodes ENABLE ROW LEVEL SECURITY;

CREATE POLICY "episodes_select_own" ON public.episodes
  FOR SELECT TO authenticated USING (auth.uid() = user_id);

CREATE POLICY "episodes_insert_own" ON public.episodes
  FOR INSERT TO authenticated WITH CHECK (auth.uid() = user_id);

CREATE POLICY "episodes_service_all" ON public.episodes
  FOR ALL TO service_role USING (true) WITH CHECK (true);


-- ============================================================================
-- 8. PLANS (versioned training plans with evaluation)
-- ============================================================================
-- Separate from weekly_plans — this table stores full plan payloads with
-- version tracking and evaluation scores.
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.plans (
  id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id               UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  plan_data             JSONB NOT NULL DEFAULT '{}'::jsonb,
  evaluation_score      INT CHECK (evaluation_score >= 0 AND evaluation_score <= 100),
  evaluation_feedback   TEXT,
  active                BOOLEAN NOT NULL DEFAULT true,
  created_at            TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE public.plans IS 'Versioned training plans with evaluation scores';

-- Indexes
CREATE INDEX idx_plans_user_active
  ON public.plans(user_id, active)
  WHERE active = true;

CREATE INDEX idx_plans_user_created
  ON public.plans(user_id, created_at DESC);

-- RLS
ALTER TABLE public.plans ENABLE ROW LEVEL SECURITY;

CREATE POLICY "plans_select_own" ON public.plans
  FOR SELECT TO authenticated USING (auth.uid() = user_id);

CREATE POLICY "plans_insert_own" ON public.plans
  FOR INSERT TO authenticated WITH CHECK (auth.uid() = user_id);

CREATE POLICY "plans_update_own" ON public.plans
  FOR UPDATE TO authenticated USING (auth.uid() = user_id);

CREATE POLICY "plans_service_all" ON public.plans
  FOR ALL TO service_role USING (true) WITH CHECK (true);


-- ============================================================================
-- 9. METRIC DEFINITIONS (agent-defined computed metrics)
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.metric_definitions (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id     UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  name        TEXT NOT NULL,
  formula     TEXT NOT NULL,
  description TEXT NOT NULL DEFAULT '',
  unit        TEXT NOT NULL DEFAULT '',
  variables   JSONB DEFAULT '{}'::jsonb,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),

  UNIQUE(user_id, name)
);

COMMENT ON TABLE public.metric_definitions IS 'Agent-defined formula-based computed metrics';

-- updated_at trigger
CREATE TRIGGER update_metric_definitions_updated_at
  BEFORE UPDATE ON public.metric_definitions
  FOR EACH ROW EXECUTE FUNCTION public.update_updated_at();

-- Indexes
CREATE INDEX idx_metric_definitions_user
  ON public.metric_definitions(user_id);

-- RLS
ALTER TABLE public.metric_definitions ENABLE ROW LEVEL SECURITY;

CREATE POLICY "metric_definitions_select_own" ON public.metric_definitions
  FOR SELECT TO authenticated USING (auth.uid() = user_id);

CREATE POLICY "metric_definitions_insert_own" ON public.metric_definitions
  FOR INSERT TO authenticated WITH CHECK (auth.uid() = user_id);

CREATE POLICY "metric_definitions_update_own" ON public.metric_definitions
  FOR UPDATE TO authenticated USING (auth.uid() = user_id);

CREATE POLICY "metric_definitions_service_all" ON public.metric_definitions
  FOR ALL TO service_role USING (true) WITH CHECK (true);


-- ============================================================================
-- 10. EVAL CRITERIA (agent-defined scoring criteria)
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.eval_criteria (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id     UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  name        TEXT NOT NULL,
  description TEXT NOT NULL DEFAULT '',
  weight      FLOAT NOT NULL DEFAULT 1.0,
  formula     TEXT NOT NULL DEFAULT '',
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),

  UNIQUE(user_id, name)
);

COMMENT ON TABLE public.eval_criteria IS 'Agent-defined scoring criteria with weights';

-- updated_at trigger
CREATE TRIGGER update_eval_criteria_updated_at
  BEFORE UPDATE ON public.eval_criteria
  FOR EACH ROW EXECUTE FUNCTION public.update_updated_at();

-- Indexes
CREATE INDEX idx_eval_criteria_user
  ON public.eval_criteria(user_id);

-- RLS
ALTER TABLE public.eval_criteria ENABLE ROW LEVEL SECURITY;

CREATE POLICY "eval_criteria_select_own" ON public.eval_criteria
  FOR SELECT TO authenticated USING (auth.uid() = user_id);

CREATE POLICY "eval_criteria_insert_own" ON public.eval_criteria
  FOR INSERT TO authenticated WITH CHECK (auth.uid() = user_id);

CREATE POLICY "eval_criteria_update_own" ON public.eval_criteria
  FOR UPDATE TO authenticated USING (auth.uid() = user_id);

CREATE POLICY "eval_criteria_service_all" ON public.eval_criteria
  FOR ALL TO service_role USING (true) WITH CHECK (true);


-- ============================================================================
-- 11. SESSION SCHEMAS (sport-specific session structure templates)
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.session_schemas (
  id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id    UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  name       TEXT NOT NULL,
  sport      TEXT NOT NULL,
  schema     JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),

  UNIQUE(user_id, sport)
);

COMMENT ON TABLE public.session_schemas IS 'Sport-specific session structure templates defined by the agent';

-- updated_at trigger
CREATE TRIGGER update_session_schemas_updated_at
  BEFORE UPDATE ON public.session_schemas
  FOR EACH ROW EXECUTE FUNCTION public.update_updated_at();

-- Indexes
CREATE INDEX idx_session_schemas_user
  ON public.session_schemas(user_id);

-- RLS
ALTER TABLE public.session_schemas ENABLE ROW LEVEL SECURITY;

CREATE POLICY "session_schemas_select_own" ON public.session_schemas
  FOR SELECT TO authenticated USING (auth.uid() = user_id);

CREATE POLICY "session_schemas_insert_own" ON public.session_schemas
  FOR INSERT TO authenticated WITH CHECK (auth.uid() = user_id);

CREATE POLICY "session_schemas_update_own" ON public.session_schemas
  FOR UPDATE TO authenticated USING (auth.uid() = user_id);

CREATE POLICY "session_schemas_service_all" ON public.session_schemas
  FOR ALL TO service_role USING (true) WITH CHECK (true);


-- ============================================================================
-- 12. PERIODIZATION MODELS (multi-phase training model definitions)
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.periodization_models (
  id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id    UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  name       TEXT NOT NULL,
  phases     JSONB NOT NULL DEFAULT '[]'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),

  UNIQUE(user_id, name)
);

COMMENT ON TABLE public.periodization_models IS 'Multi-phase training model definitions created by the agent';

-- updated_at trigger
CREATE TRIGGER update_periodization_models_updated_at
  BEFORE UPDATE ON public.periodization_models
  FOR EACH ROW EXECUTE FUNCTION public.update_updated_at();

-- Indexes
CREATE INDEX idx_periodization_models_user
  ON public.periodization_models(user_id);

-- RLS
ALTER TABLE public.periodization_models ENABLE ROW LEVEL SECURITY;

CREATE POLICY "periodization_models_select_own" ON public.periodization_models
  FOR SELECT TO authenticated USING (auth.uid() = user_id);

CREATE POLICY "periodization_models_insert_own" ON public.periodization_models
  FOR INSERT TO authenticated WITH CHECK (auth.uid() = user_id);

CREATE POLICY "periodization_models_update_own" ON public.periodization_models
  FOR UPDATE TO authenticated USING (auth.uid() = user_id);

CREATE POLICY "periodization_models_service_all" ON public.periodization_models
  FOR ALL TO service_role USING (true) WITH CHECK (true);


-- ============================================================================
-- 13. PROACTIVE TRIGGER RULES
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.proactive_trigger_rules (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id         UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  name            TEXT NOT NULL,
  condition       TEXT NOT NULL,
  action          TEXT NOT NULL,
  cooldown_hours  INT NOT NULL DEFAULT 24,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

  UNIQUE(user_id, name)
);

COMMENT ON TABLE public.proactive_trigger_rules IS 'Conditions that wake the agent proactively';

-- updated_at trigger
CREATE TRIGGER update_proactive_trigger_rules_updated_at
  BEFORE UPDATE ON public.proactive_trigger_rules
  FOR EACH ROW EXECUTE FUNCTION public.update_updated_at();

-- Indexes
CREATE INDEX idx_proactive_trigger_rules_user
  ON public.proactive_trigger_rules(user_id);

-- RLS
ALTER TABLE public.proactive_trigger_rules ENABLE ROW LEVEL SECURITY;

CREATE POLICY "proactive_trigger_rules_select_own" ON public.proactive_trigger_rules
  FOR SELECT TO authenticated USING (auth.uid() = user_id);

CREATE POLICY "proactive_trigger_rules_insert_own" ON public.proactive_trigger_rules
  FOR INSERT TO authenticated WITH CHECK (auth.uid() = user_id);

CREATE POLICY "proactive_trigger_rules_update_own" ON public.proactive_trigger_rules
  FOR UPDATE TO authenticated USING (auth.uid() = user_id);

CREATE POLICY "proactive_trigger_rules_service_all" ON public.proactive_trigger_rules
  FOR ALL TO service_role USING (true) WITH CHECK (true);


-- ============================================================================
-- 14. ALTER WEEKLY PLANS — add agent brain columns
-- ============================================================================
-- The existing weekly_plans table needs coach_message and reasoning columns
-- for the Python backend to store agent-generated plan explanations.
-- ============================================================================

ALTER TABLE public.weekly_plans
  ADD COLUMN IF NOT EXISTS coach_message TEXT,
  ADD COLUMN IF NOT EXISTS reasoning     TEXT;

COMMENT ON COLUMN public.weekly_plans.coach_message IS 'Agent-generated message explaining the plan to the user';
COMMENT ON COLUMN public.weekly_plans.reasoning     IS 'Agent internal reasoning for plan decisions';


-- ============================================================================
-- 15. GRANTS — ensure service_role has full access to new tables
-- ============================================================================

GRANT ALL ON public.profiles              TO authenticated, service_role;
GRANT ALL ON public.beliefs               TO authenticated, service_role;
GRANT ALL ON public.activities            TO authenticated, service_role;
GRANT ALL ON public.import_manifest       TO authenticated, service_role;
GRANT ALL ON public.sessions              TO authenticated, service_role;
GRANT ALL ON public.session_messages      TO authenticated, service_role;
GRANT ALL ON public.episodes              TO authenticated, service_role;
GRANT ALL ON public.plans                 TO authenticated, service_role;
GRANT ALL ON public.metric_definitions    TO authenticated, service_role;
GRANT ALL ON public.eval_criteria         TO authenticated, service_role;
GRANT ALL ON public.session_schemas       TO authenticated, service_role;
GRANT ALL ON public.periodization_models  TO authenticated, service_role;
GRANT ALL ON public.proactive_trigger_rules TO authenticated, service_role;


-- ============================================================================
-- MIGRATION COMPLETE
-- ============================================================================
-- Summary of what was created:
--
--   13 new tables:
--     profiles, beliefs, activities, import_manifest, sessions,
--     session_messages, episodes, plans, metric_definitions, eval_criteria,
--     session_schemas, periodization_models, proactive_trigger_rules
--
--   1 RPC function:
--     match_beliefs(p_user_id, p_embedding, p_match_count, p_min_confidence)
--
--   2 new columns on weekly_plans:
--     coach_message, reasoning
--
--   All tables have:
--     - UUID primary keys with gen_random_uuid() defaults
--     - TIMESTAMPTZ timestamps with now() defaults
--     - RLS enabled with auth.uid() = user_id policies
--     - service_role bypass policies for backend operations
--     - Indexes on common query patterns
--     - updated_at triggers (where applicable)
-- ============================================================================
