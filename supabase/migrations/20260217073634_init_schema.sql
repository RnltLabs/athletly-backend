-- AgenticSports Phase 1: Backend Foundation
-- Supabase schema with pgvector, RLS, and full multi-user support

-- ============================================================================
-- 1. Extensions
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA extensions;

-- ============================================================================
-- 2. Tables
-- ============================================================================

-- Athlete Profiles (replaces data/user_model/model.json → structured_core)
CREATE TABLE public.profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL UNIQUE,
    name TEXT,
    sports TEXT[] DEFAULT '{}',
    goal_event TEXT,
    goal_target_date DATE,
    goal_target_time TEXT,
    goal_type TEXT,
    estimated_vo2max REAL,
    threshold_pace_min_km REAL,
    weekly_volume_km REAL,
    fitness_trend TEXT DEFAULT 'unknown',
    training_days_per_week INT,
    max_session_minutes INT,
    available_sports TEXT[] DEFAULT '{}',
    onboarding_complete BOOLEAN DEFAULT false,
    meta JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Activities (replaces data/activities/*.json)
CREATE TABLE public.activities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
    sport TEXT NOT NULL,
    start_time TIMESTAMPTZ NOT NULL,
    duration_seconds INT,
    distance_meters REAL,
    avg_hr INT,
    max_hr INT,
    avg_pace_min_km REAL,
    elevation_gain_m REAL,
    trimp REAL,
    zone_distribution JSONB DEFAULT '{}',
    laps JSONB DEFAULT '[]',
    raw_data JSONB DEFAULT '{}',
    source TEXT DEFAULT 'manual',
    garmin_activity_id TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_activities_user_time ON public.activities(user_id, start_time DESC);
CREATE UNIQUE INDEX idx_activities_garmin_dedup ON public.activities(user_id, garmin_activity_id) WHERE garmin_activity_id IS NOT NULL;

-- Beliefs (replaces data/user_model/model.json → beliefs[])
CREATE TABLE public.beliefs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
    text TEXT NOT NULL,
    category TEXT NOT NULL CHECK (category IN (
        'preference', 'constraint', 'history', 'motivation',
        'physical', 'fitness', 'scheduling', 'personality', 'meta'
    )),
    confidence REAL DEFAULT 0.7 CHECK (confidence >= 0.0 AND confidence <= 1.0),
    stability TEXT DEFAULT 'stable',
    durability TEXT DEFAULT 'global',
    source TEXT DEFAULT 'conversation',
    source_ref TEXT,
    utility REAL DEFAULT 0.0,
    outcome_count INT DEFAULT 0,
    last_outcome TEXT,
    outcome_history JSONB DEFAULT '[]',
    embedding extensions.vector(768),
    valid_from DATE DEFAULT CURRENT_DATE,
    valid_until DATE,
    first_observed TIMESTAMPTZ DEFAULT now(),
    last_confirmed TIMESTAMPTZ DEFAULT now(),
    archived_at TIMESTAMPTZ,
    superseded_by UUID REFERENCES public.beliefs(id),
    active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_beliefs_user_active ON public.beliefs(user_id) WHERE active = true;
CREATE INDEX idx_beliefs_category ON public.beliefs(user_id, category) WHERE active = true;

-- Sessions (replaces data/sessions/*.jsonl)
CREATE TABLE public.sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
    started_at TIMESTAMPTZ DEFAULT now(),
    last_active TIMESTAMPTZ DEFAULT now(),
    compressed_summary TEXT,
    turn_count INT DEFAULT 0,
    tool_calls_total INT DEFAULT 0
);

CREATE INDEX idx_sessions_user ON public.sessions(user_id, started_at DESC);

-- Session Messages (individual turns within a session)
CREATE TABLE public.session_messages (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    session_id UUID REFERENCES public.sessions(id) ON DELETE CASCADE NOT NULL,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'model', 'tool_call', 'system')),
    content TEXT NOT NULL,
    meta JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_session_messages_session ON public.session_messages(session_id, id);

-- Training Plans (replaces data/plans/*.json)
CREATE TABLE public.plans (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
    plan_data JSONB NOT NULL,
    evaluation_score INT,
    evaluation_feedback TEXT,
    active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_plans_user_active ON public.plans(user_id) WHERE active = true;

-- Episodes (replaces data/episodes/*.json)
CREATE TABLE public.episodes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
    episode_type TEXT DEFAULT 'weekly_reflection',
    period_start DATE,
    period_end DATE,
    summary TEXT NOT NULL,
    insights JSONB DEFAULT '[]',
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_episodes_user ON public.episodes(user_id, period_end DESC);

-- Daily Usage Tracking (Rate Limiting)
CREATE TABLE public.daily_usage (
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    usage_date DATE DEFAULT CURRENT_DATE,
    request_count INT DEFAULT 0,
    token_count INT DEFAULT 0,
    PRIMARY KEY (user_id, usage_date)
);

-- Import Manifest (FIT file deduplication)
CREATE TABLE public.import_manifest (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
    file_hash TEXT NOT NULL,
    file_name TEXT,
    imported_at TIMESTAMPTZ DEFAULT now(),
    activity_id UUID REFERENCES public.activities(id),
    UNIQUE (user_id, file_hash)
);

-- ============================================================================
-- 3. Row Level Security
-- ============================================================================

ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.activities ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.beliefs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.session_messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.plans ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.episodes ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.daily_usage ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.import_manifest ENABLE ROW LEVEL SECURITY;

-- Profiles: users can CRUD their own row
CREATE POLICY profiles_own ON public.profiles FOR ALL
    USING (user_id = auth.uid())
    WITH CHECK (user_id = auth.uid());

-- Activities: users see only their own
CREATE POLICY activities_own ON public.activities FOR ALL
    USING (user_id = auth.uid())
    WITH CHECK (user_id = auth.uid());

-- Beliefs: users see only their own
CREATE POLICY beliefs_own ON public.beliefs FOR ALL
    USING (user_id = auth.uid())
    WITH CHECK (user_id = auth.uid());

-- Sessions: users see only their own
CREATE POLICY sessions_own ON public.sessions FOR ALL
    USING (user_id = auth.uid())
    WITH CHECK (user_id = auth.uid());

-- Session Messages: users see only their own
CREATE POLICY session_messages_own ON public.session_messages FOR ALL
    USING (user_id = auth.uid())
    WITH CHECK (user_id = auth.uid());

-- Plans: users see only their own
CREATE POLICY plans_own ON public.plans FOR ALL
    USING (user_id = auth.uid())
    WITH CHECK (user_id = auth.uid());

-- Episodes: users see only their own
CREATE POLICY episodes_own ON public.episodes FOR ALL
    USING (user_id = auth.uid())
    WITH CHECK (user_id = auth.uid());

-- Daily Usage: users see only their own
CREATE POLICY daily_usage_own ON public.daily_usage FOR ALL
    USING (user_id = auth.uid())
    WITH CHECK (user_id = auth.uid());

-- Import Manifest: users see only their own
CREATE POLICY import_manifest_own ON public.import_manifest FOR ALL
    USING (user_id = auth.uid())
    WITH CHECK (user_id = auth.uid());

-- ============================================================================
-- 4. Helper Functions
-- ============================================================================

-- Auto-update updated_at on profiles
CREATE OR REPLACE FUNCTION public.handle_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER on_profiles_update
    BEFORE UPDATE ON public.profiles
    FOR EACH ROW EXECUTE FUNCTION public.handle_updated_at();

-- Auto-create profile on user signup
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO public.profiles (user_id)
    VALUES (NEW.id);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- Increment daily usage counter (called from Edge Functions / backend)
CREATE OR REPLACE FUNCTION public.increment_usage(
    p_user_id UUID,
    p_tokens INT DEFAULT 0
)
RETURNS void AS $$
BEGIN
    INSERT INTO public.daily_usage (user_id, usage_date, request_count, token_count)
    VALUES (p_user_id, CURRENT_DATE, 1, p_tokens)
    ON CONFLICT (user_id, usage_date)
    DO UPDATE SET
        request_count = daily_usage.request_count + 1,
        token_count = daily_usage.token_count + p_tokens;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Cosine similarity search for beliefs (uses pgvector)
CREATE OR REPLACE FUNCTION public.match_beliefs(
    p_user_id UUID,
    p_embedding extensions.vector(768),
    p_match_count INT DEFAULT 5,
    p_min_confidence REAL DEFAULT 0.0
)
RETURNS TABLE (
    id UUID,
    text TEXT,
    category TEXT,
    confidence REAL,
    similarity REAL
) AS $$
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
      AND b.confidence >= p_min_confidence
      AND b.embedding IS NOT NULL
    ORDER BY b.embedding <=> p_embedding
    LIMIT p_match_count;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;
