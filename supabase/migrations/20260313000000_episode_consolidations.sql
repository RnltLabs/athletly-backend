-- Episode consolidations: tracks monthly consolidation runs
-- Prevents duplicate consolidation and stores metadata

create table if not exists episode_consolidations (
    id uuid primary key default gen_random_uuid(),
    user_id uuid not null references auth.users(id) on delete cascade,
    month text not null,  -- "YYYY-MM" format
    weekly_episode_count integer not null default 0,
    recurring_patterns jsonb not null default '[]'::jsonb,
    key_metrics jsonb not null default '{}'::jsonb,
    beliefs_promoted integer not null default 0,
    consolidated_at timestamptz not null default now(),

    -- Prevent duplicate consolidations for same user+month
    unique(user_id, month)
);

-- Index for efficient lookup by user
create index if not exists idx_episode_consolidations_user
    on episode_consolidations(user_id);

-- RLS policies
alter table episode_consolidations enable row level security;

create policy "Users can view own consolidations"
    on episode_consolidations for select
    using (auth.uid() = user_id);

create policy "Service role can manage consolidations"
    on episode_consolidations for all
    using (true)
    with check (true);
