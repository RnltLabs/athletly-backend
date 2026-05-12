-- Fix wrong foreign key on macrocycle_plans.user_id.
--
-- Original migration declared:
--   user_id UUID NOT NULL REFERENCES profiles(id)
-- but profiles.id is the profile row's PK, not the Supabase auth user id.
-- Every other table in this schema (activities, plans, beliefs, ...) keys
-- off auth.users.id, so saves with the correct user UUID fail here.
--
-- This migration drops the bad constraint and adds the correct one.

ALTER TABLE public.macrocycle_plans
  DROP CONSTRAINT IF EXISTS macrocycle_plans_user_id_fkey;

ALTER TABLE public.macrocycle_plans
  ADD CONSTRAINT macrocycle_plans_user_id_fkey
  FOREIGN KEY (user_id)
  REFERENCES auth.users(id)
  ON DELETE CASCADE;
