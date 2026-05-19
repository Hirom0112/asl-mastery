-- Row-level security + on-signup trigger.
-- ADR 0007: anonymous and permanent users are treated identically.
-- All policies are pure `auth.uid() = user_id`; there is no branch on
-- `auth.users.is_anonymous`.

-- On-signup trigger -------------------------------------------------
-- Fires for every auth.users insert, including anonymous sign-ins.
-- Defaults handedness to 'unspecified'; the learner sets it later
-- in settings.

create or replace function public.handle_new_auth_user()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
  insert into public.users (id, email)
  values (new.id, new.email)
  on conflict (id) do nothing;
  return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;

create trigger on_auth_user_created
  after insert on auth.users
  for each row execute function public.handle_new_auth_user();

-- Enable RLS --------------------------------------------------------

alter table public.users               enable row level security;
alter table public.attempts            enable row level security;
alter table public.mastery_state       enable row level security;
alter table public.vocabulary_items    enable row level security;
alter table public.confusion_pair_hints enable row level security;
alter table public.model_versions      enable row level security;

-- users: each user reads and updates only their own row -------------

create policy users_select_self
  on public.users for select
  using (auth.uid() = id);

create policy users_update_self
  on public.users for update
  using (auth.uid() = id)
  with check (auth.uid() = id);

-- Inserts into public.users only happen via the trigger above,
-- which runs as `security definer`. No INSERT policy is needed and
-- none is granted — the trigger bypasses RLS by design.

-- attempts: append-only by the owning user --------------------------

create policy attempts_select_self
  on public.attempts for select
  using (auth.uid() = user_id);

create policy attempts_insert_self
  on public.attempts for insert
  with check (auth.uid() = user_id);

-- No update policy: attempts are immutable history. learner_disagreed
-- is the one mutable signal and is set via a server action that runs
-- with elevated privilege (the action verifies ownership in code).

-- mastery_state: read + write by the owning user --------------------

create policy mastery_state_select_self
  on public.mastery_state for select
  using (auth.uid() = user_id);

create policy mastery_state_insert_self
  on public.mastery_state for insert
  with check (auth.uid() = user_id);

create policy mastery_state_update_self
  on public.mastery_state for update
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);

-- Read-only reference data ------------------------------------------
-- All authenticated users (including anonymous) read the catalog.

create policy vocabulary_items_read
  on public.vocabulary_items for select
  to authenticated
  using (true);

create policy confusion_pair_hints_read
  on public.confusion_pair_hints for select
  to authenticated
  using (true);

create policy model_versions_read_active
  on public.model_versions for select
  to authenticated
  using (is_active = true);

-- Grants ------------------------------------------------------------
-- RLS gates the rows, but role grants gate the table verb. We grant
-- the verbs RLS will then further restrict.

grant select, update on public.users         to authenticated;
grant select, insert on public.attempts      to authenticated;
grant select, insert, update on public.mastery_state to authenticated;
grant select on public.vocabulary_items      to authenticated;
grant select on public.confusion_pair_hints  to authenticated;
grant select on public.model_versions        to authenticated;

-- Service role bypasses RLS by default; no explicit grants needed
-- for the training pipeline / admin paths.
