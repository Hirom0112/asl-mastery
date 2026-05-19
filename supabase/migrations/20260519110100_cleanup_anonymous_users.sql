-- cleanup_inactive_anonymous_users() per ADR 0007.
--
-- Anonymous accounts (`auth.users.is_anonymous = true`) that have not
-- recorded an attempt in 30 days are deleted from auth.users. The
-- FK CASCADE chain (public.users → attempts → mastery_state) handles
-- the rest. Permanent accounts are never touched.
--
-- Scheduling: this function is invoked by pg_cron. After this
-- migration applies, enable pg_cron in the Supabase dashboard
-- (Database → Extensions → search for "pg_cron" → enable) and schedule
-- the job with the SQL at the bottom of this file. Doing it via
-- migration would require service-role privileges that the migration
-- runner does not have.

create or replace function public.cleanup_inactive_anonymous_users()
returns int
language plpgsql
security definer
set search_path = public, auth
as $$
declare
  cutoff timestamptz := now() - interval '30 days';
  deleted_count int := 0;
  uid uuid;
begin
  for uid in
    select u.id
    from auth.users u
    where u.is_anonymous = true
      and coalesce(
        (select max(submitted_at) from public.attempts a where a.user_id = u.id),
        u.created_at
      ) < cutoff
  loop
    -- auth.users does not export a stable delete API in pg, but the
    -- supabase project's `auth` schema permits deletion when running
    -- as the postgres / service_role user (security definer above).
    delete from auth.users where id = uid;
    deleted_count := deleted_count + 1;
  end loop;
  return deleted_count;
end;
$$;

-- Permissions: only service_role / supabase_admin invokes this. No
-- grants to authenticated.

-- To schedule (run once from the Supabase SQL editor as the
-- supabase_admin or via the Cron extension UI):
--
--   create extension if not exists pg_cron;
--   select cron.schedule(
--     'cleanup_inactive_anonymous_users_daily',
--     '0 4 * * *',                          -- 04:00 UTC daily
--     $$select public.cleanup_inactive_anonymous_users();$$
--   );
