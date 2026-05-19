-- Track first-time onboarding completion per user.
--
-- A null `onboarded_at` means the user has not yet seen the welcome
-- screen that explains the mastery model, the hint layers, and the
-- exit-on-mastery principle. The /practice route redirects to
-- /welcome when this is null. Once the user finishes the flow we
-- stamp the column with `now()` and they never see it again.

alter table public.users
  add column onboarded_at timestamptz;
