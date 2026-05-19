-- RLS-narrow self-update policy for attempts.learner_disagreed.
--
-- The original 20260519100100_rls.sql migration deliberately gave
-- attempts only SELECT + INSERT for authenticated users — attempts
-- are append-only history. The one exception is the
-- `learner_disagreed` flag, which the learner sets to true when they
-- believe the system was wrong about their sign (the "I think I did
-- this right" feedback button from docs/ARCHITECTURE.md §7).
--
-- We add an UPDATE policy that:
--   1. Only applies to the owning user.
--   2. Only allows flipping `learner_disagreed` from false to true.
--   3. Refuses any other column edit.
--
-- That last constraint is enforced via a row-level trigger because
-- Postgres RLS UPDATE policies cannot natively restrict which columns
-- a row update touches.

create policy attempts_update_self_disagree
  on public.attempts for update
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);

grant update (learner_disagreed) on public.attempts to authenticated;

create or replace function public.guard_attempts_self_update()
returns trigger
language plpgsql
as $$
begin
  -- Only `learner_disagreed` may transition, and only from false → true.
  -- Every other column must equal its OLD value.
  if new.user_id is distinct from old.user_id
    or new.vocab_id is distinct from old.vocab_id
    or new.prompted_at is distinct from old.prompted_at
    or new.submitted_at is distinct from old.submitted_at
    or new.predicted_class_id is distinct from old.predicted_class_id
    or new.confidence is distinct from old.confidence
    or new.passed is distinct from old.passed
    or new.hint_shown is distinct from old.hint_shown
    or new.hint_source is distinct from old.hint_source
    or new.mediapipe_detection_failed is distinct from old.mediapipe_detection_failed
    or new.model_version_id is distinct from old.model_version_id
    or new.created_at is distinct from old.created_at
  then
    raise exception 'attempts: only learner_disagreed may be updated by the learner';
  end if;
  if old.learner_disagreed = true and new.learner_disagreed = false then
    raise exception 'attempts.learner_disagreed cannot be unset';
  end if;
  return new;
end;
$$;

create trigger attempts_guard_self_update
  before update on public.attempts
  for each row execute function public.guard_attempts_self_update();
