-- Drop the vestigial `mediapipe_detection_failed` column from attempts.
--
-- The column dated to an earlier landmark pipeline. The current
-- from-scratch keypoint recognizer returns a prediction on every clip,
-- so new attempts always wrote `false` and nothing reads the column.
-- Removing it (and the guard-trigger reference to it) drops dead schema.
--
-- Idempotent. Down-migration is in the comment block at the bottom.

alter table public.attempts
  drop column if exists mediapipe_detection_failed;

-- The self-update guard (20260519110000_learner_disagreed.sql) listed the
-- dropped column in its OLD/NEW equality check; recreate it without that line
-- so the trigger still compiles. Behaviour is otherwise identical: a learner
-- may only flip `learner_disagreed` false -> true, nothing else.
create or replace function public.guard_attempts_self_update()
returns trigger
language plpgsql
as $$
begin
  if new.user_id is distinct from old.user_id
    or new.vocab_id is distinct from old.vocab_id
    or new.prompted_at is distinct from old.prompted_at
    or new.submitted_at is distinct from old.submitted_at
    or new.predicted_class_id is distinct from old.predicted_class_id
    or new.confidence is distinct from old.confidence
    or new.passed is distinct from old.passed
    or new.hint_shown is distinct from old.hint_shown
    or new.hint_source is distinct from old.hint_source
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

-- ---------------------------------------------------------------------------
-- Down-migration (manual):
--   alter table public.attempts
--     add column if not exists mediapipe_detection_failed boolean not null default false;
--   -- then restore the guard function's OLD/NEW check for that column
--   -- (see 20260519110000_learner_disagreed.sql for the original body).
-- ---------------------------------------------------------------------------
