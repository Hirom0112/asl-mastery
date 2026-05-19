-- Initial schema for ASL Mastery.
-- Mirrors docs/ARCHITECTURE.md §2.4 and the TODO.md Phase 3a field list.
-- All user-scoped tables are protected by RLS in 20260519100100_rls.sql.

create extension if not exists "citext";

-- Enums -------------------------------------------------------------

create type handedness as enum ('right', 'left', 'ambidextrous', 'unspecified');

create type sign_movement as enum ('static', 'movement');

create type mastery_status as enum ('untouched', 'learning', 'reviewing', 'mastered');

create type hint_source as enum ('confusion_pair', 'generic_failure', 'none');

-- users -------------------------------------------------------------
-- Mirrors auth.users 1:1. PK is the auth.users.id UUID, FK ON DELETE
-- CASCADE so account deletion in Supabase Auth removes the app row
-- and everything that depends on it (attempts, mastery_state).
-- Populated on auth.users insert via the trigger defined in the RLS
-- migration; never written to directly by the app.

create table public.users (
  id          uuid primary key references auth.users (id) on delete cascade,
  email       citext,
  handedness  handedness  not null default 'unspecified',
  fitzpatrick smallint    check (fitzpatrick is null or fitzpatrick between 1 and 6),
  created_at  timestamptz not null default now(),
  updated_at  timestamptz not null default now()
);

-- vocabulary_items --------------------------------------------------
-- Read-only reference data seeded from docs/VOCABULARY.md.
-- `id` matches the slug we use in code (e.g. 'thank_you').

create table public.vocabulary_items (
  id                  text primary key,
  display_gloss       text not null,
  category            text not null,
  lifeprint_lesson    text,
  asl_lex_code        text,
  asl_lex_match       text,
  wlasl_clip_count    integer,
  static_or_movement  sign_movement not null,
  flippable           boolean not null,
  parameters          jsonb,
  pre_attempt_hint    text,
  generic_failure_hint text,
  reference_video_url text,
  created_at          timestamptz not null default now()
);

-- confusion_pair_hints ----------------------------------------------
-- Authored against the model's top-3 confusion pairs per sign
-- (per docs/ARCHITECTURE.md §5 Layer B). Seeded in Phase 4c.

create table public.confusion_pair_hints (
  id                 uuid primary key default gen_random_uuid(),
  target_sign_id     text not null references public.vocabulary_items (id) on delete cascade,
  predicted_sign_id  text not null references public.vocabulary_items (id) on delete cascade,
  hint               text not null,
  created_at         timestamptz not null default now(),
  unique (target_sign_id, predicted_sign_id)
);

-- model_versions ----------------------------------------------------
-- One active row at a time. Promotion is manual for slice 1
-- (docs/ARCHITECTURE.md §2.5 step 11).

create table public.model_versions (
  id            text primary key,
  artifact_url  text not null,
  config_url    text not null,
  metrics_url   text not null,
  is_active     boolean not null default false,
  promoted_at   timestamptz,
  promoted_by   uuid references public.users (id) on delete set null,
  created_at    timestamptz not null default now()
);

create unique index model_versions_one_active
  on public.model_versions (is_active)
  where is_active = true;

-- attempts ----------------------------------------------------------
-- Append-only per learner. No frames, no keypoints — see
-- docs/ARCHITECTURE.md §2.3 step 11 (privacy commitment).

create table public.attempts (
  id                          uuid primary key default gen_random_uuid(),
  user_id                     uuid not null references public.users (id) on delete cascade,
  vocab_id                    text not null references public.vocabulary_items (id),
  prompted_at                 timestamptz not null,
  submitted_at                timestamptz not null,
  predicted_class_id          text references public.vocabulary_items (id),
  confidence                  double precision,
  passed                      boolean not null,
  hint_shown                  text,
  hint_source                 hint_source not null default 'none',
  learner_disagreed           boolean not null default false,
  mediapipe_detection_failed  boolean not null default false,
  model_version_id            text references public.model_versions (id),
  created_at                  timestamptz not null default now()
);

create index attempts_user_submitted
  on public.attempts (user_id, submitted_at desc);

create index attempts_user_vocab
  on public.attempts (user_id, vocab_id);

-- mastery_state -----------------------------------------------------
-- One row per (user, sign). Updated transactionally with each attempt
-- via the recordAttempt server action (docs/ARCHITECTURE.md §2.4).

create table public.mastery_state (
  user_id            uuid not null references public.users (id) on delete cascade,
  vocab_id           text not null references public.vocabulary_items (id),
  status             mastery_status not null default 'untouched',
  ease               double precision not null default 1.3,
  interval_days      double precision not null default 0,
  next_review_at     timestamptz,
  consecutive_passes integer not null default 0,
  total_attempts     integer not null default 0,
  total_passes       integer not null default 0,
  last_attempt_at    timestamptz,
  updated_at         timestamptz not null default now(),
  primary key (user_id, vocab_id)
);

create index mastery_state_user_next_review
  on public.mastery_state (user_id, next_review_at)
  where next_review_at is not null;

-- updated_at triggers -----------------------------------------------

create or replace function public.set_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

create trigger users_set_updated_at
  before update on public.users
  for each row execute function public.set_updated_at();

create trigger mastery_state_set_updated_at
  before update on public.mastery_state
  for each row execute function public.set_updated_at();
