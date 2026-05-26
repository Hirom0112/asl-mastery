# Data model

> Schema reference for the trainer's persistent state. Per
> `docs/VOCABULARY_TRAINER_ROADMAP.md` Phase 0 / Slice 0.2.

## Where the schema lives

The authoritative schema is the SQL in
`supabase/migrations/20260519100000_init.sql` and subsequent
timestamped migrations under `supabase/migrations/`. The
high-level relational shape is also described in prose in
`docs/ARCHITECTURE.md` §2.4. This document is the navigation
map between the two.

## Storage posture (active dev: localhost-first)

Per ADR 0012's "local only, indefinitely" rule and the active
project direction (build on localhost, deploy when ready), the
linked Supabase project (`ehrqwtvrmejozwlybndl`) and the linked
Vercel project remain in place but dormant during active
development of the from-scratch CV pipeline. Local development
uses **Supabase local dev** (`supabase start`) when persistence is
needed, falling back to in-memory state when it is not. The schema
files under `supabase/migrations/` are the single source of truth
regardless of whether they are applied to local Docker Postgres
(active dev), to the cloud project (currently dormant), or both.

The roadmap text (`docs/VOCABULARY_TRAINER_ROADMAP.md` Phase 0
Slice 0.2) calls for "SQLite via better-sqlite3 or IndexedDB." We
honor the intent (no remote write during local dev) by using
Supabase local Docker Postgres, which gives us schema parity with
production and zero schema-translation risk on the eventual
deploy. If a future ADR mandates pure browser-side IndexedDB or
file-backed SQLite, the schema in `supabase/migrations/` translates
directly — there are no Postgres-specific constructs in the
slice-1 surface beyond `gen_random_uuid()` and `citext`, both
replaceable.

## Tables (current slice-1 schema)

The slice-1 schema defines six tables. Citations are to the file
and line range in `supabase/migrations/20260519100000_init.sql`.

### `public.users`

The learner identity, linked 1:1 to Supabase Auth.

Columns of interest: `id` (FK to `auth.users.id`), `email`,
profile fields collected at onboarding (`docs/ARCHITECTURE.md`
§2.1). Row count grows with sign-ups.

### `public.vocabulary_items`

The 75-sign frozen vocabulary, one row per sign. The `id` column
is the `sign_id` from `dataset/slice1b_vocabulary.json`
(`kept_signs[].sign_id`). Other columns: `display_gloss`
(e.g. THANK-YOU), Lifeprint / ASL-LEX cross-references, and the
`flippable` boolean for augmentation safety. Row count: 75 after
the frozen seed (Phase 0); see `docs/VOCABULARY.md` § "Frozen
75-sign trainer vocabulary."

### `public.confusion_pair_hints`

Pre-authored layer-3 hint text per (target_sign, confusable_sign)
pair, seeded from ASL-LEX 2.0 phonological feature analysis. Row
count: 120 at slice-1 seed (`supabase/migrations/...seed_confusion_hints_v2.sql`).
Architecture-agnostic — survives the ADR 0011 pivot intact and is
consumed by Phase 5 Slice 5.3 (confusion-mined hint logic).

### `public.model_versions`

The deployed-model registry. Earlier model rows are
`is_active = false` per the deactivation migration
(`supabase/migrations/20260520200000_deactivate_all_models.sql`).
New rows are added in Phase 4 when the from-scratch sign-matcher
artifact ships; each new row will reference a Modal training run
and the four upstream detector model cards under
`docs/model_cards/`.

### `public.attempts`

One row per learner sign attempt: target sign, predicted sign,
similarity scores per template, pass/fail decision, timestamp,
session id. The substrate for the eval gate
(`docs/EVAL_GATE.md`), per-sign confusion-matrix reporting, and
per-Fitzpatrick fairness analysis.

### `public.mastery_state`

(user_id, vocab_id) pair with a state-machine column tracking
`unseen → learning → practicing → mastered → review` per
`docs/PEDAGOGY.md`. The substrate for the adaptive scheduler
(Phase 5 Slice 5.5) and the mastery dashboard (Slice 5.6).

## Migration policy

- Schema changes ship as **new timestamped migrations**, never as
  edits to historical migration files. State changes ship as
  migrations, not silent column updates.
- Every migration that adds or modifies a table referenced by an
  ADR is paired with an ADR update or a new ADR. Schema is part of
  the architecture record.
- Migrations applied to the cloud project are tracked in
  `supabase/migrations/` and in the Supabase project's migration
  history. The two must agree. Drift is a bug.

## What the schema does not yet model

These surfaces are not in the slice-1 schema and will arrive in
later phases:

- **Per-sign per-keypoint per-timestep template distributions**
  (Phase 4 Slice 4.2 output). Storage: `/data/templates/*.npz` on
  disk; not modeled in the relational schema. The
  `model_versions.artifact_url` column points to the bundle
  containing both the four detector ONNX files and the template
  manifest.
- **Per-sign calibrated thresholds** (Phase 4 Slice 4.3 output,
  brief Requirement 9 artifact). Stored alongside the template
  bundle and referenced by the inference path in `/web`. A
  candidate slice-2 table `sign_thresholds` would denormalize this
  for analytics if needed; not built yet.
- **Labeled-frame manifests** (Phase 1–3 outputs). Stored under
  `/data/labeled_frames/<task>/manifest.json` with `PROVENANCE.md`
  per batch; not modeled in the relational schema because the
  trainer app never reads them — they are training-time artifacts
  only.
