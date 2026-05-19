# ADR 0007: Auth providers (email magic link + Google OAuth) and demo access via anonymous sign-in

**Status:** Accepted
**Date:** 2026-05-19

---

## Context

`docs/ARCHITECTURE.md` §2.1 originally specified Supabase Auth via
**email magic links only**. Two needs surfaced during Phase 3a planning:

1. A second auth provider so the hiring partner and the demo audience
   can sign in with a Google account instead of waiting for a magic
   link email. Magic links are friction at the worst possible moment —
   the first thirty seconds of a stranger's session.
2. A "Try the demo" entry point on the landing page so a reviewer can
   experience the full practice loop (mastery state transitions,
   spaced retrieval, hint layers) without committing email or OAuth
   identity. The demo must show the actual database-backed mastery
   architecture, not a local-only stub, because that architecture is
   the substance of what we are demonstrating to Patrick and Frank.

Three demo shapes were considered:

- **(a) Anonymous sign-in.** Supabase Auth supports
  `signInAnonymously()`, which creates a real `auth.users` row with
  `is_anonymous = true`. The row has a UUID and is reachable through
  the same RLS policies that govern authenticated users.
- **(b) Shared demo account.** One fixed account everyone shares.
  Rejected because mastery state collides across concurrent demo
  visitors — the architecture's per-user state machine becomes
  unobservable.
- **(c) Local-only demo, no auth.** Progress stored only in
  IndexedDB. Rejected because the database-backed mastery state and
  RLS policies are precisely what we want a reviewer to see; hiding
  them defeats the purpose of the demo.

---

## Decision

1. **Two production auth providers from day one:**
   - **Email magic link** (already in scope per the superseded reading
     of `docs/ARCHITECTURE.md` §2.1).
   - **Google OAuth** via Supabase Auth's Google provider.
   Both providers create rows in `auth.users` with `is_anonymous = false`
   and are treated identically by the public `users` table trigger and
   by every RLS policy.

2. **Anonymous sign-in is the demo path.** The landing page exposes
   a "Try the demo" button that calls `signInAnonymously()`. The
   resulting `auth.users` row has `is_anonymous = true` but is in
   every other respect a real user account: it gets a row in
   `public.users` via the same trigger, its `attempts` and
   `mastery_state` rows are RLS-protected by `auth.uid()` exactly as
   authenticated users' rows are.

3. **Conversion path.** An anonymous user can later convert to a
   permanent account by adding email (magic link) or linking Google.
   Supabase Auth's `linkIdentity` / `updateUser` flow preserves the
   user id, so all `attempts` and `mastery_state` rows persist
   across the conversion without rewrites.

4. **Anonymous account lifecycle.** Anonymous rows that have been
   inactive for **30 days** are deleted by a Supabase scheduled
   function (`cleanup_inactive_anonymous_users()`). The deletion
   cascades through the `users → attempts → mastery_state` FK chain
   and respects the same account-delete semantics as the
   user-initiated path in `docs/PRIVACY.md` §5. The 30-day window is
   long enough that a returning demo visitor sees their previous
   progress, short enough that we are not indefinitely retaining
   data attached to an unverified identity.

---

## Rationale

- **Magic links are demo-hostile in the first thirty seconds.** A
  reviewer landing on the production URL will not switch tabs to
  their inbox. Either Google OAuth (one click) or the anonymous
  demo path (zero friction) needs to be the primary call to action.
- **Anonymous sign-in preserves the architectural demo.** A reviewer
  pressing "Try the demo" exercises the same scheduler, the same
  RLS policies, the same mastery state machine, the same hint
  layering as a permanent account. The architecture is the
  substance of the demo; hiding it behind a local-only stub would
  be self-defeating.
- **No new RLS surface area.** Anonymous users authenticate the same
  way (`auth.uid()` returns their UUID), so RLS policies do not need
  conditional branches for anonymous vs authenticated. The only new
  policy surface is the cleanup function, which is a
  service-role-only scheduled job.
- **Conversion preserves work.** A reviewer who decides mid-demo that
  they want to keep their progress can add an email or link Google
  without losing their attempt history. The user id never changes.
- **30-day retention matches the credibility contract.**
  `docs/PRIVACY.md` §5 commits to deleting attempt data on account
  deletion. Anonymous accounts are accounts; they get the same
  policy. The 30-day idle cutoff is the operational implementation
  of that policy for accounts that the user has not explicitly
  retained.

---

## What stays the same

- Every commitment in `docs/PRIVACY.md`. Local inference, no frames
  leaving the device, no third-party analytics receiving canvas
  data. Anonymous users are subject to the same data-handling rules
  as authenticated users.
- The schema in `docs/ARCHITECTURE.md` §2.4. The `users` table FK to
  `auth.users.id` does not branch on `is_anonymous`. Every downstream
  table (`attempts`, `mastery_state`) treats anonymous and permanent
  users identically.
- The eval gate (`docs/EVAL_GATE.md`). Per-demographic fairness
  reporting is still gated on consented Fitzpatrick disclosure;
  anonymous users do not provide consent to that field, so they are
  excluded from demographic accuracy denominators — same as any
  authenticated user who declines.
- The pedagogical theory (`docs/PEDAGOGY.md`). Mastery, spaced
  retrieval, the three-layer hint system, mastery-based exit —
  unchanged.

---

## Consequences

### Schema (`docs/ARCHITECTURE.md` §2.4)

- `public.users` gains no new columns. `is_anonymous` is *not*
  duplicated into `public.users` — it lives on `auth.users` and is
  consulted only by the cleanup function. Treating anonymous and
  permanent users identically in the application schema is the
  whole point of the decision.
- The on-signup trigger inserts a stub row in `public.users` for
  *every* `auth.users` insert, including anonymous ones. Default
  `handedness = 'unspecified'`. The learner can set handedness in
  settings whether anonymous or not.

### Auth UI (`docs/ARCHITECTURE.md` §2.1 update)

- The sign-in screen offers three buttons: Google, email magic link,
  Try the demo.
- The landing page's primary call to action is "Try the demo"
  (anonymous sign-in) with secondary "Sign in" linking to the auth
  screen.
- The settings page exposes "Add email" and "Link Google" for
  anonymous users, plus the existing handedness / consent controls.

### Privacy (`docs/PRIVACY.md` §5 extension)

- Anonymous accounts inherit the existing account-delete semantics:
  user-initiated delete cascades through `users → attempts →
  mastery_state` and removes contributor clips if the user had
  recorded any (which anonymous users cannot — recording is
  admin-gated).
- A scheduled `cleanup_inactive_anonymous_users()` function runs
  daily, deleting `auth.users` rows where `is_anonymous = true` and
  the most recent `attempts.submitted_at` is more than 30 days old
  (or where no attempts exist and `auth.users.created_at` is more
  than 30 days old).

### Provider configuration

- Google OAuth client id and secret live in Supabase project
  settings (server-side, not in `.env.local`). The client-side flow
  uses the Supabase JS client's `signInWithOAuth({ provider:
  'google' })` and does not need additional client secrets.

---

## Rejected alternatives

- **Shared demo account.** Rejected per the context section: mastery
  state collisions hide the substance of the demo.
- **Local-only demo with IndexedDB persistence.** Rejected: hides the
  database-backed mastery architecture, which is the substance of
  what the demo is meant to show.
- **Google OAuth only, no magic link.** Rejected: forcing Google
  identity on every learner is a privacy ask we should not make. A
  reviewer or learner who prefers not to link a Google account
  should have email magic link as an alternative.
- **No anonymous cleanup; keep anonymous rows forever.** Rejected:
  retains data tied to unverified identities indefinitely. The
  30-day cutoff matches the credibility contract in
  `docs/PRIVACY.md`.

---

## How this is verified

- Migration `supabase/migrations/0002_rls.sql` enables RLS on
  `users`, `attempts`, `mastery_state` with policies of the form
  `auth.uid() = user_id`. No conditional branch on `is_anonymous`.
- The on-signup trigger is defined once and fires on every
  `auth.users` insert; no anonymous-specific branch.
- The scheduled cleanup function is committed in a migration; its
  SQL is auditable.
- The auth UI in the Next.js app exposes all three sign-in
  affordances (Google, magic link, demo) on the dedicated `/sign-in`
  route, with the demo affordance also surfaced on `/` as the primary
  call to action.
