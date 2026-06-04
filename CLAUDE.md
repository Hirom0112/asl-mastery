# CLAUDE.md — agent rulebook for asl-mastery

Operating rules for any agent (or human) committing to this repository. Read
this before you push. The README is for users; this file is for the people and
agents who change the code.

## Git model — enforced trunk

This repo is **trunk-based**. There is one long-lived branch, `main`, and we
push directly to it — no feature branches, no pull requests, no review ceremony.

- **Direct push to `main`.** Land work as small, green, conventional commits.
- **Push to both remotes, every time.** `main` lives on two mirrors and they
  must stay in lockstep:
  - `origin` → `https://github.com/Hirom0112/asl-mastery.git`
  - `gitlab` → `https://labs.gauntletai.com/hiromalarcon/asl-mastery.git`

  Every push targets both:

  ```bash
  git push origin main
  git push gitlab main
  ```

- **The gate is the reviewer.** With no PRs, code review is mechanical. A
  versioned `pre-push` hook (`.husky/pre-push`) runs the same gating checks CI
  runs and refuses to let a red tree leave your machine. A clean push prints
  `pre-push: gate green ✓`; a failure prints `pre-push BLOCKED: <check>` and
  aborts the push. That hook — not a human reviewer — is what keeps `main`
  green, so treat a block exactly as you would a reviewer's "changes requested".

### How the gate is armed

Husky owns Git's hook path: `git config core.hooksPath` is set to `.husky/_`.
You do **not** set this by hand and you do **not** point it at `.githooks`.

The hook is **re-armed by `pnpm install`**. The `prepare` script in
`package.json` runs `husky`, which (re)creates `.husky/_` and keeps
`core.hooksPath` pointed at it. So on a **fresh clone the single step that arms
the gate is**:

```bash
pnpm install
```

`.husky/pre-push` is a tracked, executable file, so once installed it travels
with the repo and applies to everyone.

### What the gate runs

The hook mirrors the gating steps of `.github/workflows/ci.yml`, in CI order:

1. `pnpm format:check`
2. `pnpm lint`
3. `pnpm typecheck`
4. `pnpm test`
5. `pnpm build`

`build` is included because it completes in well under ~90s locally
(~22s as measured). The second CI workflow, `.github/workflows/eval-gate.yml`,
is **intentionally not mirrored**: it is API-key gated and runs only on
validation-report PRs, so it cannot and should not run on every push.

## Invariants

Never pass --no-verify to git. A blocked pre-push is a red test: fix until green, never weaken the hook. Pre-existing red found by the gate is a finding — file it, don't bypass it.

Concretely, that means:

- **Never** reach for `git push --no-verify` (or `git commit --no-verify`) to
  get around the gate. Bypassing the reviewer is not a fix.
- **Never** delete, comment out, or loosen a check in `.husky/pre-push` to make
  a push go through. The hook getting stricter is the system working.
- When the gate surfaces **pre-existing** breakage that is not yours to fix in
  this change, scope it out and record it in `TODO.md` with a one-line reason —
  then push the work that *is* green. Filing the finding is the job; silencing
  the gate is not.
