# Claude Code Handoff Instructions

> This file tells you how to bring Claude Code into the project once
> you have these documents in your repo. Read once, then follow.

The planning is done. Execution starts here. The documents in this
directory carry every decision Claude Code needs; the goal of the
handoff is to load that context cleanly and start building.

---

## 1. One-time setup

Put these documents at the root of a fresh git repository:

```
asl-mastery/
├── claude/
│   ├── CLAUDE.md
│   ├── SESSION_LOG.md
│   └── CLAUDE_CODE_HANDOFF.md  (this file)
├── docs/
│   ├── ROADMAP.md
│   ├── ARCHITECTURE.md
│   ├── PEDAGOGY.md
│   ├── MODEL.md
│   ├── DATASET.md
│   ├── EVAL_GATE.md
│   ├── PRIVACY.md
│   ├── decisions/
│   │   ├── 0001-recognition-architecture.md
│   │   ├── 0002-no-placement-test.md
│   │   └── 0003-deployment-platform.md
│   └── research/
│       ├── README.md
│       └── _template.md
└── .gitignore
```

In `.gitignore`, decide whether `claude/` is committed or ignored. The
arguments either way:

- **Commit it:** future you, your future collaborators, and Claude
  Code itself can all read the persistent context. This is what
  most teams do.
- **Ignore it:** if you want the planning to stay private and only the
  outward-facing `docs/` are visible, ignore the `claude/` directory.

I recommend committing it. The `claude/CLAUDE.md` file is exactly the
kind of artifact a senior reviewer would respect; it shows you carry
context deliberately.

Make your first commit:

```bash
git init
git add .
git commit -m "Initial scoping: pedagogy, architecture, roadmap, decisions"
```

---

## 2. Opening Claude Code in the terminal

Install Claude Code if you have not already:

```bash
npm install -g @anthropic-ai/claude-code
```

In the project root:

```bash
cd asl-mastery
claude
```

Claude Code will start a session with the working directory as the
project root.

---

## 3. The first message to Claude Code

Paste this verbatim as your first message:

> Before doing anything else, read these files in order:
>
> 1. `claude/CLAUDE.md` — the persistent context, values, decisions, and rules for this project.
> 2. `claude/SESSION_LOG.md` — the running log of what has been decided in prior sessions.
> 3. `docs/ROADMAP.md` — the phased plan for the project.
> 4. `docs/ARCHITECTURE.md` — the system architecture.
>
> After you have read those, tell me what you understand the project to be in three sentences, what phase we are starting from, and what the first concrete task is. Do not begin work until I confirm.

This gives Claude Code the full context without you re-explaining,
and the "do not begin work until I confirm" line keeps you in control
of pacing.

---

## 4. Working sessions

Every session has the same shape:

**Open:**

1. Open Claude Code in the project root.
2. Tell it to read `claude/CLAUDE.md` and `claude/SESSION_LOG.md`.
3. State what you want to work on this session.

**Middle:**

4. Direct the work. Claude Code writes code; you review and steer.
5. When a non-trivial decision arises, ask Claude Code to write an
   ADR in `docs/decisions/`.
6. When a research claim is needed for the README or pedagogy doc,
   ask Claude Code to verify the citation against the primary source
   before writing it.

**Close:**

7. Ask Claude Code to append a new "Session N" entry to
   `claude/SESSION_LOG.md` covering: decisions locked, files
   created or changed, open threads, where to start next time.
8. Commit. Push if you want.

This keeps every session reconstructable from the docs alone.

---

## 5. The recommended starting sequence

Once Claude Code has read the context, the first few sessions look
like this:

### Session N+1: Phase 0 sign-off list

Ask Claude Code to draft the candidate 75–100 sign vocabulary list,
broken into categories (greetings, pronouns, family, common verbs,
question words, colors, common nouns, numbers, days/months), with
each sign annotated for: estimated frequency in ASL 1 curricula, the
five sign parameters, whether it is static or movement-based,
flippable yes/no, and what public datasets contain it.

This list becomes the document the ASL instructor reviews and signs
off on.

### Session N+2: Repo scaffolding

Phase 2 of the roadmap. Initialize the Next.js project with:

```bash
pnpm create next-app@latest --typescript --tailwind --app
```

Configure ESLint, Prettier, Husky, GitHub Actions. Set up the Supabase
project (via dashboard, then connect with env vars). Set up the
Cloudflare R2 bucket. Deploy a "hello" page to Vercel.

Phase 2 exit criterion: `pnpm dev` works locally, the same code is
live on a Vercel preview URL, env vars are wired.

### Session N+3: Postgres schema and migrations

Following `ARCHITECTURE.md` Section 2.4, write the migrations for
`users`, `vocabulary_items`, `confusion_pair_hints`, `model_versions`,
`attempts`, `mastery_state`. Use Supabase migrations or Drizzle ORM
with proper migration tooling.

### Session N+4: The recording tool

Phase 3 of the roadmap. The admin-gated `/admin/record` route. This
needs to ship before any data collection can start.

### Session N+5+: Data collection, model training

These will be multi-week threads. The training pipeline is a separate
Python codebase that lives in the same repo under `training/` and is
not deployed (it runs on rented GPU).

---

## 6. Rules to keep Claude Code anchored

The most common failure mode of agent coding sessions is values drift
across long conversations. Mitigate with these habits:

- **Reference the docs by name when you push back.** If Claude Code
  proposes a streak counter, say "see `claude/CLAUDE.md` Section 3
  rule 2 — we do not build engagement mechanics." This is faster
  than re-arguing and Claude Code respects pointers to its own context.
- **Re-read `claude/CLAUDE.md` between long sessions.** A 4-hour
  Claude Code session can drift. Ask it to re-read the context every
  hour or so.
- **When in doubt, write it down.** New decision? ADR. New constraint?
  Update `CLAUDE.md` Section 4. New open question? `CLAUDE.md` Section
  5. The docs are not overhead; they are how you stay in control.

---

## 7. When to come back to a planning session with full Claude

Claude Code is excellent at execution. It is less good at the kind of
deep, exploratory planning we did when scoping this project. If you
hit a hard fork in the road that calls for that kind of thinking —
"do we use a multi-head model for hints in slice 2 or stay
confusion-pair-only?" — drop back into a planning conversation in
Claude or with another planning tool, then commit the decision back
to the project's `docs/decisions/` and continue execution.

The split, roughly:

- **Planning conversations:** strategy, scoping, citation verification,
  architecture trade-offs, pedagogical theory, hiring presentation prep.
- **Claude Code in terminal:** writing code, editing files, running
  builds, debugging, scaffolding, executing the roadmap.

---

## 8. What this directory looks like in three months

If you maintain the discipline:

```
claude/
├── CLAUDE.md          (lightly updated; mostly stable)
├── CLAUDE_CODE_HANDOFF.md
└── SESSION_LOG.md     (~30 session entries; full project history)

docs/
├── ROADMAP.md         (phases checked off; slice 2 added)
├── ARCHITECTURE.md    (kept in sync as system evolves)
├── PEDAGOGY.md        (stable)
├── MODEL.md           (updated with each model version)
├── DATASET.md         (updated as data grows)
├── EVAL_GATE.md       (stable; criteria refined once or twice)
├── PRIVACY.md         (stable)
├── decisions/
│   └── 0001 through 0020-ish.md
├── research/
│   └── 8-12 verified per-source notes
└── validation/
    └── v1.md, v2.md, v3.md  (one per trained model version)
```

A reviewer reading the repo three months from now sees: a project that
thought carefully before building, executed against a stated theory,
documented its decisions, measured its own quality, and improved
deliberately. That is the exact signal you want to send.