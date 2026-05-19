# ADR 0003: Deployment uses Vercel + Supabase + Cloudflare R2, not Railway

**Status:** Accepted
**Date:** scoping session

---

## Context

The hiring partner guide names Vercel, Netlify, and Railway as
example deployment platforms. The user asked which we should use for
this production-scale system, and to defend the choice.

The system has three things that need to be hosted:

1. A Next.js application.
2. A Postgres database (and auth on top of it).
3. Model artifacts and other large static files that learners download.

A single-platform approach (e.g., Railway alone) and a multi-platform
approach (Vercel + Supabase + R2) are both viable.

---

## Decision

Three platforms, each best-in-class for its slice of the system:

- **Vercel** hosts the Next.js application.
- **Supabase** hosts Postgres + auth.
- **Cloudflare R2** hosts model artifacts and instructor-recorded
  reference videos.

---

## Rationale

**Vercel for the app.**

- Built by the Next.js team. Zero-config Next.js deployment. Edge
  network. Free SSL. Per-PR preview deployments.
- The hiring partner guide explicitly names Vercel.
- Free tier covers a pilot at this scale.
- Railway can host Next.js, but you pay for what Vercel gives away.

**Supabase for Postgres + auth.**

- Managed Postgres plus auth (email magic links, OAuth, row-level
  security) plus a typed TypeScript client in one bundle. Avoiding
  three separate services for these concerns is real engineering
  savings.
- Free tier covers a pilot.
- Auth tokens flow naturally into the Next.js middleware and server
  actions.

**Cloudflare R2 for model artifacts.**

- Zero egress fees. The model file (~5-10 MB) is downloaded by every
  learner on first visit; AWS S3 egress fees would compound. R2
  costs $0 for egress, full stop.
- Edge-cached by Cloudflare automatically.
- Same R2 bucket can serve the instructor-recorded reference videos
  with the same economics.

**Why not Railway as the single platform.**

- Hosting the Next.js app on Railway means no edge network for the
  app. Slower TTFB for learners outside the Railway region.
- Railway Postgres is fine but lacks the auth + RLS bundle that
  Supabase provides; we would have to roll auth separately.
- Railway storage egress is not free.
- Net result: more work, slower app, higher cost. Worse on every
  axis for this shape of workload.

**Why not Vercel storage instead of R2.**

- Vercel has Blob storage but pricing and egress economics don't
  match R2 at the scale a model-download-per-learner implies.

---

## Total cost during pilot

- Vercel: $0 (Hobby tier).
- Supabase: $0 (free tier, well within limits for pilot scale).
- Cloudflare R2: $0–$5/month (the model artifact storage plus reference
  video storage stays under the free tier of 10 GB).
- Vercel domain (optional): ~$15/year if we want a custom domain.
- GPU rental for training (Modal, Vast.ai, RunPod, or similar): pay
  per training run, expected ~$5–20 per run, 5–20 runs total.

Pilot total: under $50 in infrastructure costs across the entire
project.

---

## Production-scale defense

If this system were to scale to thousands or millions of learners:

- **Vercel** scales horizontally with the edge network; this is its
  bread-and-butter use case.
- **Supabase** scales vertically (instance size) and horizontally (read
  replicas); a million-user workload is well within its capacity tier.
- **R2** scales effectively without bound; egress remains free.

The architecture does not need to be re-platformed to handle scale.
That is a real production-grade property.

---

## Consequences

- Three vendor relationships instead of one.
- Three sets of credentials to manage; we use Vercel environment
  variables for client and server, Supabase service-role keys for
  server-only operations, and R2 access keys for the training
  pipeline only.
- Vendor lock-in is moderate: Next.js can deploy elsewhere; Postgres
  is portable; R2 has standard S3-compatible API so artifacts could
  move to S3 or any S3-clone.

---

## Rejected alternatives

- **Railway for everything.** Defended against above.
- **Vercel + Vercel Postgres + Vercel Blob.** Plausible all-Vercel
  alternative, but Postgres pricing and Blob egress economics are
  worse than the chosen stack at this workload's shape (large file
  served on every first visit).
- **Self-hosted on a VPS.** Worse on every axis for a pilot:
  setup time, reliability, scaling, security posture, SSL management.