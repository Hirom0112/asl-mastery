# Research Index

> Every paper, dataset, or external source this project cites or
> depends on, with verification status and notes. One file per source
> in this directory; this README is the index.

The discipline: a claim in user-facing documentation must trace back
to a source in this index whose status is at least `VERIFIED`. If the
status is `PENDING`, the citation must be verified before the doc
ships. If the status is `WITHDRAWN`, the claim must be removed.

---

## Verification tiers

We use two tiers so that web-search verification (cheap, fast, fit for
slice 1 drafting) is distinguished from primary-source reading
(expensive, slow, required before a hiring-partner demo).

- **`VERIFIED` (Tier 1).** The paper exists at the cited venue; the
  DOI, authors, year, and journal are accurate; the central claim we
  attribute to it is corroborated against primary or high-quality
  secondary sources via web search. Sufficient to ship internal docs
  and to draft user-facing copy. **Not** sufficient for the final
  presentation to the hiring partner.
- **`DEEP-VERIFIED` (Tier 2).** Tier 1 plus an exact-quote pull from
  the original published source with page or section reference,
  committed in a per-source file in `docs/research/<slug>.md`. Required
  before the demo (`docs/ROADMAP.md` Phase 8) for any citation that
  appears in the README, the presentation, or the validation report.

`NOTED` and `PENDING` and `WITHDRAWN` are unchanged.

---

## How to add a new source

1. Create `docs/research/<short-slug>.md` (e.g., `hattie-timperley-2007.md`)
   when promoting to `DEEP-VERIFIED`; for `VERIFIED` (Tier 1) only the
   index row below is required.
2. Mark the source `PENDING` on creation.
3. Promote to `VERIFIED` once the venue, DOI, and central claim are
   web-corroborated.
4. Promote to `DEEP-VERIFIED` once the per-source file with an exact
   quote and page/section reference is committed.
5. If the source does not support the claim it was cited for, mark
   `WITHDRAWN` and edit the doc that cited it.

---

## Pedagogy citations

Source-of-truth status for the citations referenced in `PEDAGOGY.md`.
Six are currently `VERIFIED` (Tier 1) from web-search corroboration
during Session 2 against primary literature; one is `NOTED` as
historical reference; one remains `PENDING`. All six `VERIFIED` rows
are scheduled for `DEEP-VERIFIED` promotion in Phase 8 (presentation
prep) before the demo.

| Slug | Source | Claim attributed | Status |
|---|---|---|---|
| `bloom-1984-2-sigma` | Bloom (1984), Educational Researcher | 2-sigma effect of mastery learning + tutoring (~1 sigma for mastery alone) | VERIFIED |
| `hattie-timperley-2007-feedback` | Hattie & Timperley (2007), Review of Educational Research; Wisniewski, Zierer, Hattie (2019), Frontiers in Psychology | Feedback type moderates effect; replication finds larger effect for motor-skill outcomes | VERIFIED |
| `karpicke-roediger-2008-testing` | Karpicke & Roediger (2008), Science | Retrieval practice > re-study for long-term retention (foreign vocabulary paradigm) | VERIFIED |
| `cepeda-2006-spacing-effect` | Cepeda, Pashler, Vul, Wixted, Rohrer (2006), Psychological Bulletin | Distributed practice meta-analysis; optimal inter-study interval scales with retention interval | VERIFIED |
| `shea-morgan-1979-contextual` | Shea & Morgan (1979), J Exp Psych: Human Learning and Memory | Contextual interference: interleaved > blocked for motor skill retention | VERIFIED |
| `sweller-1988-cognitive-load` | Sweller (1988), Cognitive Science; Sweller (2010), Educational Psychology Review | Cognitive load theory; three-component model articulated in 2010 paper | VERIFIED |
| `ebbinghaus-and-replication` | Ebbinghaus (1885) + Murre & Dros (2015), PLOS ONE | Forgetting curve and spaced retrieval re-stabilization | NOTED |
| `anderson-corbett-1995-cognitive-tutors` | Anderson, Corbett, Koedinger, Pelletier (1995), J Learning Sciences | ITS systems can approach 1:1 tutoring outcomes | PENDING |

---

## Datasets we plan to use

These are sources of training data, not academic claims. Verification
here means confirming the dataset exists, the license terms, and the
fit to our 75–100 vocabulary.

| Slug | Source | Status |
|---|---|---|
| `wlasl-dataset` | Li, Rodriguez, Xu, Joze, Wang, Kacorri, Yamada (2020), WACV | PENDING |
| `ms-asl-dataset` | Joze & Koller (2018), Microsoft Research | PENDING |
| `asl-lex-2` | Sehyr, Caselli, Cohen-Goldberg, Emmorey (2021), J Deaf Studies and Deaf Education | PENDING |

---

## Sign linguistics references

For the five-parameter model of ASL signs (handshape, location, palm
orientation, movement, non-manual markers) used throughout the hint
system and vocabulary metadata.

| Slug | Source | Status |
|---|---|---|
| `stokoe-1960-sign-structure` | Stokoe (1960), Sign Language Structure (foundational; verify quoted claims) | PENDING |
| `valli-lucas-mulrooney` | Valli, Lucas, Mulrooney (textbook on ASL linguistics, multiple editions) | PENDING |

---

## What the statuses mean in practice

**`VERIFIED` (Tier 1)** requires:

1. The paper exists at the cited venue (DOI resolves; or the venue's
   archive lists it; or multiple independent secondary sources agree on
   venue, year, and authors).
2. The central claim we attribute to the source is corroborated by
   primary or high-quality secondary sources accessed via web search.
3. Any subsequent qualification or replication of substance (e.g., a
   later meta-analysis tightening an effect size) is recorded in
   `PEDAGOGY.md` alongside the original citation.

**`DEEP-VERIFIED` (Tier 2)** additionally requires a per-source file
at `docs/research/<slug>.md` containing:

1. Bibliographic record (full citation, DOI, authors, year, venue).
2. An exact quote from the published source supporting the attributed
   claim, with page or section reference.
3. A "what we took from this" note in plain English.
4. A "subsequent qualifications" note recording any later replication
   or meta-analytic update that changes how the claim should be read.

If step 2 of `DEEP-VERIFIED` fails (the source does not actually
support the claim), the status drops to `WITHDRAWN` and the citing
doc must be edited.

---

## Why this discipline exists

A well-documented EdTech failure mode is projects that sound
rigorous but make claims they cannot defend. We will not be that
project. The pedagogical theory we claim is exactly as strong as
the citations behind it; if a citation cannot be verified, the
claim does not ship.
