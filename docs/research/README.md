# Research Index

> Every paper, dataset, or external source this project cites or
> depends on, with verification status and notes. One file per source
> in this directory; this README is the index.

The discipline: a claim in user-facing documentation must trace back
to a file in this directory with status `VERIFIED`. If the file's
status is `PENDING`, the citation must be verified before the doc
ships. If the file's status is `WITHDRAWN`, the claim must be removed.

---

## How to add a new source

1. Create `docs/research/<short-slug>.md` (e.g., `hattie-timperley-2007.md`).
2. Fill the template (see `_template.md`).
3. Mark the source `PENDING` on creation.
4. Read the actual source.
5. Update notes, exact-quote-pull, and status to `VERIFIED`.
6. If the source does not support the claim it was cited for, mark
   `WITHDRAWN` and edit the doc that cited it.

---

## Sources we have committed to verify

These are referenced as `[PENDING]` in `PEDAGOGY.md` and must be
resolved during Phase 1 of the roadmap.

| Slug | Source | Claim attributed | Status |
|---|---|---|---|
| `bloom-1984-2-sigma` | Bloom (1984), Educational Researcher | 2-sigma effect of mastery learning + tutoring | PENDING |
| `hattie-timperley-2007-feedback` | Hattie & Timperley (2007), Review of Educational Research | Feedback type matters; task feedback > self feedback | PENDING |
| `ebbinghaus-and-replication` | Ebbinghaus (1885) + modern replication (likely Murre & Dros 2015) | Forgetting curve and spaced retrieval re-stabilization | PENDING |
| `cepeda-2006-spacing-effect` | Cepeda, Pashler, Vul, Wixted, Rohrer (2006), Psychological Bulletin | Distributed practice > massed practice for retention | PENDING |
| `karpicke-roediger-2008-testing` | Karpicke & Roediger (2008), Science | Retrieval practice > re-study for long-term retention | PENDING |
| `shea-morgan-1979-contextual` | Shea & Morgan (1979), J Exp Psych: Human Learning and Memory | Contextual interference: interleaved > blocked for motor skill retention | PENDING |
| `sweller-1988-cognitive-load` | Sweller (1988), Cognitive Science | Cognitive load theory: intrinsic / extraneous / germane | PENDING |
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

## What "verified" means

A source is `VERIFIED` when a human team member has:

1. Located the actual published source (not a summary, not a textbook
   reference, not a citation in another paper).
2. Read at least the abstract, introduction, and conclusions.
3. Located the specific passage that supports the claim we attribute
   to it.
4. Recorded an exact quote in the per-source file, with page or
   section reference.
5. Checked whether the claim has been subsequently overturned or
   substantially qualified by later work.

If step 3 fails (the source does not actually support the claim), the
status becomes `WITHDRAWN` and the citing doc must be edited.

If step 5 reveals a substantial later qualification, the per-source
file records the qualification and the citing doc reflects it.

---

## Why this discipline exists

Patrick's published criticism of EdTech includes the failure mode of
projects that sound rigorous but make claims they cannot defend. We
will not be that project. The pedagogical theory we claim is exactly
as strong as the citations behind it; if a citation cannot be
verified, the claim does not ship.
