# Vocabulary (slice 1)

> The 96-sign list the recognition model trains on. Drawn from
> Lifeprint ASL 1 (Lessons 1–15) and cross-referenced against
> ASL-LEX 2.0 (phonological coverage) and WLASL (training-clip
> availability). Per `docs/decisions/0004-public-sources-only.md`,
> this list is sourced from public corpora only for the pilot;
> Deaf-instructor sign-off is a slice-2 production-deployment
> requirement.

The brief (Requirement 2) calls for 75–100 vocabulary items. This
list is 96. Alphabet, fingerspelled letters, and numbers are
**excluded** per `claude/CLAUDE.md` §4 ("Alphabet inclusion:
EXCLUDED. Vocabulary must be life-usable content signs only.").

---

## Column key

| Column | Meaning |
|---|---|
| `Gloss` | English uppercase gloss form (display form in the app may add hyphens or punctuation, e.g. THANK-YOU). |
| `Category` | Functional grouping for category-balance review. Not a database field. |
| `Lifeprint Lesson` | The Lifeprint ASL 1 lesson where the sign is taught (`Lx` = Lesson x). All citations are to https://www.lifeprint.com/asl101/lessons/lessonNN.htm fetched 2026-05-19. |
| `ASL-LEX 2.0 Code` | The `Code` column from ASL-LEX 2.0 `signdata.csv` (e.g. `C_01_006`). Blank if the gloss has no canonical ASL-LEX entry. |
| `ASL-LEX Match` | The `EntryID` (lemma) we matched on. Where Lifeprint and ASL-LEX use different surface forms (MOM → `mother`, DAD → `father`, GRANDMA → `grandmother`), the ASL-LEX lemma is recorded here. |
| `WLASL Clips` | Instance count for this gloss in `WLASL_v0.3.json` from the [official WLASL repo](https://github.com/dxli94/WLASL). Counts raw video instances; not all instances are necessarily downloadable, and the cleaning pipeline (`docs/DATASET.md` §3) will filter further. |
| `MS-ASL Clips` | Marked `PENDING` for every row. MS-ASL's class manifests are gated behind a Microsoft Research license-acceptance flow that is not automatable here; see "Open follow-ups" below. |
| `Static / Movement` | `static` = ASL-LEX `Movement.2.0` is `None` and `RepeatedMovement.2.0` is `0`; otherwise `movement`. Per `docs/DATASET.md` §7, slice-1 prioritization favors more `static` signs initially. |
| `Flippable` | `yes` = safe horizontal-flip augmentation; `no` = flipping would change or destroy meaning. Computed from ASL-LEX `SignType.2.0` (signs whose type starts with `Asymmetrical` or is a `DominanceViolation`/`SymmetryViolation` are not flippable) plus a deny list of pronouns and directional verbs whose meaning is encoded by handedness or pointing direction. |

---

## Category balance

| Category | Count |
|---|---|
| greetings-social | 7 |
| wh-question | 6 |
| yes-no | 5 |
| pronouns | 6 |
| family | 11 |
| deaf-culture | 9 |
| feelings | 6 |
| colors | 6 |
| verbs-common | 10 |
| places | 5 |
| food | 5 |
| animals | 4 |
| time | 7 |
| descriptors | 6 |
| objects | 3 |
| **Total** | **96** |

---

## The list

| # | Gloss | Category | Lifeprint Lesson | ASL-LEX 2.0 Code | ASL-LEX Match | WLASL Clips | MS-ASL Clips | Static / Movement | Flippable |
|---|---|---|---|---|---|---|---|---|---|
| 1 | NICE | greetings-social | L1 | G_03_059 | nice | 14 | PENDING | movement | no |
| 2 | MEET | greetings-social | L1 | D_02_018 | meet | 19 | PENDING | movement | no |
| 3 | THANK-YOU | greetings-social | L1 | H_02_053 | thank_you | 14 | PENDING | movement | no |
| 4 | SORRY | greetings-social | L4 | C_01_020 | sorry | 14 | PENDING | movement | yes |
| 5 | EXCUSE | greetings-social | L4 | D_03_052 | excuse | 9 | PENDING | movement | no |
| 6 | FINE | greetings-social | L3 | C_01_043 | fine_1 | 22 | PENDING | movement | yes |
| 7 | AGAIN | greetings-social | L1 | J_02_013 | again | 16 | PENDING | movement | no |
| 8 | WHAT | wh-question | L1 | D_02_084 | what_1 | 21 | PENDING | movement | yes |
| 9 | WHERE | wh-question | L1 | B_02_035 | where | 15 | PENDING | movement | yes |
| 10 | WHO | wh-question | L1 | C_01_041 | who | 25 | PENDING | movement | yes |
| 11 | WHY | wh-question | L1 | D_01_067 | why | 17 | PENDING | movement | yes |
| 12 | HOW | wh-question | L2 | D_02_082 | how | 18 | PENDING | movement | yes |
| 13 | WHEN | wh-question | L6 | C_03_086 | when | 14 | PENDING | movement | no |
| 14 | YES | yes-no | L1 | G_03_074 | yes | 22 | PENDING | movement | yes |
| 15 | NO | yes-no | L1 | C_03_041 | no | 22 | PENDING | movement | yes |
| 16 | HAVE | yes-no | L2 | B_03_057 | have | 17 | PENDING | movement | yes |
| 17 | NEED | yes-no | L3 | C_02_034 | need | 18 | PENDING | movement | yes |
| 18 | WANT | yes-no | L3 | C_03_013 | want | 19 | PENDING | movement | yes |
| 19 | MY | pronouns | L2 | C_01_060 | my | 12 | PENDING | movement | no |
| 20 | YOUR | pronouns | L1 | F_01_052 | your | 15 | PENDING | movement | no |
| 21 | OUR | pronouns | L2 | G_02_067 | our | 7 | PENDING | movement | no |
| 22 | THEY | pronouns | L1 | E_01_044 | they_1 | 9 | PENDING | movement | no |
| 23 | WE | pronouns | L1 | G_03_019 | we | 10 | PENDING | movement | no |
| 24 | HIS | pronouns | L2 | F_02_058 | his | 8 | PENDING | movement | no |
| 25 | MOM | family | L2 | B_02_008 | mother | 11 | PENDING | movement | yes |
| 26 | DAD | family | L2 | B_01_077 | father | 10 | PENDING | movement | yes |
| 27 | BROTHER | family | L2 | B_01_065 | brother | 17 | PENDING | static | yes |
| 28 | SISTER | family | L2 | D_03_050 | sister | 14 | PENDING | static | yes |
| 29 | FAMILY | family | L3 | C_01_025 | family | 20 | PENDING | movement | yes |
| 30 | BABY | family | L4 | C_01_017 | baby | 14 | PENDING | movement | yes |
| 31 | GRANDMA | family | L2 | C_02_027 | grandmother | 8 | PENDING | static | yes |
| 32 | FRIEND | family | L4 | D_03_010 | friend | 16 | PENDING | movement | yes |
| 33 | MAN | family | L14 | C_01_040 | man | 20 | PENDING | static | yes |
| 34 | WOMAN | family | L14 | C_02_028 | woman | 21 | PENDING | movement | yes |
| 35 | PEOPLE | family | L11 | C_01_009 | people | 13 | PENDING | movement | yes |
| 36 | DEAF | deaf-culture | L1 | F_03_038 | deaf_1 | 23 | PENDING | movement | yes |
| 37 | HEARING | deaf-culture | L1 | C_02_023 | hearing | 20 | PENDING | movement | yes |
| 38 | SIGN | deaf-culture | L1 | E_03_050 | sign | 15 | PENDING | movement | yes |
| 39 | NAME | deaf-culture | L1 | D_01_021 | name | 16 | PENDING | movement | no |
| 40 | LEARN | deaf-culture | L1 | B_01_042 | learn | 17 | PENDING | static | no |
| 41 | UNDERSTAND | deaf-culture | L1 | C_01_006 | understand | 14 | PENDING | static | yes |
| 42 | INTERPRETER | deaf-culture | L13 | D_02_066 | interpreter | 9 | PENDING | movement | no |
| 43 | TEACHER | deaf-culture | L1 | A_02_033 | teacher | 17 | PENDING | movement | yes |
| 44 | STUDENT | deaf-culture | L1 | A_02_012 | student | 15 | PENDING | static | no |
| 45 | HAPPY | feelings | L4 | C_03_078 | happy | 16 | PENDING | movement | yes |
| 46 | SAD | feelings | L4 | B_02_053 | sad | 14 | PENDING | movement | yes |
| 47 | ANGRY | feelings | L4 | B_01_070 | angry | 11 | PENDING | movement | yes |
| 48 | LOVE | feelings | L4 | G_01_068 | love | 12 | PENDING | static | yes |
| 49 | HURT | feelings | L4 | E_01_002 | hurt | 12 | PENDING | movement | yes |
| 50 | FEEL | feelings | L4 | C_02_001 | feel | 15 | PENDING | movement | yes |
| 51 | RED | colors | L6 | C_02_090 | red | 17 | PENDING | movement | yes |
| 52 | BLUE | colors | L6 | C_02_062 | blue | 20 | PENDING | movement | yes |
| 53 | GREEN | colors | L6 | B_02_085 | green | 17 | PENDING | movement | yes |
| 54 | YELLOW | colors | L6 | D_01_020 | yellow | 18 | PENDING | movement | yes |
| 55 | BLACK | colors | L6 | B_02_006 | black | 21 | PENDING | movement | yes |
| 56 | WHITE | colors | L6 | E_03_046 | white | 20 | PENDING | movement | yes |
| 57 | GO | verbs-common | L3 | C_03_056 | go | 26 | PENDING | movement | yes |
| 58 | COME | verbs-common | L3 | C_03_074 | come | 8 | PENDING | movement | yes |
| 59 | EAT | verbs-common | L7 | B_02_002 | eat_1 | 19 | PENDING | movement | yes |
| 60 | DRINK | verbs-common | L7 | B_02_012 | drink | 35 | PENDING | movement | yes |
| 61 | WORK | verbs-common | L2 | B_03_059 | work | 19 | PENDING | movement | no |
| 62 | LIVE | verbs-common | L2 | B_03_046 | live_1 | 15 | PENDING | movement | yes |
| 63 | HELP | verbs-common | L4 | D_01_042 | help | 22 | PENDING | movement | no |
| 64 | KNOW | verbs-common | L10 | C_01_048 | know | 16 | PENDING | movement | yes |
| 65 | THINK | verbs-common | L3 | C_03_053 | think | 14 | PENDING | movement | yes |
| 66 | PLAY | verbs-common | L5 | C_01_066 | play | 19 | PENDING | movement | yes |
| 67 | HOME | places | L5 | B_03_063 | home | 16 | PENDING | movement | yes |
| 68 | SCHOOL | places | L3 | C_03_089 | school | 19 | PENDING | movement | no |
| 69 | HOUSE | places | L3 | D_03_086 | house | 15 | PENDING | movement | yes |
| 70 | CHURCH | places | L5 | C_02_018 | church | 13 | PENDING | movement | no |
| 71 | BATHROOM | places | L3 | A_02_051 | bathroom | 16 | PENDING | movement | yes |
| 72 | WATER | food | L7 | A_02_031 | water | 18 | PENDING | movement | yes |
| 73 | MILK | food | L7 | A_01_051 | milk | 14 | PENDING | movement | yes |
| 74 | HUNGRY | food | L7 | C_01_010 | hungry | 12 | PENDING | movement | yes |
| 75 | APPLE | food | L7 | A_03_054 | apple | 19 | PENDING | movement | yes |
| 76 | PIZZA | food | L7 | G_01_013 | pizza_1 | 19 | PENDING | movement | yes |
| 77 | DOG | animals | L10 | A_01_056 | dog | 20 | PENDING | movement | yes |
| 78 | CAT | animals | L10 | A_01_023 | cat | 17 | PENDING | movement | yes |
| 79 | BIRD | animals | L10 | A_01_041 | bird | 19 | PENDING | movement | yes |
| 80 | FISH | animals | L10 | A_03_040 | fish | 20 | PENDING | movement | no |
| 81 | NOW | time | L6 | C_03_062 | now | 21 | PENDING | movement | yes |
| 82 | TOMORROW | time | L6 | F_02_040 | tomorrow | 13 | PENDING | movement | yes |
| 83 | YESTERDAY | time | L9 | D_01_024 | yesterday | 17 | PENDING | movement | yes |
| 84 | MORNING | time | L12 | C_02_012 | morning | 13 | PENDING | movement | yes |
| 85 | NIGHT | time | L12 | A_01_003 | night | 13 | PENDING | movement | no |
| 86 | DAY | time | L12 | C_01_044 | day | 15 | PENDING | movement | yes |
| 87 | WEEK | time | L12 | B_02_079 | week | 17 | PENDING | movement | no |
| 88 | BIG | descriptors | L3 | F_02_054 | big | 12 | PENDING | movement | yes |
| 89 | SMALL | descriptors | L3 | D_01_030 | small | 15 | PENDING | movement | yes |
| 90 | GOOD | descriptors | L3 | B_01_052 | good | 16 | PENDING | movement | yes |
| 91 | BAD | descriptors | L3 | B_02_082 | bad | 16 | PENDING | movement | yes |
| 92 | HOT | descriptors | L14 | F_02_093 | hot | 21 | PENDING | movement | yes |
| 93 | COLD | descriptors | L14 | C_02_068 | cold | 16 | PENDING | movement | yes |
| 94 | CAR | objects | L5 | D_02_043 | car | 13 | PENDING | movement | yes |
| 95 | BOOK | objects | L10 | A_01_027 | book | 40 | PENDING | movement | yes |
| 96 | COMPUTER | objects | L5 | A_01_054 | computer | 30 | PENDING | movement | yes |

---

## Sources and verification

**Lifeprint ASL 1 curriculum.** Dr. Bill Vicars, Lifeprint /
ASL University. Lessons 1 through 15. Each lesson's vocabulary
list was fetched directly from its page at
`https://www.lifeprint.com/asl101/lessons/lessonNN.htm` on
**2026-05-19** and read into the cross-reference script. Lifeprint
groups ASL 1 into Units 1–3 (Lessons 1–5, 6–10, 11–15). The user's
original brief read "Units 1–6," which spans into Lifeprint's ASL 2
(Lessons 16–30); we scoped to ASL 1 (Lessons 1–15) per the
2026-05-19 clarification that aligns with `claude/CLAUDE.md` §2's
"college ASL 1" framing.

**ASL-LEX 2.0.** Sevcikova Sehyr, Z., Caselli, N. K., Cohen-Goldberg,
A. M., & Emmorey, K. (2021). *The ASL-LEX 2.0 Project: A Database
of Lexical and Phonological Properties for 2,723 Signs in American
Sign Language.* The Journal of Deaf Studies and Deaf Education,
26(2), 263–277. We use the `signdata.csv` file from the OSF
supplementary materials (https://osf.io/zpha4/, file
https://osf.io/download/9nygd/), 2,722 sign entries plus header,
fetched 2026-05-19. The `Code` column in our table is the ASL-LEX
2.0 `Code` field; the `Match` column is its `EntryID`. The
`Static / Movement` column is derived from `Movement.2.0` and
`RepeatedMovement.2.0`; the `Flippable` column is derived from
`SignType.2.0` plus a deny list for pronouns and directional verbs.

**WLASL.** Li, D., Rodriguez Opazo, C., Yu, X., & Li, H. (2020).
*Word-level Deep Sign Language Recognition from Video: A New
Large-scale Dataset and Methods Comparison.* WACV 2020. We use
`WLASL_v0.3.json` from the official repository
https://github.com/dxli94/WLASL (commit on `master`, file fetched
2026-05-19). Clip counts are raw `instances[]` lengths per gloss;
downloadability filtering happens in the data-cleaning pipeline
(`docs/DATASET.md` §3) and may reduce the per-sign count further.

**MS-ASL.** Vaezi Joze, H. R., & Koller, O. (2018, BMVC 2019).
*MS-ASL: A Large-Scale Data Set and Benchmark for Understanding
American Sign Language.* arXiv:1812.01053. The MS-ASL class
manifests are distributed by Microsoft Research and require
license acceptance through a gated download flow; this could not
be completed in the current session, so every row's `MS-ASL Clips`
field is marked `PENDING`. See "Open follow-ups" below.

**Reproducibility.** The Python scripts that generated this table
from the four sources above live at `/tmp/asl-vocab/build.py` and
`/tmp/asl-vocab/select.py` on the author's machine. They are not
committed (the working scratchpad is intentionally outside the
repo). The intermediate JSON outputs (`crossref.json`, `selected.json`)
are also local-only. A clean re-derivation of this table is a small
Phase 3 work item: port `build.py` to `training/data/build_vocabulary.py`
when the training codebase is initialized.

---

## Open follow-ups (Phase 3, not blocking docs)

1. **MS-ASL class manifest verification.** Complete the Microsoft
   Research download flow for MS-ASL and replace every `PENDING` in
   the table with an actual per-gloss instance count. Update this
   doc with the manifest version and fetch date.
2. **WLASL downloadability filter.** Many WLASL instances point at
   YouTube URLs that have rotted since 2020. The cleaning pipeline
   needs to verify each per-gloss count against the actually-
   downloadable subset and re-issue the count in `docs/DATASET.md`.
3. **ASL-LEX `Static / Movement` audit.** The static/movement
   derivation is a phonology-based heuristic, not a
   pedagogical/visual judgment. A small audit pass against the
   ASL-LEX videos (which are public) should sanity-check the 8
   signs currently marked `static` and flag any obviously-wrong
   labels.
4. **Flippable audit for two-handed asymmetric signs the deny list
   missed.** The deny list is conservative on pronouns and
   directional verbs but does not exhaustively cover every
   handedness-encoding sign; a manual review pass before training
   should catch edge cases (e.g. directional verbs not in the deny
   list).
5. **Slice-2 instructor review (per ADR 0004).** The full list
   reviewed by a Deaf ASL instructor with authority to remove signs,
   re-classify static/movement, and adjust flippable judgments.
   Production deployment requires this; the pilot ships without it.

---

## What this list intentionally does not include

- **The alphabet and fingerspelled letters.** Per
  `claude/CLAUDE.md` §4, vocabulary must be life-usable content
  signs only.
- **Numbers.** Same constraint; numbers are excluded from the 75–100
  vocabulary budget.
- **Sentences or phrases.** Per `docs/ROADMAP.md` "What is
  intentionally not on this roadmap," sentence/phrase recognition
  is out of scope.
- **Variants of a single sign.** ASL-LEX 2.0 lists multiple
  variants for common signs (e.g. `fine_1`, `fine_2`). We commit to
  one canonical variant per gloss for the slice-1 model; the
  variant chosen is recorded in `ASL-LEX Match`. Multi-variant
  recognition is a slice-2 candidate.
