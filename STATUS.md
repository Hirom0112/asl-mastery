# STATUS — what's true right now (2026-05-21)

> This project pivoted architectures twice. Many docs in the repo
> describe one of the older architectures. **Use this file as the
> single source of truth for "what is current."**

---

## Architecture (post-second-pivot)

**Recognition system:** four from-scratch CV detectors feeding a
distribution-of-templates sign matcher.

```
webcam frame
  ↓
hand detector (320×320 CenterNet-style, 1–3M params)
  ↓ per detected hand
hand landmark regressor (224×224 small CNN, 21 keypoints, 3–4M params)
  ↓
            +  pose regressor (256×256, 8 upper-body keypoints)
            +  face detector (320×320 single-class CenterNet, for framing only)
  ↓
trajectory accumulated over 1–3 s capture window
  ↓
template matcher (Mahalanobis distance per sign, softmax → confidence)
  ↓
per-sign calibrated threshold (target precision ≥ 0.95)
  ↓
pass / fail / retry → hint engine (Phase 5, not yet built)
```

Every weight in every CV component is trained by this project. No
MediaPipe, no OpenPose, no pretrained backbones, no foreign
`load_state_dict`. Labels come from public human-annotated datasets
(FreiHAND, CMU HandDB, COCO-WholeBody, MPII, WIDER FACE, Multiview
Hand Pose — all audited under ADR 0015) plus self-training
pseudo-labels on the project's 12K ASL clip corpus.

---

## Read in this order

| # | Path | Purpose |
|---|---|---|
| 1 | `docs/decisions/0011-landmarks-and-templates-pivot.md` | The governing architecture decision |
| 2 | `docs/decisions/0012-strict-from-scratch-cv-constraint.md` | The constraint perimeter — what's from-scratch and why |
| 3 | `docs/decisions/0015-external-cv-datasets-provenance.md` | Label-source acceptance criteria |
| 4 | `docs/VOCABULARY_TRAINER_ROADMAP.md` | Current strategic roadmap (the "strict" one) |
| 5 | `TODO2.md` | Operational task tracker — what's done, what's in flight |
| 6 | `docs/data/external_datasets_audit.md` | Per-dataset provenance findings + download plan |
| 7 | `training/detectors/` source files | The actually-built code |

---

## Superseded / stale — read for historical context only

| Path | What it describes | Superseded by |
|---|---|---|
| `docs/decisions/0001-recognition-architecture.md` | Original end-to-end RGB CNN (Path B) | ADR 0006 (revoked) and then ADR 0011 |
| `docs/decisions/0006-recognition-architecture-revised.md` | First pivot: MediaPipe + BiLSTM | ADR 0010 (reversed it) |
| `docs/decisions/0010-reversal-of-adr-0006.md` | Reinstated ADR 0001 Path B under strict reading | ADR 0011 (superseded the reinstatement) |
| `docs/ROADMAP.md` | Phase-based pre-pivot roadmap | `docs/VOCABULARY_TRAINER_ROADMAP.md` |
| `docs/ARCHITECTURE.md` | Pre-pivot 5-surface architecture (most sections still valid in principle; §2.3 inference and §2.5 training pipeline are stale) | This STATUS file + ADR 0011 |
| `docs/MODEL.md` | Pre-pivot single-classifier spec | ADR 0011 § "Decision" |
| `docs/DATASET.md` | ASL clip corpus only (no external CV datasets) | ADR 0015 + `docs/data/external_datasets_audit.md` |
| `docs/EVAL_GATE.md` | Hard criteria referencing the pre-pivot architecture; criterion 10 was removed by ADR 0010; per-keypoint and per-detector criteria pending under ADR 0011 | Pending update — eval-gate v2 ADR |
| `training/README.md` | Describes the MediaPipe-era ingestion + classifier pipeline | This STATUS + `training/detectors/` README (to add) |
| `TODO.md` | Pre-pivot triage list | `TODO2.md` |

---

## Production state right now

- **Live deploy:** `https://asl-mastery.vercel.app` continues to serve the deactivated v2.1.0 artifact behind the ADR 0010 stub-fallback + offline banner. No model_versions rows reactivated.
- **All v1.0.x, v2.0.0, v2.1.0:** `is_active = false` per `supabase/migrations/20260520200000_deactivate_all_models.sql`. They remain in R2 + Postgres as historical records (no silent revisions).
- **No v3.x artifact has been trained yet under ADR 0011.** Phase 1 hand-detector training is the next thing to fire on Modal once downloads finish.
- **Supabase + Vercel:** linked, dormant. No infrastructure unlinked.

---

## Pipeline state (Session 14 close)

| Stage | Status |
|---|---|
| External-dataset audit + download plan | ✅ committed (`f26bedf`) |
| FreiHAND, CMU HandDB, Multiview, WIDER annotations downloaded | ✅ on disk |
| MPII, COCO 2017 downloads | 🔄 in flight (background) |
| Annotation normalizers for all 6 datasets | ✅ committed (`8cd650b`) |
| `normalize_external.py` manifest builder | ✅ committed |
| Hand detector (Phase 1) — arch, loss, dataset, train, Modal entrypoint | ✅ committed (`8e8ea0c`) |
| Hand landmark regressor (Phase 2) — arch, dataset, augment, train, Modal | ✅ committed (`1e56b95`) |
| Pose regressor (Phase 3.1) — arch | ✅ scaffolded; Modal entrypoint stubbed |
| Face detector (Phase 3.2) — alias + Modal entrypoint | ✅ committed |
| Trajectory extraction (Phase 4.1) | ✅ committed |
| Template fitter (Phase 4.2) | ✅ committed |
| Sign matcher + per-sign threshold calibration (Phase 4.3) | ✅ committed (`4afefa4`) |
| Self-training pseudo-labeler | ✅ committed |
| Modal volume push script + overnight launcher | ✅ committed |
| **First training run on Modal** | ⏳ pending downloads |
| ONNX export for any detector | ❌ not written |
| Browser inference (TS) — 4-detector pipeline + sign matcher | ❌ not written |
| Phase 5 (pedagogy) | ❌ not started |
| Phase 6 (avatar + TTS) | ❌ not started |
| Phase 7 (validation + bias eval + final polish) | ❌ not started |

---

## When in doubt

Read ADR 0011 first. If a doc in this repo conflicts with ADR 0011,
ADR 0011 wins and the doc is stale.

If `STATUS.md` (this file) conflicts with ADR 0011, ADR 0011 wins
and this file needs updating.
