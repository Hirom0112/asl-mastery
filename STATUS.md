# STATUS — what's true right now (2026-05-22, Session 19)

> This project pivoted architectures twice. Many docs in the repo
> describe one of the older architectures. **Use this file as the
> single source of truth for "what is current."**

---

## Architecture (post-second-pivot)

**Recognition system:** five from-scratch CV components feeding a learned
TCN/transformer classifier head. The Mahalanobis template matcher (ADR 0011
original spec) was kept as a fallback but a learned head substantially
outperforms it on the realistic confusion set.

```
webcam frame
  ↓
hand detector (320×320 CenterNet, hand_det v2 ~2.3M params)
  ↓ per detected hand
            +  hand landmark regressor (224×224, 21 keypoints, ~4.3M params)
            +  pose regressor (256×256, 8 upper-body kpts, fed upper-body crop derived from hand bboxes)
            +  HandshapeEncoder (224×224, ~11M params, 128D L2-norm embedding)  ← Phase 4.6
            +  face detector (320×320 CenterNet, for framing only)
  ↓
per-frame feature vector (356D = 2 hands × 21kpts + 2 hands × 128D embed + 8 pose kpts)
  ↓
trajectory accumulated over 32-frame (≈ 2 s @ 15 fps) capture window
  ↓
classifier head: TCN (~2M params) or transformer (Phase 4.7 A/B parked — both ≈11% at 100D)
  ↓
softmax over the active sign slice → per-sign calibrated threshold
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
| 5 | `TODO.md` | Operational task tracker — what's done, what's in flight |
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

---

## Production state right now

- **Live deploy:** `https://asl-mastery.vercel.app` continues to serve the deactivated v2.1.0 artifact behind the ADR 0010 stub-fallback + offline banner. No model_versions rows reactivated.
- **All v1.0.x, v2.0.0, v2.1.0:** `is_active = false` per `supabase/migrations/20260520200000_deactivate_all_models.sql`. They remain in R2 + Postgres as historical records (no silent revisions).
- **No v3.x artifact has been trained yet under ADR 0011.** Phase 1 hand-detector training is the next thing to fire on Modal once downloads finish.
- **Supabase + Vercel:** linked, dormant. No infrastructure unlinked.

---

## Pipeline state (Session 19)

| Stage | Status |
|---|---|
| External-dataset audit + download plan | ✅ committed |
| All external datasets downloaded + extracted on Modal volume | ✅ |
| Annotation normalizers (FreiHAND, CMU HandDB, COCO pose, MPII, WIDER) | ✅ |
| `normalize_external.py` manifest builder + Modal `build_manifests` | ✅ |
| **Hand detector v2** (HaGRID + CMU + FreiHAND, 181k records) | ✅ Session 16 — `runs/hand_det_v2_hagrid_fromscratch/best_ema.pt`, val=0.79, ~2.3M params, ~$15.40 |
| **Hand landmark regressor v0** | ✅ Session 15 — `runs/hand_landmarks_v0_20260521_105603Z/best.pt`, val=0.036/12.82 px, ~4.3M params |
| **Pose regressor v0** | ✅ Session 15-trained, Session 18 **inference-bug fix**: was trained on tight person-bbox crops but receiving full frames at inference. Fix: `_pose_inference.upper_body_bbox_from_hands()` derives upper-body crop from hand_det v2 bboxes. Visual confirmation in `data/probe/pose_*_verify_*/compare.png`. |
| **Face detector v0** | ✅ Session 15 — bbox detector only (no face landmarks yet) |
| Trajectory extraction (Phase 4.1) | ✅ Session 18 — `extract_trajectories_v2.py` with batched inference, `_detect_hands_batched` quality filter (raised threshold to 0.15, max_aspect_ratio=3.5; 0% degenerate bboxes in v7 vs 48% in v6) |
| Template fitter + threshold calibration (Phase 4.2-4.3) | ✅ Session 18 — templates_v9 at mean P=19.7% R=28.5%; **Mahalanobis matcher at architectural ceiling**, learned classifier substituted as the recognition head |
| **Sign classifier head v0 (TCN, 100D features)** | ✅ Session 17 — `runs/sign_classifier_v0/best.pt`, 960K params, **10.1% top-1 / 23.6% top-3**; structural improvement over template matcher |
| **Phase 4.7 transformer classifier A/B** | ✅ Session 19 — fair-comp (192/3/4, 975K) hit 11.1%, upsized (256/4/8, 2.24M) hit 10.9%. Both ≈ TCN baseline. **Temporal attention is NOT the bottleneck** — feature representation is. Phase 4.7 parked. |
| **Phase 4.9 data corpus expansion** | ✅ Session 19 — v4 manifest **12,752 clips, 80/80 signs covered, 79 of 80 ≥100**. Sources: sem_lex 7,657 + asl_citizen 2,979 + wlasl 772 + msasl 450 + lifeprint 235 + ytsearch_v2 560 + project_clean 99. ADR-0015 audit entries for all. Only `bad` still <100 at 91. |
| **Phase 4.6 HandshapeEncoder (handshape image embedding)** | ✅ Session 19 — `runs/handshape_v0d/best.pt`, 11.3M-param ResNet-style CNN trained via NT-Xent contrastive on 182K hand crops. Best val_loss=2.77 (3× improvement from random init). Cost $4.62 H100 (after 4 attempts; root causes: CUDAGraphs, /dev/shm, num_workers). |
| **Linear probe (encoder quality diagnostic)** | ✅ Session 19 — clip-pool top-1 **7.18%** (chance 1.27%, 5.7× above chance). Yellow-green signal; real handshape information present but single-frame insufficient. |
| **356D feature pipeline (kpts + 128D embed per hand + pose)** | ✅ Session 19 — `fit_templates._frame_to_features(with_embedding=True)`, `_mirror_features` extended, `trajectory_from_frames` auto-detects, `train_classifier` auto-detects feat dim |
| **v4 trajectory extract WITH encoder** | 🔥 Session 19 in flight — A100 maxspeed (cpu=32, decode_workers=24) on 12,752-clip v4 manifest. ~40% through, 0 errors, ETA ~15 min, projected cost ~$1.30 |
| **Final v4 classifier retrain on 356D** | ⏳ pending — gated on extract completion + explicit spend approval (~$0.40 A100, ~5-10 min) |
| **ONNX export script + 3 exported detectors** | ✅ Session 15 — `artifacts/onnx/{hand_detector,hand_landmarks,face_detector}_v0.onnx`, parity OK. HandshapeEncoder ONNX export still owed. |
| Model cards (hand_detector_v0, hand_landmarks_v0, face_detector_v0, pose_v0) | ✅ Session 15 + 18 |
| Live demo (`scripts/live_demo.py`) | ✅ Session 18 + 19 — pose-bbox aspect-ratio clamp + `PoseEMA` temporal smoothing added Session 19 |
| Browser inference (TS) — 5-component pipeline | ⚠️ partial — `lib/inference/sign_matcher.ts` started Session 18 but needs 356D update before Phase 5 ship |
| Phase 5 (pedagogy / live trainer UX) | ❌ not started |
| Phase 6 (avatar + TTS) | ❌ not started |
| Phase 7 (validation + bias eval + final polish) | ❌ not started |

---

## Modal spend cumulative (sessions 13-19)

Approximate: ~$50 across all training runs. Session 19 alone ~$7 (Phase 4.7 transformer A/B $0.50 + encoder training including 4 attempts $7.62 → only $4.62 produced a usable artifact + v4 extract in flight $1.30 + MSAsl Modal scrape attempt failed $0.05). See TODO.md Session 19 entry for detailed breakdown.

---

## Compliance notes — HaGRID ingestion (Session 16, added 2026-05-21, revised after HF-mirror switch)

HaGRID is approved for hand-detector training under ADR-0015. The compliance posture is **structural, not defensive** because we ingest via the `cj-mills/hagrid-sample-500k-384p` HuggingFace mirror, which **already excludes the disqualified fields**:

- **What's in our data**: `image`, `bboxes`, `labels`, `leading_hand`, `leading_conf`, `user_id`. Bboxes are human-drawn by Yandex.Toloka + ABC Elementary crowdworkers (4-stage pipeline; preserved by the mirror).
- **What's NOT in our data**: `hand_landmarks` (MediaPipe-generated in upstream) and `meta` (FairFace + MiVOLO-generated in upstream). The mirror omits these fields entirely — they cannot enter our pipeline because they do not exist in our data source.
- **Defense in depth**: `training/detectors/external_loaders/hagrid.py` still ships `_check_no_landmark_drift()`. It's vestigial under the current HF-mirror path but kept active so that if we ever switch sources back to upstream Sbercloud, the tripwire catches accidental schema widening.
- **Ingest entrypoint**: `training/modal_app.py::download_hagrid_from_hf`. Why HF mirror over upstream: Sbercloud Moscow throttles US-Modal at ~40 MiB/s; HF Cloudflare CDN delivers ~200–1000 MiB/s. 13.4 GB vs 119 GB.
- **Compliance rationale + provenance review**: `docs/data/external_datasets_audit.md` entry **I. HaGRID**.

## When in doubt

Read ADR 0011 first. If a doc in this repo conflicts with ADR 0011,
ADR 0011 wins and the doc is stale.

If `STATUS.md` (this file) conflicts with ADR 0011, ADR 0011 wins
and this file needs updating.
