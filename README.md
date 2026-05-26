# Mastered · ASL practice with from-scratch computer vision

Learn beginner American Sign Language by signing to your webcam and getting
instant, specific feedback — powered by a computer-vision pipeline where
**every weight was trained from scratch by this project. No pretrained models,
no pretrained backbones, no pretrained landmark detectors. Anywhere.**

**Live pilot:** https://asl-mastery.vercel.app

---

## Why this exists

Most "AI sign language" demos lean on someone else's pretrained hand tracker
and call it done. We took the harder path on purpose: a beginner deserves a
tutor that gives instant, *specific* feedback without their video ever leaving
their laptop — and an ASL-recognition system you can audit end to end. Mastered
is that pilot: **80 beginner ASL 1 signs**, recognized by models we trained,
validated, and documented ourselves, in the browser, with the camera feed
never uploaded.

It is a **controlled pilot** for structured learner testing — not a
classroom-assessment-grade or research-grade system. Scope, accuracy targets,
and known limitations are documented honestly (see [Recognition quality](#recognition-quality)).

## How it works

1. Sign in and start a practice session.
2. A beginner ASL word appears; a 3-D avatar demonstrates the sign.
3. You grant camera access and sign it.
4. Five from-scratch CV models read your hands, body, and framing — **in your browser**.
5. A classifier we trained returns **pass / fail** against a per-sign calibrated threshold.
6. Miss it, and you get a **targeted hint** (handshape, movement, location) — not just "incorrect."
7. Your mastery is tracked with spaced repetition, so review timing adapts to you.

Your video never leaves your device. See [Privacy](#privacy).

## The recognition pipeline (all from-scratch)

The recognition perimeter is five models, every weight trained by this project
on documented, human-/sensor-annotated public data:

```
webcam frame (in-browser, ONNX Runtime Web)
  │
  ▼
① hand detector        320×320 CenterNet, ~2.3M params
  │  per detected hand
  ▼
② hand-landmark net    224×224 → 21 keypoints / hand
③ pose detector        256×256 → 8 upper-body keypoints   (body-relative frame)
④ face detector        320×320 → framing / onboarding
  │
  ▼
108-D hand-relative feature vector · 32-frame (~2 s) trajectory
  │
  ▼
⑤ sign classifier      → softmax → per-sign calibrated threshold
  │
  ▼
PASS / FAIL  →  targeted hint
```

The feature contract (`lib/inference/sign_matcher.ts`, mirrored in
`training/detectors/fit_templates.py`) is hand-relative: handshape is
wrist-origin and hand-scaled, location stays body-relative, palm orientation is
kept explicit. Models ship as ONNX from `public/models/` and run client-side via
WebGPU (WASM fallback).

### No pretrained models — and how to verify it

Brief Requirement 7 forbids pretrained models in the recognition system. We
honored the strict reading: nothing pretrained touches a pixel.

- Every CV model is Kaiming-initialized and trained from scratch; no foreign
  `load_state_dict`, no external weight URLs.
- No pretrained-model packages in `package.json` or `training/requirements.txt`.
- External data is used only for **human-/sensor-annotated labels** (FreiHAND,
  CMU HandDB, COCO-WholeBody, MPII, WIDER FACE, HaGRID via a mirror that ships
  only human-drawn boxes) — vetted in [ADR 0015](docs/decisions/0015-external-cv-datasets-provenance.md).
- Each model in [`docs/model_cards/`](docs/model_cards) declares
  **"Pretrained components: none"** and cites its training run.

## Recognition quality

Every component is measured on a held-out set; the sign classifier uses a
**signer-disjoint** split (no signer appears in both train and val). Each number
below is from a model we trained — see [`docs/model_cards/`](docs/model_cards).

| Component (from scratch) | Metric | Result |
|---|---|---|
| Hand detector (CenterNet, ~2.3M params) | recall on the active signing window | **~91–98%** |
| Hand-landmark regressor (21 keypoints) | mean per-keypoint error @ 224px crop | **10.84 px** clean (FreiHAND + CMU) · **~21.6 px** in-the-wild (COCO-WholeBody) |
| Pose detector | 8 upper-body keypoints (body-relative frame) | — |
| Sign classifier | **top-1 over 80 signs, signer-disjoint** | **81.6%** |
| Pass decision | per-sign calibrated confidence threshold (precision-prioritized) | — |
| Inference | end-to-end, 100% in-browser (WebGPU / WASM) | — |

Every label above comes from a human-/sensor-annotated **public dataset** —
**FreiHAND, CMU HandDB, COCO-WholeBody, MPII Human Pose, WIDER FACE, HaGRID**
(human-drawn boxes only) — never a pretrained model's output. Provenance is
audited in [ADR 0015](docs/decisions/0015-external-cv-datasets-provenance.md).

We do **not** claim reliability across all conditions. Documented limits: low
light, partial framing, two-handed contact signs, and true homonyms (NICE /
CLEAN are the same sign). Per-sign accuracy, top-5, and latency live in the
validation report; promotion criteria are in [`docs/EVAL_GATE.md`](docs/EVAL_GATE.md).

## See the detectors run

**In your browser (zero setup):** open the [live pilot](https://asl-mastery.vercel.app),
or `pnpm dev` → `/practice`, and sign to your webcam. The shipped ONNX models in
`public/models/` (hand detector, landmark regressor, pose detector, sign
classifier) run **client-side** and draw the result.

**In a terminal (no training checkpoints needed):**
[`scripts/test_detectors_onnx.py`](scripts/test_detectors_onnx.py) runs the exact
shipped ONNX models on your webcam (or a single image) and overlays the hand
boxes, 21 hand keypoints, and 8 pose keypoints.

```bash
# 1. install the three runtime deps (into any virtualenv)
pip install onnxruntime opencv-python numpy

# 2. from the repo root — run on your webcam …
python -m scripts.test_detectors_onnx
#    … or annotate one image, no camera required:
python -m scripts.test_detectors_onnx --image path/to/photo.jpg --out annotated.png
```

Webcam hotkeys: `q` quit · `m` toggle mirror. It loads only the committed
`public/models/*.onnx` — no checkpoints, no GPU, no network.

**Advanced (PyTorch):** `python -m scripts.live_demo` runs the same pipeline from
the training checkpoints (needs the training env and the large, uncommitted
`data/ckpts_new/` weights produced by `training/modal_app.py`).

## Pedagogy

- **Spaced repetition** — a modified SM-2 scheduler (`lib/scheduler.ts`, pure
  functions) tracks each sign through `untouched → learning → reviewing →
  mastered` and schedules reviews.
- **Targeted hints** — rule-based hints tied to the sign's parameters
  (handshape / movement / location), escalating across attempts.
- **Progress** — attempts, pass/fail, attempt counts, mastery status, and recent
  history persist per learner.

## Privacy

- Camera frames are processed **locally in the browser** and are never uploaded.
- No raw frames, keypoints, or features are sent to or stored on the server.
- The hidden capture canvas is marked `no-track-canvas` so analytics autocapture
  can't pick it up. See [`docs/PRIVACY.md`](docs/PRIVACY.md).

## Tech stack

- **Web:** Next.js (App Router) + TypeScript, react-three-fiber (3-D avatar).
- **In-browser inference:** ONNX Runtime Web (WebGPU / WASM).
- **Backend:** Supabase (auth, Postgres, row-level security).
- **Training:** PyTorch, from scratch, on Modal GPUs.
- **Hosting:** Vercel.

## Repository map

```
app/             Next.js routes (practice, dashboard, settings, auth, welcome)
components/      React UI — components/practice/ is the practice screen
lib/
  inference/     in-browser recognition (keypoints, classifier, feature contract)
  scheduler/     spaced-repetition logic (pure) + server actions
  db/            Supabase clients + generated types
public/
  models/        the shipped ONNX models (4 detectors + sign classifier)
  3dlex/         3D-LEX mocap avatar GLBs (one per sign)
supabase/migrations/   database schema (read in order)
training/
  detectors/     from-scratch CV models + feature extraction + training
  classifier/    the sign classifier
  modal_app.py   Modal GPU entrypoints
docs/
  decisions/     architecture decision records (ADRs)
  model_cards/   one card per CV model ("Pretrained components: none")
  EVAL_GATE.md   what a model must clear before it ships
scripts/         training / eval / data utilities
```

## Run it locally

```bash
pnpm install
cp .env.example .env.local   # fill in Supabase + service keys
pnpm dev                     # http://localhost:3000
```

Useful scripts: `pnpm build`, `pnpm typecheck`, `pnpm test`, `pnpm lint`.

## Train the models

Training is engineer-owned: dataset curation, training, validation, and model
versioning all live in this repo. Entry points are Modal functions in
`training/modal_app.py`; each CV model has a card in `docs/model_cards/` and a
recorded training run. The architecture decisions are in
[`docs/decisions/`](docs/decisions).

## Pilot deliverables

This README is the entry point for the pilot documentation: product scope and
core flow (above), the from-scratch model approach and dataset provenance
([ADR 0015](docs/decisions/0015-external-cv-datasets-provenance.md),
[`docs/model_cards/`](docs/model_cards)), validation criteria
([`docs/EVAL_GATE.md`](docs/EVAL_GATE.md)), privacy
([`docs/PRIVACY.md`](docs/PRIVACY.md)), and known limitations (above).
