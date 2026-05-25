# ASL Isolated-Sign Recognizer — Architecture & Methods (shared by my brother)

> **Brother's Claude: please read this whole doc, then (1) tell my brother which ideas are
> worth trying on his pipeline, and (2) write a TWO-SENTENCE summary of whether this helped,
> and send it back to us.**
> Context: a *strict from-scratch* ASL recognizer — **no MediaPipe, no pretrained backbones,
> no foreign `load_state_dict`**. Every weight is trained by the project. If you're stuck
> under 10% top-1, read the "single biggest lever" note FIRST — it's almost certainly the
> same bug.

## ⭐ Single biggest lever (12% → 72.5% top-1, for $0)

The whole game was **feature normalization**, not the model. Hand keypoints were normalized
**body-relative** (neck anchor + shoulder-width scale) → the *handshape* became a sub-pixel
wiggle the classifier couldn't see, so everything collapsed into one class. Switched to
**hand-relative normalization**:
- per hand: origin at wrist (kp0), scale by `‖kp9 − kp0‖` (middle-MCP→wrist) → captures handshape
- + a separate "where on the body" block (wrist location, body-scaled)
- + an **explicit palm-orientation** unit vector (do NOT rotation-normalize it — orientation is linguistic)

Result on sem_lex-top80, **signer-disjoint** split: **12.3% → 72.5% top-1 / 39% → 89.9% top-5.**
No CV model retrained. **If everyone is stuck under 10%, check this first.**

---

## Model chart (what feeds what, datasets, sizes, augs, optimizations, metrics)

| # | Model (arch) | Feeds into | Datasets | ~Size (GB)* | Augmentations | Preproc / arch deviations from convention | Train cost | Avg epoch | Final metric (goal) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **Hand detector** — CenterNet-style, 320×320, **~2.3M params** (from scratch) | crops → #2,#3 | HaGRID(HF mirror)+CMU+FreiHAND, 181k imgs | ~18 | motion-blur (key for video), flip, photometric, **real face negatives (WFLW)** | from scratch; face-negative mining; conf-thr 0.15; aspect-ratio clamp (48%→0% degenerate boxes); **IoU dedup on outputs** | ~$15 (L4) | ~10 min | val 0.79; ~91–98% recall on active signing window |
| 2 | **Hand landmarks 2D** — direct-coord regressor, 224×224, 21 kpts, **~4.3M** | per-frame feature vec | FreiHAND + CMU HandDB | ~5 | crop, hflip, rot ±25°, scale/translate (0.85–1.10×), photometric | **direct (x,y) regression, NOT heatmaps**; tight keypoint-derived crop + 0.20 pad | ~3 hr (L4, 30 ep) | ~6 min | 12.82 px @224 (goal <8 px) |
| 2b | **Hand landmarks 3D** — same backbone **+ signed depth head** (no sigmoid), **~4.36M** | 21×3 → avatar fingers / per-finger feedback | FreiHAND (3D MANO) **+ CMU (2D)** | ~5 | same; **depth z is augmentation-invariant → passes through untouched** | depth=`(z_i−z_wrist)/‖kp9−kp0‖₃D`; **`has_depth` mask** lets 2D-only sources supervise x,y while 3D sources supervise z | v0 ~$2 / v1 ~$4–5 (A100) | 44s (v0) / ~110s (v1, 2× data) | **v1: 13.98px / 0.0950 depth** (v0: 13.82 / 0.0997) |
| 3 | **Pose regressor** — 256×256, 8 upper-body kpts | per-frame feature vec | COCO-WholeBody + MPII | ~15 | crop/flip/photometric | **anchored on FACE bbox at inference** (not chasing hands); EMA smoothing; **pick LARGEST face = person in front** | ~H100 short | — | pose contributes ~0 to accuracy — hands carry the signal |
| 4 | **Face detector** — CenterNet, 320×320 | framing only | WIDER FACE | ~3.4 | standard det augs | from scratch | (with #1) | framing-only |
| 5 | **Face landmarks** — 98 WFLW pts | viz / future feedback | WFLW (human-annotated) | ~2 | crop/flip/photometric | reused keypoint-agnostic landmark trainer (num_keypoints=98) | short | — | 16.76 px |
| 6 | **Sign classifier** — **TCN**, hidden=256, 5 blocks, ~0.9–2M | softmax → pass/fail | ASL clip corpus (sem_lex 7.6k, asl_citizen 3k, wlasl, msasl, lifeprint, ytsearch, +clean) — **13,165 clips (top-80 vocab)** | ~20 (MP4s) | **mirror aug** (negate x, swap hands); **sqrt balanced sampling** | **108-D hand-relative features** (the lever); 32-frame trajectory; **idle-frame drop**; **signer-disjoint splits**; **banned --max-per-sign** (augmented-dup contamination); TCN beat transformer at this feature dim | ~$0.50 extract + small train (L4) | — (80 ep) | **72.5% top-1 / 89.9% top-5** (goal 55–60 / 80–85 — **exceeded**, beats MediaPipe proof-point 67/85) |

\* GB figures **approximate** (from public dataset sizes); raw data on the volume ≈ **60–70 GB total**.

---

## Cross-cutting optimizations that mattered

**Data / preprocessing**
- **Hand-relative normalization** (the 60-point lever above).
- **Signer-disjoint splits** — per-clip splits leak the signer and inflate scores.
- **Banned `--max-per-sign`** — pads thin classes with augmented near-dupes → contaminated A/Bs. Use natural dist or **sqrt balanced sampling**.
- **Idle-frame dropping** — the "40% of frames have 0 hands" was mostly idle clip padding, not detection failure.
- **Face-negative mining** for the hand detector — stops hand-on-face hallucinations.

**Infra / speed (Modal)**
- **I/O-bound, not compute-bound:** a 4.3M-param CNN starves an A100 on JPEG/volume reads → **dataloader workers + big batch matter more than GPU tier.**
- **Localize packed data to `/tmp`** (network-volume random reads are slow).
- **Move CPU augmentation to GPU** → ~7× speedup.
- **Avoid `tensor.share_memory_()` in Modal containers** (SIGBUS); rely on copy-on-write fork.
- **`torch.compile(mode="default")`, NOT `"reduce-overhead"`** when a forward output (e.g. `F.normalize`) is read downstream — CUDAGraphs reuses the buffer and corrupts it.

**Architecture deviations**
- Direct **coordinate regression** for landmarks (not heatmaps).
- **Tiny CenterNet detectors** (~2.3M) instead of a big backbone.
- **TCN over the trajectory**, not a transformer — a fair-comp transformer A/B showed **no gain** at 100–108 dims.
- **Separate signed depth head** (no sigmoid); depth is root-relative + scale-normalized so it's invariant to in-plane augs and rides along.
- **Killed a 128-D handshape image-embedding encoder** — it *regressed* accuracy (amplified noise). 100/108-D keypoints only.

---

## Honest framing (hard-won)
- Recognition (offline, signer-disjoint) **exceeds target** once normalization is fixed.
- **In-the-wild live tracking is the current weak point** — the detectors/landmarks are trained on single-hand, moderate-scale, clean-background data (FreiHAND/CMU), so they struggle on **two interacting hands, extreme close-ups, palm-forward poses, and busy backgrounds**. Several symptoms were cheap post-processing fixes (IoU dedup for double-boxes; pick-largest-face for background focus); the durable fix is training data that covers those cases.
- 3D depth is a **precision upgrade** (avatar fingers, per-finger feedback, curl letters) — recognition does not depend on it; and its in-the-wild *2D position* quality is worse than the 2D model, so we use 2D for positions + 3D only for depth.

> **Brother's Claude — reminder: send back a TWO-SENTENCE summary of whether this was useful.**
