# ASL Vocabulary Trainer — Strict From-Scratch Roadmap

## Operating Rules (Set By You, Held Hard)

1. **No pretrained models anywhere in the CV / recognition stack.** No MediaPipe, no pretrained hand or pose detectors, no pretrained feature extractors, no pretrained backbones, no pretrained classifiers, no pretrained anything that touches a pixel. Every weight in every model that processes the camera feed is trained by you for this project.
2. **No stubs in the recognition path.** No synthetic landmark trajectories, no hand-authored coordinate fallbacks, no "we'll swap it later" — the detector is built first and the pedagogy layer cannot be developed until real landmarks are flowing from your real model.
3. **Local only, indefinitely.** Nothing on any server. Nothing on any URL. The app runs on localhost. Data lives on your disk. No deployment of any kind until you explicitly say otherwise.
4. **Strict-but-scoped reading of Requirement 7.** Non-CV libraries (PyTorch, Three.js, Web Speech API, SQLite, React, etc.) are permitted per the brief's explicit allowance for "programming frameworks, data processing libraries, and machine learning libraries." The from-scratch rule governs the recognition system, not the surrounding infrastructure. If you ever feel this drift, re-read Requirement 7 and stop.

## Timeline Reality

- **Total realistic timeline:** 11-13 months
- **First real detector output (one hand, indoor lighting, your own face):** End of Month 4
- **All four CV components working (hands, pose, face, sign classifier):** End of Month 7
- **75-sign vocabulary working end-to-end with pedagogy layer:** Month 10
- **Tier 3 polish (avatar, TTS, mastery, validation report):** Month 12-13

If this number is too long, the answer isn't to relax the constraints — it's to reduce the vocabulary count (50 instead of 75) or accept a simpler pedagogy layer. The constraints are load-bearing.

## Project Structure on Your Local Machine

```
asl-trainer/
├── /ml                    # All training code. PyTorch.
│   ├── /datasets          # Dataset classes for hand kpts, pose kpts, sign clips
│   ├── /models            # Architecture definitions: hand_detector, hand_landmarks, pose, face, sign_classifier
│   ├── /training          # Training loops, loss functions, schedulers
│   ├── /eval              # Validation metrics, confusion matrices, bias evaluation
│   └── /export            # PyTorch → ONNX → ONNX.js conversion for browser
├── /data                  # Datasets and labeled artifacts. Gitignored.
│   ├── /raw_clips         # Your Modal-cleaned 12K clips
│   ├── /labeled_frames    # Frames you've keypoint-labeled
│   ├── /templates         # Reference sign trajectories
│   └── /splits            # train/val/test manifests
├── /web                   # The browser app. React + Vite. Localhost only.
│   ├── /detector          # ONNX.js inference wrapper
│   ├── /pedagogy          # Template matching, hint logic, mastery model
│   ├── /ui                # React components
│   └── /storage           # Local IndexedDB / SQLite layer
├── /labeling              # Internal tool for keypoint labeling
├── /docs                  # ADRs, model cards, validation reports
└── /scripts               # Modal jobs, batch inference, dataset prep
```

Everything lives on your laptop. The only thing that touches a server is Modal for training compute — and Modal is processing *your* data, not running anyone else's pretrained model.

---

## Phase 0 — Foundation (Weeks 1-2)

**Goal:** The scaffolding the entire project hangs off. Slower than typical, because nothing about the next year is recoverable if Phase 0 is sloppy.

### Week 1

- Repo, monorepo structure, gitignored data directories, ADR template committed
- Local Python environment with PyTorch, CUDA verified
- Modal account configured, GPU access verified, your existing data cleaning job confirmed complete
- ADR 0011: "Pivot from end-to-end RGB classifier to landmarks + templates" — supersedes ADRs 0001 and 0010
- ADR 0012: "Strict from-scratch CV constraint and scoping" — documents the strict-but-scoped reading of Requirement 7, lists every CV component that must be from scratch, lists every non-CV dependency that's permitted. This is the most important ADR you'll write. Show it to yourself every time you're tempted to drift.
- Vocabulary frozen: pick the final 75 signs from your existing slice1b_vocabulary.json, prioritizing ASL 1 beginner concepts. Document selection criteria.

### Week 2

- Local-only React + Vite skeleton: opens camera, displays feed, displays a "no detector loaded" placeholder. No detection yet.
- Local storage layer: SQLite via better-sqlite3 (Node) or IndexedDB (browser). Schema for: users, signs, attempts, sessions. Document the schema in `/docs/data_model.md`.
- Decision gate: you should be able to start the dev server with `npm run dev`, open localhost, and see your webcam feed in a styled UI. No detection. Just the shell. Confirm this works before moving to Phase 1.

---

## Phase 1 — Hand Detector (Weeks 3-8)

**Goal:** Train a from-scratch model that takes a webcam frame and outputs bounding boxes around visible hands. This is the gateway component — everything downstream depends on it.

### Why hand detection first

Landmark regression depends on having a cropped hand image to operate on. You can't train the landmark model meaningfully without a hand detector running upstream. Build the spine vertebra by vertebra.

### Week 3 — Data acquisition strategy for hand detection

You need images labeled with hand bounding boxes. Sources, in order of leverage:

1. **Sample frames from your 12K-clip corpus.** Run ffmpeg to extract 1 frame per second from each clip. That gives you ~100,000+ frames featuring real signing hands. This is your image pool.
2. **Annotate hand bounding boxes** using CVAT or LabelStudio. Target: ~3,000-5,000 labeled frames for hand detection. Box per visible hand, including partial hands at frame edges.
3. **Diversify:** also record your own footage in different lighting, distances, backgrounds. Label those too. Aim for 500-1000 self-recorded frames to cover conditions your 12K corpus doesn't.

Labeling target for Week 3-4: ~3,000 frames. At 15 seconds per frame for bbox-only labeling, that's ~12 hours of labeling work. Spread across the week alongside model architecture design.

### Week 4 — Architecture and first training run

Architecture: **single-stage anchor-free detector, custom backbone, trained from scratch.**

- Backbone: ~6 conv blocks, residual connections, batch norm. Target 1-3M parameters. Built in PyTorch from `nn.Module` primitives.
- Detection head: CenterNet-style — heatmap of hand centers + size regression. Simpler than anchor-based detectors and works well for the 1-2 hands typical of signing.
- Loss: focal loss on heatmap + L1 on size regression
- Input resolution: 320×320 (gives you both speed and enough resolution for distant hands)

Train on Modal with your labeled data. First runs will be bad. That's normal. Document the architecture and first run results in `/docs/model_cards/hand_detector_v0.md`.

### Week 5-6 — Iteration

- Augmentation: random crops, brightness/contrast/saturation, motion blur, color jitter, horizontal flips (carefully — flipping changes which hand is which)
- Hard negative mining if false positives are an issue
- Validation set: 500 held-out frames, with signers/conditions not in training
- Metric: AP@IoU=0.5 for hand detection. Target: >0.85 on held-out validation.

If you're not at 0.85 by end of Week 6, options: label another 1000-2000 frames focused on failure modes, scale up the backbone slightly, or accept lower accuracy and document it.

### Week 7 — Browser inference

Convert PyTorch → ONNX → test in ONNX Runtime Web. Benchmark on your laptop and on whatever lowest-end machine you can find. Target: 30+ FPS at 320×320 input, <200ms cold start.

If too slow: quantize to int8, prune low-magnitude weights, or distill to a smaller model.

### Week 8 — Integration and model card

- Wire `HandDetector` into the React app. Webcam frame → bounding boxes → render boxes in green on overlay canvas.
- Write the model card: architecture, parameter count, training data (with provenance), training process, validation metrics, known failure modes, bias notes (which subgroups underperform, what you'd do about it later).
- ADR 0013: "Hand detector v1 — architecture and acceptance criteria."

**Decision gate end of Phase 1:** Open localhost, see your webcam, see green bounding boxes drawn around your hands as you move them. Real model, real weights, all yours. If this works, you have your first piece of provable from-scratch CV.

---

## Phase 2 — Hand Landmark Detector (Weeks 9-16)

**Goal:** Train a model that takes a cropped hand image and outputs 21 keypoints. This is the hardest single component of the project. Budget accordingly.

### Week 9-10 — The labeling problem

Keypoint labeling is the cliff. You need ~5,000-10,000 hand-cropped images with all 21 keypoints labeled, per hand orientation/pose diversity.

**Realistic labeling rate:** ~45-60 seconds per hand at 21 keypoints with practice. So 8,000 labels = ~120 hours of labeling work. **Don't underestimate this. It is the single biggest schedule risk in the project.**

Mitigations:

- **Bootstrapping:** Label 500 frames manually. Train an initial weak landmark model. Use it to pre-label the next batch; you correct rather than label from scratch. Each generation reduces per-frame labeling time by ~40%.
- **Sampling strategy:** Don't label random frames. Label frames that maximize diversity — different hand orientations, different signers, different lighting, different distances. Use the hand detector you just trained to crop hands from your 12K corpus and sample diversely.
- **Don't strive for perfect labels.** Reasonable consistency beats perfect accuracy. Document your labeling rubric and stick to it.

### Week 11-12 — Architecture

- Input: 224×224 cropped hand image (output of hand detector)
- Architecture: small CNN backbone (5-6 conv blocks, ~1-2M parameters) + regression head that outputs 42 numbers (21 × 2 coordinates) normalized to [0, 1]
- Loss: L1 or smooth L1 on normalized coordinates, with optional visibility prediction per keypoint
- Train from scratch on Modal

Avoid heatmap regression for v1. Direct coordinate regression is simpler, faster, and works fine for hand keypoints when training data is sufficient. Heatmap regression is a v2 upgrade if accuracy isn't enough.

### Week 13-14 — Iteration

- Augmentation: random rotations (carefully — keypoints rotate too), scale variations, photometric augmentation
- Temporal smoothing: in the browser, apply exponential moving average across frames to reduce jitter (post-processing, not part of the model)
- Hard cases: occluded keypoints, self-occlusion, motion blur. Identify systematic failures and label more training data targeted at those.
- Validation metric: mean per-keypoint pixel error on held-out hands. Target: <8 pixels at 224×224 resolution.

### Week 15 — Browser inference

ONNX conversion. Benchmark end-to-end: hand detector + landmark detector combined. Target: 20+ FPS on your dev laptop. If pipeline lags, profile and optimize.

### Week 16 — Integration

- Wire into the app. Webcam → hand detector → for each detected hand, crop → landmark detector → 21 keypoints per hand
- Render green skeleton overlay on the camera feed: dots at each keypoint, lines connecting them in canonical hand topology
- This is the first time you have something that looks like the screenshot you originally showed me. Take a screenshot for the portfolio.
- Model card + ADR 0014

**Decision gate end of Phase 2:** Real-time hand skeleton overlay on your webcam feed, all from-scratch. The "exactly this" from your earlier message is now visually achievable.

---

## Phase 3 — Pose and Face Detection (Weeks 17-20)

**Goal:** Two smaller from-scratch models for upper body pose (shoulders, elbows, wrists) and face detection (for framing bracket).

These are simpler than hands because: pose has 6-8 keypoints, not 21, and is less articulated; face detection is single-class with high contrast features.

### Week 17 — Pose detector

- Same general approach as hand landmarks: small CNN, direct coordinate regression
- Keypoints: nose, neck, left shoulder, right shoulder, left elbow, right elbow, left wrist, right wrist (8 keypoints)
- Labeling: pose annotations on ~2,000 frames. Faster than hand keypoints — bigger features, less articulation. ~15-20 seconds per frame.
- Why we need it: signs are located in 3D space relative to the body (chin, chest, forehead, etc.). Pose gives us the reference frame for "is the hand near the chin or near the chest." Without this, "location" hints are impossible.

### Week 18 — Face detector

- Simple single-class detector. Architecture: tiny CNN, single anchor or center heatmap.
- Labeling: face bounding boxes on ~1,500 frames. Fast — ~10 seconds per frame.
- Why we need it: the framing bracket in the original screenshot is a UI affordance for the learner ("you're in frame, you're at the right distance"). It also enables the onboarding flow's framing calibration.

### Week 19-20 — Integration + multi-task experiments (optional stretch)

- Wire pose and face into the app. Camera feed now shows hand skeleton + pose stick figure + face bracket.
- Optional: try training a shared-backbone multi-task model that predicts hand bboxes + pose + face from one forward pass. Faster inference, possibly better representations. If it doesn't beat the separate models, scrap it. Document either way.

**Decision gate end of Phase 3:** All four CV components working from scratch and integrated. You now have everything you need to start building the pedagogy layer.

---

## Phase 4 — Sign Recognition (Weeks 21-26)

**Goal:** Given a landmark trajectory (hand + pose over a 1-3 second window), output a sign label with calibrated confidence.

### Week 21 — Run all detectors across the corpus

Run hand + pose detectors across your full 12K-clip corpus. Output: per-clip JSON of landmark trajectories. This is your sign classifier dataset.

Storage: ~12K clips × ~3 seconds × 15 FPS × (21 hand keypoints × 2 hands + 8 pose keypoints) × 2 floats ≈ tens of MB of trajectory data. Trivial compared to the video.

### Week 22 — Template-based recognition (the NVIDIA approach)

For each of the 75 signs, build a distribution-of-templates from the clips of that sign:

- Cluster landmark trajectories per sign (most signs cluster cleanly; some have legitimate variants)
- Compute mean trajectory + per-keypoint per-timestep variance
- Store as the "template" for that sign

At inference time: learner's landmark trajectory → compute Mahalanobis distance to each sign's template distribution → softmax over distances → confidence per sign.

### Week 23 — Per-sign threshold calibration

For each sign, compute:

- Positive distribution: similarity scores when correct clips of that sign match against its own template
- Negative distribution: similarity scores when clips of other signs match against this template

Pick per-sign thresholds that prioritize precision (false-pass is worse than false-fail in a learning context — you don't want learners thinking they got it right when they didn't).

Document each per-sign threshold and the data behind it. This is the artifact Requirement 9 demands.

### Week 24-25 — Optional: train a learned sign classifier head

If templates aren't accurate enough, add a learned classifier on top of the landmark trajectories. Small temporal model — TCN or 1D conv stack — operating on the landmark sequence. Trained from scratch on your 12K clips with their class labels.

This is the only part of the project where you'd train a model on the full corpus the way v3.0 was supposed to. Difference: input is now landmark sequences, not raw RGB. Vastly more sample-efficient.

Evaluate both: templates-only vs. templates + learned head. Ship whichever is better, document why.

### Week 26 — Buffer + model card

ADR 0015: sign recognition approach. Model card for the recognition system.

**Decision gate end of Phase 4:** End-to-end recognition working. You can sign at the webcam and get a sign label with a confidence score back. Still no pedagogy, just classification.

---

## Phase 5 — Pedagogy Layer (Weeks 27-32)

**Goal:** Wrap the recognition system in a learning loop. Templates → pass/fail → hints → retry → mastery.

### Week 27 — Pass/fail logic

Wire up the calibrated thresholds. For each attempt: similarity to target sign's template + similarity to all other signs' templates. Pass if target is highest and above threshold. Fail otherwise. Uncertain ("close but below threshold") triggers retry without penalty.

### Week 28 — Diagnostic feature extraction

When an attempt fails, decompose the trajectory similarity into the four ASL parameters from Stokoe linguistics (also what NVIDIA uses):

1. **Handshape** — finger configuration at key moments. Compare to template's handshape signature.
2. **Location** — where the hand is relative to body landmarks (head, chin, chest, neutral space). Use pose keypoints as the reference frame.
3. **Movement** — trajectory shape (path through space, direction, curvature). Compare against template's motion primitive.
4. **Palm orientation** — hand normal direction. Inferred from keypoint geometry.

Output per attempt: similarity score per ASL parameter. The worst parameter is the hint target.

### Week 29 — Confusion-mined hint logic

Run your recognition system across the full corpus to generate a confusion matrix. For each (target sign, confusable sign) pair, identify which of the four parameters most distinguishes them.

Build the hint template library:

- "Your hand shape is close — try [specific shape change]."
- "Try signing closer to [body location]."
- "The motion should go [direction]."
- "Turn your palm to face [direction]."
- Specific confusion-aware hints: "Your attempt looks like [other sign]. The difference is [specific parameter change]."

These are real, defensible, pedagogically grounded. Each hint can be traced back to a specific feature deviation in the model output.

### Week 30 — Mastery model

State machine per (user, sign): `unseen → learning → practicing → mastered → review`. Transition rules grounded in mastery learning theory. Document the criteria with citations.

### Week 31 — Adaptive scheduler

Session-level sign selection algorithm. Priority order: signs in learning state, signs in review due to spacing, signs in practicing state nearing mastery. Spaced repetition kicks in for mastered signs at increasing intervals.

### Week 32 — Mastery dashboard

Per-user view: signs mastered, learning queue, review queue, recent attempts. **No streak counter, no time-spent metric, no engagement framing.** This is your stated value system on screen.

**Decision gate end of Phase 5:** Full learning loop working. The brief's pass/fail + targeted hints + saved progress + mastery tracking is all functional. You have a real Tier 2 pilot.

---

## Phase 6 — Avatar, TTS, Multimodal Feedback (Weeks 33-40)

**Goal:** The full vision — avatar demonstrates, animates corrections, speaks warm hints, subtitles render the text.

### Week 33-34 — Avatar pipeline

- Ready Player Me avatar or Mixamo-rigged free model (these are 3D assets, not pretrained ML models — allowed)
- Three.js scene: avatar centered on the left side, soft background, smooth idle
- Camera, lighting, materials, neutral palette

### Week 35-36 — Sign animations

Two approaches:
1. **Procedural from your templates:** Map recorded landmark trajectories onto avatar rig via inverse kinematics. Faster but jankier.
2. **Hand-keyed in Blender:** Polish the top 20 most-common signs by hand. Slower but cleaner.

Combine: procedural for all 75, hand-polished for the 20 most-frequent.

### Week 37 — TTS integration

Web Speech API for v1. SpeechSynthesisUtterance from hint text. Free, browser-native, runs locally. No API keys, no network calls.

Pick the warmest available voice. Test with different rates and pitches. Save user preferences locally.

### Week 38 — Correction animation

When learner fails on parameter X, avatar:
- Re-plays canonical sign at half speed
- Highlights the failed parameter (e.g., glow on the hand if handshape was wrong)
- Optionally: animates the corrected version of the learner's attempt, side by side with canonical

This is the trickiest piece. Budget the full week.

### Week 39 — Subtitle bubble + multimodal sync

Subtitle appears below avatar synchronized to TTS. High contrast, large font, dismissible. Stays 2 seconds after TTS ends.

### Week 40 — Polish and integration testing

Full multimodal experience working: target sign → avatar demonstrates → learner attempts → camera tracks hands → pass or fail → on fail, avatar speaks warm hint + subtitle + correction animation → retry.

**Decision gate end of Phase 6:** The full vision is on screen. Locally. On localhost. Not deployed.

---

## Phase 7 — Validation, Documentation, Final Polish (Weeks 41-end)

**Goal:** The artifacts Patrick and Frank will read.

### Validation report

- Per-sign accuracy across the held-out validation set
- Confusion matrix (visualization included)
- False-pass rate, false-fail rate, broken down by sign
- Performance by lighting, distance, signer
- Known limitations document

### Bias evaluation

The brief lists "research-grade bias analysis" as out of scope, but it doesn't forbid pilot-level bias evaluation, and NVIDIA's model card explicitly reports "None" for bias mitigation. Doing even a modest bias analysis differentiates you from the reference.

Construct a small test set with deliberate diversity (skin tone, hand size, lighting). Run pass/fail. Report per-subgroup accuracy. Honest about gaps.

### Pilot documentation

Per Requirement 15: product scope, model approach, dataset approach, validation criteria, privacy assumptions, known limitations, and **evidence that pretrained models were not used.** This last one is the easiest in your case: every weight has a training run on Modal, every dataset has provenance, every model card documents the from-scratch architecture.

### Onboarding and edge cases

- First-run flow: camera permission, framing check, lighting check
- Camera denied / unavailable handling per Requirement 4
- Accessibility audit: keyboard navigation, screen reader labels, contrast ratios

### Final ADR consolidation

Read every ADR you've written from 0011 onward. Make sure they tell a coherent story. Make sure decisions you superseded are clearly marked as superseded. Make sure future-you (or Patrick reading this) can reconstruct your reasoning.

---

## What Happens If You Get Stuck

The honest answer: you will, probably more than once. The from-scratch landmark detector is the most likely place. If you hit a wall:

1. **Don't add a pretrained component to escape.** That's the constraint failure mode that ruins the entire project's story.
2. **Cut scope instead.** Drop from 75 to 50 signs. Simplify the architecture. Reduce keypoint count. Accept lower validation accuracy and document it as a known limitation per Requirement 8 — the brief explicitly says the pilot doesn't need to be classroom-grade.
3. **Write the wall as an ADR.** Document what didn't work, what you tried, what you'd do with more time. This becomes a *credibility artifact* in the interview, not a weakness — it shows you encountered a real problem and made an honest engineering call.

The brief itself gives you cover for limitations: Requirement 8 demands you document "known limitations and failure cases," not that you eliminate them. Section 4 explicitly says the pilot doesn't need to be classroom-grade. Use that latitude honestly.

---

## The Single Most Important Thing in This Document

If you only remember one thing from this roadmap, make it this:

**The hardest constraint to hold isn't technical — it's the constraint to not drift.**

In Month 4, when the hand detector isn't converging and MediaPipe is sitting right there, your hand will move toward `npm install @mediapipe/hands` and you'll feel a story forming in your head about how you'll "just use it for now and replace later."

Don't.

The entire value of this project to Superbuilders is the strict from-scratch story. The minute you compromise it, you've built a worse version of NVIDIA Signs instead of a from-scratch pilot. The constraint is the moat.

Read ADR 0012 (the strict-but-scoped constraint ADR) on the bad days. That's literally why you wrote it.

---

## Friday Question

Same as before:

> "If I had to show this to Patrick and Frank Monday morning, what would I show them, and what would the story be?"

For the first 4 months, the answer is "the detector training pipeline and the labeled dataset I'm building, plus the model cards for what I've trained so far." That's not nothing. That's the work. The screenshot-quality product comes in Month 10. The story of how you built it comes from Month 1.
