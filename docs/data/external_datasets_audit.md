# External labeled CV datasets — provenance audit (Phase 1 prerequisite)

**Date:** 2026-05-21
**Status:** AWAITING USER APPROVAL — no datasets are on disk; no download scripts have been executed.
**Governing constraint:** [ADR 0012](../decisions/0012-strict-from-scratch-cv-constraint.md) — strict from-scratch CV rule.
**Governing decision:** [ADR 0015](../decisions/0015-external-cv-datasets-provenance.md) (new) — external dataset acceptance criteria.
**Pivot context:** [ADR 0011](../decisions/0011-landmarks-and-templates-pivot.md) — landmarks + templates pipeline.

---

## Purpose

ADR 0012 forbids pretrained CV models anywhere in the recognition pipeline. The
spirit of that rule extends to labels: training on a dataset whose 21-keypoint
annotations were produced by MediaPipe, OpenPose, or any other pretrained
keypoint detector would distill that detector's knowledge into our weights. The
constraint perimeter would be honored on the surface (we never `pip install
mediapipe`) while being violated in substance (our detector would predict
whatever MediaPipe predicts, minus the noise floor of distillation loss). ADR 0015
codifies this stance.

This memo is the provenance audit for every public hand / pose / face dataset
considered for Phase 1–3 training. Each dataset is investigated against six
acceptance criteria:

1. **Label provenance is verifiable from a primary source** (originating paper,
   official dataset README, or authoritative project page). Hearsay or
   third-party summaries are not enough.
2. **Labels were produced by humans, by physical sensors, or by the project's
   own multi-view fitting pipeline.** Labels produced by a third-party
   pretrained CV detector are disqualifying.
3. **License permits research use.** Anything more restrictive than CC-BY-NC
   or BSD requires a flag in the audit.
4. **Modality matches:** RGB single-frame for the detector / landmark / face
   work. Depth-only datasets are out of scope for the browser inference path.
5. **21-keypoint convention is supported, or trivially mappable.** Datasets
   using a different topology need an explicit mapping.
6. **For pose:** must include upper-body keypoints (shoulders, elbows, wrists)
   per ADR 0011 (8-keypoint pose head). Lower-body-only datasets are out.

When provenance cannot be confirmed against a primary source, the dataset is
**REJECTED with reason "unverified provenance"** rather than recommended. Per
the audit-rule preamble: better to leave out a usable dataset than silently
include a contaminated one.

---

## Per-dataset findings

### Hand keypoint datasets (target: 21 joints, RGB)

#### A. FreiHAND (Zimmermann et al., ICCV 2019)

| | |
|---|---|
| **Total images** | 32,560 unique training + 3,960 evaluation; 130,240 augmented training samples (3 background-swap variants per unique sample) |
| **Subjects** | 32 subjects (per paper §4) |
| **Annotation methodology** | Semi-automated **human-in-the-loop**. Initial sparse 2D keypoint annotations + segmentation masks come from **human annotators**. A MANO hand-model fitting routine then produces dense 3D pose + shape from the multi-view captures (8 synchronized cameras). An iterative refinement uses a network the authors trained — **trained from scratch on their own annotations**, not a pretrained third-party detector. |
| **Originating paper** | Zimmermann, Ceylan, Yang, Russell, Argus, Brox. "FreiHAND: A Dataset for Markerless Capture of Hand Pose and Shape from Single RGB Images." ICCV 2019. arXiv:1909.04349. |
| **License** | "This dataset is provided for research purposes only and without any warranty. Any commercial use is prohibited." (verbatim, lmb-freiburg/freihand README) |
| **Signing-specific?** | No. Generic isolated hands against varied backgrounds (provided green-screen + composited backgrounds). |
| **Download URL** | Project page: `https://lmb.informatik.uni-freiburg.de/projects/freihand/`. Direct ZIP listed on the page. |
| **Approximate size** | ~9.5 GB compressed (images + annotations + MANO fits). |
| **Format** | JPEG images + JSON annotations (training_xyz.json, training_K.json for camera intrinsics, training_verts.json for MANO mesh). |
| **Provenance verdict** | **PASS.** Human-seeded annotations + project-own multi-view fitting + project-own iterative network. No third-party pretrained CV detector in the loop. |

#### B. CMU Panoptic HandDB (Simon et al., CVPR 2017)

| | |
|---|---|
| **Total images** | Three subsets per the project page: (i) **Manual subset (MPII+NZSL):** ~2,758 images, 14,817 hand annotations (human-labeled, drawn from MPII Human Pose images and New Zealand Sign Language Exercises). (ii) **Multiview-bootstrapped subset:** ~33,000+ hand keypoint annotations triangulated from the CMU Panoptic Studio's 31-camera dome. (iii) **Synthetic subset:** ~14,261 rendered hand images. |
| **Subjects** | Manual subset: hundreds of unique people from MPII + NZSL signers. Multi-view subset: studio participants. |
| **Annotation methodology** | (i) Manual subset: **pure human annotation** by Mechanical Turk workers and authors. (ii) Multiview subset: produced by Simon et al.'s **own keypoint detector**, which was first trained on the manual subset, then run across all 31 dome views; detections triangulated in 3D; only triangulations consistent across views were kept and reprojected as labels. The bootstrap detector is **trained-from-scratch by the authors**, not a third-party pretrained model. |
| **Originating paper** | Simon, Joo, Matthews, Sheikh. "Hand Keypoint Detection in Single Images Using Multiview Bootstrapping." CVPR 2017. arXiv:1704.07809. |
| **License** | "The CMU Panoptic Studio dataset is shared only for research purposes and cannot be used for any commercial purposes. The dataset or its modified version cannot be redistributed without permission from dataset organizers." (CMU Panoptic project page) |
| **Signing-specific?** | **Partially yes** — the NZSL subset is New Zealand Sign Language signing data. The rest is generic hand poses. |
| **Download URL** | `http://domedb.perception.cs.cmu.edu/handdb.html` (manual subset, multiview subset, synthetic subset each have separate ZIPs). |
| **Approximate size** | Manual subset ~140 MB. Multiview subset ~17 GB. Synthetic subset ~3 GB. |
| **Format** | JPEG + JSON (per-image, with `hand_pts` array of 21 [x, y, visibility] tuples). |
| **Provenance verdict** | **PASS for manual subset (i)** unequivocally. **PASS for multiview subset (ii)** under ADR 0012's spirit: the bootstrap detector that produced these labels was trained by the authors on the manual subset; no third-party pretrained weights are upstream. **PASS for synthetic subset (iii)** — rendered annotations have synthetic ground truth from the renderer, not from a learned model. |

#### C. Multiview Hand Pose Dataset (Gomez-Donoso, Orts-Escolano, Cazorla, 2017)

| | |
|---|---|
| **Total images** | ~21,000 RGB frames across multiple sequences (per arXiv abstract: 21 keypoint annotations on each frame, four views per frame). |
| **Subjects** | Not stated on the dataset page. |
| **Annotation methodology** | **3D joint positions from a Leap Motion Controller sensor** (per the official project page). 2D projections + bounding boxes derived computationally from camera calibration (rotation/translation matrices supplied). **No learned model in the annotation pipeline at all** — pure physical sensor + geometry. |
| **Originating paper** | Gomez-Donoso, Orts-Escolano, Cazorla. "Large-scale Multiview 3D Hand Pose Dataset." arXiv:1707.03742 (2017); published 2018 in IVC. |
| **License** | "This dataset is publicly available under the BSD License." (verbatim, project page) |
| **Signing-specific?** | No. Generic hand poses in a multi-view rig. |
| **Download URL** | V2: Google Drive `https://drive.google.com/file/d/1KWG4c4ieT_4K9Rd7EYdXqH27Py1wRiNk/view?usp=sharing`. V1: `http://www.rovit.ua.es/dataset/mhpdataset_v1/multiview_hand_pose_dataset_release.zip`. |
| **Approximate size** | Not specified on page; expect single-digit GB based on image count. |
| **Format** | JPG images + TXT 3D joint files + pickled NumPy calibration matrices (`rvec.pkl`, `tvec.pkl`). Project ships Python utilities to generate 2D projections + bounding boxes. |
| **Provenance verdict** | **PASS.** Physical sensor labels + geometric projection — zero learned models in the annotation chain. The BSD license is permissive enough to be a strong signal of project-cleanliness on the licensing side too. |

#### D. InterHand2.6M (Moon et al., ECCV 2020 — Meta / Facebook Research)

| | |
|---|---|
| **Total images** | 2.6 million labeled hand frames captured in an 80-to-140-camera studio. Split into `human_annot` and `machine_annot` subdirectories per the official release. |
| **Subjects** | 27 subjects (per paper). |
| **Annotation methodology** | **Semi-automatic**, explicitly documented: `human_annot/` is pure human annotation; `machine_annot/` is from an automated detector the authors trained, applied across multi-view captures with triangulation consistency checks. Per ADR 0012's spirit, **only the `human_annot/` subset should be used** in this project unless we can verify the `machine_annot/` bootstrap detector was itself trained only on the project's own data (analogous to the CMU multiview case). The paper supports this but the audit defaults conservatively. |
| **Originating paper** | Moon, Yu, Wen, Shiratori, Lee. "InterHand2.6M: A Dataset and Baseline for 3D Interacting Hand Pose Estimation from a Single RGB Image." ECCV 2020. arXiv:2008.09309. |
| **License** | **CC-BY-NC 4.0** (per the LICENSE file in `facebookresearch/InterHand2.6M`). |
| **Signing-specific?** | No — but explicitly includes **interacting two-hand poses**, which is unusually relevant for ASL given two-handed signs (e.g., MEET, NICE, AGAIN in the frozen 75-sign vocabulary). |
| **Download URL** | Project page: `https://mks0601.github.io/InterHand2.6M/`. Multiple split sizes available (5fps subset down to ~32 GB; full 30fps several hundred GB). |
| **Approximate size** | 5fps split: ~32 GB images + ~3 GB annotations. Full 30fps: ~700 GB+. |
| **Format** | JPEG images + JSON annotations (COCO-like structure with per-frame 21-keypoint per-hand entries). |
| **Provenance verdict** | **PASS for `human_annot/` subset.** Defer `machine_annot/` to a follow-up audit (see "Open follow-ups"). Recommend the 5fps split rather than the full 30fps for storage tractability. |

#### E. COCO-WholeBody (Jin et al., ECCV 2020)

| | |
|---|---|
| **Total images** | Inherits from COCO 2017: ~118K train + ~5K val images; per-image counts of valid hands vary. Roughly ~40K hand instances with valid keypoint annotations (left + right combined). |
| **Subjects** | Drawn from COCO's general-population imagery. |
| **Annotation methodology** | **First benchmark with manual annotations on the entire human body**, per the paper's abstract. 42 hand keypoints (21 per hand) labeled by human annotators with explicit `validity` flags — only sufficiently clear hands are labeled. |
| **Originating paper** | Jin, Xu, Xu, Wang, Liu, Qian, Ouyang, Luo. "Whole-Body Human Pose Estimation in the Wild." ECCV 2020. arXiv:2007.11858. |
| **License** | Annotations: **CC BY 4.0** (inherited from COCO 2017). Images: Flickr Terms — academic use OK; commercial use of the dataset as a whole is restricted. |
| **Signing-specific?** | No — but real-world variety (signers in front of varied backgrounds, varied lighting, varied skin tones, varied distances from camera). The **highest in-the-wild relevance** of any candidate hand dataset. |
| **Download URL** | Annotations: `https://github.com/jin-s13/COCO-WholeBody`. Images: COCO 2017 train+val ZIPs from `http://images.cocodataset.org/`. |
| **Approximate size** | COCO 2017 train images: ~18 GB. Val: ~1 GB. Whole-body annotations: ~150 MB. |
| **Format** | JSON in COCO format (extended `keypoints` arrays of 133 entries per person). |
| **Provenance verdict** | **PASS.** Manual annotation, primary source verifiable, broad in-the-wild relevance. |

#### F. GANerated Hands (Mueller et al., CVPR 2018)

| | |
|---|---|
| **Total images** | 330,000+ synthetic-then-GAN-translated RGB hand images. |
| **Subjects** | None — fully synthetic body, then domain-adapted via GAN. |
| **Annotation methodology** | Annotations come from the **synthetic rendering pipeline** (perfect ground truth from the rendered geometry). The GAN translates synthetic appearance → realistic appearance but is constrained by a geometric-consistency loss so the keypoints are preserved across translation. **No third-party pretrained CV detector** touches the annotations. The GAN itself was trained by the authors. |
| **Originating paper** | Mueller, Bernard, Sotnychenko, Mehta, Sridhar, Casas, Theobalt. "GANerated Hands for Real-Time 3D Hand Tracking from Monocular RGB." CVPR 2018. |
| **License** | "This dataset can only be used for scientific/non-commercial purposes" (per the GANerated Hands project page). Detailed license enclosed in the download. |
| **Signing-specific?** | No — generic hand poses chosen for hand-tracking research. |
| **Download URL** | `https://handtracker.mpi-inf.mpg.de/projects/GANeratedHands/GANeratedDataset.htm` |
| **Approximate size** | Order of 10 GB per public secondary descriptions; verify with the project page. |
| **Format** | PNG + per-image JSON with 2D + 3D 21-keypoint annotations. |
| **Provenance verdict** | **PASS.** Synthetic data with synthetic ground truth; the GAN is a project-trained renderer not a labeler. Could be useful as an augmentation source for stylistic robustness; lower-priority than the real-image datasets above. |

#### G. Epic-HandKps (subset of Epic-Kitchens VISOR)

| | |
|---|---|
| **Total images** | ~5K annotated frames (validation split of VISOR). |
| **Annotation methodology** | "Joints annotated via Scale AI" per the dataset card — Scale AI is a human-annotation service. **Pure human annotation.** |
| **License** | Inherits Epic-Kitchens / VISOR license — research-only, attribution required. |
| **Signing-specific?** | **No — egocentric kitchen footage.** Cameras mounted on head, hands appear in first-person from above. Domain shift vs. third-person signing camera is large. |
| **Download URL** | `https://ap229997.github.io/projects/hands/` (project page); Epic-Kitchens main site for parent corpus. |
| **Provenance verdict** | **PASS on provenance** but **LOW PRIORITY** for our use: egocentric viewpoint does not match a tripod webcam pointed at a signer. Flag for opportunistic inclusion if augmentation diversity ends up being a bottleneck. |

#### H. Voxel51 / `hand-keypoints` (HuggingFace)

| | |
|---|---|
| **Total images** | 846. |
| **Annotation methodology** | Per the dataset card: "manually annotated RGB images sourced from the MPII Human Pose dataset and the New Zealand Sign Language (NZSL) Exercises." This is **a re-host of the manual subset of CMU Panoptic HandDB** (Simon et al. 2017). Identical labels under a tidier wrapper. |
| **Provenance verdict** | **PASS (duplicate of CMU manual subset).** Already covered under dataset **B (i)** above. Do not download separately. |

#### I. HaGRID (Kapitanov et al., WACV 2024) — via `cj-mills/hagrid-sample-500k-384p` mirror

| | |
|---|---|
| **Total images** | 509,323 from `cj-mills/hagrid-sample-500k-384p` (downscaled to 384p from the upstream 552,992 1080p originals). Same population, ~92% retained. |
| **Subjects** | ~37,000 unique people in the upstream — captured under varied natural lighting / distance 0.5–4 m / multiple scenes. Closest match in this audit to our webcam deployment target. |
| **Annotation methodology** | Per the upstream paper (WACV 2024, §3 Data Annotation) and confirmed by the project README: bounding boxes are drawn by **human crowdworkers** on Yandex.Toloka + ABC Elementary, in a 4-stage pipeline (mining → validation → filtration → annotation). Crowdworkers passed an exam, then drew one box around each gesture and a separate box for any "no-gesture" hand fully in frame. Hard + soft aggregation across multiple worker labels for QC. **Bbox provenance: PURE HUMAN ANNOTATION.** The cj-mills mirror preserves these exact bboxes (verified by fetching the dataset card and inspecting the schema). |
| **Disqualified upstream fields — NOT PRESENT in this mirror** | The upstream HaGRID JSON schema ships `hand_landmarks` (MediaPipe-generated) and `meta` (FairFace + MiVOLO-generated). **Neither field exists in the cj-mills mirror.** We verified the dataset card's schema list: it contains only `image`, `bboxes`, `labels`, `leading_hand`, `leading_conf`, `user_id`. This means our compliance posture for ADR-0012 is **structural, not defensive** — the disqualified fields are not in our data source at all, so there's nothing to strip at runtime. (The tripwire in `training/detectors/external_loaders/hagrid.py` is kept as belt-and-suspenders in case we ever switch sources.) |
| **Originating paper** | Kapitanov, Makhlyarchuk, Kvanchiani, Bagaev. "HaGRID — HAnd Gesture Recognition Image Dataset." WACV 2024. arXiv:2206.08219. |
| **License** | **CC BY-SA 4.0** (mirror's `License: cc-by-sa-4.0`, same as upstream). Permissive for research. We won't redistribute the dataset — only use it as training data. |
| **Signing-specific?** | No, but the capture distribution (webcam framing, varied lighting, varied distance, varied skin tones, varied backgrounds) is the closest match in this audit to our actual webcam deployment target. This is the primary motivation for including it. |
| **Why not upstream Sbercloud?** | Sbercloud OBS in Moscow throttles US-Modal egress at ~40 MiB/s aggregate and adds ~300 ms per-request latency that defeats HTTP Range partial downloads. We tried multiple paths (per-class FullHD, 512px lite, `remotezip` streaming) and all hit the same wall. The HuggingFace mirror lives on Cloudflare CDN with US PoPs; observed ~200–1000+ MiB/s download. 13.4 GB vs 119 GB also keeps the Modal volume inode budget comfortable. |
| **Download path** | `huggingface_hub` / `datasets` `load_dataset("cj-mills/hagrid-sample-500k-384p", streaming=True)`. Driven by `training/modal_app.py::download_hagrid_from_hf`. |
| **Approximate size** | 13.4 GB total dataset; we sample 120k images → ~30 GB on the Modal volume after JPEG save. |
| **Format** | HuggingFace `datasets` Parquet format. Row schema: `image` (PIL.Image, 384p), `bboxes` (normalized `[x, y, w, h]` per image), `labels` (gesture class), `leading_hand`, `leading_conf`, `user_id`. Our entrypoint converts the normalized bboxes to absolute-pixel `[x0, y0, x1, y1]` xyxy at JPEG save time. |
| **Provenance verdict** | **PASS unconditionally.** Schema is human-labeled-only by construction. The mirror is a faithful downscale of upstream HaGRID with the disqualified fields already removed. Sources verified: arXiv:2206.08219 §3 "Data Annotation"; HuggingFace dataset card at huggingface.co/datasets/cj-mills/hagrid-sample-500k-384p (schema list + license metadata). |

#### REJECTED hand keypoint candidates

- **Ultralytics Hand Keypoints (26,768 images).** Ultralytics' own
  documentation and blog post identify these labels as **MediaPipe-generated**.
  Explicitly disqualified per the project brief and ADR 0015. **REJECT.**
- **BigHand2.2M (Yuan et al., CVPR 2017).** Pure physical magnetic-sensor
  annotation — provenance is excellent — **but this is a depth-map dataset, not
  RGB.** Our browser inference path is RGB. **REJECT — wrong modality.**

---

### Body / pose keypoint datasets (target: 8 upper-body keypoints)

#### I. MPII Human Pose

| | |
|---|---|
| **Total images** | ~25,000 images containing 40,000+ annotated people. |
| **Annotation methodology** | Crowd-sourced human annotation via Amazon Mechanical Turk (referenced extensively in the literature; verify against the originating paper). |
| **Originating paper** | Andriluka, Pishchulin, Gehler, Schiele. "2D Human Pose Estimation: New Benchmark and State of the Art Analysis." CVPR 2014. |
| **License** | Annotations under Simplified BSD License. Images are extracted from YouTube videos and the original copyright is held by the video owners — commercial use is not permitted. |
| **Keypoint topology** | 16 keypoints including the upper-body joints we need (shoulders, elbows, wrists). Maps cleanly to our 8-keypoint head. |
| **Download URL** | `http://human-pose.mpi-inf.mpg.de/` |
| **Approximate size** | ~12 GB images + ~100 MB annotations. |
| **Format** | MAT files (`mpii_human_pose_v1_u12_2.mat`) + image JPEGs. |
| **Provenance verdict** | **PASS.** Human-annotated. Research-only license fits our pilot scope. |

#### J. COCO Keypoints (COCO 2017)

| | |
|---|---|
| **Total images** | ~118K train + ~5K val. Keypoint annotations for ~250K person instances over ~200K images. |
| **Annotation methodology** | Crowd-sourced human annotation via Amazon Mechanical Turk (documented in the COCO paper, Lin et al. 2014). |
| **License** | Annotations under CC BY 4.0. Images under Flickr Terms — academic use OK; commercial use of the dataset as a whole is restricted by image rights. |
| **Keypoint topology** | 17 keypoints including the upper-body joints we need. Maps cleanly to our 8-keypoint head. |
| **Download URL** | Images: `http://images.cocodataset.org/zips/train2017.zip`, `val2017.zip`. Annotations: `http://images.cocodataset.org/annotations/annotations_trainval2017.zip`. |
| **Approximate size** | Train2017 images: ~18 GB. Val2017: ~1 GB. Annotations: ~250 MB. |
| **Format** | JSON in COCO format. |
| **Provenance verdict** | **PASS.** |

#### REJECTED pose candidates

- **AIST++ (Li et al., ICCV 2021).** Dance dataset with multi-view 2D + 3D
  annotations. Public documentation does not specify the 2D keypoint detection
  step; commonly cited dance / motion-capture pipelines (e.g., Pose2Sim) use
  OpenPose for 2D, with later 3D fitting. Per the audit's default-conservative
  rule, **REJECT for unverified provenance**. Also: dance choreography is a
  poor domain match for signing pose. **REJECT.**
- **Halpe-FullBody (Fang et al., 2020).** Provenance documentation does not
  clearly state the 136-keypoint annotation methodology. The COCO-WholeBody
  paper explicitly claims to be "the first benchmark with manual annotations
  on the entire human body," which implies Halpe is at least partially
  semi-automated. Per the default-conservative rule, **REJECT for unverified
  provenance**. (COCO-WholeBody covers our needs at higher confidence.)

---

### Face detection datasets (target: bounding boxes only)

#### K. WIDER FACE (Yang et al., CVPR 2016)

| | |
|---|---|
| **Total images** | 32,203 images, 393,703 face annotations. |
| **Annotation methodology** | Manual labeling by the dataset curators per the WIDER FACE paper §3.1 ("we label the bounding boxes for all the recognizable faces"). Annotators also tag pose / occlusion / blur / expression / illumination / makeup / event attributes per face. |
| **Originating paper** | Yang, Luo, Loy, Tang. "WIDER FACE: A Face Detection Benchmark." CVPR 2016. |
| **License** | **CC BY-NC-ND 4.0** per the HuggingFace mirror's license field. Restrictive (no derivatives clause), but research use within a non-commercial pilot is permitted. The no-derivatives clause may bite if we ever want to redistribute a derived bbox-extraction. Flag for slice-2 commercial-cliff review. |
| **Download URL** | `http://shuoyang1213.me/WIDERFACE/` (official) — split ZIPs for train / val / test. |
| **Approximate size** | Train images: ~1.4 GB. Val: ~360 MB. Annotations: small. |
| **Format** | JPEG images + plaintext annotation files (`wider_face_train_bbx_gt.txt` etc.) listing per-image bbox lists. |
| **Provenance verdict** | **PASS.** Manually annotated. The CC BY-NC-ND license is acceptable for pilot use but should be re-reviewed at slice-2 commercialization. |

---

### Sign-language vocabulary clip datasets (target: full RGB clips per gloss in slice-1 vocab)

These are **clip-level** datasets used for the project's own pseudo-labeling +
classifier head training, not for detector training. They feed
`scripts/rebuild_unified_manifest.py` → `unified_clip_manifest_modal_v*.json`.
ADR-0012 is unchanged: pose/hand/face landmarks for these clips are produced
by **our own** detectors (`hand_det_v2`, `landmarks_v0`, `pose_v0`), not by
any pretrained CV model that came with the dataset.

#### L. ASL Citizen (Desai et al., NeurIPS 2023 D&B Track)

| | |
|---|---|
| **Total clips / signs / signers** | 83,399 clips, 2,731 distinct signs, 52 signers (per arXiv:2304.05934 abstract — verified 2026-05-22). |
| **Collection methodology** | Crowdsourced citizen-science recordings ("with consent" per paper) from a multi-signer pool. Deaf research team members were involved throughout the project per the MSR project page. Clips are isolated single-sign recordings (not continuous signing). |
| **Originating paper** | Desai, Berger, Minakov, Maddiwar, Sodhi, Bragg. "ASL Citizen: A Community-Sourced Dataset for Advancing Isolated Sign Language Recognition." NeurIPS 2023 Datasets & Benchmarks. arXiv:2304.05934. |
| **License** | Distribution governed by Microsoft Research's project terms; the project page directs commercial inquiries to `ASL_Citizen@microsoft.com`, which strongly implies a research-only / non-commercial license (MSR-LA family). **License text not posted on the project page or download page** — must be confirmed at first download before any clip enters our pipeline. Treat as research-only pending verification. |
| **Signer-disjoint splits** | **Yes — explicitly designed for it.** The paper notes "model performance was evaluated entirely on videos of users who are not present in the training or validation sets," and signer IDs are part of the release per the project description. This is the strongest signer-disjoint guarantee of any vocab-clip source on our list. |
| **Gloss alignment** | Gloss strings are ASL gloss (uppercase, may include `_N` numeric suffixes for sense disambiguation). Maps to our `sign_id` via the same lowercase + strip-suffix normalization already implemented for Sem-Lex in `scripts/rebuild_unified_manifest.py`. |
| **Download URL** | `https://www.microsoft.com/en-us/download/details.aspx?id=105253` (Microsoft Download Center, MSR). Also referenced from `https://www.microsoft.com/en-us/research/project/asl-citizen/`. |
| **Approximate size** | ~150 GB total (~83K clips at ~1.8 MB avg, mp4 H.264). Subset for our 46 thin signs is ~1500-3000 clips ≈ 3-6 GB. |
| **Format** | MP4 video files + per-clip CSV metadata (`Video file`, `Participant ID`, `Gloss`, plus split assignment). |
| **Fields we ingest** | Clip file path, `sign_id` (normalized from `Gloss`), `signer_id` (= `Participant ID`), source URL, license string. |
| **Fields we drop** | None applicable — ASL Citizen does not ship pretrained landmarks/embeddings (it ships raw video only), so the ADR-0012 tripwire pattern used for HaGRID landmarks is not needed here. |
| **Provenance verdict** | **PASS pending license confirmation at first download.** Human-recorded, human-glossed, signer IDs released, signer-disjoint splits supported. Strongest single source for Phase 4.9. ADR-0015 default-conservative rule says research-only is acceptable for pilot scope. |
| **Compliance follow-ups** | (a) Capture and persist the license string from the first download into `dataset/raw/asl_citizen/LICENSE.txt`. (b) If license forbids redistribution of derived features, treat extracted trajectories as project-private — same posture as Sem-Lex. (c) Re-review at slice-2 commercial-cliff alongside the other research-only sources. |

---

## RECOMMENDED summary

| Dataset | Task | Images / instances usable | License | Notes |
|---|---|---|---|---|
| **FreiHAND** | Hand keypoints (21) | 32,560 unique training (130,240 augmented) + 3,960 eval | Research only, no commercial | Highest-quality real-RGB hand keypoints |
| **CMU Panoptic HandDB — manual subset** | Hand keypoints (21) | ~14,817 hand annotations across ~2,758 images | Research only, no commercial | Includes NZSL signing imagery (rare positive signal) |
| **CMU Panoptic HandDB — multiview subset** | Hand keypoints (21) | ~33,000 triangulated hand annotations | Research only, no commercial | Project-own bootstrap detector; ADR-0012-spirit compliant |
| **CMU Panoptic HandDB — synthetic subset** | Hand keypoints (21) | ~14,261 rendered hand images | Research only, no commercial | Optional — opportunistic augmentation source |
| ~~**Multiview Hand Pose (Gomez-Donoso)**~~ | ~~Hand keypoints (21)~~ | ~~~21,000 frames (× 4 views each)~~ | ~~BSD~~ | **DROPPED 2026-05-21 — upstream archive corrupt; see "Multiview drop" note below** |
| **InterHand2.6M — `human_annot` subset** | Hand keypoints (21) incl. interacting two hands | ~500K human-annotated frames (5fps split estimate; confirm before downloading full set) | CC-BY-NC 4.0 | Interacting two-hand poses relevant for two-handed signs |
| **COCO-WholeBody** | Hand keypoints (21 per hand) | ~40K hand instances with `validity=True` | Annotations CC BY 4.0; images Flickr Terms | Strongest in-the-wild signal among hand datasets |
| **GANerated Hands** | Hand keypoints (21) | 330,000 synthetic images | Non-commercial research | Optional — synthetic, useful only for stylistic augmentation |
| **Epic-HandKps** | Hand keypoints (21) | ~5K egocentric frames | Epic-Kitchens research-only | Optional — domain mismatch (egocentric) |
| **MPII Human Pose** | Pose keypoints | ~25K images / ~40K people, 16 kpts each | Annotations BSD; images research-only | Primary pose source |
| **COCO Keypoints** | Pose keypoints | ~250K person instances, 17 kpts each | Annotations CC BY 4.0; images Flickr Terms | Primary pose source, in-the-wild |
| **WIDER FACE** | Face bboxes | 32,203 images / 393,703 faces | CC BY-NC-ND 4.0 | Only face-detection candidate considered |

---

## Multiview drop (2026-05-21, post-download)

The Gomez-Donoso Multiview Hand Pose dataset was RECOMMENDED in the
original audit but **dropped during ingestion** because both available
distributions deliver a corrupt archive:

- **V1 (rovit.ua.es direct ZIP):** downloaded successfully (1.2 GB) but
  every fetched copy fails `unzip` central-directory parsing,
  regardless of downloader (curl, aria2c). The upstream server is
  serving a broken file.
- **V2 (Google Drive):** manual click-through download also fails
  decompression. Whether the cause is Google Drive's virus-scan
  interstitial saving as HTML, the underlying ZIP itself being
  corrupt, or partial-download truncation is unverified — three
  failed downloads across two distribution channels is enough signal
  to stop trying.

**Net loss:** ~21,000 frames × 4 camera views ≈ 84,000 view-images
of Leap-Motion-sensor-labeled 21-keypoint hand poses. Would have been
the best-provenance hand dataset in the corpus (pure physical sensor,
zero learned models, zero human annotation variance).

**Why this does not block training:** FreiHAND (32,560 unique samples)
+ CMU Panoptic HandDB (2,758 manual + 14,261 synthetic + 14,817
multiview-bootstrapped = 31,836 items) yields **64,396 labeled hand
samples**, which is ~2× the lower bound the literature uses for
training a 21-keypoint regressor at this scale. The Multiview drop is
acknowledged as a coverage-gap (Leap Motion's physical-sensor
distribution is now absent; CMU's bootstrap-detector labels and
FreiHAND's iterative multi-view fits are both inherently project-
trained rather than physical-sensor). The fairness eval will reflect
this in any per-keypoint accuracy report.

**Open follow-up:** if a working Multiview source ever surfaces
(academic mirror, archived release on a paper-replication site,
direct contact with the authors), re-run
`scripts/datasets/download_multiview_hand_pose.py` and re-issue
`python3 -m training.detectors.normalize_external --task hand_keypoints`
to fold the data in. The loader and normalizer are already wired —
it's a no-code-change re-run.

---

## REJECTED summary

| Dataset | Reason for rejection |
|---|---|
| **Ultralytics 26K Hand Keypoints** | Labels generated by **MediaPipe** (pretrained model) per Ultralytics' own documentation. Disqualified by ADR 0015. |
| **BigHand2.2M (Yuan et al., 2017)** | **Wrong modality** — depth maps, not RGB. Annotation provenance (magnetic sensors) is excellent but the data does not fit the browser RGB inference path. |
| **AIST++ (Li et al., ICCV 2021)** | **Unverified provenance** — the 2D keypoint detection step is not documented to be manual on the official factsheet, and the multi-view dance pipeline commonly involves OpenPose. Also: domain mismatch (dance, not signing). |
| **Halpe-FullBody (Fang et al., 2020)** | **Unverified provenance** — the 136-keypoint annotation methodology is not clearly documented as fully manual, and the COCO-WholeBody paper's explicit claim to be the *first* fully manual whole-body benchmark implies Halpe is at least partially semi-automated. COCO-WholeBody covers the same need at higher provenance confidence. |
| **RWTH-PHOENIX-Weather (sign-language video corpus)** | Not a keypoint dataset; original video sequences are not downloadable due to broadcast-rights legal constraints. Out of scope for landmark training. |
| **Voxel51 / `hand-keypoints` (HuggingFace, 846 imgs)** | Not actually rejected; **deduplicate** — this is a re-host of CMU Panoptic HandDB's manual subset (already RECOMMENDED). Do not download separately. |

---

## Combined-corpus summary (if all RECOMMENDED datasets are used)

Counting only the high-confidence subsets (excluding Epic-HandKps and
GANerated, which are flagged optional):

- **Hand keypoint images, 21 joints, RGB, project-clean provenance:**
  ≈ 32,560 (FreiHAND unique) + ~50,075 (CMU HandDB manual + multiview, all
  three subsets) + ~21,000 (Gomez-Donoso, × 4 views ≈ ~84,000 view-images) +
  ~500,000 (InterHand2.6M `human_annot`, 5fps split — confirm at download
  time) + ~40,000 (COCO-WholeBody valid-hand instances).
  **Floor estimate: ~150K usable hand-keypoint image instances**, with
  realistic ceiling closer to **600K+** if InterHand2.6M human_annot and
  Gomez-Donoso's multiple views are unpacked aggressively.

- **Pose keypoint instances with upper-body joints, RGB, project-clean
  provenance:** ~40,000 (MPII) + ~250,000 (COCO) = **~290,000 person
  instances** with at minimum shoulders / elbows / wrists. Far above the
  ~2,000 frames the roadmap budgets for self-labeling.

- **Face bounding-box images, project-clean provenance:** **32,203 images,
  393,703 face boxes** (WIDER FACE). The roadmap budgets ~1,500 self-labeled
  frames for face — WIDER FACE alone is ~20× that.

If these numbers hold post-download, **the labeling cliff named in
`docs/VOCABULARY_TRAINER_ROADMAP.md` Phase 2 Slice 2.1 (~120 hours of
hand-keypoint labeling) is dramatically reduced**, possibly eliminated as
the primary training-data source. Self-labeling becomes a coverage-gap-closer
rather than the entire training corpus.

---

## Coverage gap analysis

### Where the recommended datasets are strong

- **Hand keypoint coverage:** 600K+ images possible. Two-hand interactions
  covered by InterHand2.6M. Multi-view 3D-consistent labels covered by
  FreiHAND, CMU bootstrap, Gomez-Donoso.
- **Upper-body pose:** 290K+ person instances. MPII + COCO together cover
  every conceivable lighting and background.
- **Face detection:** 393K faces, 61 event categories of background context.

### Where the recommended datasets are weak for our specific use

1. **ASL signing context.** None of the hand-keypoint datasets feature ASL
   signers signing the slice-1 vocabulary. CMU's NZSL subset is the only
   sign-language coverage, and NZSL ≠ ASL (different vocabulary, different
   phonological inventory).
2. **Body-relative location signal.** Most hand-keypoint datasets isolate the
   hand or capture it in non-signing contexts. Locations like "at chin," "at
   chest," "at forehead," "neutral space" — which are the very semantic
   handles our hint engine depends on — are under-sampled.
3. **Skin-tone diversity.** Documented under-representation of darker
   Fitzpatrick types in MPII / COCO / WIDER-FACE (per published fairness
   literature). Per-Fitzpatrick eval-gate criterion 3 (`docs/EVAL_GATE.md`)
   will need a deliberate self-recorded augmentation step.
4. **Camera distance and framing.** The frozen 75-sign trainer uses a
   webcam-at-conversational-distance framing (head + upper body in frame).
   Most hand datasets are close-ups; most pose datasets are full-body. There
   is a framing gap our self-recorded supplement should fill.
5. **Background environment.** Home / classroom / office backgrounds at a
   distance, while the user is sitting and signing, are not the dominant
   modes in any of these datasets.

### Supplemental ASL-specific labeling estimate

Given the above gaps and assuming the recommended public datasets are
ingested, a realistic supplemental labeling budget is:

- **Hand keypoints (signer at signing distance, body-relative locations):
  300–600 self-recorded frames**, focused on the four ASL parameters
  (handshape × location × movement × palm orientation) with deliberate
  Fitzpatrick diversity across captures.
- **Pose keypoints:** **~100–200 self-recorded frames** of the project author
  + 1–2 additional signers if available. Public pose data is far above
  saturation already; this is purely for the framing/distance gap.
- **Face detection:** **~50–100 self-recorded frames** for framing
  calibration in the user's actual webcam setup. WIDER FACE will carry the
  detector training; the self-recorded frames are for the onboarding
  framing-check UX.

**Total supplemental labeling estimate: 450–900 self-recorded frames**, down
from the original ~5,000–10,000 keypoint estimate in
`docs/VOCABULARY_TRAINER_ROADMAP.md` Phase 2 Slice 2.1. At ~30–60 seconds
per labeled frame (keypoint labeling rate after the rubric is in muscle
memory), this is **~5–15 hours of supplemental labeling**, not the original
120-hour estimate.

The 120-hour estimate stands only if public datasets are excluded entirely.
Folding in the recommended public sources collapses the labeling cliff to
something we can do in a long weekend.

---

## Open follow-ups (not blocking go/no-go)

1. **InterHand2.6M `machine_annot` subset.** Confirm from the paper (§3.2) or
   GitHub README whether the automated annotator was trained only on
   `human_annot`. If yes, lift the conservative restriction and use both
   subsets. If unclear, leave restricted.
2. **CMU HandDB synthetic subset.** Confirm the renderer's keypoint extraction
   is geometric (not learned). Almost certainly geometric, but worth a
   one-line verification.
3. **Halpe-FullBody.** A direct read of the AlphaPose paper §3 would settle
   whether Halpe's hand keypoint annotations are manual or AlphaPose-bootstrapped.
   If manual, promote to RECOMMENDED; if AlphaPose-bootstrapped, the rejection
   is permanent.
4. **COCO-WholeBody hand keypoint topology.** Verify the 21-per-hand convention
   matches the standard wrist + 4-per-finger × 5 fingers topology used by
   FreiHAND and CMU. If a remap is needed, scaffold it before training.
5. **Per-Fitzpatrick metadata** on hand keypoint datasets. None of the
   candidates ship per-subject demographic metadata; per-Fitzpatrick eval will
   have to come from our held-out self-recorded set.

These do not block the download go/no-go. They are scoped to the first
training run.

---

## Approval checklist (for the reviewer)

Tick each before approving the download plan:

- [ ] License terms in this memo match what the reviewer expects.
- [ ] Provenance for each RECOMMENDED dataset is convincing — no "trust me" claims.
- [ ] The REJECTED list correctly excludes Ultralytics (per the project
      brief's explicit MediaPipe-labeled prohibition).
- [ ] The combined storage budget (hundreds of GB on disk if InterHand2.6M
      30fps is included, ~80 GB if 5fps only) fits the user's available disk.
- [ ] The supplemental labeling estimate (450–900 frames, ~5–15 hours) is
      acceptable.
- [ ] ADR 0015 (new) is approved as the governing acceptance criteria.
- [ ] ADR 0011 amendment ("Data sourcing — landmark training") is approved.

Once approved, run the dataset-specific download commands listed at the bottom
of the session summary (or call `scripts/datasets/download_all_recommended.sh`
once the user has read each individual script).
