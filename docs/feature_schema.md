# Feature schema — per-frame classifier/template feature vector

> The explicit contract for `_frame_to_features` (`training/detectors/fit_templates.py`)
> and `trajectory_from_frames` (`training/detectors/sign_matcher.py`). Any change
> here is versioned and must update: the classifier input dim, `_mirror_features`,
> the P0 ablation index ranges (`scripts/p0_signer_baseline.py`), and this doc.
>
> Background: the hand keypoints are currently normalized **body-relative**, which
> represents *where* the hand is but NOT the *handshape* (finger geometry). Per the
> 2026-05-23 review, that — together with the missing Z — is why curl-differentiated
> letters (A/S/E/N/M) confuse. v2 below fixes the normalization; Z is a later (v3)
> upgrade gated on measuring how much curl confusion survives v2.

Hand keypoint topology (21, MediaPipe-compatible, trained from scratch):
`0 = wrist`, `9 = middle-finger MCP`. **Hand scale = ‖kp9 − kp0‖** (wrist→middle MCP).

<!-- NOTE: this file was partially lost in a lint-staged stash mishap on
     2026-05-23 and reconstructed. The text ABOVE this marker is verbatim from
     the original. The layout table BELOW is re-derived from the actual code in
     fit_templates.py (authoritative). Any v2/v3 normalization DESIGN notes that
     followed in the original are NOT recovered here — restore from your editor
     copy if you have it. -->

## Current layout (re-derived from `fit_templates.py`)

Baseline (no handshape embedding): **100 floats / frame**

| Block | Indices | Contents |
|---|---|---|
| slot 0 hand | `0..42` | 21 keypoints × (x, y), body-anchored + shoulder-scaled |
| slot 1 hand | `42..84` | 21 keypoints × (x, y) |
| pose | `84..100` | 8 keypoints × (x, y): nose, neck, r_sh, l_sh, r_el, l_el, r_wr, l_wr |

With handshape embedding (Phase 4.6): **356 floats / frame**

| Block | Indices | Contents |
|---|---|---|
| slot 0 | `0..170` | 42 kpts + 128-d handshape embedding |
| slot 1 | `170..340` | 42 kpts + 128-d embedding |
| pose | `340..356` | 8 keypoints × (x, y) |

Constants in code: `NUM_HAND_KP=21`, `NUM_POSE_KP=8`, `HAND_EMBED_DIM=128`,
`HAND_SLOT_DIMS=42` (kpts) / `170` (kpts+embed), `POSE_BASE=84` / `340`.

**Normalization:** translation = subtract pose anchor (neck → shoulder-midpoint
→ nose fallback); scale = ‖r_sh − l_sh‖ (shoulder span), fallback 1.5·‖nose − neck‖,
floor 1.0. **Hand slot:** slot 0 if `wrist.x < anchor.x` (left side of frame),
slot 1 otherwise; mirror-aware (inference scores the hflipped trajectory too).
**Z (depth):** not present today — see [[project_sign_avatar]]; requires a 3D
landmark-detector retrain (head `21×3`, MANO labels from FreiHAND/CMU).
