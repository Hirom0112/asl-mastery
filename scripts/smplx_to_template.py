"""Convert SMPLest-X motion (data/smplx_motion/out/<sign>.npz) → avatar templates.

Runs the SMPL-X layer forward on the extracted per-frame params to get 3D joint
positions, maps them into the existing avatar template schema
(public/templates/<sign>.json: per-frame pose[8] + hand0[21] + hand1[21]), and
writes them in **body space** (neck-anchored, shoulder-scaled) — the same space
the current avatar is calibrated for, but now 3D (keypoints are [x, y, z]) and
with REAL finger keypoints (so `ENABLE_FINGERS` can be turned on).

After global_orient, SMPL-X joints are camera-aligned (x right, y down, z depth),
which already matches the template/image convention, so the transform is just
(kp - neck) / shoulder_span.

Hand 21-kp order (MANO/OpenPose, matches sign-avatar.tsx):
  0 wrist, 1-4 thumb, 5-8 index, 9-12 middle, 13-16 ring, 17-20 pinky
  (each finger: joint1, joint2, joint3, tip)

Usage: training/.venv/bin/python scripts/smplx_to_template.py [--only sign ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import smplx
from smplx.joint_names import JOINT_NAMES
from scipy.ndimage import gaussian_filter1d

SMOOTH_SIGMA = 2.2  # temporal smoothing on joint trajectories — kills the
                    # per-frame SMPL-X jitter (esp. the palm) that made the hand twitch
LOOP_BLEND = 6      # crossfade the last N frames toward frame 0 so the loop is
                    # seamless (no snap / "back and forth" jump at the wrap)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "smplx_motion" / "out"
DST = ROOT / "public" / "templates"
MODELS = ROOT / "data" / "signavatars" / "human_model_files"
T_OUT = 32  # frames per template (matches existing schema + avatar playback)

POSE_NAMES = ["nose", "neck", "right_shoulder", "left_shoulder",
              "right_elbow", "left_elbow", "right_wrist", "left_wrist"]
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]


def _hand_names(side: str) -> list[str]:
    # 21-kp: wrist, then per finger [1,2,3,tip]; tip = the finger-name w/o number
    names = [f"{side}_wrist"]
    for f in FINGERS:
        names += [f"{side}_{f}1", f"{side}_{f}2", f"{side}_{f}3", f"{side}_{f}"]
    return names


L_HAND = _hand_names("left")
R_HAND = _hand_names("right")
EXTRA = ["pelvis"]  # for the torso-local frame
JIDX = {n: JOINT_NAMES.index(n) for n in set(POSE_NAMES + L_HAND + R_HAND + EXTRA)}

# Per-hand chirality sign for the PALMAR normal. The cross product
# (pinky->index) x (wrist->mid) yields OPPOSITE signs for the two hands (mirror
# anatomy), so we flip one hand to get a consistent "toward-palm" normal. Anchored
# to the SMPL-X rest T-pose (both palms face -y / down) and cross-checked against
# the wrist joint's global rotation x the MANO rest palmar axis (agree to <0.1deg).
PALM_SIGN = {"left": -1.0, "right": 1.0}


def _resample(arr: np.ndarray, T: int) -> np.ndarray:
    n = arr.shape[0]
    if n == T:
        return arr
    src = np.linspace(0, 1, n)
    dst = np.linspace(0, 1, T)
    out = np.empty((T,) + arr.shape[1:], np.float32)
    flat = arr.reshape(n, -1)
    o = np.empty((T, flat.shape[1]), np.float32)
    for c in range(flat.shape[1]):
        o[:, c] = np.interp(dst, src, flat[:, c])
    return o.reshape((T,) + arr.shape[1:])


def forward_joints(model, d) -> np.ndarray:
    """(F, 144, 3) joint positions from the stored SMPL-X params."""
    def t(key):
        return torch.tensor(np.nan_to_num(d[key]), dtype=torch.float32)
    F = d["smplx_root_pose"].shape[0]
    z = torch.zeros(F, 3, dtype=torch.float32)
    out = model(
        global_orient=t("smplx_root_pose"), body_pose=t("smplx_body_pose"),
        left_hand_pose=t("smplx_lhand_pose"), right_hand_pose=t("smplx_rhand_pose"),
        jaw_pose=t("smplx_jaw_pose"), leye_pose=z, reye_pose=z,
        betas=t("smplx_shape"), expression=t("smplx_expr"), return_verts=False,
    )
    return out.joints.detach().numpy()


Z_DAMP = 0.35  # SMPLest-X camera depth is exaggerated; damp it (avatar FORWARD_BIAS
               # already pushes hands forward — we just want subtle relative depth)


def to_bodyspace(J: np.ndarray) -> np.ndarray:
    """(F,144,3) → (F,144,3) neck-anchored + shoulder-scaled, in CAMERA axes.

    After global_orient the body is already camera-aligned (x right, y down, z
    depth), which matches the template/avatar convention — so we use raw camera
    coords directly. (A torso-local frame tilts 'up' by the body's lean and
    cancels vertical reach, which collapsed face-signs to chest level.) We only
    fix depth: raw camera +z points away from the camera, but the avatar's +z is
    toward it, so flip z to put the signing hands in front.
    """
    neck = J[:, JIDX["neck"]][:, None, :]
    rsh = J[:, JIDX["right_shoulder"]]
    lsh = J[:, JIDX["left_shoulder"]]
    span = np.maximum(np.linalg.norm(lsh - rsh, axis=1), 1e-6)[:, None, None]
    out = ((J - neck) / span).astype(np.float32)  # x, y(down), z(depth)

    lw, rw = JIDX["left_wrist"], JIDX["right_wrist"]
    if (out[:, lw, 2].mean() + out[:, rw, 2].mean()) < 0:  # hands toward cam (raw -z) → make +
        out[..., 2] *= -1.0
    out[..., 2] *= Z_DAMP
    return out


def kp(J: np.ndarray, name: str) -> np.ndarray:
    return J[:, JIDX[name], :]  # (F,3)


def palm_normal(B: np.ndarray, side: str) -> np.ndarray:
    """(F,3) per-frame PALMAR unit normal (points OUT of the palm) for one
    anatomical hand, in body space — chirality-correct via PALM_SIGN.

    Built from the palm plane: cross of 'across' (index1 - pinky1) with 'along'
    (middle1 - wrist), so the avatar can set the wrist's full roll instead of
    guessing a single global facing. The earlier data-driven
    attempt twisted because it used ONE cross-product formula for both hands;
    mirrored anatomy flips its sign, so the non-dominant hand got the DORSAL
    normal (palm 180° backwards → wrist rolled toward the face). PALM_SIGN fixes
    that per hand.
    """
    w = kp(B, f"{side}_wrist")
    idx = kp(B, f"{side}_index1")
    mid = kp(B, f"{side}_middle1")
    pky = kp(B, f"{side}_pinky1")
    n = np.cross(idx - pky, mid - w)            # (across) x (along)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.where(ln > 1e-6, PALM_SIGN[side] * n / np.maximum(ln, 1e-9), 0.0)
    return n.astype(np.float32)


def build_template(model, sign: str, d, one_handed_set: set, trim=None) -> dict:
    J = forward_joints(model, d)            # (F,144,3)
    F = J.shape[0]
    B = to_bodyspace(J)                     # body space
    if F >= 3:                              # temporal smoothing (de-jitter)
        B = gaussian_filter1d(B, SMOOTH_SIGMA, axis=0, mode="nearest")

    def pts(names):  # (F, len(names), 3)
        return np.stack([kp(B, n) for n in names], axis=1)

    pose = pts(POSE_NAMES)                  # (F,8,3)
    lhand = pts(L_HAND)                     # (F,21,3)
    rhand = pts(R_HAND)                     # (F,21,3)
    lpalm = palm_normal(B, "left")          # (F,3) chirality-correct palmar normals
    rpalm = palm_normal(B, "right")

    # Per-sign clip trim: keep only [start_frac, end_frac] of the raw timeline,
    # dropping the idle reach-in/return that makes the looped hand fall to the
    # lap (e.g. understand: the signer raises the arm to the temple and lowers
    # it again; only the middle hold is the citation form).
    if trim:
        i0 = max(0, int(round(trim[0] * F)))
        i1 = min(F, int(round(trim[1] * F)))
        if i1 - i0 >= 3:
            pose, lhand, rhand = pose[i0:i1], lhand[i0:i1], rhand[i0:i1]
            lpalm, rpalm = lpalm[i0:i1], rpalm[i0:i1]

    # slot assignment: slot0 = hand whose wrist is more screen-left (smaller x),
    # matching fit_templates' convention so the avatar's L/R logic lines up. The
    # palm normals ride the SAME swap so palm{N} always matches hand{N}.
    lx = lhand[:, 0, 0].mean()
    rx = rhand[:, 0, 0].mean()
    if lx <= rx:
        hand0, hand1, palm0, palm1 = lhand, rhand, lpalm, rpalm
    else:
        hand0, hand1, palm0, palm1 = rhand, lhand, rpalm, lpalm

    # one-handed: authoritative from ASL-LEX (SMPLest-X often mirrors the resting
    # hand, so motion-based auto-detect is unreliable; the curated list wins).
    one_handed = sign in one_handed_set

    # resample to T_OUT
    pose = _resample(pose, T_OUT)
    hand0 = _resample(hand0, T_OUT)
    hand1 = _resample(hand1, T_OUT)
    palm0 = _resample(palm0, T_OUT)
    palm1 = _resample(palm1, T_OUT)

    def _renorm(a):  # (T,3) re-unit-length after interpolation (lerp shrinks)
        ln = np.linalg.norm(a, axis=1, keepdims=True)
        return np.where(ln > 1e-6, a / np.maximum(ln, 1e-9), 0.0).astype(np.float32)
    palm0, palm1 = _renorm(palm0), _renorm(palm1)

    # loop-close: crossfade the last LOOP_BLEND frames toward frame 0 so the
    # animation wraps seamlessly (no snap / back-and-forth jump).
    def _loop_close(a):
        T = a.shape[0]
        L = min(LOOP_BLEND, T - 1)
        for i in range(L):
            idx = T - L + i
            w = (i + 1) / L  # 0→1, full blend on the last frame
            a[idx] = (1 - w) * a[idx] + w * a[0]
        return a
    pose, hand0, hand1 = _loop_close(pose), _loop_close(hand0), _loop_close(hand1)
    palm0, palm1 = _renorm(_loop_close(palm0)), _renorm(_loop_close(palm1))

    def enc(arr):  # (T,K,3) -> list of frames of [[x,y,z],...]
        return [[[round(float(x), 4) for x in p] for p in frame] for frame in arr]

    def encv(arr):  # (T,3) -> list of [x,y,z] per frame
        return [[round(float(x), 4) for x in v] for v in arr]

    return {
        "sign": sign,
        "frames": [
            {"hand0": h0, "hand1": h1, "pose": ps, "palm0": p0, "palm1": p1}
            for h0, h1, ps, p0, p1 in zip(
                enc(hand0), enc(hand1), enc(pose), encv(palm0), encv(palm1)
            )
        ],
        "n_clips": 1,
        "T": T_OUT,
        "one_handed": bool(one_handed),
        "source": "smplx",  # SMPLest-X whole-body motion (3D, fingers)
        "schema": {
            "pose_order": ["nose", "neck", "r_sh", "l_sh", "r_el", "l_el", "r_wr", "l_wr"],
            "coord_space": "neck_anchored_shoulder_scaled_3d",
            "hand_order": "mano21: wrist, thumb(1,2,3,tip), index..., middle..., ring..., pinky...",
            "palm": "palm{N}: per-frame PALMAR unit normal (body space, points out "
                    "of the palm) for hand{N}; drives the wrist roll",
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--dst", type=Path, default=DST)
    args = ap.parse_args()

    files = sorted(args.src.glob("*.npz"))
    if args.only:
        keep = set(args.only)
        files = [f for f in files if f.stem in keep]

    oh_path = ROOT / "dataset" / "one_handed_top80.json"
    one_handed_set = set(json.loads(oh_path.read_text()).get("one_handed", [])) if oh_path.exists() else set()
    hs_path = ROOT / "dataset" / "sign_handshapes_top80.json"
    handshapes = json.loads(hs_path.read_text()) if hs_path.exists() else {}
    ov_path = ROOT / "dataset" / "handshape_overrides.json"
    if ov_path.exists():  # per-sign fixes (handshape-change signs etc.)
        handshapes.update({k: v for k, v in json.loads(ov_path.read_text()).items()
                           if not k.startswith("_")})
    tr_path = ROOT / "dataset" / "clip_trim_overrides.json"
    trims = ({k: v for k, v in json.loads(tr_path.read_text()).items() if not k.startswith("_")}
             if tr_path.exists() else {})

    model = smplx.create(str(MODELS), "smplx", gender="neutral", use_pca=False,
                         use_face_contour=True, flat_hand_mean=False, batch_size=1)
    args.dst.mkdir(parents=True, exist_ok=True)
    n_oh = 0
    for f in files:
        d = np.load(f)
        tpl = build_template(model, f.stem, d, one_handed_set, trim=trims.get(f.stem))
        tpl["handshape"] = handshapes.get(f.stem)  # ASL-LEX canonical handshape
        n_oh += tpl["one_handed"]
        (args.dst / f"{f.stem}.json").write_text(json.dumps(tpl, separators=(",", ":")))
        print(f"  {f.stem}: T={tpl['T']} one_handed={tpl['one_handed']}")
    print(f"\nwrote {len(files)} templates ({n_oh} one-handed) → {args.dst}")


if __name__ == "__main__":
    main()
