"""Graft WiLoR crisp finger pose into the SMPLest-X body npz.

WiLoR (data/wilor_motion/<sign>.npz) gives per-frame MANO hand_pose (45-dim
axis-angle) for left/right hands, frame-aligned to the body npz (SAME clip, SAME
`ffmpeg -vsync 0` extraction). We replace `smplx_{l,r}hand_pose` in the body npz
with WiLoR's crisp fingers, keeping SMPLest-X's body + wrist orientation. Output
overwrites data/smplx_motion/out/<sign>.npz (a one-time backup is kept under
out_prewilor/). Then re-bake with scripts/smplx_to_glb.py --finger-gain 1.0.

Missing-frame hands (WiLoR didn't detect a hand) are filled from the nearest valid
frame for continuity. The MANO hands-mean convention between WiLoR and the baker's
SMPL-X (flat_hand_mean=False) may differ → optional `--hands-mean {none,add,sub}`
(decide empirically by rendering one sign with a flat/open handshape).

  training/.venv/bin/python scripts/graft_wilor_fingers.py --only animal
  training/.venv/bin/python scripts/graft_wilor_fingers.py            # all wilor npz
"""
import argparse
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BODY = ROOT / "data" / "smplx_motion" / "out"
WILOR = ROOT / "data" / "wilor_motion"
BACKUP = ROOT / "data" / "smplx_motion" / "out_prewilor"
MODELS = ROOT / "data" / "signavatars" / "human_model_files"


def fill_missing(pose, valid):
    """Forward/back-fill the 45-dim pose over frames where the hand wasn't detected."""
    F = pose.shape[0]
    idx = np.where(valid)[0]
    if len(idx) == 0:
        return np.zeros_like(pose), False
    out = pose.copy()
    last = None
    for f in range(F):
        if valid[f]:
            last = pose[f]
        elif last is not None:
            out[f] = last
    first = pose[idx[0]]
    for f in range(idx[0]):
        out[f] = first
    return out, True


def get_hands_mean():
    import smplx

    m = smplx.create(str(MODELS), "smplx", gender="neutral", use_pca=False,
                     use_face_contour=True, flat_hand_mean=False, batch_size=1)
    return (m.left_hand_mean.detach().numpy().reshape(-1),
            m.right_hand_mean.detach().numpy().reshape(-1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--hands-mean", choices=["none", "add", "sub"], default="none")
    args = ap.parse_args()
    signs = args.only or sorted(p.stem for p in WILOR.glob("*.npz"))
    lmean, rmean = get_hands_mean()
    BACKUP.mkdir(parents=True, exist_ok=True)

    for s in signs:
        bp = BODY / f"{s}.npz"
        wp = WILOR / f"{s}.npz"
        if not bp.exists() or not wp.exists():
            print(f"SKIP {s}: missing body({bp.exists()}) or wilor({wp.exists()}) npz")
            continue
        body = dict(np.load(bp))
        wil = np.load(wp)
        Fb = body["smplx_root_pose"].shape[0]
        Fw = wil["lhand_pose"].shape[0]
        F = min(Fb, Fw)
        if Fb != Fw:
            print(f"WARN {s}: frame mismatch body={Fb} wilor={Fw} -> truncate {F}")

        lp, lok = fill_missing(wil["lhand_pose"][:F], wil["lvalid"][:F])
        rp, rok = fill_missing(wil["rhand_pose"][:F], wil["rvalid"][:F])
        if args.hands_mean == "add":
            lp, rp = lp + lmean, rp + rmean
        elif args.hands_mean == "sub":
            lp, rp = lp - lmean, rp - rmean

        bkp = BACKUP / f"{s}.npz"
        if not bkp.exists():
            shutil.copy(bp, bkp)

        for k in list(body.keys()):
            if body[k].ndim >= 1 and body[k].shape[0] == Fb:
                body[k] = body[k][:F]
        if lok:
            body["smplx_lhand_pose"] = lp.astype(np.float32)
        if rok:
            body["smplx_rhand_pose"] = rp.astype(np.float32)
        np.savez_compressed(bp, **body)
        print(f"graft {s}: F={F} L={lok}({int(wil['lvalid'][:F].sum())}/{F}) "
              f"R={rok}({int(wil['rvalid'][:F].sum())}/{F}) mean={args.hands_mean}")


if __name__ == "__main__":
    main()
