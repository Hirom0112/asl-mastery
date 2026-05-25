"""Live webcam demo of the from-scratch pipeline.

Loads:
  - hand_det v2 (best_ema.pt)        — heatmap+size CenterNet
  - hand_landmarks v0 (best.pt)       — 21 keypoints / hand
  - pose v0 (best.pt)                 — 8 body keypoints
  - sign_classifier v0 (best.pt)      — TCN classifier (optional, --no-classifier to disable)

Runs an OpenCV window with overlays drawn on the webcam frame.
A 32-frame rolling trajectory buffer feeds the classifier; the top-3
sign predictions appear in the corner with their probabilities.

Hotkeys:
  q  quit
  m  toggle mirror (laptop webcams are usually mirrored)
  c  toggle classifier display

Usage:
  python -m scripts.live_demo
  python -m scripts.live_demo --camera 1 --no-classifier
"""
from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from time import time
from typing import Optional

import cv2
import numpy as np
import torch

from training.detectors.hand_detector import HandDetector
from training.detectors.hand_landmarks import HandLandmarkRegressor
from training.detectors.pose_detector import PoseRegressor
from training.detectors.sign_classifier import SignClassifier
from training.detectors.sign_matcher import trajectory_from_frames
from training.detectors._pose_inference import (
    pose_on_upper_body_crop, pose_batched_with_bboxes, PoseEMA)
from training.detectors.face_detector import FaceDetector
from training.detectors._hand_tracking import HandTracker
from training.detectors.extract_trajectories_v2 import (
    _detect_hands_batched, _landmarks_batched)

TIME_STEPS = 32

# Below this top-face confidence we treat the face as absent (raw_face=None)
# so the smoothed box decays and we render nothing rather than freeze on a
# stale/broken anchor (the 3 face dots + cyan body dots key off face_bbox).
FACE_CONF_THRESHOLD = 0.30

# 21-point hand skeleton (MediaPipe-style topology)
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]
POSE_EDGES = [
    (0, 1),  # nose -> neck
    (1, 2), (1, 3),  # neck -> shoulders
    (2, 4), (3, 5),  # shoulders -> elbows
    (4, 6), (5, 7),  # elbows -> wrists
]


def _load_pt(path: Path, model: torch.nn.Module, device: str) -> torch.nn.Module:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    # strip "module." prefix if present
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  missing keys in {path.name}: {len(missing)} (first few: {missing[:3]})")
    if unexpected:
        print(f"  unexpected keys in {path.name}: {len(unexpected)} (first few: {unexpected[:3]})")
    model.eval()
    return model.to(device)


def _frame_to_tensor(bgr: np.ndarray, device: str) -> torch.Tensor:
    """BGR uint8 (H,W,3) → CHW float [0,1] on device."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0).to(device, non_blocking=True)


@torch.no_grad()
def _box_iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0:
        return 0.0
    aa = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    ba = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    return inter / (aa + ba - inter + 1e-9)


def detect_hands(detector: HandDetector, frame_chw: torch.Tensor,
                 threshold: float = 0.15, top_k: int = 4,
                 second_hand_threshold: float = 0.22,
                 max_hands: int = 2,
                 iou_dedup: float = 0.35,
                 ):
    """Returns (boxes, scores). Thresholds raised vs the old 0.10/0.10: the
    fine-tuned detector is better-calibrated, so a higher bar — especially on
    the 2nd hand — cuts the phantom/noise detections."""
    H, W = frame_chw.shape[-2:]
    x = torch.nn.functional.interpolate(frame_chw, size=(320, 320),
                                        mode="bilinear", align_corners=False)
    out = detector(x)
    prob = torch.sigmoid(out["heatmap"])  # (1,1,80,80)
    pooled = torch.nn.functional.max_pool2d(prob, 3, 1, 1)
    is_peak = (prob == pooled) & (prob >= threshold)
    peak_scores = torch.where(is_peak, prob, torch.full_like(prob, -1.0))
    flat = peak_scores.flatten(1)
    vals, idxs = flat.topk(top_k, dim=1)
    sx = W / 320.0
    sy = H / 320.0
    stride = HandDetector.STRIDE
    grid = prob.shape[-1]
    scored = []
    for k in range(top_k):
        s = float(vals[0, k].item())
        if s < 0:
            continue
        flat_idx = int(idxs[0, k].item())
        cy, cx = divmod(flat_idx, grid)
        w = float(out["size"][0, 0, cy, cx].item())
        h = float(out["size"][0, 1, cy, cx].item())
        xc = (cx + 0.5) * stride
        yc = (cy + 0.5) * stride
        scored.append((s, ((xc - w / 2) * sx, (yc - h / 2) * sy,
                           (xc + w / 2) * sx, (yc + h / 2) * sy)))
    # Keep the top box, then add further boxes only if they clear the 2nd-hand
    # bar AND don't overlap an already-kept box (IoU dedup) — otherwise a single
    # large/near hand throws two peaks and we draw two boxes on one hand.
    kept: list = []
    for s, box in scored:
        if kept and s < second_hand_threshold:
            break
        if any(_box_iou(box, kb) > iou_dedup for _, kb in kept):
            continue  # same hand as an already-kept box → drop the duplicate
        kept.append((s, box))
        if len(kept) >= max_hands:
            break
    boxes = [b for _, b in kept]
    scores = [s for s, _ in kept]
    return boxes, scores


@torch.no_grad()
def landmarks_for_bboxes(regressor: HandLandmarkRegressor,
                         frame_chw: torch.Tensor,
                         bboxes: list[tuple[float, float, float, float]],
                         pad_frac: float = 0.20,
                         return_visibility: bool = False,
                         return_depth: bool = False,
                         ):
    """Returns list of 21-keypoint lists in original-image pixel coords.
    If return_visibility, also returns per-hand 21 visibility probs (0-1).
    If return_depth (3D model only), also returns per-hand 21 depth values
    (root-relative, palm-scaled z); None per hand if the model has no z head."""
    if not bboxes:
        empties = [[]]
        if return_visibility:
            empties.append([])
        if return_depth:
            empties.append([])
        return empties[0] if len(empties) == 1 else tuple(empties)
    _, _, H, W = frame_chw.shape
    crops = []
    crop_meta = []
    for x0, y0, x1, y1 in bboxes:
        bw, bh = x1 - x0, y1 - y0
        pad = pad_frac * max(bw, bh)
        cx0 = max(0, int(x0 - pad))
        cy0 = max(0, int(y0 - pad))
        cx1 = min(W, int(x1 + pad))
        cy1 = min(H, int(y1 + pad))
        if cx1 <= cx0 or cy1 <= cy0:
            cx0, cy0, cx1, cy1 = 0, 0, W, H
        crop = frame_chw[0, :, cy0:cy1, cx0:cx1].unsqueeze(0)
        crop = torch.nn.functional.interpolate(
            crop, size=(HandLandmarkRegressor.INPUT_SIZE,
                        HandLandmarkRegressor.INPUT_SIZE),
            mode="bilinear", align_corners=False,
        )
        crops.append(crop)
        crop_meta.append((cx0, cy0, cx1 - cx0, cy1 - cy0))
    batch = torch.cat(crops, 0)
    out = regressor(batch)
    coords = out["coords"].cpu().numpy()  # (K, 21, 2) in [0,1]
    vis = None
    if "visibility" in out:
        vis = torch.sigmoid(out["visibility"]).cpu().numpy()  # (K, 21) in [0,1]
    depth = out["depth"].cpu().numpy() if "depth" in out else None  # (K, 21)
    result, vis_result, depth_result = [], [], []
    for i, ((cx0, cy0, cw, ch), kps) in enumerate(zip(crop_meta, coords)):
        result.append([[float(cx0 + kx * cw), float(cy0 + ky * ch)]
                       for kx, ky in kps])
        vis_result.append([float(v) for v in vis[i]] if vis is not None
                          else [1.0] * len(kps))
        depth_result.append([float(z) for z in depth[i]] if depth is not None
                            else None)
    outs = [result]
    if return_visibility:
        outs.append(vis_result)
    if return_depth:
        outs.append(depth_result)
    return outs[0] if len(outs) == 1 else tuple(outs)


@torch.no_grad()
def pose_for_frame(regressor: PoseRegressor,
                   frame_chw: torch.Tensor,
                   hand_bboxes: list[tuple[float, float, float, float]],
                   face_bbox: tuple[float, float, float, float] | None = None,
                   ) -> list[list[float]]:
    # Prefer a FACE-anchored crop (stable, independent of hand gestures) so
    # the body stops chasing the hands; fall back to the hand-derived crop.
    return pose_on_upper_body_crop(regressor, frame_chw, hand_bboxes,
                                   face_bbox=face_bbox)


def anchor_pose_wrists(pose_kps: list[list[float]],
                       hand_kps: list[list[list[float]]]) -> list[list[float]]:
    """Snap the pose wrist points (idx 6=r_wr, 7=l_wr) to each hand's actual
    WRIST landmark (hand keypoint 0 = wrist, the base of the hand) — NOT the
    palm/box center. This connects the arm to the wrist anatomically instead
    of overshooting to the palm."""
    if not pose_kps or not hand_kps:
        return pose_kps
    # hand keypoint 0 is the wrist in the 21-point topology.
    wrists = [k[0] for k in hand_kps if k and len(k) > 0]
    used = set()
    for wi in (6, 7):
        if wi >= len(pose_kps):
            continue
        wx, wy = pose_kps[wi]
        best, bd = None, 1e18
        for ci, (hx, hy) in enumerate(wrists):
            if ci in used:
                continue
            d = (wx - hx) ** 2 + (wy - hy) ** 2
            if d < bd:
                bd, best = d, ci
        if best is not None:
            pose_kps[wi] = [float(wrists[best][0]), float(wrists[best][1])]
            used.add(best)
    return pose_kps


def _depth_color(z: float, zmin: float = -0.8, zmax: float = 0.8) -> tuple:
    """Map root-relative palm-scaled depth z → BGR. Near-camera (low z) = red,
    far (high z) = blue. (Sign convention may flip; what matters is that a
    keypoint changes color as it moves in depth.)"""
    t = max(0.0, min(1.0, (z - zmin) / (zmax - zmin)))
    return (int(255 * t), 0, int(255 * (1 - t)))  # BGR: red→blue


def draw_overlays(canvas: np.ndarray,
                  bboxes: list,
                  hand_kps: list[list[list[float]]],
                  pose_kps: list[list[float]],
                  scores: list | None = None,
                  pose_mode: str = "simple",
                  hand_vis: list | None = None,
                  hand_depth: list | None = None) -> None:
    H, W = canvas.shape[:2]
    pc = (255, 200, 0)
    if pose_mode == "full":
        # Legacy: all 8 pose points + arm chain (shoulder->elbow->wrist).
        for x, y in pose_kps:
            cv2.circle(canvas, (int(x), int(y)), 4, pc, -1)
        for a, b in POSE_EDGES:
            if a < len(pose_kps) and b < len(pose_kps):
                xa, ya = pose_kps[a]
                xb, yb = pose_kps[b]
                cv2.line(canvas, (int(xa), int(ya)), (int(xb), int(yb)), pc, 2)
    # Simple mode: neck + shoulders are NOT drawn from the (collapsing) pose
    # regressor — they're drawn face-anchored in the main loop
    # (body_points_from_face), together with the head/forehead/nose/chin dots.
    # Hands
    palette = [(0, 255, 0), (0, 200, 255)]
    for hi, (bbox, kps) in enumerate(zip(bboxes, hand_kps)):
        color = palette[hi % len(palette)]
        vis = hand_vis[hi] if (hand_vis and hi < len(hand_vis)) else None
        x0, y0, x1, y1 = [int(v) for v in bbox]
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)
        if scores is not None and hi < len(scores):
            cv2.putText(canvas, f"{scores[hi]:.2f}", (x0, max(12, y0 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        # Live visibility readout: lowest-confidence (most-occluded) keypoint.
        # Curl a finger — if the head works, this number should drop.
        if vis is not None:
            cv2.putText(canvas, f"vis_min {min(vis):.2f}", (x0, y1 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        def _occ(i):  # keypoint i occluded (visibility model says low)?
            return vis is not None and i < len(vis) and vis[i] < 0.5
        for a, b in HAND_EDGES:  # hand skeleton lines (back on)
            if a < len(kps) and b < len(kps):
                ec = (110, 110, 110) if (_occ(a) or _occ(b)) else color
                cv2.line(canvas, (int(kps[a][0]), int(kps[a][1])),
                         (int(kps[b][0]), int(kps[b][1])), ec, 1)
        # Depth-viz: color each keypoint by predicted z (root-relative,
        # palm-scaled). Ramp red=near-camera → blue=far. Curl a finger and
        # its tip should change color as it moves in depth.
        depth = hand_depth[hi] if (hand_depth and hi < len(hand_depth)) else None
        if depth is not None:
            # Same restraint as normal mode: uncertain (occluded) keypoints
            # draw as small hollow rings, confident ones as small solid dots —
            # just colored by depth instead of the flat palette. (Avoids the
            # "all 21 huge solid dots" look that made errors scream.)
            for ki, (x, y) in enumerate(kps):
                if ki >= len(depth):
                    continue
                dc = _depth_color(depth[ki])
                if _occ(ki):
                    cv2.circle(canvas, (int(x), int(y)), 3, dc, 1)      # hollow
                else:
                    cv2.circle(canvas, (int(x), int(y)), 2, dc, -1)     # solid
            # numeric z only at CONFIDENT fingertips (thumb..pinky)
            for tip in (4, 8, 12, 16, 20):
                if tip < len(kps) and tip < len(depth) and not _occ(tip):
                    cv2.putText(canvas, f"{depth[tip]:+.2f}",
                                (int(kps[tip][0]) + 4, int(kps[tip][1])),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            continue
        for ki, (x, y) in enumerate(kps):
            if _occ(ki):   # occluded → hollow grey dot (model is unsure)
                cv2.circle(canvas, (int(x), int(y)), 3, (110, 110, 110), 1)
            else:          # visible → solid colored dot
                cv2.circle(canvas, (int(x), int(y)), 2, color, -1)


def _frac_inside(box, face) -> float:
    """Fraction of `box`'s area that overlaps `face`. Used to drop hand
    false-positives that land on the ear/jaw/beard when the head turns."""
    ax1, ay1, ax2, ay2 = box
    bx1, by1, bx2, by2 = face
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    area = max(1e-6, (ax2 - ax1) * (ay2 - ay1))
    return (iw * ih) / area


def smooth_box(state: dict, box, alpha: float = 0.4, max_miss: int = 8):
    """EMA-smooth a detector box across frames so derived points don't jitter.
    Holds the last box for up to `max_miss` missed frames before clearing."""
    if box is None:
        state["miss"] = state.get("miss", 0) + 1
        if state["miss"] > max_miss:
            state["box"] = None
        return state.get("box")
    prev = state.get("box")
    sm = tuple(box) if prev is None else tuple(
        alpha * float(b) + (1 - alpha) * float(p) for b, p in zip(box, prev))
    state["box"], state["miss"] = sm, 0
    return sm


def order_pose_lr(pose_kps):
    """Lock shoulders/elbows/wrists into a consistent screen left/right order.
    The pose regressor swaps which point is 'right' vs 'left' shoulder between
    frames; without this, the dots cross AND the EMA averages across swapped
    identities (jitter amplified). Sorting by x first makes both behave."""
    pts = [list(p) for p in pose_kps]
    if len(pts) >= 4 and pts[2][0] > pts[3][0]:  # kp2/3 = shoulders
        pts[2], pts[3] = pts[3], pts[2]
        if len(pts) >= 6:                         # kp4/5 = elbows
            pts[4], pts[5] = pts[5], pts[4]
        if len(pts) >= 8:                         # kp6/7 = wrists
            pts[6], pts[7] = pts[7], pts[6]
    return pts


def body_points_from_face(face_bbox):
    """Neck + shoulders estimated geometrically from the (smoothed) face box.
    The pose regressor collapses/inverts under gesture and frame-edge motion;
    the face box is stable, so these reference dots never jump. Returns
    (neck, left_shoulder, right_shoulder) in screen coords, or None."""
    if face_bbox is None:
        return None
    fx0, fy0, fx1, fy1 = face_bbox
    fw, fh = fx1 - fx0, fy1 - fy0
    if fw <= 0 or fh <= 0:
        return None
    cx = (fx0 + fx1) / 2.0
    sh_y = int(fy1 + 0.55 * fh)   # raised (was 0.85*fh — sat too low)
    neck = (int(cx), int(fy1 + 0.25 * fh))
    l_sh = (int(cx - 1.15 * fw), sh_y)
    r_sh = (int(cx + 1.15 * fw), sh_y)
    return neck, l_sh, r_sh


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--ckpt-dir", type=Path, default=Path("data/ckpts"))
    ap.add_argument("--no-classifier", action="store_true")
    ap.add_argument("--pose-mode", choices=["simple", "full"], default="simple",
                    help="simple = head+shoulders ref + shoulder->hand-wrist "
                         "(drops low-value pose elbow/wrist); full = legacy arm chain.")
    ap.add_argument("--device", default=None)
    ap.add_argument("--depth-viz", action="store_true",
                    help="load the 3D landmark model (hand_landmarks_3d_v1_best.pt) "
                         "and color keypoints by predicted depth (red=near, blue=far)")
    ap.add_argument("--crop-pad", type=float, default=0.20,
                    help="padding fraction around the detector box before the "
                         "landmark crop (training used ~0.20). Lower it (e.g. 0.05) "
                         "to test whether the detector over-boxes — if keypoints "
                         "stop bunching in the palm, the box was too loose.")
    ap.add_argument("--match-extraction", action="store_true",
                    help="Run the EXACT training extraction pipeline "
                         "(extract_trajectories_v2): hand-anchored pose, NO face "
                         "detector / tracker / EMA, detector thresh 0.02 / top-2 / "
                         "no-dedup. This is what the v2/v3 classifier was actually "
                         "trained AND measured on (75.8/92.7). Compare it against "
                         "the default face-anchored demo to judge which the browser "
                         "should use.")
    args = ap.parse_args()
    match_extraction = args.match_extraction
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    print("loading models…")
    hand_det = _load_pt(args.ckpt_dir / "hand_det_v2_best_ema.pt",
                        HandDetector(), device)
    # Positions ALWAYS come from the proven 2D model (best in-the-wild tracking).
    hand_lm = _load_pt(args.ckpt_dir / "hand_landmarks_v0_best.pt",
                      HandLandmarkRegressor(), device)
    # Depth-viz: load the 3D model ONLY for its depth output; it colors the
    # 2D model's keypoints. (The 3D model localizes worse in-the-wild, so we
    # do NOT use it for positions.)
    hand_lm_3d = None
    if args.depth_viz:
        _lm3d = args.ckpt_dir / "hand_landmarks_3d_v1_best.pt"
        if not _lm3d.exists():
            _lm3d = args.ckpt_dir / "hand_landmarks_3d_v0_best.pt"
        hand_lm_3d = _load_pt(_lm3d, HandLandmarkRegressor(predict_z=True), device)
        print(f"DEPTH-VIZ: positions from 2D model (stable tracking); "
              f"depth color from {_lm3d.name} (red=near, blue=far)")
    pose = _load_pt(args.ckpt_dir / "pose_v0_best.pt", PoseRegressor(), device)
    pose_ema = PoseEMA(alpha=0.25, n_kp=8)  # heavier smoothing (was 0.4)
    tracker = HandTracker()  # stable hand identity across frames
    # Face detector anchors the pose crop on the (stable) face instead of the
    # hands. Optional — if the ckpt isn't present, pose falls back to hands.
    face_det = None
    _face_ckpt = args.ckpt_dir / "face_det_v0_best.pt"
    if _face_ckpt.exists():
        face_det = _load_pt(_face_ckpt, FaceDetector(), device)
        print("face detector loaded (pose anchored on face)")
    else:
        print("no face detector ckpt — pose falls back to hand-anchored crop")
    # 98-point face landmark regressor (chin/forehead/cheek/...) — for viz.
    face_lm = None
    _face_lm_ckpt = args.ckpt_dir / "face_landmarks_v0_best.pt"
    if _face_lm_ckpt.exists():
        face_lm = _load_pt(_face_lm_ckpt,
                           HandLandmarkRegressor(num_keypoints=98), device)
        print("face landmark regressor loaded (98 pts)")
    classifier: Optional[SignClassifier] = None
    classes: list[str] = []
    clf_norm = "body"  # feature normalization the loaded classifier expects
    if not args.no_classifier:
        # Prefer the v2 hand-relative classifier (108D, sem_lex_top80, 72.5%
        # top-1 / 89.9% top-5 signer-disjoint). Fall back to the old v0
        # (100D body-norm) if the v2 artifact isn't present.
        _v2_pt = args.ckpt_dir / "sign_classifier_v2_best.pt"
        if _v2_pt.exists():
            ckpt = torch.load(_v2_pt, map_location=device, weights_only=False)
            classes = ckpt["classes"]
            num_features = ckpt["num_features"]  # 108
            clf_norm = "hand" if ckpt.get("feature_version", "").endswith(
                "108d") else "body"
            classifier = SignClassifier(num_features=num_features,
                                        num_classes=len(classes),
                                        hidden=256, num_blocks=5)
            classifier.load_state_dict(ckpt["model"])
            classifier.eval().to(device)
            print(f"classifier v2 loaded — {len(classes)} classes, "
                  f"{num_features}D, norm={clf_norm} "
                  f"({ckpt.get('feature_version')})")
        else:
            classes = json.loads(
                (args.ckpt_dir / "sign_classifier_v0_classes.json").read_text()
            )
            # v0 checkpoint config (best result: 10.1% top-1)
            classifier = SignClassifier(num_features=100,
                                        num_classes=len(classes),
                                        hidden=192, num_blocks=4)
            ckpt = torch.load(args.ckpt_dir / "sign_classifier_v0_best.pt",
                              map_location=device, weights_only=False)
            classifier.load_state_dict(ckpt["state_dict"])
            classifier.eval().to(device)
            print(f"classifier v0 loaded — {len(classes)} classes (100D, "
                  f"norm=body)")

    # Try multiple backends — macOS often needs AVFoundation explicitly.
    backends = [
        ("AVFOUNDATION", cv2.CAP_AVFOUNDATION),
        ("ANY", cv2.CAP_ANY),
        ("V4L2", getattr(cv2, "CAP_V4L2", 200)),
        ("DSHOW", getattr(cv2, "CAP_DSHOW", 700)),
    ]
    cap = None
    for cam_idx in [args.camera, 0, 1, 2]:
        for name, backend in backends:
            test = cv2.VideoCapture(cam_idx, backend)
            if test.isOpened():
                ok, _ = test.read()
                if ok:
                    print(f"opened camera {cam_idx} via {name}")
                    cap = test
                    break
                test.release()
            else:
                test.release()
        if cap is not None:
            break
    if cap is None:
        print("could not open any camera with any backend.")
        print("On macOS: System Settings -> Privacy & Security -> Camera")
        print("  -> enable the terminal app you're running this from")
        print("  -> then RELAUNCH the terminal (permission is per-process).")
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    traj_buffer: deque = deque(maxlen=TIME_STEPS * 2)
    face_box_state: dict = {"box": None, "miss": 0}
    mirror = True
    show_classifier = bool(classifier is not None)
    last_pred_text = ""
    last_pred_time = 0.0

    if match_extraction:
        print("MATCH-EXTRACTION MODE: hand-anchored pose, NO face/tracker/EMA — "
              "the EXACT pipeline the v2/v3 classifier (75.8/92.7) was trained on.")
    print("press 'q' to quit, 'm' to toggle mirror, 'c' to toggle classifier")
    fps_t = time()
    fps = 0.0
    frame_i = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if mirror:
            frame = cv2.flip(frame, 1)
        t0 = time()
        frame_chw = _frame_to_tensor(frame, device)
        try:
            if match_extraction:
                # EXACTLY the training extraction pipeline
                # (extract_trajectories_v2): hand-anchored pose, NO face
                # detector / tracker / EMA, detector thresh 0.02 / top-2 /
                # no-dedup. What the v2/v3 classifier was trained + measured on.
                per_b = _detect_hands_batched(hand_det, frame_chw, device)
                per_h = _landmarks_batched(hand_lm, frame_chw, per_b, device)
                per_p = pose_batched_with_bboxes(pose, frame_chw, per_b, device)
                bboxes = list(per_b[0])
                hand_kps = [hd["keypoints"] for hd in per_h[0]]
                hand_vis = [[1.0] * 21 for _ in hand_kps]
                hand_depth = None
                pose_kps = per_p[0]
                face_bbox = None
                scores = []
            else:
                # Face first (smoothed): anchors the pose crop AND lets us reject
                # hand false-positives that land on the face/ear when the head turns.
                raw_face = None
                if face_det is not None:
                    # Get several face candidates, then anchor on the PERSON IN
                    # FRONT = the LARGEST face above threshold (a background face
                    # scores high too but is small) — stops the pose/face dots from
                    # jumping to people/objects behind you.
                    f_boxes, f_scores = detect_hands(
                        face_det, frame_chw, max_hands=3,
                        second_hand_threshold=FACE_CONF_THRESHOLD)
                    cands = [b for b, s in zip(f_boxes, f_scores)
                             if s >= FACE_CONF_THRESHOLD]
                    if cands:
                        raw_face = max(cands, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
                face_bbox = (smooth_box(face_box_state, raw_face)
                             if face_det is not None else None)

                bboxes, scores = detect_hands(hand_det, frame_chw)
                # Drop ear/jaw false-positives WITHOUT dropping real signing hands
                # at the face (many ASL signs are face-located). An ear blob is
                # SMALL relative to the face; a real hand is large. So reject only
                # boxes that are both mostly inside the (inflated) face AND small.
                if face_bbox is not None:
                    fgx1, fgy1, fgx2, fgy2 = face_bbox
                    face_area = max(1e-6, (fgx2 - fgx1) * (fgy2 - fgy1))
                    mw, mh = 0.35 * (fgx2 - fgx1), 0.25 * (fgy2 - fgy1)
                    face_guard = (fgx1 - mw, fgy1 - mh, fgx2 + mw, fgy2 + mh)
                    bboxes, scores = list(bboxes), list(scores)
                    keep = [i for i, b in enumerate(bboxes)
                            if not (_frac_inside(b, face_guard) > 0.6
                                    and (b[2] - b[0]) * (b[3] - b[1]) < 0.22 * face_area)]
                    bboxes = [bboxes[i] for i in keep]
                    scores = [scores[i] for i in keep]
                hand_kps, hand_vis = landmarks_for_bboxes(
                    hand_lm, frame_chw, bboxes, return_visibility=True,
                    pad_frac=args.crop_pad)
                hand_depth = None
                if args.depth_viz and hand_lm_3d is not None:
                    # positions stay from the 2D model; pull only depth from the 3D
                    # model (per-keypoint z, same 21-index topology → colors line up)
                    _, hand_depth = landmarks_for_bboxes(
                        hand_lm_3d, frame_chw, bboxes, return_depth=True,
                        pad_frac=args.crop_pad)
                    depth_by_box = {tuple(b): d for b, d in zip(bboxes, hand_depth)}
                # Map visibility to each bbox BEFORE the tracker reorders (the
                # tracker leaves bbox values untouched, so we re-align by value).
                vis_by_box = {tuple(b): v for b, v in zip(bboxes, hand_vis)}
                # Stable hand identity across frames (fixes the cross/swap/360).
                bboxes, hand_kps = tracker.update(bboxes, hand_kps,
                                                  frame.shape[1], frame.shape[0])
                hand_vis = [vis_by_box.get(tuple(b), [1.0] * 21) for b in bboxes]
                if args.depth_viz:
                    hand_depth = [depth_by_box.get(tuple(b)) for b in bboxes]
                pose_kps = pose_for_frame(pose, frame_chw, bboxes, face_bbox=face_bbox)
                pose_kps = order_pose_lr(pose_kps)   # consistent L/R before smoothing
                pose_kps = pose_ema.update(pose_kps)
                # Snap pose wrists to each hand's WRIST landmark (kp 0), not palm.
                pose_kps = anchor_pose_wrists(pose_kps, hand_kps)
        except Exception as e:
            cv2.putText(frame, f"err: {e}", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow("asl-live", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue
        draw_overlays(frame, bboxes, hand_kps, pose_kps, scores,
                      pose_mode=args.pose_mode, hand_vis=hand_vis,
                      hand_depth=hand_depth)
        # Three STABLE face points derived from the face-detector bbox:
        # forehead (top-center), nose (center), chin (bottom-center). We do NOT
        # render the 98-pt regressor cloud — it's noisy (16.76px) and the WFLW
        # 98-pt scheme has no forehead point, so its raw output spins/collapses.
        if face_bbox is not None:
            fx1, fy1, fx2, fy2 = face_bbox
            fcx = int((fx1 + fx2) / 2)
            fh = fy2 - fy1
            forehead = (fcx, int(fy1 + 0.12 * fh))
            nose = (fcx, int((fy1 + fy2) / 2))
            chin = (fcx, int(fy2 - 0.05 * fh))
            for pt in (forehead, nose, chin):
                cv2.circle(frame, pt, 4, (255, 0, 255), -1)  # magenta, stable
            # Neck + shoulders, anchored to the stable face box (cyan).
            if args.pose_mode == "simple":
                body = body_points_from_face(face_bbox)
                if body is not None:
                    for pt in body:
                        cv2.circle(frame, pt, 5, (255, 255, 0), -1)  # cyan

        if match_extraction and pose_kps and len(pose_kps) >= 4:
            # No face detector in this mode — draw the body anchor the CLASSIFIER
            # actually uses: the pose regressor's nose (head, magenta) + neck and
            # both shoulders (cyan). These ride the hand-anchored upper-body crop,
            # so they move with the hands — exactly the (less stable) signal the
            # trained model consumed. Contrast with the steady face-anchored dots.
            nose_pt = pose_kps[0]
            cv2.circle(frame, (int(nose_pt[0]), int(nose_pt[1])), 4, (255, 0, 255), -1)
            for idx in (1, 2, 3):  # neck, r_shoulder, l_shoulder
                bx, by = pose_kps[idx]
                cv2.circle(frame, (int(bx), int(by)), 5, (255, 255, 0), -1)

        # Build a frame record in the same shape extract_trajectories_v2 writes
        rec = {
            "frame_idx": frame_i,
            "hands": [{"bbox": list(b), "keypoints": kp}
                      for b, kp in zip(bboxes, hand_kps)],
            "pose": pose_kps,
        }
        traj_buffer.append(rec)
        frame_i += 1

        if classifier is not None and show_classifier and len(traj_buffer) >= TIME_STEPS:
            feats = trajectory_from_frames(list(traj_buffer)[-TIME_STEPS:],
                                           TIME_STEPS, norm=clf_norm)
            x = torch.from_numpy(feats).unsqueeze(0).float().to(device)
            with torch.no_grad():
                logits = classifier(x)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            top3 = probs.argsort()[-3:][::-1]
            lines = [f"{classes[i]}: {probs[i]*100:.1f}%" for i in top3]
            last_pred_text = " | ".join(lines)
            last_pred_time = time()

        # HUD
        infer_ms = (time() - t0) * 1000
        if time() - fps_t > 1.0:
            fps = frame_i / max(time() - fps_t + 1e-6, 1e-6)
            fps_t = time()
            frame_i = 0
        _conf = max(scores) if scores else 0.0
        hud = (f"{infer_ms:.0f}ms/frame  {fps:.1f} fps  hands={len(bboxes)}  "
               f"conf={_conf:.2f}  mirror={mirror}")
        cv2.putText(frame, hud, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 2)
        if show_classifier and last_pred_text:
            for li, line in enumerate(last_pred_text.split(" | ")):
                cv2.putText(frame, line, (8, 60 + li * 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (50, 255, 255), 2)

        cv2.imshow("asl-live", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("m"):
            mirror = not mirror
        elif key == ord("c"):
            show_classifier = not show_classifier

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
