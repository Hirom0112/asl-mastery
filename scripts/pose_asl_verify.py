"""Verify person-bbox fix on ASL deployment frames.

Pipeline per frame:
  1. hand_det v2 → hand bbox(es)
  2. Construct upper-body bbox heuristically from hand bboxes:
       - union all hand bboxes
       - expand UP by 3.5 × hand_h (head ~3 hand-heights above wrist)
       - expand L/R by 1.5 × hand_w (shoulders ~2 hand-widths wider than wrists)
       - expand DOWN by 0.5 × hand_h
       - pad 20% (matches training)
  3. Crop to that bbox, resize 256×256, run pose
  4. Draw overlays on both full-frame baseline and person-cropped variant

If person-cropped pose locks on, fix path is confirmed for deployment too.
"""
from __future__ import annotations

import argparse
import random
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

from training.detectors.hand_detector import HandDetector
from training.detectors.pose_detector import PoseRegressor

POSE_EDGES = [(0, 1), (1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7)]
KPT_NAMES = ["nose", "neck", "r_sh", "l_sh", "r_el", "l_el", "r_wr", "l_wr"]


def load_pose(ckpt: Path, device: str) -> PoseRegressor:
    model = PoseRegressor()
    obj = torch.load(ckpt, map_location=device, weights_only=False)
    sd = obj.get("model", obj.get("state_dict", obj))
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    return model.eval().to(device)


def load_hand_det(ckpt: Path, device: str) -> HandDetector:
    model = HandDetector()
    obj = torch.load(ckpt, map_location=device, weights_only=False)
    sd = obj.get("model", obj.get("state_dict", obj.get("model_state_dict", obj)))
    if isinstance(sd, dict) and "ema_state_dict" in obj:
        sd = obj["ema_state_dict"]
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    return model.eval().to(device)


@torch.no_grad()
def detect_hands(detector: HandDetector, bgr: np.ndarray, device: str,
                 threshold: float = 0.10, max_hands: int = 2,
                 ) -> list[tuple[float, float, float, float]]:
    H, W = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    t = torch.nn.functional.interpolate(t, size=(320, 320),
                                        mode="bilinear", align_corners=False).to(device)
    out = detector(t)
    prob = torch.sigmoid(out["heatmap"])
    pooled = torch.nn.functional.max_pool2d(prob, 3, 1, 1)
    is_peak = (prob == pooled) & (prob >= threshold)
    peak_scores = torch.where(is_peak, prob, torch.full_like(prob, -1.0))
    vals, idxs = peak_scores.flatten(1).topk(4, dim=1)
    sx = W / 320.0; sy = H / 320.0
    stride = HandDetector.STRIDE
    grid = prob.shape[-1]
    scored = []
    for k in range(4):
        s = float(vals[0, k].item())
        if s < 0: continue
        flat_idx = int(idxs[0, k].item())
        cy, cx = divmod(flat_idx, grid)
        w = float(out["size"][0, 0, cy, cx].item())
        h = float(out["size"][0, 1, cy, cx].item())
        xc = (cx + 0.5) * stride; yc = (cy + 0.5) * stride
        scored.append((s, ((xc - w / 2) * sx, (yc - h / 2) * sy,
                           (xc + w / 2) * sx, (yc + h / 2) * sy)))
    return [b for _, b in scored[:max_hands]]


def upper_body_bbox(hand_bboxes: list[tuple[float, float, float, float]],
                    img_w: int, img_h: int
                    ) -> tuple[int, int, int, int] | None:
    if not hand_bboxes:
        return None
    xs0 = [b[0] for b in hand_bboxes]; ys0 = [b[1] for b in hand_bboxes]
    xs1 = [b[2] for b in hand_bboxes]; ys1 = [b[3] for b in hand_bboxes]
    x0, y0, x1, y1 = min(xs0), min(ys0), max(xs1), max(ys1)
    hw = max(b[2] - b[0] for b in hand_bboxes)
    hh = max(b[3] - b[1] for b in hand_bboxes)
    # Expand to estimated upper-body
    x0 -= 1.5 * hw
    x1 += 1.5 * hw
    y0 -= 3.5 * hh   # reach above head
    y1 += 0.5 * hh
    # 20% pad (matches training crop)
    w = x1 - x0; h = y1 - y0
    pad = 0.20 * max(w, h)
    x0 -= pad; y0 -= pad; x1 += pad; y1 += pad
    x0 = max(0, int(x0)); y0 = max(0, int(y0))
    x1 = min(img_w, int(x1)); y1 = min(img_h, int(y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


@torch.no_grad()
def pose_on_crop(model: PoseRegressor, bgr_crop: np.ndarray,
                 device: str) -> list[list[float]]:
    h, w = bgr_crop.shape[:2]
    rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    t = torch.nn.functional.interpolate(t, size=(256, 256),
                                        mode="bilinear", align_corners=False)
    out = model(t.to(device))
    coords = out["coords"][0].cpu().numpy()
    return [[float(kx * w), float(ky * h)] for kx, ky in coords]


def draw_kps(canvas: np.ndarray, kps: list[list[float]]) -> None:
    for a, b in POSE_EDGES:
        cv2.line(canvas, (int(kps[a][0]), int(kps[a][1])),
                 (int(kps[b][0]), int(kps[b][1])), (0, 200, 255), 2)
    for i, (x, y) in enumerate(kps):
        cv2.circle(canvas, (int(x), int(y)), 4, (0, 255, 0), -1)
        cv2.putText(canvas, KPT_NAMES[i], (int(x) + 5, int(y) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)


def plausibility(kps: list[list[float]]) -> float:
    n, neck, rsh, lsh, rel, lel, rwr, lwr = kps
    checks = [
        n[1] < neck[1],
        neck[1] < (rsh[1] + lsh[1]) / 2,
        abs(rsh[1] - lsh[1]) < 0.2 * abs(rsh[0] - lsh[0] + 1e-3) + 20,
        rel[1] > rsh[1] - 20,
        lel[1] > lsh[1] - 20,
    ]
    return sum(checks) / len(checks)


def sample_asl_frames(clips_dir: Path, n: int, rng: random.Random
                      ) -> list[tuple[str, np.ndarray]]:
    mp4s = list(clips_dir.rglob("*.mp4"))
    if not mp4s:
        return []
    chosen = rng.sample(mp4s, min(n, len(mp4s)))
    frames = []
    for mp4 in chosen:
        cap = cv2.VideoCapture(str(mp4))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n_frames <= 0:
            cap.release(); continue
        mid = n_frames // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
        ok, fr = cap.read()
        cap.release()
        if ok and fr is not None:
            frames.append((f"ASL/{mp4.stem[:30]}", fr))
    return frames


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose-ckpt", type=Path,
                    default=Path("data/ckpts/pose_v0_best.pt"))
    ap.add_argument("--hand-ckpt", type=Path,
                    default=Path("data/ckpts/hand_det_v2_best_ema.pt"))
    ap.add_argument("--asl-dir", type=Path,
                    default=Path("dataset/clean/v2-yt/normalized_videos"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/probe"))
    ap.add_argument("--num-asl", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)

    print(f"[device] {device}")
    pose = load_pose(args.pose_ckpt, device)
    hand = load_hand_det(args.hand_ckpt, device)

    inputs = sample_asl_frames(args.asl_dir, args.num_asl, rng)
    print(f"[sample] {len(inputs)} ASL frames")
    if not inputs:
        print("[err] no inputs"); return 1

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir / f"pose_asl_verify_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = []
    print(f"\n{'image':40s} {'hands':>6s} {'full_s':>8s} {'crop_s':>8s}")
    for name, img in inputs:
        H, W = img.shape[:2]
        # FULL FRAME (current broken inference)
        side = min(H, W)
        cy0 = (H - side) // 2; cx0 = (W - side) // 2
        sq = img[cy0:cy0 + side, cx0:cx0 + side]
        kps_sq = pose_on_crop(pose, sq, device)
        kps_full = [[cx0 + p[0], cy0 + p[1]] for p in kps_sq]
        canvas_full = img.copy()
        draw_kps(canvas_full, kps_full)
        score_full = plausibility(kps_full)

        # PERSON-CROP (proposed fix)
        hand_bboxes = detect_hands(hand, img, device)
        bbox = upper_body_bbox(hand_bboxes, W, H)
        if bbox is None:
            score_crop = float("nan")
            canvas_crop = img.copy()
            cv2.putText(canvas_crop, "no hands detected", (8, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            bx0, by0, bx1, by1 = bbox
            crop = img[by0:by1, bx0:bx1]
            kps_pc = pose_on_crop(pose, crop, device)
            canvas_crop = crop.copy()
            draw_kps(canvas_crop, kps_pc)
            score_crop = plausibility(kps_pc)
            # Also draw bbox on full frame
            cv2.rectangle(canvas_full, (bx0, by0), (bx1, by1), (255, 0, 255), 2)
            for hb in hand_bboxes:
                cv2.rectangle(canvas_full,
                              (int(hb[0]), int(hb[1])),
                              (int(hb[2]), int(hb[3])), (0, 255, 255), 2)

        pairs.append((name, canvas_full, canvas_crop, score_full, score_crop))
        print(f"{name:40s} {len(hand_bboxes):>6d} {score_full:>8.2f} {score_crop:>8.2f}")

    # Side-by-side grid
    cell_h = 340
    pad_y = 30
    rows = len(pairs)
    grid_h = pad_y + rows * cell_h
    grid_w = 1100
    grid = np.full((grid_h, grid_w, 3), 240, dtype=np.uint8)
    cv2.putText(grid, "FULL FRAME (current broken inference)", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    cv2.putText(grid, "UPPER-BODY CROP (proposed fix)", (560, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    for ri, (name, full, crop, sf, sc) in enumerate(pairs):
        y0 = pad_y + ri * cell_h
        for ci, (im, score) in enumerate([(full, sf), (crop, sc)]):
            h, w = im.shape[:2]
            s = min(540 / w, (cell_h - 24) / h)
            nh, nw = int(h * s), int(w * s)
            rs = cv2.resize(im, (nw, nh))
            x0 = ci * 550 + 10
            grid[y0:y0 + nh, x0:x0 + nw] = rs
            label = f"{name}  score={score:.2f}" if not np.isnan(score) else name
            color = (0, 130, 0) if (not np.isnan(score) and score >= 0.7) else (0, 0, 200)
            cv2.putText(grid, label, (x0, y0 + nh + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    grid_path = out_dir / "compare.png"
    cv2.imwrite(str(grid_path), grid)

    full_scores = [p[3] for p in pairs if not np.isnan(p[3])]
    crop_scores = [p[4] for p in pairs if not np.isnan(p[4])]
    full_mean = np.mean(full_scores) if full_scores else float("nan")
    crop_mean = np.mean(crop_scores) if crop_scores else float("nan")
    print(f"\n[headline]")
    print(f"  mean plausibility — full-frame (current) : {full_mean:.2f}")
    print(f"  mean plausibility — upper-body crop (fix): {crop_mean:.2f}")
    print(f"\n[out]    {grid_path}")
    if crop_mean >= 0.7 and crop_mean - full_mean >= 0.2:
        print("\n[CONFIRMED] inference fix works on ASL frames too.")
        print("    Path: implement upper-body bbox stage in pose call sites.")
    elif crop_mean < 0.5:
        print("\n[REJECTED] inference fix doesn't help on ASL frames.")
    else:
        print("\n[PARTIAL] some lift but mediocre.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
