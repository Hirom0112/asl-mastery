"""Verify the pose_v0 collapse hypothesis: model was trained on person crops,
inference feeds full center-square frames.

We replicate the training crop pipeline (bbox from keypoint min/max, pad 20%,
resize 256x256) and re-run pose_v0. If pose locks onto the person properly,
the model is fine and only inference is wrong.
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

from training.detectors.pose_detector import PoseRegressor
from training.detectors.external_loaders.mpii_pose import load_pose_keypoints

POSE_EDGES = [(0, 1), (1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7)]
KPT_NAMES = ["nose", "neck", "r_sh", "l_sh", "r_el", "l_el", "r_wr", "l_wr"]


def load_pose(ckpt: Path, device: str) -> PoseRegressor:
    model = PoseRegressor()
    obj = torch.load(ckpt, map_location=device, weights_only=False)
    sd = obj.get("model", obj.get("state_dict", obj))
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    return model.eval().to(device)


@torch.no_grad()
def pose_on_crop(model: PoseRegressor, bgr_crop: np.ndarray,
                 device: str) -> list[list[float]]:
    """Run pose on an arbitrary-aspect crop. Replicates training pipeline:
    resize to 256x256 (not letterbox; training did interpolate-distort).
    Returns keypoints in crop pixel coordinates."""
    h, w = bgr_crop.shape[:2]
    rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    t = torch.nn.functional.interpolate(t, size=(256, 256),
                                        mode="bilinear", align_corners=False)
    out = model(t.to(device))
    coords = out["coords"][0].cpu().numpy()  # (8, 2) in [0,1]
    return [[float(kx * w), float(ky * h)] for kx, ky in coords]


def crop_with_pad(img: np.ndarray, bbox: tuple[float, float, float, float],
                  pad_frac: float = 0.20) -> tuple[np.ndarray, tuple[int, int]]:
    H, W = img.shape[:2]
    x0, y0, x1, y1 = bbox
    bw, bh = x1 - x0, y1 - y0
    pad = pad_frac * max(bw, bh)
    cx0 = max(0, int(x0 - pad))
    cy0 = max(0, int(y0 - pad))
    cx1 = min(W, int(x1 + pad))
    cy1 = min(H, int(y1 + pad))
    return img[cy0:cy1, cx0:cx1].copy(), (cx0, cy0)


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=Path("data/ckpts/pose_v0_best.pt"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/probe"))
    ap.add_argument("--num-mpii", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)

    print(f"[device] {device}")
    print(f"[load]   {args.ckpt}")
    model = load_pose(args.ckpt, device)

    print("[mpii]   loading annotations…")
    items = load_pose_keypoints()
    items_with_persons = [it for it in items if it.get("persons")]
    print(f"[mpii]   {len(items_with_persons)} items with persons")
    if not items_with_persons:
        print("[err] no MPII items"); return 1

    rng.shuffle(items_with_persons)
    chosen = items_with_persons[:args.num_mpii]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir / f"pose_crop_verify_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parent.parent

    pairs = []  # (full_with_overlay, crop_with_overlay, score_full, score_crop)
    print("\n[results]")
    print(f"{'image':30s} {'full_score':>12s} {'crop_score':>12s}")
    for it in chosen:
        img_path = repo_root / it["image_path"]
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        person = it["persons"][0]
        bbox = person["bbox"]

        # (1) Center-square-crop (current inference path) — full frame baseline
        H, W = img.shape[:2]
        side = min(H, W)
        cy0 = (H - side) // 2; cx0 = (W - side) // 2
        sq = img[cy0:cy0 + side, cx0:cx0 + side]
        kps_sq = pose_on_crop(model, sq, device)
        canvas_full = img.copy()
        # Map sq-relative kps back to original frame
        kps_full = [[cx0 + p[0], cy0 + p[1]] for p in kps_sq]
        draw_kps(canvas_full, kps_full)
        cv2.rectangle(canvas_full, (int(bbox[0]), int(bbox[1])),
                      (int(bbox[2]), int(bbox[3])), (255, 0, 255), 2)
        score_full = plausibility(kps_full)

        # (2) Person-bbox crop (replicates training path)
        person_crop, (cx0p, cy0p) = crop_with_pad(img, bbox, 0.20)
        kps_pc = pose_on_crop(model, person_crop, device)
        canvas_crop = person_crop.copy()
        draw_kps(canvas_crop, kps_pc)
        score_crop = plausibility(kps_pc)

        name = Path(it["image_path"]).stem
        pairs.append((name, canvas_full, canvas_crop, score_full, score_crop))
        print(f"{name:30s} {score_full:>12.2f} {score_crop:>12.2f}")

    # Build side-by-side grid: 2 columns (full, person_crop) x N rows
    cell_h = 360
    pad_y = 30
    rows = len(pairs)
    grid_h = pad_y + rows * cell_h
    grid_w = 1100
    grid = np.full((grid_h, grid_w, 3), 240, dtype=np.uint8)
    cv2.putText(grid, "FULL FRAME (current inference)", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(grid, "PERSON CROP (replicates training)", (560, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    for ri, (name, full, crop, sf, sc) in enumerate(pairs):
        y0 = pad_y + ri * cell_h
        for ci, (im, score) in enumerate([(full, sf), (crop, sc)]):
            h, w = im.shape[:2]
            s = min(540 / w, (cell_h - 20) / h)
            nh, nw = int(h * s), int(w * s)
            rs = cv2.resize(im, (nw, nh))
            x0 = ci * 550 + 10
            grid[y0:y0 + nh, x0:x0 + nw] = rs
            cv2.putText(grid, f"{name}  score={score:.2f}", (x0, y0 + nh + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 130, 0) if score >= 0.7 else (0, 0, 200), 1)

    grid_path = out_dir / "compare.png"
    cv2.imwrite(str(grid_path), grid)

    full_mean = np.mean([p[3] for p in pairs])
    crop_mean = np.mean([p[4] for p in pairs])
    print(f"\n[headline]")
    print(f"  mean plausibility — full-frame inference : {full_mean:.2f}")
    print(f"  mean plausibility — person-cropped       : {crop_mean:.2f}")
    print(f"\n[out]    {grid_path}")
    if crop_mean >= 0.7 and crop_mean - full_mean >= 0.3:
        print("\n[hypothesis CONFIRMED] pose_v0 is fine. Inference is wrong.")
        print("    Fix: add person detection (or extrapolated upper-body crop")
        print("    from hand_det v2 hand bboxes) before pose. No retrain needed.")
    elif crop_mean < 0.5:
        print("\n[hypothesis REJECTED] pose_v0 fails even on person crops.")
        print("    Model is genuinely broken. Path A retrain stands.")
    else:
        print("\n[hypothesis PARTIAL] crops help but pose still mediocre.")
        print("    Mixed result: inference fix is necessary, retrain may still help.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
