"""Pose v0 probe: training-distribution vs deployment-distribution vs framing transforms.

Goal: answer Session 18's gating question — is pose_v0 broken because of
the deployment input distribution (close-upper-body webcam framing) or
because the model is genuinely undertrained?

We feed pose_v0 three kinds of input and check anatomical plausibility:
  1. MPII validation images (training distribution — should work if model is OK)
  2. ASL clip mid-frames at native framing (deployment distribution — known to fail)
  3. ASL clip mid-frames under framing transforms (identity, zoom-out 1.5x,
     zoom-out 2x, full-body-pad) — if any transform recovers pose, it's
     a framing problem (→ Path B). If none does, it's a training problem (→ Path A).

Output:
  - data/probe/pose_probe_<ts>/grid.png — rows = inputs, cols = transforms,
    each cell has pose overlay drawn on the image.
  - data/probe/pose_probe_<ts>/summary.json — per-cell anatomical plausibility score.

Usage:
  python -m scripts.pose_probe
  python -m scripts.pose_probe --num-asl 6 --num-mpii 4
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


POSE_EDGES = [(0, 1), (1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7)]
KPT_NAMES = ["nose", "neck", "r_shoulder", "l_shoulder",
             "r_elbow", "l_elbow", "r_wrist", "l_wrist"]


def load_pose(ckpt: Path, device: str) -> PoseRegressor:
    model = PoseRegressor()
    obj = torch.load(ckpt, map_location=device, weights_only=False)
    sd = obj.get("state_dict", obj.get("model_state_dict", obj))
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    return model.eval().to(device)


@torch.no_grad()
def run_pose(model: PoseRegressor, bgr: np.ndarray, device: str) -> list[list[float]]:
    """Return 8 keypoints in original-image pixel coords."""
    H, W = bgr.shape[:2]
    side = min(H, W)
    cy0 = (H - side) // 2
    cx0 = (W - side) // 2
    sq = bgr[cy0:cy0 + side, cx0:cx0 + side]
    rgb = cv2.cvtColor(sq, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    t = torch.nn.functional.interpolate(t, size=(PoseRegressor.INPUT_SIZE,) * 2,
                                        mode="bilinear", align_corners=False)
    out = model(t.to(device))
    coords = out["coords"][0].cpu().numpy()  # (8, 2) in [0,1]
    return [[float(cx0 + kx * side), float(cy0 + ky * side)] for kx, ky in coords]


def draw_pose(canvas: np.ndarray, kps: list[list[float]]) -> None:
    for a, b in POSE_EDGES:
        xa, ya = kps[a]; xb, yb = kps[b]
        cv2.line(canvas, (int(xa), int(ya)), (int(xb), int(yb)), (0, 200, 255), 2)
    for i, (x, y) in enumerate(kps):
        cv2.circle(canvas, (int(x), int(y)), 4, (0, 255, 0), -1)
        cv2.putText(canvas, KPT_NAMES[i][:3], (int(x) + 6, int(y) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)


def plausibility(kps: list[list[float]]) -> tuple[float, dict]:
    """Anatomical sanity heuristic. Returns (score in [0,1], per-check dict)."""
    nose, neck, r_sh, l_sh, r_el, l_el, r_wr, l_wr = kps
    checks = {
        "nose_above_neck": nose[1] < neck[1],
        "neck_above_shoulders": neck[1] < (r_sh[1] + l_sh[1]) / 2,
        "shoulders_level": abs(r_sh[1] - l_sh[1]) < 0.15 * abs(r_sh[0] - l_sh[0] + 1e-3) * 2 + 30,
        "shoulders_apart_x": abs(r_sh[0] - l_sh[0]) > 20,
        "r_elbow_below_r_shoulder": r_el[1] > r_sh[1] - 30,
        "l_elbow_below_l_shoulder": l_el[1] > l_sh[1] - 30,
        "left_right_x_order": l_sh[0] > r_sh[0] or r_sh[0] > l_sh[0],
    }
    return sum(checks.values()) / len(checks), checks


def t_identity(img: np.ndarray) -> np.ndarray:
    return img.copy()


def t_zoom_out(img: np.ndarray, factor: float) -> np.ndarray:
    """Pad the image to simulate stepping back from camera."""
    H, W = img.shape[:2]
    nH, nW = int(H * factor), int(W * factor)
    canvas = np.full((nH, nW, 3), 128, dtype=img.dtype)  # gray pad
    y0 = (nH - H) // 2
    x0 = (nW - W) // 2
    canvas[y0:y0 + H, x0:x0 + W] = img
    return canvas


def t_full_body_pad(img: np.ndarray) -> np.ndarray:
    """Simulate MPII-ish framing: heavy vertical pad so person occupies ~40% of frame height."""
    H, W = img.shape[:2]
    nH = int(H * 2.2)
    nW = max(W, nH)  # keep aspect roughly square
    canvas = np.full((nH, nW, 3), 128, dtype=img.dtype)
    y0 = int(nH * 0.2)
    x0 = (nW - W) // 2
    canvas[y0:y0 + H, x0:x0 + W] = img
    return canvas


TRANSFORMS = [
    ("identity", t_identity),
    ("zoom_out_1.5x", lambda im: t_zoom_out(im, 1.5)),
    ("zoom_out_2x", lambda im: t_zoom_out(im, 2.0)),
    ("full_body_pad", t_full_body_pad),
]


def sample_asl_frames(clips_dir: Path, n: int, rng: random.Random) -> list[tuple[str, np.ndarray]]:
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


def sample_mpii_frames(mpii_dir: Path, n: int, rng: random.Random) -> list[tuple[str, np.ndarray]]:
    jpgs = list(mpii_dir.rglob("*.jpg"))
    if not jpgs:
        return []
    chosen = rng.sample(jpgs, min(n, len(jpgs)))
    frames = []
    for j in chosen:
        img = cv2.imread(str(j))
        if img is not None:
            frames.append((f"MPII/{j.stem[:20]}", img))
    return frames


def make_grid(cells: list[list[np.ndarray]], col_labels: list[str],
              row_labels: list[str], cell_size: int = 280) -> np.ndarray:
    n_rows, n_cols = len(cells), len(cells[0])
    pad_left, pad_top = 140, 40
    H = pad_top + n_rows * cell_size
    W = pad_left + n_cols * cell_size
    grid = np.full((H, W, 3), 255, dtype=np.uint8)
    for ci, label in enumerate(col_labels):
        x = pad_left + ci * cell_size + 8
        cv2.putText(grid, label, (x, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    for ri, label in enumerate(row_labels):
        y = pad_top + ri * cell_size + cell_size // 2
        cv2.putText(grid, label, (4, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    for ri in range(n_rows):
        for ci in range(n_cols):
            cell = cells[ri][ci]
            h, w = cell.shape[:2]
            s = min(cell_size / h, cell_size / w)
            nh, nw = int(h * s), int(w * s)
            resized = cv2.resize(cell, (nw, nh))
            y0 = pad_top + ri * cell_size + (cell_size - nh) // 2
            x0 = pad_left + ci * cell_size + (cell_size - nw) // 2
            grid[y0:y0 + nh, x0:x0 + nw] = resized
    return grid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path,
                    default=Path("data/ckpts/pose_v0_best.pt"))
    ap.add_argument("--asl-dir", type=Path,
                    default=Path("dataset/clean/v2-yt/normalized_videos"))
    ap.add_argument("--mpii-dir", type=Path,
                    default=Path("data/external/mpii_pose/images"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/probe"))
    ap.add_argument("--num-asl", type=int, default=5)
    ap.add_argument("--num-mpii", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)

    print(f"[device] {device}")
    print(f"[load]   {args.ckpt}")
    model = load_pose(args.ckpt, device)

    print(f"[sample] {args.num_mpii} MPII + {args.num_asl} ASL")
    inputs: list[tuple[str, np.ndarray]] = []
    inputs += sample_mpii_frames(args.mpii_dir, args.num_mpii, rng)
    inputs += sample_asl_frames(args.asl_dir, args.num_asl, rng)
    if not inputs:
        print("[err] no input frames found"); return 1

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir / f"pose_probe_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cells: list[list[np.ndarray]] = []
    summary: dict = {"checkpoint": str(args.ckpt), "inputs": []}

    for name, img in inputs:
        row_cells: list[np.ndarray] = []
        row_entry = {"name": name, "transforms": {}}
        for t_name, t_fn in TRANSFORMS:
            transformed = t_fn(img)
            kps = run_pose(model, transformed, device)
            canvas = transformed.copy()
            draw_pose(canvas, kps)
            cv2.putText(canvas, t_name, (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            score, checks = plausibility(kps)
            cv2.putText(canvas, f"score={score:.2f}", (8, 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 255, 0) if score >= 0.7 else (0, 0, 255), 2)
            row_cells.append(canvas)
            row_entry["transforms"][t_name] = {
                "score": score,
                "checks": checks,
                "keypoints": kps,
            }
        cells.append(row_cells)
        summary["inputs"].append(row_entry)
        print(f"  {name:35s}  " +
              "  ".join(f"{tn}={row_entry['transforms'][tn]['score']:.2f}"
                        for tn, _ in TRANSFORMS))

    grid = make_grid(cells,
                     col_labels=[tn for tn, _ in TRANSFORMS],
                     row_labels=[n for n, _ in inputs])
    grid_path = out_dir / "grid.png"
    cv2.imwrite(str(grid_path), grid)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Headline numbers per (distribution, transform)
    mpii_inputs = [r for r in summary["inputs"] if r["name"].startswith("MPII")]
    asl_inputs = [r for r in summary["inputs"] if r["name"].startswith("ASL")]
    print("\n[headline] mean plausibility score per transform")
    print(f"{'transform':<20s} {'MPII':>8s} {'ASL':>8s}")
    for tn, _ in TRANSFORMS:
        m = np.mean([r["transforms"][tn]["score"] for r in mpii_inputs]) if mpii_inputs else float("nan")
        a = np.mean([r["transforms"][tn]["score"] for r in asl_inputs]) if asl_inputs else float("nan")
        print(f"{tn:<20s} {m:>8.2f} {a:>8.2f}")

    print(f"\n[out]    {grid_path}")
    print(f"[out]    {out_dir / 'summary.json'}")
    print("\nInterpretation:")
    print("  - If MPII rows score high (>=0.7) and ASL identity scores low → distribution shift.")
    print("    Then: if any ASL transform jumps to >=0.7 → framing problem → Path B.")
    print("    If no ASL transform helps → training-data problem → Path A.")
    print("  - If even MPII rows score low → model is genuinely broken → Path A (rebuild).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
