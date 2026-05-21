"""Phase 4 Slice 4.1 — extract landmark trajectories for the 12K ASL corpus.

Pipeline (per clip):
  1. Decode N frames (1 FPS default; configurable).
  2. Run hand detector → bbox(es) per frame.
  3. For each detected hand, crop + run hand landmark regressor → 21 kpts.
  4. Run pose regressor on a fixed full-frame crop → 8 upper-body kpts.
  5. Accumulate per-frame per-hand kpts + pose kpts into a single
     trajectory record + write JSON beside the clip.

Output per clip: data/trajectories/<sign_id>/<clip_stem>.json

This is the dataset input for Phase 4 Slice 4.2 (template fitting).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torchvision.io as tvio

from training.detectors.face_detector import FaceDetector  # noqa: F401  (kept for symmetric loader API)
from training.detectors.hand_detector import HandDetector
from training.detectors.hand_landmarks import HandLandmarkRegressor
from training.detectors.pose_detector import PoseRegressor


def _decode_frames(clip_path: Path, fps: float) -> list[torch.Tensor]:
    """Use ffmpeg to extract frames at the requested rate, return as a list
    of (3, H, W) float32 [0, 1] tensors in temporal order.
    """
    with tempfile.TemporaryDirectory(prefix="traj_") as tmp:
        pattern = Path(tmp) / "f%05d.jpg"
        cmd = ["ffmpeg", "-loglevel", "error", "-y", "-i", str(clip_path),
               "-vf", f"fps={fps}", "-q:v", "3", str(pattern)]
        subprocess.check_call(cmd)
        frames = sorted(Path(tmp).glob("f*.jpg"))
        return [tvio.read_image(str(f), mode=tvio.ImageReadMode.RGB).float() / 255.0
                for f in frames]


def _detect_hand_bboxes(
    detector: HandDetector,
    img: torch.Tensor,
    device: str,
    threshold: float = 0.4,
    max_hands: int = 2,
) -> list[tuple[float, float, float, float]]:
    H, W = img.shape[1], img.shape[2]
    x = torch.nn.functional.interpolate(img.unsqueeze(0), size=(320, 320),
                                        mode="bilinear", align_corners=False).to(device)
    with torch.no_grad():
        out = detector(x)
    prob = torch.sigmoid(out["heatmap"][0, 0])
    pooled = torch.nn.functional.max_pool2d(
        prob.unsqueeze(0).unsqueeze(0), kernel_size=3, stride=1, padding=1
    ).squeeze()
    keep = (prob == pooled) & (prob >= threshold)
    ys, xs = torch.where(keep)
    cands = []
    for cx, cy in zip(xs.tolist(), ys.tolist()):
        score = float(prob[cy, cx].item())
        w = float(out["size"][0, 0, cy, cx].item())
        h = float(out["size"][0, 1, cy, cx].item())
        x_center = (cx + 0.5) * HandDetector.STRIDE
        y_center = (cy + 0.5) * HandDetector.STRIDE
        # Back to original-image pixels
        sx = W / 320
        sy = H / 320
        cands.append((score, (x_center - w / 2) * sx, (y_center - h / 2) * sy,
                      (x_center + w / 2) * sx, (y_center + h / 2) * sy))
    cands.sort(key=lambda t: -t[0])
    return [(b[1], b[2], b[3], b[4]) for b in cands[:max_hands]]


def _crop_resize(img: torch.Tensor, bbox: tuple[float, float, float, float],
                 size: int, pad_frac: float = 0.20) -> torch.Tensor:
    _, H, W = img.shape
    x0, y0, x1, y1 = bbox
    bw, bh = x1 - x0, y1 - y0
    pad = pad_frac * max(bw, bh)
    cx0 = max(0, int(x0 - pad))
    cy0 = max(0, int(y0 - pad))
    cx1 = min(W, int(x1 + pad))
    cy1 = min(H, int(y1 + pad))
    if cx1 <= cx0 or cy1 <= cy0:
        cx0, cy0, cx1, cy1 = 0, 0, W, H
    crop = img[:, cy0:cy1, cx0:cx1]
    return torch.nn.functional.interpolate(
        crop.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False
    ).squeeze(0)


def extract_clip(
    clip_path: Path,
    out_path: Path,
    sign_id: str,
    detectors: dict,
    device: str,
    fps: float = 15.0,
) -> dict:
    frames = _decode_frames(clip_path, fps=fps)
    if not frames:
        print(f"  no frames decoded from {clip_path}")
        return {}

    hand_det = detectors["hand_detector"]
    landmark_reg = detectors["hand_landmarks"]
    pose_reg = detectors["pose"]

    per_frame: list[dict] = []
    for fi, img in enumerate(frames):
        hand_bboxes = _detect_hand_bboxes(hand_det, img, device)
        # Run landmark regressor per detected hand
        hands_out = []
        for bbox in hand_bboxes:
            crop = _crop_resize(img, bbox, size=HandLandmarkRegressor.INPUT_SIZE).unsqueeze(0).to(device)
            with torch.no_grad():
                lr_out = landmark_reg(crop)
            coords = lr_out["coords"][0].cpu().numpy()
            # Map back to original image pixels
            x0, y0, x1, y1 = bbox
            bw, bh = x1 - x0, y1 - y0
            abs_coords = []
            for kx, ky in coords:
                abs_coords.append([float(x0 + kx * bw), float(y0 + ky * bh)])
            hands_out.append({"bbox": list(bbox), "keypoints": abs_coords})

        # Run pose regressor on the full frame (square-center crop to 256)
        _, H, W = img.shape
        side = min(H, W)
        cy0 = (H - side) // 2
        cx0 = (W - side) // 2
        center_crop = img[:, cy0:cy0 + side, cx0:cx0 + side]
        pose_in = torch.nn.functional.interpolate(
            center_crop.unsqueeze(0), size=(PoseRegressor.INPUT_SIZE,) * 2,
            mode="bilinear", align_corners=False
        ).to(device)
        with torch.no_grad():
            p_out = pose_reg(pose_in)
        pose_coords = p_out["coords"][0].cpu().numpy()
        abs_pose = [[float(cx0 + kx * side), float(cy0 + ky * side)] for kx, ky in pose_coords]

        per_frame.append(
            {
                "frame_idx": fi,
                "hands": hands_out,
                "pose": abs_pose,
            }
        )

    record = {
        "version": 1,
        "clip_path": str(clip_path),
        "sign_id": sign_id,
        "fps": fps,
        "num_frames": len(per_frame),
        "frame_size": [int(frames[0].shape[2]), int(frames[0].shape[1])],
        "frames": per_frame,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record))
    return record


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips-dir", type=Path, required=True,
                    help="Root containing cleaned .mp4 clips (recursively scanned).")
    ap.add_argument("--out-dir", type=Path, default=Path("data/trajectories"))
    ap.add_argument("--hand-detector-ckpt", type=Path, required=True)
    ap.add_argument("--hand-landmarks-ckpt", type=Path, required=True)
    ap.add_argument("--pose-ckpt", type=Path, required=True)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--sign-from-filename-regex", type=str, default=r"^([^_]+)_",
                    help="Regex with one capture group extracting sign_id from clip stem.")
    args = ap.parse_args()

    import re
    rx = re.compile(args.sign_from_filename_regex)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    detectors = {
        "hand_detector": HandDetector().to(device),
        "hand_landmarks": HandLandmarkRegressor().to(device),
        "pose": PoseRegressor().to(device),
    }
    detectors["hand_detector"].load_state_dict(
        torch.load(args.hand_detector_ckpt, map_location=device).get("model"))
    detectors["hand_landmarks"].load_state_dict(
        torch.load(args.hand_landmarks_ckpt, map_location=device).get("model"))
    detectors["pose"].load_state_dict(
        torch.load(args.pose_ckpt, map_location=device).get("model"))
    for m in detectors.values():
        m.eval()

    clips = sorted(args.clips_dir.rglob("*.mp4"))
    print(f"extracting trajectories for {len(clips)} clips → {args.out_dir}")
    n_ok, n_skip = 0, 0
    for i, clip in enumerate(clips, 1):
        m = rx.search(clip.stem)
        sign = m.group(1) if m else "unknown"
        out = args.out_dir / sign / f"{clip.stem}.json"
        if out.exists():
            n_skip += 1
            continue
        try:
            extract_clip(clip, out, sign, detectors, device, fps=args.fps)
            n_ok += 1
        except subprocess.CalledProcessError as e:
            print(f"  ffmpeg failed for {clip}: {e}")
        if i % 50 == 0:
            print(f"  [{i}/{len(clips)}] ok={n_ok} skip={n_skip}")
    print(f"done: ok={n_ok} skip={n_skip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
