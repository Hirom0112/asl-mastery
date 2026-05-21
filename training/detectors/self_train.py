"""Self-training (pseudo-labeling) for the from-scratch detectors.

The strategy per ADR 0011/0015 §"Data sourcing - landmark training":

    1. Train detector M_0 from scratch on public human-labeled datasets.
    2. Run M_0 across all frames of our 12K-clip ASL corpus.
    3. Keep only predictions that pass BOTH:
         - peak-confidence filter (heatmap peak >= conf_threshold)
         - temporal-smoothness filter (prediction consistent with the
           neighboring frames in the same clip within a tolerance)
    4. Emit a pseudo-labeled manifest.
    5. Combine with the public-data manifest → retrain → M_1.
    6. Optionally repeat (noisy student).

This module is task-agnostic at the API level but currently implements
hand-bbox pseudo-labeling (the easiest case — confidence is the heatmap
peak; temporal smoothness is bbox-center distance across frames). The
hand-keypoint, pose-keypoint, and face-bbox variants are stubbed for
future commits once those models exist.

Run on Modal (recommended — ASL corpus is 12K clips) or locally on a
subset for smoke-testing.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchvision.io as tvio
from torch.utils.data import DataLoader, Dataset

from training.detectors.hand_detector import HandDetector


# ---------------------------------------------------------------------------
# Confidence + temporal filters
# ---------------------------------------------------------------------------


def _heatmap_peaks(
    heatmap_logits: torch.Tensor,
    size_pred: torch.Tensor,
    *,
    threshold: float,
    max_peaks_per_frame: int,
    nms_radius_px: int,
) -> list[tuple[float, float, float, float, float]]:
    """Return [(x0, y0, x1, y1, score)] in INPUT-image (320) pixel space.

    Simple peak-NMS over a single batch sample.
    """
    # heatmap_logits: (1, H, W); size_pred: (2, H, W)
    prob = torch.sigmoid(heatmap_logits.squeeze(0))  # (H, W)
    H, W = prob.shape
    scores: list[tuple[float, float, float, float, float]] = []

    # 3x3 NMS by max-pool
    pooled = torch.nn.functional.max_pool2d(
        prob.unsqueeze(0).unsqueeze(0), kernel_size=3, stride=1, padding=1
    ).squeeze()
    keep = (prob == pooled) & (prob >= threshold)
    ys, xs = torch.where(keep)

    candidates = []
    for cx, cy in zip(xs.tolist(), ys.tolist()):
        score = float(prob[cy, cx].item())
        w = float(size_pred[0, cy, cx].item())
        h = float(size_pred[1, cy, cx].item())
        x_center = (cx + 0.5) * HandDetector.STRIDE
        y_center = (cy + 0.5) * HandDetector.STRIDE
        candidates.append((score, x_center - w / 2, y_center - h / 2,
                           x_center + w / 2, y_center + h / 2, w, h))

    # Hard cap on detections per frame; sort by score desc
    candidates.sort(key=lambda t: -t[0])
    for cand in candidates[:max_peaks_per_frame]:
        score, x0, y0, x1, y1, _, _ = cand
        scores.append((x0, y0, x1, y1, score))
    return scores


def _temporal_consistent(
    per_frame_dets: list[list[tuple[float, float, float, float, float]]],
    *,
    center_tol_px: float,
    min_consistent_frames: int,
) -> list[list[tuple[float, float, float, float, float]]]:
    """Drop detections whose bbox center is more than center_tol_px from
    any detection in the previous OR next frame. Drops the whole prediction
    list if a clip has fewer than min_consistent_frames consistent
    detections (signal is too weak).
    """
    def center(b):
        return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)

    n = len(per_frame_dets)
    if n < 3:
        # Not enough temporal context — drop the clip entirely
        return [[] for _ in range(n)]

    kept: list[list[tuple[float, float, float, float, float]]] = []
    consistent_count = 0
    for i, dets in enumerate(per_frame_dets):
        kept_here = []
        neighbors = []
        if i > 0:
            neighbors.extend(per_frame_dets[i - 1])
        if i < n - 1:
            neighbors.extend(per_frame_dets[i + 1])
        if not neighbors:
            kept.append([])
            continue
        for det in dets:
            cx, cy = center(det)
            for nb in neighbors:
                ncx, ncy = center(nb)
                if (cx - ncx) ** 2 + (cy - ncy) ** 2 <= center_tol_px ** 2:
                    kept_here.append(det)
                    break
        kept.append(kept_here)
        if kept_here:
            consistent_count += 1

    if consistent_count < min_consistent_frames:
        return [[] for _ in range(n)]
    return kept


# ---------------------------------------------------------------------------
# Frame loader
# ---------------------------------------------------------------------------


class _FrameDirDataset(Dataset):
    """Reads JPGs from a flat per-clip directory."""

    def __init__(self, frame_paths: list[Path]) -> None:
        self.paths = frame_paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, str, tuple[int, int]]:
        p = self.paths[i]
        img = tvio.read_image(str(p), mode=tvio.ImageReadMode.RGB).float() / 255.0
        h0, w0 = img.shape[1], img.shape[2]
        img = torch.nn.functional.interpolate(
            img.unsqueeze(0),
            size=(HandDetector.STRIDE * 80, HandDetector.STRIDE * 80),  # 320x320
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        return img, str(p), (w0, h0)


def _frames_per_clip(frames_root: Path) -> dict[str, list[Path]]:
    """Group frames by clip stem (everything before the trailing _fNNNN.jpg).

    Returns dict: clip_id -> sorted list of frame paths.
    """
    groups: dict[str, list[Path]] = defaultdict(list)
    for p in frames_root.rglob("*.jpg"):
        # extract_frames.py writes <clip_stem>_f<NNNN>.jpg
        stem = p.stem
        if "_f" in stem:
            clip_id, _ = stem.rsplit("_f", 1)
        else:
            clip_id = stem
        groups[clip_id].append(p)
    for clip_id in groups:
        groups[clip_id].sort()
    return groups


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def pseudo_label_hand_bbox(
    checkpoint: Path,
    frames_root: Path,
    out_manifest: Path,
    *,
    conf_threshold: float = 0.5,
    max_peaks_per_frame: int = 2,
    nms_radius_px: int = 12,
    temporal_center_tol_px: float = 40.0,
    min_consistent_frames_per_clip: int = 5,
    batch_size: int = 8,
    num_workers: int = 4,
    device: str | None = None,
) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = HandDetector().to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state.get("model", state))
    model.eval()
    print(f"loaded checkpoint {checkpoint} on {device}")

    clip_groups = _frames_per_clip(frames_root)
    print(f"found {len(clip_groups)} clips covering {sum(len(v) for v in clip_groups.values())} frames")

    items: list[dict] = []
    kept_clips = 0
    for clip_id, frame_paths in clip_groups.items():
        ds = _FrameDirDataset(frame_paths)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        per_frame_dets: list[list[tuple[float, float, float, float, float]]] = []
        per_frame_meta: list[tuple[str, tuple[int, int]]] = []
        with torch.no_grad():
            for imgs, paths, sizes in loader:
                imgs = imgs.to(device, non_blocking=True)
                out = model(imgs)
                for b in range(imgs.size(0)):
                    dets = _heatmap_peaks(
                        out["heatmap"][b],
                        out["size"][b],
                        threshold=conf_threshold,
                        max_peaks_per_frame=max_peaks_per_frame,
                        nms_radius_px=nms_radius_px,
                    )
                    per_frame_dets.append(dets)
                    per_frame_meta.append((paths[b], (int(sizes[0][b]), int(sizes[1][b]))))

        kept = _temporal_consistent(
            per_frame_dets,
            center_tol_px=temporal_center_tol_px,
            min_consistent_frames=min_consistent_frames_per_clip,
        )
        if not any(kept):
            continue
        kept_clips += 1

        # Convert from 320-space bboxes back to source-image pixel space
        for (frame_path, (w0, h0)), dets in zip(per_frame_meta, kept):
            if not dets:
                continue
            sx = w0 / (HandDetector.STRIDE * 80)
            sy = h0 / (HandDetector.STRIDE * 80)
            bboxes = [
                [d[0] * sx, d[1] * sy, d[2] * sx, d[3] * sy]
                for d in dets
            ]
            items.append(
                {
                    "image_path": str(Path(frame_path).resolve()),
                    "width": w0,
                    "height": h0,
                    "bboxes": bboxes,
                    "pseudo_label_source": str(checkpoint.name),
                    "pseudo_label_confidence": [float(d[4]) for d in dets],
                }
            )

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "task": "hand_bbox",
                "items": items,
                "pseudo_label_metadata": {
                    "checkpoint": str(checkpoint),
                    "conf_threshold": conf_threshold,
                    "temporal_center_tol_px": temporal_center_tol_px,
                    "min_consistent_frames_per_clip": min_consistent_frames_per_clip,
                    "kept_clips": kept_clips,
                    "total_clips": len(clip_groups),
                    "kept_frames": len(items),
                },
            },
            indent=2,
        )
    )
    print(f"wrote {out_manifest}  ({len(items)} frames from {kept_clips}/{len(clip_groups)} clips)")
    return {
        "kept_frames": len(items),
        "kept_clips": kept_clips,
        "total_clips": len(clip_groups),
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--frames-root", type=Path, required=True)
    p.add_argument("--out-manifest", type=Path, required=True)
    p.add_argument("--conf-threshold", type=float, default=0.5)
    p.add_argument("--temporal-center-tol-px", type=float, default=40.0)
    p.add_argument("--min-consistent-frames-per-clip", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    pseudo_label_hand_bbox(
        checkpoint=args.checkpoint,
        frames_root=args.frames_root,
        out_manifest=args.out_manifest,
        conf_threshold=args.conf_threshold,
        temporal_center_tol_px=args.temporal_center_tol_px,
        min_consistent_frames_per_clip=args.min_consistent_frames_per_clip,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
