"""Job B Phase 1 — pseudo-label hand bboxes on the ASL clip corpus.

Reads an ASL clip manifest (MP4 paths), runs hand_det_v0 on every frame,
keeps frames where the heatmap peak ≥ conf_threshold, applies a
temporal-smoothness filter (kept bbox centers move ≤ tol_px across
adjacent kept frames), saves the kept frames as 320×320 JPGs, and
emits a manifest in the same FrameRecord format as
`data/labeled_frames/hand_bbox/external_train.json`.

The resulting manifest is concatenated with the public-data train
manifest in Phase 2 to produce a CMU+FreiHAND+pseudo_ASL training mix.

Per ADR 0011 & 0012, every pseudo-label comes from a from-scratch
detector trained on our public-source data — no MediaPipe, no OpenPose,
no foreign weights anywhere in the path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torchvision.io as tvio

from training.detectors.hand_detector import HandDetector


def _decode_clip_at_320(clip_path: Path, fps: float = 15.0) -> tuple[torch.Tensor, int, int]:
    """Decode clip → (N, 3, 320, 320) float[0,1] + (orig_w, orig_h).

    Resamples temporally to `fps` (~match extract_trajectories_v2 default).
    Returns empty tensor on decode failure.
    """
    video, _, info = tvio.read_video(str(clip_path), pts_unit="sec")
    if video.numel() == 0:
        return torch.empty(0, 3, 320, 320), 0, 0
    src_fps = float(info.get("video_fps") or 30.0)
    step = max(1, int(round(src_fps / fps)))
    idxs = list(range(0, video.shape[0], step)) or [0]
    sampled = video[idxs].permute(0, 3, 1, 2).float() / 255.0  # (N, 3, H, W)
    _, _, H, W = sampled.shape
    resized = torch.nn.functional.interpolate(
        sampled, size=(320, 320), mode="bilinear", align_corners=False
    )
    return resized, W, H


@torch.no_grad()
def _run_detector(
    detector: HandDetector,
    frames: torch.Tensor,        # (N, 3, 320, 320) already resized
    device: str,
    threshold: float,
    nms_kernel: int = 3,
) -> list[list[tuple[float, float, float, float, float]]]:
    """Returns per-frame [(x0, y0, x1, y1, score)] in 320-space, peaks
    above `threshold`. NMS via 3×3 max-pool equality.
    """
    if frames.numel() == 0:
        return []
    frames = frames.to(device, non_blocking=True)
    out = detector(frames)
    prob = torch.sigmoid(out["heatmap"])  # (N, 1, 80, 80)
    pooled = torch.nn.functional.max_pool2d(
        prob, kernel_size=nms_kernel, stride=1, padding=nms_kernel // 2
    )
    keep = (prob == pooled) & (prob >= threshold)
    size = out["size"]  # (N, 2, 80, 80)
    stride = HandDetector.STRIDE
    N = frames.shape[0]
    per_frame: list[list[tuple[float, float, float, float, float]]] = []
    keep_cpu = keep.cpu()
    prob_cpu = prob.cpu()
    size_cpu = size.cpu()
    for n in range(N):
        ys, xs = torch.where(keep_cpu[n, 0])
        peaks = []
        for cy, cx in zip(ys.tolist(), xs.tolist()):
            score = float(prob_cpu[n, 0, cy, cx].item())
            w = float(size_cpu[n, 0, cy, cx].item())
            h = float(size_cpu[n, 1, cy, cx].item())
            if w < 4.0 or h < 4.0:
                continue
            xc = (cx + 0.5) * stride
            yc = (cy + 0.5) * stride
            peaks.append((xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2, score))
        peaks.sort(key=lambda t: -t[4])
        per_frame.append(peaks[:2])  # max 2 hands per frame
    return per_frame


def _temporal_filter(
    per_frame_peaks: list[list[tuple[float, float, float, float, float]]],
    center_tol_px: float,
) -> list[bool]:
    """Returns boolean per-frame keep mask. A frame is kept if it has at
    least one peak AND that peak's center is within `center_tol_px` of
    the closest neighboring kept frame's strongest peak. Suppresses
    isolated false-positive frames.
    """
    n = len(per_frame_peaks)
    if n == 0:
        return []
    # First pass: candidate set is frames with at least one peak
    have_peak = [bool(p) for p in per_frame_peaks]
    if not any(have_peak):
        return [False] * n

    def center(p):
        return ((p[0] + p[2]) / 2.0, (p[1] + p[3]) / 2.0)

    # For each candidate frame, check whether SOME peak is consistent
    # with the top peak of the nearest candidate frame on either side.
    kept = [False] * n
    candidates = [i for i, hp in enumerate(have_peak) if hp]
    # If only one candidate, keep it (no neighbors to validate against).
    if len(candidates) == 1:
        kept[candidates[0]] = True
        return kept
    for idx_pos, i in enumerate(candidates):
        # Find left/right candidate neighbors
        left = candidates[idx_pos - 1] if idx_pos > 0 else None
        right = candidates[idx_pos + 1] if idx_pos + 1 < len(candidates) else None
        i_top = per_frame_peaks[i][0]
        ci = center(i_top)
        ok = False
        for neighbor in (left, right):
            if neighbor is None:
                continue
            n_top = per_frame_peaks[neighbor][0]
            cn = center(n_top)
            if abs(ci[0] - cn[0]) <= center_tol_px and abs(ci[1] - cn[1]) <= center_tol_px:
                ok = True
                break
        if ok:
            kept[i] = True
    return kept


def pseudo_label_clip(
    detector: HandDetector,
    clip_path: Path,
    clip_id: str,
    out_frames_dir: Path,
    *,
    device: str,
    fps: float,
    conf_threshold: float,
    temporal_tol_px: float,
    min_kept_frames: int,
) -> list[dict]:
    """Run detector on one clip, save kept frames as JPGs, return manifest
    records [{image_path, width, height, bboxes}].
    """
    frames, ow, oh = _decode_clip_at_320(clip_path, fps=fps)
    if frames.numel() == 0:
        return []
    peaks = _run_detector(detector, frames, device, conf_threshold)
    kept_mask = _temporal_filter(peaks, center_tol_px=temporal_tol_px)
    if sum(kept_mask) < min_kept_frames:
        return []

    out_frames_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for i, (kept, fp_peaks) in enumerate(zip(kept_mask, peaks)):
        if not kept:
            continue
        # Save the frame (320×320 uint8) and record its label
        img_uint8 = (frames[i].clamp(0, 1) * 255.0).to(torch.uint8)
        out_path = out_frames_dir / f"{clip_id}_f{i:04d}.jpg"
        try:
            tvio.write_jpeg(img_uint8, str(out_path), quality=85)
        except Exception:
            continue
        bboxes = [list(p[:4]) for p in fp_peaks]
        records.append({
            "image_path": str(out_path),
            "width": 320,
            "height": 320,
            "bboxes": bboxes,
        })
    return records


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True,
                    help="Path to hand_det best.pt to use for pseudo-labeling.")
    ap.add_argument("--clip-manifest", type=Path, required=True,
                    help="ASL clip manifest (unified_clip_manifest.json or "
                         "dataset_v3_manifest.json).")
    ap.add_argument("--repo-root", type=Path, default=Path("."),
                    help="Resolves relative clip_path entries in the manifest.")
    ap.add_argument("--out-frames-dir", type=Path, required=True,
                    help="Where to write kept JPGs (one subdir per sign).")
    ap.add_argument("--out-manifest", type=Path, required=True,
                    help="Where to write the pseudo-label manifest JSON.")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--conf-threshold", type=float, default=0.5)
    ap.add_argument("--temporal-tol-px", type=float, default=40.0)
    ap.add_argument("--min-kept-frames", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0,
                    help="Optional: only process the first N clips (smoke test).")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    detector = HandDetector().to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    detector.load_state_dict(state.get("model", state))
    detector.eval()
    print(f"loaded {args.checkpoint}")

    manifest = json.loads(args.clip_manifest.read_text())
    # Manifest schemas in use:
    #   v3 ASL pipeline: {"records": [{"normalized_video_path", "sign_id", "clip_id", ...}]}
    #   unified clip:    {"clips": [{"clip_path", "sign_id", ...}]}
    #   FrameRecord:     {"items": [...]}
    clips = (
        manifest.get("records")
        or manifest.get("clips")
        or manifest.get("items")
        or (manifest if isinstance(manifest, list) else [])
    )
    if args.limit > 0:
        clips = clips[: args.limit]
    print(f"processing {len(clips)} clips")

    all_records: list[dict] = []
    n_clips_with_records = 0
    for i, c in enumerate(clips):
        clip_rel = (
            c.get("normalized_video_path")
            or c.get("clip_path")
            or c.get("path")
            or c.get("local_path")
        )
        if not clip_rel:
            continue
        sign_id = c.get("sign_id") or c.get("gloss") or "unknown"
        clip_id = c.get("clip_id") or Path(clip_rel).stem
        # If the manifest path is absolute (Modal-side schema), use as-is;
        # else join under repo_root.
        clip_path = Path(clip_rel) if Path(clip_rel).is_absolute() else args.repo_root / clip_rel
        if not clip_path.exists():
            continue
        sign_frames_dir = args.out_frames_dir / sign_id
        try:
            records = pseudo_label_clip(
                detector, clip_path, clip_id, sign_frames_dir,
                device=device, fps=args.fps, conf_threshold=args.conf_threshold,
                temporal_tol_px=args.temporal_tol_px,
                min_kept_frames=args.min_kept_frames,
            )
        except Exception as e:
            print(f"  ERROR {clip_id}: {e}")
            continue
        if records:
            all_records.extend(records)
            n_clips_with_records += 1
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(clips)} clips, {n_clips_with_records} kept, "
                  f"{len(all_records)} frames total")

    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.out_manifest.write_text(json.dumps({
        "version": 1,
        "task": "hand_bbox_pseudo_asl",
        "source_checkpoint": str(args.checkpoint),
        "conf_threshold": args.conf_threshold,
        "n_clips_processed": len(clips),
        "n_clips_with_records": n_clips_with_records,
        "items": all_records,
    }, indent=2))
    print(f"wrote {args.out_manifest}")
    print(f"summary: {n_clips_with_records} kept clips, {len(all_records)} kept frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
