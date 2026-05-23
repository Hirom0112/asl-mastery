"""Phase 4 Slice 4.1 v2 — fast trajectory extraction with batched inference.

Speedups over `extract_trajectories.py`:
  1. In-memory video decode via torchvision.io.read_video (no ffmpeg→JPG→reread roundtrip).
  2. All frames in a clip batched through hand_detector in ONE forward pass.
  3. All detected hand crops batched through hand_landmarks in ONE forward pass.
  4. All frames batched through pose_detector in ONE forward pass.
  5. Manifest-driven (mp4 + webm + any container ffmpeg can decode), not filename-regex.
  6. Optional CPU decode prefetch via torch DataLoader workers.

Pipeline (per clip, but inside the loop all GPU work is one batch per detector):
    decode → (N, 3, H, W) frames
    → hand_det forward(batch=N) → per-frame hand bboxes
    → gather crops → hand_lm forward(batch=K) → 21 kpts per crop
    → pose_det forward(batch=N) → 8 kpts per frame

Output per clip: <out_dir>/<sign_id>/<clip_stem>.json   (same schema as v1)

Inputs:
    --manifest path/to/unified_clip_manifest.json  (the v2 input)
    OR --clips-dir path/  (back-compat with v1; uses filename regex)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torchvision.io as tvio
from torch.utils.data import DataLoader, Dataset

from training.detectors.hand_detector import HandDetector
from training.detectors.hand_landmarks import HandLandmarkRegressor
from training.detectors.pose_detector import PoseRegressor
from training.detectors._pose_inference import pose_batched_with_bboxes
from training.detectors.handshape_encoder import HandshapeEncoder


def _resize_batch(frames: torch.Tensor, size: int) -> torch.Tensor:
    """Resize (N, 3, H, W) → (N, 3, size, size) bilinear."""
    return torch.nn.functional.interpolate(
        frames, size=(size, size), mode="bilinear", align_corners=False
    )


def _decode_clip(clip_path: Path, fps: float) -> torch.Tensor:
    """Decode entire clip into a (N, 3, H, W) float32 [0,1] tensor at requested fps.

    Uses tvio.read_video then resamples temporally to the target fps.
    Raises RuntimeError / OSError on decode failure (caller handles).
    """
    # read_video returns (T, H, W, 3) uint8 + dict with 'video_fps'
    video, _, info = tvio.read_video(str(clip_path), pts_unit="sec")
    if video.numel() == 0:
        return torch.empty(0, 3, 0, 0)
    src_fps = float(info.get("video_fps") or 30.0)
    # Temporal resample by index picking
    T = video.shape[0]
    step = max(1, int(round(src_fps / fps)))
    idxs = list(range(0, T, step))
    if not idxs:
        idxs = [0]
    sampled = video[idxs].permute(0, 3, 1, 2).float() / 255.0  # (N, 3, H, W)
    return sampled


def _detect_hands_batched(
    detector: HandDetector,
    frames: torch.Tensor,
    device: str,
    threshold: float = 0.02,  # very low floor; primary filter is top-K
    max_hands_per_frame: int = 2,
    top_k: int = 2,
    second_hand_threshold: float = 0.15,
    min_bbox_px: int = 8,
    max_aspect_ratio: float = 3.5,
) -> list[list[tuple[float, float, float, float]]]:
    """Run hand detector on ALL frames in one forward pass, returning the
    top-K NMS peaks per frame regardless of low confidence. This is the
    Session-15 fix for hand_det's undercalibration (max heatmap prob ~0.27
    on real ASL frames, far below the prior 0.4 threshold). ASL clips
    always have hands visible, so guaranteeing a fixed K per frame buys us
    template-grade landmark sequences instead of NaN-dominated ones.

    Returns: list of length N, each entry = list of up-to-top_k bboxes in
    original-image pixel coordinates, sorted high→low score.
    """
    N, _, H, W = frames.shape
    x = _resize_batch(frames, 320).to(device, non_blocking=True)
    with torch.no_grad():
        out = detector(x)
    prob = torch.sigmoid(out["heatmap"])  # (N, 1, 80, 80)
    pooled = torch.nn.functional.max_pool2d(prob, kernel_size=3, stride=1, padding=1)
    is_peak = (prob == pooled) & (prob >= threshold)
    # Mask non-peaks to -1 so they never beat real peaks in top-K
    peak_scores = torch.where(is_peak, prob, torch.full_like(prob, -1.0))  # (N, 1, 80, 80)
    peak_flat = peak_scores.flatten(1)  # (N, 6400)
    # Top-K per frame on the flattened heatmap
    vals, idxs = peak_flat.topk(top_k, dim=1)  # (N, K)

    sx = W / 320.0
    sy = H / 320.0
    stride = HandDetector.STRIDE
    GRID = prob.shape[-1]  # 80
    vals_cpu = vals.cpu()
    idxs_cpu = idxs.cpu()
    size_cpu = out["size"].cpu()  # (N, 2, 80, 80)

    per_frame: list[list[tuple[float, float, float, float]]] = []
    for n in range(N):
        scored: list[tuple[float, tuple[float, float, float, float]]] = []
        for k in range(top_k):
            score = float(vals_cpu[n, k].item())
            if score < 0:  # padding from masked non-peaks (rare for ASL frames)
                continue
            flat = int(idxs_cpu[n, k].item())
            cy, cx = divmod(flat, GRID)
            w = float(size_cpu[n, 0, cy, cx].item())
            h = float(size_cpu[n, 1, cy, cx].item())
            xc = (cx + 0.5) * stride
            yc = (cy + 0.5) * stride
            bbox = ((xc - w / 2) * sx, (yc - h / 2) * sy,
                    (xc + w / 2) * sx, (yc + h / 2) * sy)
            # Drop degenerate bboxes (size head occasionally emits ~1px-tall
            # boxes at low-confidence locations — ~48% of trajectories_v6).
            bw = bbox[2] - bbox[0]; bh = bbox[3] - bbox[1]
            if bw < min_bbox_px or bh < min_bbox_px:
                continue
            if max(bw, bh) / max(min(bw, bh), 1e-6) > max_aspect_ratio:
                continue
            scored.append((score, bbox))
        # Top-1 always taken; second hand only if its score crosses a soft threshold.
        # Prevents hallucinated "second hand" patches in one-handed signs.
        boxes = [b for _, b in scored[:1]]
        if len(scored) > 1 and scored[1][0] >= second_hand_threshold:
            boxes.append(scored[1][1])
        per_frame.append(boxes[:max_hands_per_frame])
    return per_frame


def _crop_for_landmarks(img: torch.Tensor,
                        bbox: tuple[float, float, float, float],
                        size: int,
                        pad_frac: float = 0.20) -> torch.Tensor:
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


def _landmarks_batched(
    regressor: HandLandmarkRegressor,
    frames: torch.Tensor,
    per_frame_bboxes: list[list[tuple[float, float, float, float]]],
    device: str,
) -> list[list[dict]]:
    """For each frame, run landmark regressor on every detected hand. All crops
    across the clip get batched into one forward pass.
    Returns: list of length N, each entry = list of dicts {bbox, keypoints (absolute pixels)}.
    """
    crops: list[torch.Tensor] = []
    index: list[tuple[int, int]] = []  # (frame_idx, hand_idx)
    for fi, bboxes in enumerate(per_frame_bboxes):
        for hi, bbox in enumerate(bboxes):
            crops.append(_crop_for_landmarks(frames[fi], bbox,
                                             size=HandLandmarkRegressor.INPUT_SIZE))
            index.append((fi, hi))

    out_per_frame: list[list[dict]] = [[] for _ in per_frame_bboxes]
    if not crops:
        return out_per_frame

    batch = torch.stack(crops, 0).to(device, non_blocking=True)
    with torch.no_grad():
        lr_out = regressor(batch)
    coords_all = lr_out["coords"].cpu().numpy()  # (K, 21, 2) crop-relative

    for k, (fi, hi) in enumerate(index):
        bbox = per_frame_bboxes[fi][hi]
        x0, y0, x1, y1 = bbox
        bw, bh = x1 - x0, y1 - y0
        abs_coords = [[float(x0 + kx * bw), float(y0 + ky * bh)]
                      for kx, ky in coords_all[k]]
        out_per_frame[fi].append({"bbox": list(bbox), "keypoints": abs_coords})
    return out_per_frame


def _pose_batched(
    regressor: PoseRegressor,
    frames: torch.Tensor,
    device: str,
) -> list[list[list[float]]]:
    """Run pose regressor on all frames at once (square-center crop)."""
    N, _, H, W = frames.shape
    side = min(H, W)
    cy0 = (H - side) // 2
    cx0 = (W - side) // 2
    cropped = frames[:, :, cy0:cy0 + side, cx0:cx0 + side]
    x = _resize_batch(cropped, PoseRegressor.INPUT_SIZE).to(device, non_blocking=True)
    with torch.no_grad():
        out = regressor(x)
    coords = out["coords"].cpu().numpy()  # (N, 8, 2)
    return [[[float(cx0 + kx * side), float(cy0 + ky * side)] for kx, ky in frame]
            for frame in coords]


def _encode_hands_batched(
    encoder: HandshapeEncoder,
    frames: torch.Tensor,                       # (N, 3, H, W) float32 [0,1]
    per_frame_bboxes: list[list[list[float]]],  # frame → hands → [x0,y0,x1,y1]
    device: str,
    crop_size: int = 224,
    pad_frac: float = 0.15,
) -> list[list[list[float]]]:
    """Run HandshapeEncoder on every detected hand bbox, return per-frame
    embeddings parallel to `per_frame_bboxes`.

    Crops each bbox + 15% pad, resizes to 224×224, runs all crops through the
    encoder in ONE batched forward pass. Output is float32 lists (len=128 per
    hand) for JSON serialization.
    """
    crops: list[torch.Tensor] = []
    slots: list[tuple[int, int]] = []  # (frame_idx, hand_idx)
    N, _, H, W = frames.shape
    for fi, hands_in_frame in enumerate(per_frame_bboxes):
        for hi, bbox in enumerate(hands_in_frame):
            x0, y0, x1, y1 = bbox
            bw = x1 - x0; bh = y1 - y0
            pad = pad_frac * max(bw, bh)
            cx0 = max(0, int(round(x0 - pad)))
            cy0 = max(0, int(round(y0 - pad)))
            cx1 = min(W, int(round(x1 + pad)))
            cy1 = min(H, int(round(y1 + pad)))
            if cx1 - cx0 < 4 or cy1 - cy0 < 4:
                continue
            crop = frames[fi, :, cy0:cy1, cx0:cx1].unsqueeze(0)
            crop = torch.nn.functional.interpolate(
                crop, size=(crop_size, crop_size), mode="bilinear", align_corners=False
            ).squeeze(0)
            crops.append(crop)
            slots.append((fi, hi))

    per_frame_embeds: list[list[list[float]]] = [[None] * len(h) for h in per_frame_bboxes]  # type: ignore
    if not crops:
        return per_frame_embeds

    batch = torch.stack(crops, 0).to(device, non_blocking=True)
    with torch.no_grad():
        z = encoder(batch)  # (K, 128) L2-normalized
    z_np = z.detach().float().cpu().numpy()
    for (fi, hi), vec in zip(slots, z_np):
        per_frame_embeds[fi][hi] = vec.tolist()
    return per_frame_embeds


def _spatial_crop_frames(frames: torch.Tensor, x_frac: float, crop_frac: float = 0.90) -> torch.Tensor:
    """Slice a horizontal subwindow of `crop_frac` width starting at `x_frac`.

    Produces materially different per-frame inputs to the detectors: hand
    bbox + pose keypoints land at different absolute positions across
    crops. The classifier's input feature is x-y based, so these become
    distinct trajectories (not just augmentation noise on the same one).

    x_frac in [0, 1 - crop_frac]; crop_frac in (0, 1]. For our 3-crop
    setup: x_frac ∈ {0.0, 0.05, 0.10} with crop_frac=0.90 → left / center
    / right framings of the same 90%-width window.
    """
    if x_frac == 0.0 and crop_frac == 1.0:
        return frames
    _, _, h, w = frames.shape
    new_w = max(2, int(w * crop_frac))
    x0 = max(0, min(w - new_w, int(round(w * x_frac))))
    return frames[:, :, :, x0:x0 + new_w].contiguous()


def extract_one(
    clip_path: Path,
    out_path: Path,
    sign_id: str,
    detectors: dict,
    device: str,
    fps: float = 15.0,
    n_spatial_crops: int = 1,
    out_path_template: str | None = None,
) -> dict | list[dict] | None:
    """Extract trajectories for a clip.

    n_spatial_crops=1: original behavior, single trajectory at `out_path`.
    n_spatial_crops=N (>1): N trajectories from horizontally-shifted
    crops of the same source frames; output paths are derived from
    `out_path_template` (a str format taking {crop_idx}, e.g.
    ``"<sign>__<basename>_crop{crop_idx}.json"``). Every variant carries
    `source_clip_id` = the original basename so train/val split logic can
    keep all crops of one source clip in the same split.
    """
    try:
        frames = _decode_clip(clip_path, fps=fps)
    except Exception as e:  # noqa: BLE001 — corrupt clip
        print(f"  decode failed for {clip_path}: {e!r}")
        return None
    if frames.numel() == 0 or frames.shape[0] == 0:
        return None

    source_clip_id = clip_path.stem
    # 3-crop preset: left / center / right; for N=1 use [0.0] (full frame).
    if n_spatial_crops <= 1:
        crop_offsets = [(0.0, 1.0)]  # (x_frac, crop_frac) — full frame, no crop
    elif n_spatial_crops == 2:
        crop_offsets = [(0.0, 0.90), (0.10, 0.90)]
    else:
        # N=3: left/center/right. >3 evenly spaces across [0, 1-crop_frac].
        crop_frac = 0.90
        max_x = 1.0 - crop_frac
        crop_offsets = [(max_x * i / (n_spatial_crops - 1), crop_frac)
                        for i in range(n_spatial_crops)]

    records: list[dict] = []
    encoder = detectors.get("handshape_encoder")
    for ci, (x_frac, crop_frac) in enumerate(crop_offsets):
        cropped = _spatial_crop_frames(frames, x_frac, crop_frac)
        per_frame_bboxes = _detect_hands_batched(detectors["hand_detector"], cropped, device)
        per_frame_hands = _landmarks_batched(detectors["hand_landmarks"], cropped,
                                             per_frame_bboxes, device)
        per_frame_pose = pose_batched_with_bboxes(detectors["pose"], cropped,
                                                  per_frame_bboxes, device)
        per_frame_embeds = (
            _encode_hands_batched(encoder, cropped, per_frame_bboxes, device)
            if encoder is not None else None
        )
        if per_frame_embeds is not None:
            # Attach embedding to the matching hand dict (parallel index)
            for fi in range(cropped.shape[0]):
                for hi, hand_dict in enumerate(per_frame_hands[fi]):
                    if hi < len(per_frame_embeds[fi]) and per_frame_embeds[fi][hi] is not None:
                        hand_dict["embedding"] = per_frame_embeds[fi][hi]
        per_frame_records = [
            {"frame_idx": fi, "hands": per_frame_hands[fi], "pose": per_frame_pose[fi]}
            for fi in range(cropped.shape[0])
        ]
        record = {
            "version": 2,
            "clip_path": str(clip_path),
            "sign_id": sign_id,
            "source_clip_id": source_clip_id,
            "spatial_crop": {"x_frac": x_frac, "crop_frac": crop_frac, "idx": ci,
                             "n_crops": n_spatial_crops},
            "fps": fps,
            "num_frames": cropped.shape[0],
            "frame_size": [int(cropped.shape[3]), int(cropped.shape[2])],
            "frames": per_frame_records,
        }
        if n_spatial_crops == 1:
            target = out_path
        else:
            assert out_path_template is not None, "multi-crop requires out_path_template"
            target = out_path.parent / out_path_template.format(crop_idx=ci)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(record))
        records.append(record)

    return records[0] if n_spatial_crops == 1 else records


class _ClipDataset(Dataset):
    """DataLoader-driven CPU decode pool: workers do tvio.read_video in parallel,
    main process consumes (clip_path, sign_id, frames) tuples for GPU work."""

    def __init__(self, manifest_clips: list[dict], repo_root: Path, fps: float) -> None:
        self.clips = manifest_clips
        self.repo_root = repo_root
        self.fps = fps

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int):
        entry = self.clips[idx]
        clip_path = Path(entry["clip_path"])
        if not clip_path.is_absolute():
            clip_path = self.repo_root / clip_path
        try:
            frames = _decode_clip(clip_path, fps=self.fps)
        except Exception as e:  # noqa: BLE001
            return {"clip_path": str(clip_path), "sign_id": entry["sign_id"],
                    "frames": None, "error": repr(e)}
        return {"clip_path": str(clip_path), "sign_id": entry["sign_id"],
                "frames": frames, "error": None}


def _collate(batch):
    # batch_size=1 — return the single dict
    return batch[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=None,
                    help="Unified clip manifest JSON (v2 mode).")
    ap.add_argument("--clips-dir", type=Path, default=None,
                    help="Back-compat: scan dir for *.mp4 and infer sign from filename.")
    ap.add_argument("--out-dir", type=Path, default=Path("data/trajectories"))
    ap.add_argument("--hand-detector-ckpt", type=Path, required=True)
    ap.add_argument("--hand-landmarks-ckpt", type=Path, required=True)
    ap.add_argument("--pose-ckpt", type=Path, required=True)
    ap.add_argument("--handshape-encoder-ckpt", type=Path, default=None,
                    help="Optional: if set, emit a 128D handshape embedding per hand "
                         "(Phase 4.6). Output trajectories will have hand.embedding fields.")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--decode-workers", type=int, default=4,
                    help="CPU workers for parallel video decode (DataLoader).")
    ap.add_argument("--sign-from-filename-regex", type=str, default=r"^([^_]+)_")
    ap.add_argument("--repo-root", type=Path, default=Path("."),
                    help="Resolves relative clip_path entries in the manifest.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    detectors = {
        "hand_detector": HandDetector().to(device).eval(),
        "hand_landmarks": HandLandmarkRegressor().to(device).eval(),
        "pose": PoseRegressor().to(device).eval(),
    }
    detectors["hand_detector"].load_state_dict(
        torch.load(args.hand_detector_ckpt, map_location=device, weights_only=False)["model"])
    detectors["hand_landmarks"].load_state_dict(
        torch.load(args.hand_landmarks_ckpt, map_location=device, weights_only=False)["model"])
    detectors["pose"].load_state_dict(
        torch.load(args.pose_ckpt, map_location=device, weights_only=False)["model"])

    # Phase 4.6: optional handshape encoder.
    if args.handshape_encoder_ckpt is not None:
        enc = HandshapeEncoder().to(device).eval()
        ckpt = torch.load(args.handshape_encoder_ckpt, map_location=device, weights_only=False)
        enc.load_state_dict(ckpt["state_dict"])
        detectors["handshape_encoder"] = enc
        print(f"loaded handshape encoder from {args.handshape_encoder_ckpt}; "
              f"trajectories will include 128D embeddings per hand.")

    # Build the clip list
    if args.manifest is not None:
        m = json.loads(args.manifest.read_text())
        clips_meta = m["clips"]
        print(f"v2: manifest mode, {len(clips_meta)} clips from {args.manifest}")
    elif args.clips_dir is not None:
        import re
        rx = re.compile(args.sign_from_filename_regex)
        clips_meta = []
        for p in sorted(args.clips_dir.rglob("*.mp4")):
            mm = rx.search(p.stem)
            sign = mm.group(1) if mm else "unknown"
            clips_meta.append({"clip_path": str(p), "sign_id": sign})
        print(f"v2: dir-scan mode, {len(clips_meta)} clips from {args.clips_dir}")
    else:
        raise SystemExit("provide --manifest or --clips-dir")

    ds = _ClipDataset(clips_meta, repo_root=args.repo_root, fps=args.fps)
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        num_workers=args.decode_workers, collate_fn=_collate)

    n_ok, n_skip, n_err = 0, 0, 0
    for i, item in enumerate(loader, 1):
        clip_path = Path(item["clip_path"])
        sign_id = item["sign_id"]
        out = args.out_dir / sign_id / f"{clip_path.stem}.json"
        if out.exists():
            n_skip += 1
            continue
        frames = item["frames"]
        if item["error"] or frames is None or frames.numel() == 0:
            print(f"  [skip-decode-fail] {clip_path}: {item.get('error')}")
            n_err += 1
            continue
        try:
            per_frame_bboxes = _detect_hands_batched(detectors["hand_detector"], frames, device)
            per_frame_hands = _landmarks_batched(detectors["hand_landmarks"], frames,
                                                 per_frame_bboxes, device)
            per_frame_pose = pose_batched_with_bboxes(detectors["pose"], frames,
                                                     per_frame_bboxes, device)
            encoder = detectors.get("handshape_encoder")
            if encoder is not None:
                per_frame_embeds = _encode_hands_batched(encoder, frames, per_frame_bboxes, device)
                for fi in range(frames.shape[0]):
                    for hi, hand_dict in enumerate(per_frame_hands[fi]):
                        if hi < len(per_frame_embeds[fi]) and per_frame_embeds[fi][hi] is not None:
                            hand_dict["embedding"] = per_frame_embeds[fi][hi]
            record = {
                "version": 2 if encoder is not None else 1,
                "clip_path": str(clip_path),
                "sign_id": sign_id,
                "fps": args.fps,
                "num_frames": frames.shape[0],
                "frame_size": [int(frames.shape[3]), int(frames.shape[2])],
                "frames": [
                    {"frame_idx": fi, "hands": per_frame_hands[fi], "pose": per_frame_pose[fi]}
                    for fi in range(frames.shape[0])
                ],
            }
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(record))
            n_ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"  [error] {clip_path}: {e!r}")
            n_err += 1
        if i % 50 == 0:
            print(f"  [{i}/{len(clips_meta)}] ok={n_ok} skip={n_skip} err={n_err}")

    print(f"done: ok={n_ok} skip={n_skip} err={n_err}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
