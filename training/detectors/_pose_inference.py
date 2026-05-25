"""Pose inference helpers — shared by live_demo + extract_trajectories.

pose_v0 was trained on tight person bbox crops (landmarks_dataset.py); feeding
it raw center-cropped full frames is OOD and produces near-constant nonsense
output (confirmed by data/probe/pose_crop_verify_*/compare.png on MPII and
data/probe/pose_asl_verify_*/compare.png on ASL). The fix is to crop to an
upper-body bbox derived from hand_det v2 bboxes before pose.

Topology assumption: pose_v0 emits (nose, neck, r_sh, l_sh, r_el, l_el,
r_wr, l_wr). Hands are 2-of-8 keypoints (wrists) and bound the lower half
of the upper-body silhouette; we extrapolate UP for head, OUT for shoulders.
"""
from __future__ import annotations

import torch

from training.detectors.pose_detector import PoseRegressor


def upper_body_bbox_from_hands(
    hand_bboxes: list[tuple[float, float, float, float]],
    img_w: int,
    img_h: int,
    *,
    up_factor: float = 3.5,
    side_factor: float = 1.5,
    down_factor: float = 0.5,
    pad_frac: float = 0.20,
    min_size_frac: float = 0.40,
) -> tuple[int, int, int, int] | None:
    """Heuristic upper-body bbox from hand detections.

    Heuristic constants are tuned so the resulting bbox roughly matches the
    keypoint-min/max bbox used during pose training (mpii_pose.py:137):
      - up_factor: head sits ~3 hand-heights above wrists
      - side_factor: shoulders sit ~1.5 hand-widths wider than wrists
      - pad_frac: 20% pad matches landmarks_dataset.py:114
      - min_size_frac: guards the "hands at face level" degenerate case
        where the union-of-hands bbox is too tight to contain the head

    Returns None if no hands were detected.
    """
    if not hand_bboxes:
        return None
    xs0 = [b[0] for b in hand_bboxes]
    ys0 = [b[1] for b in hand_bboxes]
    xs1 = [b[2] for b in hand_bboxes]
    ys1 = [b[3] for b in hand_bboxes]
    x0, y0, x1, y1 = min(xs0), min(ys0), max(xs1), max(ys1)

    hw = max(b[2] - b[0] for b in hand_bboxes)
    hh = max(b[3] - b[1] for b in hand_bboxes)

    x0 -= side_factor * hw
    x1 += side_factor * hw
    y0 -= up_factor * hh
    y1 += down_factor * hh

    w = x1 - x0
    h = y1 - y0
    pad = pad_frac * max(w, h)
    x0 -= pad; y0 -= pad; x1 += pad; y1 += pad

    # Enforce a minimum bbox size relative to the frame so the "hands at face
    # level" case doesn't produce a tiny crop. Expand around the bbox center.
    min_side = min_size_frac * min(img_w, img_h)
    cur_w, cur_h = x1 - x0, y1 - y0
    if cur_w < min_side:
        cx = (x0 + x1) / 2
        x0, x1 = cx - min_side / 2, cx + min_side / 2
    if cur_h < min_side:
        cy = (y0 + y1) / 2
        y0, y1 = cy - min_side / 2, cy + min_side / 2

    x0 = max(0, int(x0))
    y0 = max(0, int(y0))
    x1 = min(img_w, int(x1))
    y1 = min(img_h, int(y1))
    if x1 <= x0 or y1 <= y0:
        return None

    # Fix-up for the "hands at face level" case: when y0 was clamped to 0,
    # the bbox can end up too short vs wide. The pose net was trained on
    # roughly 1:1 person crops, so a 3:1 wide-short bbox squashes the body
    # under the 256×256 resize and pose collapses. Cap aspect-ratio
    # distortion at 1.6× by expanding the shorter dimension (within frame
    # bounds, centered on the bbox).
    MAX_ASPECT = 1.6
    w_, h_ = x1 - x0, y1 - y0
    if w_ > h_ * MAX_ASPECT:
        # too wide — pad height (downward first, then upward if blocked)
        need = int(w_ / MAX_ASPECT) - h_
        room_down = img_h - y1
        room_up = y0
        add_down = min(need, room_down)
        y1 += add_down
        add_up = min(need - add_down, room_up)
        y0 -= add_up
    elif h_ > w_ * MAX_ASPECT:
        # too tall — pad width symmetrically
        need = int(h_ / MAX_ASPECT) - w_
        half = need // 2
        room_left, room_right = x0, img_w - x1
        add_left = min(half, room_left)
        add_right = min(need - add_left, room_right)
        x0 -= add_left
        x1 += add_right

    return (x0, y0, x1, y1)


class PoseEMA:
    """Per-keypoint exponential moving average for temporal pose smoothing in
    live demos. Drops noisy single-frame jitter without lagging meaningful
    motion much. Reset between sign attempts.

    alpha = blend toward new sample; 0.4 is responsive without jitter.
    Treats (0,0) keypoints as "missing" — passes prior value through.
    """

    def __init__(self, alpha: float = 0.4, n_kp: int = 8):
        self.alpha = alpha
        self.n_kp = n_kp
        self._state: list[list[float]] | None = None

    def reset(self) -> None:
        self._state = None

    def update(self, kpts: list[list[float]]) -> list[list[float]]:
        if not kpts:
            return kpts
        if self._state is None:
            self._state = [list(kp) for kp in kpts]
            return [list(kp) for kp in self._state]
        out: list[list[float]] = []
        for i, kp in enumerate(kpts):
            prev = self._state[i] if i < len(self._state) else None
            if prev is None or (kp[0] == 0 and kp[1] == 0):
                out.append(prev if prev is not None else list(kp))
                continue
            x = self.alpha * kp[0] + (1 - self.alpha) * prev[0]
            y = self.alpha * kp[1] + (1 - self.alpha) * prev[1]
            self._state[i] = [x, y]
            out.append([x, y])
        return out


@torch.no_grad()
def upper_body_bbox_from_face(
    face_bbox: tuple[float, float, float, float] | None,
    img_w: int,
    img_h: int,
    *,
    down_factor: float = 4.2,
    side_factor: float = 1.9,
    up_factor: float = 0.5,
    pad_frac: float = 0.12,
) -> tuple[int, int, int, int] | None:
    """Upper-body bbox anchored on the FACE, not the hands.

    The face is a stable body landmark that does NOT move when the user
    gestures — so a face-anchored crop keeps the pose on the body regardless
    of where the hands go (fixes "the body runs to the hand"). Geometry:
    head occupies the top ~1/5 of the upper body; extrapolate down for torso
    and out for shoulders from the face box.
    """
    if face_bbox is None:
        return None
    fx0, fy0, fx1, fy1 = face_bbox
    fw, fh = fx1 - fx0, fy1 - fy0
    if fw <= 0 or fh <= 0:
        return None
    cx = (fx0 + fx1) / 2.0
    x0 = cx - side_factor * fw
    x1 = cx + side_factor * fw
    y0 = fy0 - up_factor * fh
    y1 = fy1 + down_factor * fh
    pad = pad_frac * max(x1 - x0, y1 - y0)
    x0 -= pad; y0 -= pad; x1 += pad; y1 += pad
    x0 = max(0, int(x0)); y0 = max(0, int(y0))
    x1 = min(img_w, int(x1)); y1 = min(img_h, int(y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def pose_on_upper_body_crop(
    pose: PoseRegressor,
    frame_chw: torch.Tensor,
    hand_bboxes: list[tuple[float, float, float, float]],
    face_bbox: tuple[float, float, float, float] | None = None,
) -> list[list[float]]:
    """Run pose on an upper-body crop. Prefers a FACE-anchored crop (stable,
    independent of hand gestures); falls back to the hand-derived crop when
    no face is detected.

    frame_chw is (1, 3, H, W) on the model's device, values in [0, 1].
    Returns 8 keypoints in full-image pixel coordinates.
    """
    _, _, H, W = frame_chw.shape
    bbox = upper_body_bbox_from_face(face_bbox, W, H)
    if bbox is None:
        bbox = upper_body_bbox_from_hands(hand_bboxes, W, H)
    if bbox is None:
        return []
    bx0, by0, bx1, by1 = bbox
    crop = frame_chw[:, :, by0:by1, bx0:bx1]
    x = torch.nn.functional.interpolate(
        crop, size=(PoseRegressor.INPUT_SIZE, PoseRegressor.INPUT_SIZE),
        mode="bilinear", align_corners=False,
    )
    out = pose(x)
    coords = out["coords"][0].cpu().numpy()  # (8, 2) in [0,1] of crop
    cw, ch = bx1 - bx0, by1 - by0
    return [[float(bx0 + kx * cw), float(by0 + ky * ch)] for kx, ky in coords]


@torch.no_grad()
def pose_batched_with_bboxes(
    pose: PoseRegressor,
    frames: torch.Tensor,
    per_frame_hand_bboxes: list[list[tuple[float, float, float, float]]],
    device: str,
) -> list[list[list[float]]]:
    """Batched version for trajectory extraction.

    Builds an upper-body crop per frame (skipping frames with no hands),
    stacks the resized crops, runs pose once, maps keypoints back. Frames
    with no hand detection get pose=[] (caller fills via interpolation if
    needed).
    """
    N, _, H, W = frames.shape
    crops = []
    indices: list[int] = []
    bboxes: list[tuple[int, int, int, int]] = []
    for fi in range(N):
        bbox = upper_body_bbox_from_hands(per_frame_hand_bboxes[fi], W, H)
        if bbox is None:
            continue
        bx0, by0, bx1, by1 = bbox
        crop = frames[fi:fi + 1, :, by0:by1, bx0:bx1]
        x = torch.nn.functional.interpolate(
            crop, size=(PoseRegressor.INPUT_SIZE, PoseRegressor.INPUT_SIZE),
            mode="bilinear", align_corners=False,
        )
        crops.append(x)
        indices.append(fi)
        bboxes.append(bbox)

    out_per_frame: list[list[list[float]]] = [[] for _ in range(N)]
    if not crops:
        return out_per_frame
    batch = torch.cat(crops, 0).to(device, non_blocking=True)
    out = pose(batch)
    coords = out["coords"].cpu().numpy()  # (K, 8, 2)
    for k, fi in enumerate(indices):
        bx0, by0, bx1, by1 = bboxes[k]
        cw, ch = bx1 - bx0, by1 - by0
        out_per_frame[fi] = [[float(bx0 + kx * cw), float(by0 + ky * ch)]
                             for kx, ky in coords[k]]
    return out_per_frame


@torch.no_grad()
def pose_batched_with_face_bboxes(
    pose: PoseRegressor,
    frames: torch.Tensor,
    per_frame_face_bbox: list[tuple[float, float, float, float] | None],
    per_frame_hand_bboxes: list[list[tuple[float, float, float, float]]],
    device: str,
) -> list[list[list[float]]]:
    """FACE-anchored batched pose for trajectory extraction — the offline twin
    of live_demo's pose_for_frame(face_bbox=...). For each frame, build the
    upper-body crop from the (smoothed) face box; fall back to the hand-derived
    crop when no face was detected. Frames with neither get pose=[].

    This is the experiment lever: the v2/v3 classifier (75.8) was trained on the
    HAND-anchored crop (pose_batched_with_bboxes); this swaps in the
    face-anchored crop the terminal demo uses (the "clean" look).
    """
    N, _, H, W = frames.shape
    crops = []
    indices: list[int] = []
    bboxes: list[tuple[int, int, int, int]] = []
    for fi in range(N):
        bbox = upper_body_bbox_from_face(per_frame_face_bbox[fi], W, H)
        if bbox is None:
            bbox = upper_body_bbox_from_hands(per_frame_hand_bboxes[fi], W, H)
        if bbox is None:
            continue
        bx0, by0, bx1, by1 = bbox
        crop = frames[fi:fi + 1, :, by0:by1, bx0:bx1]
        x = torch.nn.functional.interpolate(
            crop, size=(PoseRegressor.INPUT_SIZE, PoseRegressor.INPUT_SIZE),
            mode="bilinear", align_corners=False,
        )
        crops.append(x)
        indices.append(fi)
        bboxes.append(bbox)

    out_per_frame: list[list[list[float]]] = [[] for _ in range(N)]
    if not crops:
        return out_per_frame
    batch = torch.cat(crops, 0).to(device, non_blocking=True)
    out = pose(batch)
    coords = out["coords"].cpu().numpy()  # (K, 8, 2)
    for k, fi in enumerate(indices):
        bx0, by0, bx1, by1 = bboxes[k]
        cw, ch = bx1 - bx0, by1 - by0
        out_per_frame[fi] = [[float(bx0 + kx * cw), float(by0 + ky * ch)]
                             for kx, ky in coords[k]]
    return out_per_frame
