"""Augmentations for hand-bbox detector training.

Per docs/VOCABULARY_TRAINER_ROADMAP.md Phase 1 Slice 1.3:
- random crops
- brightness / contrast / saturation
- motion blur (simple kernel, applied randomly)
- horizontal flips (carefully — flipping does change which hand is which,
  but for detection-only training this is fine; the detector is
  hand-vs-not-hand without left/right semantics, which arrive in
  the downstream landmark / pose stages).

All ops operate on a (3, H, W) tensor in [0, 1] and a list of
[x_min, y_min, x_max, y_max] bboxes in the same pixel space.
No torchvision.models. torchvision.transforms.functional only.
"""

from __future__ import annotations

import random

import torch
import torchvision.transforms.functional as TF


def _rand(low: float, high: float) -> float:
    return low + (high - low) * random.random()


def random_horizontal_flip(
    img: torch.Tensor, bboxes: list[list[float]], p: float = 0.5
) -> tuple[torch.Tensor, list[list[float]]]:
    if random.random() >= p:
        return img, bboxes
    _, _, w = img.shape
    img = TF.hflip(img)
    flipped = [[w - b[2], b[1], w - b[0], b[3]] for b in bboxes]
    return img, flipped


def random_photometric(
    img: torch.Tensor, bboxes: list[list[float]]
) -> tuple[torch.Tensor, list[list[float]]]:
    img = TF.adjust_brightness(img, _rand(0.7, 1.3))
    img = TF.adjust_contrast(img, _rand(0.7, 1.3))
    img = TF.adjust_saturation(img, _rand(0.7, 1.3))
    img = TF.adjust_hue(img, _rand(-0.05, 0.05))
    return img.clamp(0, 1), bboxes


def random_motion_blur(
    img: torch.Tensor, bboxes: list[list[float]], p: float = 0.35
) -> tuple[torch.Tensor, list[list[float]]]:
    """Angled line-kernel motion blur. Re-enabled for P3: the hand detector
    was dropping ~40% of ASL frames to motion blur during fast signing, and
    this aug was previously disabled. Arbitrary angle (not just h/v) + larger
    kernels better match real signing motion than the old separable kernel.
    """
    if random.random() >= p:
        return img, bboxes
    k = random.choice([5, 7, 9, 11])
    angle = random.uniform(0, 3.14159)  # radians, 0..pi
    kernel = torch.zeros(k, k)
    c = (k - 1) / 2.0
    dx, dy = torch.cos(torch.tensor(angle)), torch.sin(torch.tensor(angle))
    for t in torch.linspace(-c, c, steps=k * 2):
        x = int(round(c + (dx * t).item()))
        y = int(round(c + (dy * t).item()))
        if 0 <= x < k and 0 <= y < k:
            kernel[y, x] = 1.0
    if kernel.sum() == 0:
        kernel[int(c), :] = 1.0
    kernel = (kernel / kernel.sum()).view(1, 1, k, k).repeat(3, 1, 1, 1)
    blurred = torch.nn.functional.conv2d(
        img.unsqueeze(0), kernel, padding=k // 2, groups=3
    ).squeeze(0)
    return blurred.clamp(0, 1), bboxes


def random_crop_resize(
    img: torch.Tensor,
    bboxes: list[list[float]],
    scale_range: tuple[float, float] = (0.7, 1.0),
    p: float = 0.5,
) -> tuple[torch.Tensor, list[list[float]]]:
    """Random square crop covering scale_range[0]..1.0 of the image,
    then resized back to original. Bboxes are clipped to the crop and
    dropped if they degenerate.
    """
    if random.random() >= p:
        return img, bboxes
    _, h, w = img.shape
    s = _rand(*scale_range)
    cw = int(w * s)
    ch = int(h * s)
    x0 = random.randint(0, w - cw)
    y0 = random.randint(0, h - ch)
    cropped = img[:, y0 : y0 + ch, x0 : x0 + cw]
    resized = torch.nn.functional.interpolate(
        cropped.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False
    ).squeeze(0)

    sx = w / cw
    sy = h / ch
    new_bboxes = []
    for bx0, by0, bx1, by1 in bboxes:
        nx0 = max(bx0 - x0, 0) * sx
        ny0 = max(by0 - y0, 0) * sy
        nx1 = min(bx1 - x0, cw) * sx
        ny1 = min(by1 - y0, ch) * sy
        if nx1 - nx0 < 4 or ny1 - ny0 < 4:
            continue
        new_bboxes.append([nx0, ny0, nx1, ny1])
    return resized, new_bboxes


def random_zoom_out(
    img: torch.Tensor,
    bboxes: list[list[float]],
    scale_range: tuple[float, float] = (1.0, 3.0),
    p: float = 0.5,
) -> tuple[torch.Tensor, list[list[float]]]:
    """Paste the image onto a larger gray canvas (scale ≥ 1.0), then
    resize back to original. Simulates hands smaller relative to frame —
    the OOD case that webcam-deployment training data lacks.
    """
    if random.random() >= p:
        return img, bboxes
    _, h, w = img.shape
    s = _rand(*scale_range)
    if s <= 1.001:
        return img, bboxes
    new_h, new_w = int(h * s), int(w * s)
    canvas = torch.full((3, new_h, new_w), 0.5, dtype=img.dtype)
    ox = random.randint(0, new_w - w)
    oy = random.randint(0, new_h - h)
    canvas[:, oy:oy + h, ox:ox + w] = img
    # Resize canvas back to (h, w)
    resized = torch.nn.functional.interpolate(
        canvas.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False
    ).squeeze(0)
    sx = w / new_w
    sy = h / new_h
    new_bboxes = [[(b[0] + ox) * sx, (b[1] + oy) * sy,
                   (b[2] + ox) * sx, (b[3] + oy) * sy] for b in bboxes]
    return resized, new_bboxes


def random_letterbox_aspect(
    img: torch.Tensor,
    bboxes: list[list[float]],
    aspect_range: tuple[float, float] = (9.0 / 16.0, 16.0 / 9.0),
    p: float = 0.3,
) -> tuple[torch.Tensor, list[list[float]]]:
    """Re-fit the image into a non-square aspect ratio via letterbox
    padding (gray bars), then re-square. Simulates webcam capture aspect
    drift — CMU data is mostly square-cropped, real cams are 16:9.
    """
    if random.random() >= p:
        return img, bboxes
    _, h, w = img.shape
    target_aspect = _rand(*aspect_range)  # w/h
    if target_aspect > 1.0:
        new_w = w
        new_h = int(w / target_aspect)
    else:
        new_h = h
        new_w = int(h * target_aspect)
    new_h = max(new_h, 1)
    new_w = max(new_w, 1)
    inner = torch.nn.functional.interpolate(
        img.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False
    ).squeeze(0)
    canvas = torch.full((3, h, w), 0.5, dtype=img.dtype)
    oy = (h - new_h) // 2
    ox = (w - new_w) // 2
    canvas[:, oy:oy + new_h, ox:ox + new_w] = inner
    sx = new_w / w
    sy = new_h / h
    new_bboxes = [[b[0] * sx + ox, b[1] * sy + oy,
                   b[2] * sx + ox, b[3] * sy + oy] for b in bboxes]
    return canvas, new_bboxes


def default_train_augment(
    img: torch.Tensor, bboxes: list[list[float]]
) -> tuple[torch.Tensor, list[list[float]]]:
    # Each aug must justify its CPU cost against real webcam OOD risk:
    #   - random_zoom_out (p=0.5): user-at-distance ✓
    #   - random_letterbox_aspect (p=0.15, was 0.3): 16:9 webcam → 320×320
    #     squash. EVERY real user hits this. Kept but dial-backed; the
    #     epoch-30 plateau audit suggested it was over-active.
    #   - random_crop_resize (p=0.5): hand-not-centered framing ✓
    #   - random_horizontal_flip (p=0.5): left- vs right-handed ✓
    #   - random_photometric: lighting + skin-tone variation ✓
    #   - random_motion_blur (p=0.35): RE-ENABLED for P3 with angled kernels.
    #     The detector was missing ~40% of ASL frames to signing motion blur
    #     (measured: scripts/p1_landmark_diag.py). This is the highest-value
    #     aug change for closing that gap.
    img, bboxes = random_zoom_out(img, bboxes)
    img, bboxes = random_letterbox_aspect(img, bboxes, p=0.15)
    img, bboxes = random_crop_resize(img, bboxes)
    img, bboxes = random_horizontal_flip(img, bboxes)
    img, bboxes = random_motion_blur(img, bboxes)
    img, bboxes = random_photometric(img, bboxes)
    return img, bboxes
