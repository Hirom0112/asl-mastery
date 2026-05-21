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
    img: torch.Tensor, bboxes: list[list[float]], p: float = 0.2
) -> tuple[torch.Tensor, list[list[float]]]:
    if random.random() >= p:
        return img, bboxes
    k = random.choice([3, 5, 7])
    direction = random.choice(["h", "v"])
    kernel = torch.zeros(k, k)
    if direction == "h":
        kernel[k // 2, :] = 1.0 / k
    else:
        kernel[:, k // 2] = 1.0 / k
    kernel = kernel.view(1, 1, k, k).repeat(3, 1, 1, 1)
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


def default_train_augment(
    img: torch.Tensor, bboxes: list[list[float]]
) -> tuple[torch.Tensor, list[list[float]]]:
    img, bboxes = random_crop_resize(img, bboxes)
    img, bboxes = random_horizontal_flip(img, bboxes)
    img, bboxes = random_photometric(img, bboxes)
    img, bboxes = random_motion_blur(img, bboxes)
    return img, bboxes
