"""Keypoint-aware augmentations for hand landmark training.

Operates on (img, coords, visibility):
  - img: (3, 224, 224) in [0, 1]
  - coords: (21, 2) in [0, 1] of crop
  - visibility: (21,) in {0.0, 1.0}

All transforms preserve the coord/visibility correspondence — when
the image flips, rotates, or scales, the keypoints move accordingly.
torchvision.transforms.functional only (no .models).
"""

from __future__ import annotations

import math
import random

import torch
import torchvision.transforms.functional as TF


def _rand(lo: float, hi: float) -> float:
    return lo + (hi - lo) * random.random()


# Topology pairs for left/right swap on horizontal flip. Hand kinematics
# don't have left/right paired keypoints (only one hand per crop), but the
# semantic of "left vs right hand" inverts. The keypoint INDICES inside one
# hand stay the same. So hflip is a coordinate-only transform.


def random_hflip(img, coords, vis, p: float = 0.5):
    if random.random() >= p:
        return img, coords, vis
    img = TF.hflip(img)
    coords = coords.clone()
    coords[:, 0] = 1.0 - coords[:, 0]
    return img, coords, vis


def random_rotation(img, coords, vis, max_deg: float = 25.0):
    angle = _rand(-max_deg, max_deg)
    img = TF.rotate(img, angle)
    # Rotate keypoints around image center (0.5, 0.5)
    rad = math.radians(angle)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    coords = coords.clone()
    xs = coords[:, 0] - 0.5
    ys = coords[:, 1] - 0.5
    # torchvision rotate is COUNTER-clockwise positive; the inverse to apply
    # to coords is also CCW (point follows the image).
    new_x = cos_a * xs - sin_a * ys + 0.5
    new_y = sin_a * xs + cos_a * ys + 0.5
    coords[:, 0] = new_x
    coords[:, 1] = new_y
    # Mark keypoints that fell outside [0, 1] as invisible
    in_bounds = (coords[:, 0] >= 0) & (coords[:, 0] <= 1) & \
                (coords[:, 1] >= 0) & (coords[:, 1] <= 1)
    vis = vis * in_bounds.float()
    return img, coords, vis


def random_scale_translate(img, coords, vis, scale_range=(0.65, 1.40), trans_frac=0.08):
    s = _rand(*scale_range)
    tx = _rand(-trans_frac, trans_frac)
    ty = _rand(-trans_frac, trans_frac)
    H, W = img.shape[-2:]
    img = TF.affine(img, angle=0, translate=[int(tx * W), int(ty * H)], scale=s, shear=[0.0, 0.0])
    coords = coords.clone()
    coords[:, 0] = (coords[:, 0] - 0.5) * s + 0.5 + tx
    coords[:, 1] = (coords[:, 1] - 0.5) * s + 0.5 + ty
    in_bounds = (coords[:, 0] >= 0) & (coords[:, 0] <= 1) & \
                (coords[:, 1] >= 0) & (coords[:, 1] <= 1)
    vis = vis * in_bounds.float()
    return img, coords, vis


def random_photometric(img, coords, vis):
    img = TF.adjust_brightness(img, _rand(0.7, 1.3))
    img = TF.adjust_contrast(img, _rand(0.7, 1.3))
    img = TF.adjust_saturation(img, _rand(0.7, 1.3))
    img = TF.adjust_hue(img, _rand(-0.04, 0.04))
    return img.clamp(0, 1), coords, vis


def default_train_augment(img, coords, vis):
    img, coords, vis = random_hflip(img, coords, vis)
    img, coords, vis = random_rotation(img, coords, vis)
    img, coords, vis = random_scale_translate(img, coords, vis)
    img, coords, vis = random_photometric(img, coords, vis)
    return img, coords, vis
