"""GPU-side batched augmentations + target rendering.

The CPU augmentation chain in augment.py runs at ~6 ms/sample in
__getitem__; with Modal's spawn-start DataLoader workers, effective
parallelism caps at ~1.7× even with 8 workers, giving the 211 samples/s
we measured. Moving augs to the GPU lifts the bottleneck off the
data-loader entirely — they run on the assembled batch tensor in ~ms
per batch of 256 instead of ~6 ms × 256 = 1.5 s per batch on CPU.

Design:
- __getitem__ returns just the resized uint8 image + a (N_max, 4)
  padded bbox tensor + (N_max,) mask. No augment, no target render.
- _collate stacks these into batched tensors.
- After `.to(device)` in train.py we call `gpu_augment_batch(images,
  bboxes, mask)` which applies (in order): hflip, affine warp
  (combines zoom_out + crop_resize + letterbox), photometric.
- `render_targets_gpu(bboxes, mask)` produces the (B, 1, H, W)
  Gaussian heatmap + (B, 2, H, W) size map + (B, 1, H, W) mask in
  vectorized torch ops on GPU.

No pretrained components, no torchvision.models. Only
torch.nn.functional + torchvision.transforms.functional are used —
the same allowlist as augment.py under ADR-0012.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# Per-batch maximum bboxes. The hand_bbox dataset has 1 bbox per record
# in the manifest, but we pad to N_MAX_BBOXES to keep the collate fixed-shape.
N_MAX_BBOXES = 4


# ---------------------------------------------------------------------------
# Bbox padding (used by dataset / collate)
# ---------------------------------------------------------------------------


def pad_bboxes(boxes: list[list[float]], n_max: int = N_MAX_BBOXES) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad a list of [x0, y0, x1, y1] to a fixed (n_max, 4) tensor plus a
    (n_max,) bool mask marking which slots are valid.
    """
    out = torch.zeros((n_max, 4), dtype=torch.float32)
    mask = torch.zeros((n_max,), dtype=torch.bool)
    for i, b in enumerate(boxes[:n_max]):
        out[i, 0] = b[0]
        out[i, 1] = b[1]
        out[i, 2] = b[2]
        out[i, 3] = b[3]
        mask[i] = True
    return out, mask


# ---------------------------------------------------------------------------
# GPU-side augmentations on assembled batches
# ---------------------------------------------------------------------------


def gpu_random_hflip(
    images: torch.Tensor,    # (B, 3, H, W)
    bboxes: torch.Tensor,    # (B, N_max, 4) xyxy
    p: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample random horizontal flip. Cheap — no resampling."""
    B, _, _, W = images.shape
    device = images.device
    flip = torch.rand(B, device=device) < p   # (B,)
    flip_b = flip[:, None, None, None]
    images = torch.where(flip_b, torch.flip(images, dims=[3]), images)
    # bbox: new_x0 = W - x1, new_x1 = W - x0
    fx0 = W - bboxes[..., 2:3]
    fx1 = W - bboxes[..., 0:1]
    flipped_bb = torch.cat([fx0, bboxes[..., 1:2], fx1, bboxes[..., 3:4]], dim=-1)
    flip_bb = flip[:, None, None]
    bboxes = torch.where(flip_bb, flipped_bb, bboxes)
    return images, bboxes


def _build_affine_grid(
    B: int,
    H: int,
    W: int,
    scale: torch.Tensor,     # (B,) in (0, 1] — fraction of source covered
    tx: torch.Tensor,        # (B,) normalized [-1, 1]
    ty: torch.Tensor,
    aspect: torch.Tensor,    # (B,) target_w/target_h (1.0 = square)
    device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample affine matrix for combined zoom + crop + aspect warp.

    Returns (grid, theta) where:
      - `grid`: (B, H, W, 2) for `F.grid_sample`
      - `theta`: (B, 2, 3) the affine matrix in normalized [-1, 1] coords,
        so bboxes can be transformed the same way.

    Geometry: sample from a sub-region of the source image whose width
    is W·scale·aspect, height is H·scale, offset by (tx, ty) (clamped
    so the sub-region stays inside [-1, 1]).
    """
    # The affine grid maps output normalized coords → input normalized
    # coords. A scale of 0.5 means the output is sampled from the middle
    # half of the input — i.e. a 2× zoom-in. Combined with offset, this
    # spans random_zoom_out (scale<1 makes hand smaller relative to
    # canvas), random_crop_resize (scale<1 with offset), and aspect
    # changes (different sx vs sy).
    sx = scale * aspect.clamp(min=1.0)    # widen if aspect > 1
    sy = scale * (1.0 / aspect.clamp(min=1.0))  # narrow if aspect > 1
    # Wait — that doesn't handle aspect < 1. Fix: split into sx, sy
    # such that sx/sy = aspect, both ≤ 1 max(scale).
    sx = scale * torch.where(aspect >= 1.0, aspect, torch.ones_like(aspect))
    sy = scale * torch.where(aspect >= 1.0, torch.ones_like(aspect), 1.0 / aspect)
    # clamp offsets so the sample region stays inside [-1, 1]
    tx = tx * (1.0 - sx).clamp(min=0.0)
    ty = ty * (1.0 - sy).clamp(min=0.0)

    theta = torch.zeros(B, 2, 3, device=device)
    theta[:, 0, 0] = sx
    theta[:, 1, 1] = sy
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, (B, 1, H, W), align_corners=False)
    return grid, theta


def _affine_bboxes(
    bboxes: torch.Tensor,    # (B, N_max, 4) xyxy in pixel coords [0, W] × [0, H]
    theta: torch.Tensor,     # (B, 2, 3) normalized affine
    H: int,
    W: int,
) -> torch.Tensor:
    """Apply the inverse of `theta` to bboxes — because `theta` maps
    output→input for sampling, the bboxes in the *output* image come
    from inverting that mapping.
    """
    B, N, _ = bboxes.shape
    # Convert pixel-space bboxes → normalized [-1, 1] xyxy
    x0 = bboxes[..., 0] / W * 2.0 - 1.0
    y0 = bboxes[..., 1] / H * 2.0 - 1.0
    x1 = bboxes[..., 2] / W * 2.0 - 1.0
    y1 = bboxes[..., 3] / H * 2.0 - 1.0

    # Inverse of theta = [[sx, 0, tx], [0, sy, ty]] is
    #   [[1/sx, 0, -tx/sx], [0, 1/sy, -ty/sy]]
    sx = theta[:, 0, 0]    # (B,)
    sy = theta[:, 1, 1]
    tx = theta[:, 0, 2]
    ty = theta[:, 1, 2]
    inv_sx = 1.0 / sx
    inv_sy = 1.0 / sy
    inv_tx = -tx / sx
    inv_ty = -ty / sy

    # Apply per-sample (broadcast over N)
    nx0 = x0 * inv_sx[:, None] + inv_tx[:, None]
    ny0 = y0 * inv_sy[:, None] + inv_ty[:, None]
    nx1 = x1 * inv_sx[:, None] + inv_tx[:, None]
    ny1 = y1 * inv_sy[:, None] + inv_ty[:, None]

    # Order min/max (flip-safe even though hflip ran first)
    nlx = torch.minimum(nx0, nx1)
    nly = torch.minimum(ny0, ny1)
    nhx = torch.maximum(nx0, nx1)
    nhy = torch.maximum(ny0, ny1)

    # Convert back to pixel space + clip to image rect
    px0 = ((nlx + 1.0) * 0.5 * W).clamp(0.0, float(W))
    py0 = ((nly + 1.0) * 0.5 * H).clamp(0.0, float(H))
    px1 = ((nhx + 1.0) * 0.5 * W).clamp(0.0, float(W))
    py1 = ((nhy + 1.0) * 0.5 * H).clamp(0.0, float(H))
    return torch.stack([px0, py0, px1, py1], dim=-1)


def gpu_random_affine(
    images: torch.Tensor,
    bboxes: torch.Tensor,
    p: float = 0.7,
    scale_range: tuple[float, float] = (0.5, 1.0),
    offset_range: tuple[float, float] = (-1.0, 1.0),
    aspect_range: tuple[float, float] = (1.0, 1.7),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combined zoom_out + crop_resize + aspect-distortion in one warp.

    Per-sample random parameters:
      - scale ∈ scale_range (1.0 = identity, <1.0 = sub-region sampled)
      - tx, ty ∈ offset_range (× the remaining headroom)
      - aspect ∈ aspect_range (1.0 = square)

    With p% probability per sample the affine is applied; otherwise the
    sample passes through unchanged.
    """
    B, _, H, W = images.shape
    device = images.device
    do = torch.rand(B, device=device) < p

    scale = torch.empty(B, device=device).uniform_(*scale_range)
    tx = torch.empty(B, device=device).uniform_(*offset_range)
    ty = torch.empty(B, device=device).uniform_(*offset_range)
    aspect = torch.empty(B, device=device).uniform_(*aspect_range)

    # Identity where do=False
    scale = torch.where(do, scale, torch.ones_like(scale))
    tx = torch.where(do, tx, torch.zeros_like(tx))
    ty = torch.where(do, ty, torch.zeros_like(ty))
    aspect = torch.where(do, aspect, torch.ones_like(aspect))

    grid, theta = _build_affine_grid(B, H, W, scale, tx, ty, aspect, device)
    images = F.grid_sample(images, grid, mode="bilinear",
                           padding_mode="zeros", align_corners=False)
    bboxes = _affine_bboxes(bboxes, theta, H, W)
    return images, bboxes


def gpu_random_photometric(images: torch.Tensor) -> torch.Tensor:
    """Batched per-image photometric. Brightness / contrast / saturation
    in pure broadcasting tensor math — one kernel each, no Python loop.

    The previous implementation looped over B and called torchvision's
    `TF.adjust_*` per image, each requiring `.item()` on the per-image
    factor (forcing GPU↔CPU sync) + 4 kernel launches per image. At
    batch 256 this was 1024 kernel launches + 1024 sync points per
    batch — the dominant cost in Modal training at ~500 ms/batch.

    Hue rotation is dropped — its RGB↔HSV math is non-trivial to batch
    cleanly and webcam hue variance is small. Brightness/contrast/
    saturation cover the meaningful color OOD.
    """
    B = images.shape[0]
    device = images.device
    dtype = images.dtype

    # Per-image factors as (B, 1, 1, 1) tensors for broadcasting
    bri = torch.empty(B, 1, 1, 1, device=device, dtype=dtype).uniform_(0.7, 1.3)
    con = torch.empty(B, 1, 1, 1, device=device, dtype=dtype).uniform_(0.7, 1.3)
    sat = torch.empty(B, 1, 1, 1, device=device, dtype=dtype).uniform_(0.7, 1.3)

    # --- Brightness: x * factor
    x = images * bri

    # --- Contrast: (x - mean) * factor + mean
    # `mean` is the per-image RGB-luminance grayscale mean, broadcast back.
    gray_w = torch.tensor([0.2989, 0.5870, 0.1140],
                          device=device, dtype=dtype).view(1, 3, 1, 1)
    gray = (x * gray_w).sum(dim=1, keepdim=True)              # (B, 1, H, W)
    mean_per_image = gray.mean(dim=[2, 3], keepdim=True)      # (B, 1, 1, 1)
    x = (x - mean_per_image) * con + mean_per_image

    # --- Saturation: lerp(grayscale_3ch, x, factor)
    gray = (x * gray_w).sum(dim=1, keepdim=True)              # recompute on post-contrast
    gray_3ch = gray.expand_as(x)
    x = gray_3ch + sat * (x - gray_3ch)

    return x.clamp(0.0, 1.0)


def gpu_random_motion_blur(
    images: torch.Tensor,    # (B, 3, H, W)
    p: float = 0.35,
    k: int = 11,
) -> torch.Tensor:
    """Per-sample angled line-kernel motion blur, fully on GPU.

    P3: the detector misses ~40% of ASL frames to signing motion blur, and
    this aug only existed (disabled) on the CPU path. Implemented here with
    the per-sample depthwise-conv trick: each image gets its own random
    angle, non-selected images get an identity kernel — one grouped conv for
    the whole batch, no Python-per-image loop.
    """
    B, C, H, W = images.shape
    device = images.device
    c = (k - 1) / 2.0
    ang = torch.rand(B, device=device) * math.pi
    do = torch.rand(B, device=device) < p
    steps = torch.linspace(-c, c, steps=2 * k, device=device)        # (S,)
    xs = (c + torch.cos(ang)[:, None] * steps[None, :]).round().long().clamp(0, k - 1)
    ys = (c + torch.sin(ang)[:, None] * steps[None, :]).round().long().clamp(0, k - 1)
    kernels = torch.zeros(B, k, k, device=device)
    bidx = torch.arange(B, device=device)[:, None].expand_as(xs)
    kernels[bidx.reshape(-1), ys.reshape(-1), xs.reshape(-1)] = 1.0
    kernels = kernels / kernels.sum(dim=(1, 2), keepdim=True).clamp(min=1.0)
    ident = torch.zeros(B, k, k, device=device)
    ident[:, int(c), int(c)] = 1.0
    kernels = torch.where(do[:, None, None], kernels, ident)
    weight = kernels[:, None, :, :].repeat(1, C, 1, 1).reshape(B * C, 1, k, k)
    out = F.conv2d(images.reshape(1, B * C, H, W), weight,
                   padding=k // 2, groups=B * C)
    return out.reshape(B, C, H, W).clamp(0.0, 1.0)


def gpu_augment_batch(
    images: torch.Tensor,
    bboxes: torch.Tensor,
    photometric: bool = True,
    motion_blur: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the whole train-time augmentation chain on GPU."""
    images, bboxes = gpu_random_hflip(images, bboxes)
    images, bboxes = gpu_random_affine(images, bboxes)
    if motion_blur:
        images = gpu_random_motion_blur(images)
    if photometric:
        images = gpu_random_photometric(images)
    return images, bboxes


# ---------------------------------------------------------------------------
# GPU target rendering — vectorized over the batch
# ---------------------------------------------------------------------------


def render_targets_gpu(
    bboxes: torch.Tensor,    # (B, N_max, 4) xyxy in pixel space (post-aug)
    mask: torch.Tensor,      # (B, N_max) bool — which bboxes are valid
    input_size: int = 320,
    stride: int = 4,
    min_overlap: float = 0.7,
    device: str | torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Render the CenterNet training targets directly on GPU.

    Returns the same dict shape the CPU `_render_targets` produced:
      - heatmap:     (B, 1, H, W)
      - size:        (B, 2, H, W)
      - center_mask: (B, 1, H, W)

    All ops are vectorized — no per-bbox Python loop. The cost is
    dominated by the (H, W) broadcast for the Gaussian, which on a
    256-batch at H=W=80 is one ~5 ms kernel.
    """
    if device is None:
        device = bboxes.device
    B = bboxes.shape[0]
    H = W = input_size // stride

    # Per-bbox center and size in feature-map coords
    cx = (bboxes[..., 0] + bboxes[..., 2]) / (2.0 * stride)
    cy = (bboxes[..., 1] + bboxes[..., 3]) / (2.0 * stride)
    bw = (bboxes[..., 2] - bboxes[..., 0]).clamp(min=1.0)
    bh = (bboxes[..., 3] - bboxes[..., 1]).clamp(min=1.0)

    # Integer center pixel (inside the feature map)
    ix = cx.long().clamp(0, W - 1)
    iy = cy.long().clamp(0, H - 1)

    # CenterNet Gaussian radius formula (vectorized)
    fmap_w = bw / stride
    fmap_h = bh / stride
    radius = _gaussian_radius_batched(fmap_w, fmap_h, min_overlap)
    sigma = (radius * 2.0 + 1.0) / 6.0

    # Build per-(B, N) Gaussian heatmaps via outer-product grid
    yy = torch.arange(H, device=device, dtype=torch.float32)
    xx = torch.arange(W, device=device, dtype=torch.float32)
    # Shapes: gauss (B, N_max, H, W)
    dy2 = (yy[None, None, :, None] - cy[:, :, None, None]) ** 2
    dx2 = (xx[None, None, None, :] - cx[:, :, None, None]) ** 2
    sigma2 = (sigma[:, :, None, None] ** 2).clamp(min=1e-3)
    gauss = torch.exp(-(dy2 + dx2) / (2.0 * sigma2))
    # Mask out invalid slots
    gauss = gauss * mask[:, :, None, None].float()
    # Per-batch max across N_max → (B, H, W)
    heatmap = gauss.amax(dim=1, keepdim=True)   # (B, 1, H, W)

    # Snap exact-1.0 at center pixels (focal-loss positive marker)
    # Scatter ones into heatmap at (iy, ix) for each valid bbox.
    b_idx = torch.arange(B, device=device)[:, None].expand_as(ix)
    valid_flat = mask
    # Build a center-mask tensor by scatter
    center_mask = torch.zeros((B, 1, H, W), device=device)
    # Index put: for each valid (b, n), set center_mask[b, 0, iy, ix] = 1
    flat_b = b_idx[valid_flat]
    flat_iy = iy[valid_flat]
    flat_ix = ix[valid_flat]
    if flat_b.numel() > 0:
        center_mask[flat_b, 0, flat_iy, flat_ix] = 1.0
        # also force heatmap=1.0 at centers
        heatmap = heatmap.clone()
        heatmap[flat_b, 0, flat_iy, flat_ix] = 1.0

    # Size map: (B, 2, H, W) zero outside centers
    size_map = torch.zeros((B, 2, H, W), device=device)
    if flat_b.numel() > 0:
        size_map[flat_b, 0, flat_iy, flat_ix] = bw[valid_flat]
        size_map[flat_b, 1, flat_iy, flat_ix] = bh[valid_flat]

    return {"heatmap": heatmap, "size": size_map, "center_mask": center_mask}


def _gaussian_radius_batched(w: torch.Tensor, h: torch.Tensor, min_overlap: float = 0.7) -> torch.Tensor:
    """Vectorized CenterNet Gaussian radius — same formula as dataset.py's
    `_gaussian_radius` but operates on whole tensors at once.
    """
    a1 = 1.0
    b1 = (h + w)
    c1 = w * h * (1.0 - min_overlap) / (1.0 + min_overlap)
    r1 = (b1 - torch.sqrt((b1 * b1 - 4 * a1 * c1).clamp(min=0.0))) / (2.0 * a1)

    a2 = 4.0
    b2 = 2.0 * (h + w)
    c2 = (1.0 - min_overlap) * w * h
    r2 = (b2 - torch.sqrt((b2 * b2 - 4 * a2 * c2).clamp(min=0.0))) / (2.0 * a2)

    a3 = 4.0 * min_overlap
    b3 = -2.0 * min_overlap * (h + w)
    c3 = (min_overlap - 1.0) * w * h
    r3 = (b3 + torch.sqrt((b3 * b3 - 4 * a3 * c3).clamp(min=0.0))) / (2.0 * a3)

    return torch.minimum(torch.minimum(r1, r2), r3).clamp(min=0.0)
