"""Batched GPU augmentation for landmark training (keypoint-aware).

Moves the per-sample CPU augmentation (the bottleneck once data is packed)
onto the GPU as one batched op: a single affine resample (rotation + scale +
translate + optional hflip) via affine_grid/grid_sample, plus vectorized
photometric jitter. Keypoints are transformed by the SAME forward affine so
image and coords stay in correspondence (verified in __main__).

Contract matches landmarks_augment.default_train_augment but BATCHED:
  imgs:   (B,3,H,W) float [0,1] on device
  coords: (B,K,2)   in [0,1]
  vis:    (B,K)
Returns augmented (imgs, coords, vis). Out-of-frame keypoints → vis 0.
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def _photometric(img: torch.Tensor, lo: float = 0.7, hi: float = 1.3) -> torch.Tensor:
    B = img.shape[0]
    dev = img.device
    def f():
        return torch.rand(B, 1, 1, 1, device=dev) * (hi - lo) + lo
    img = img * f()                                   # brightness
    m = img.mean(dim=(2, 3), keepdim=True)
    img = (img - m) * f() + m                         # contrast
    g = img.mean(dim=1, keepdim=True)
    img = (img - g) * f() + g                         # saturation
    return img.clamp(0, 1)


def gpu_augment_batch(imgs: torch.Tensor, coords: torch.Tensor, vis: torch.Tensor,
                      max_deg: float = 25.0, scale_range=(0.65, 1.40),
                      trans_frac: float = 0.08, hflip_p: float = 0.5):
    B, C, H, W = imgs.shape
    dev = imgs.device
    ang = (torch.rand(B, device=dev) * 2 - 1) * math.radians(max_deg)
    s = torch.rand(B, device=dev) * (scale_range[1] - scale_range[0]) + scale_range[0]
    tx = (torch.rand(B, device=dev) * 2 - 1) * trans_frac * 2.0   # normalized units
    ty = (torch.rand(B, device=dev) * 2 - 1) * trans_frac * 2.0
    flip = torch.where(torch.rand(B, device=dev) < hflip_p,
                       torch.tensor(-1.0, device=dev), torch.tensor(1.0, device=dev))

    cos, sin = torch.cos(ang), torch.sin(ang)
    # FORWARD affine on a normalized point n∈[-1,1]: n' = A @ n + t,  A = R(ang) @ diag(flip*s, s)
    Sx, Sy = flip * s, s
    A00, A01 = cos * Sx, -sin * Sy
    A10, A11 = sin * Sx, cos * Sy
    # grid_sample needs OUTPUT→INPUT mapping = inverse forward: n = A^{-1}(n' - t)
    det = A00 * A11 - A01 * A10
    iA00, iA01 = A11 / det, -A01 / det
    iA10, iA11 = -A10 / det, A00 / det
    bx = -(iA00 * tx + iA01 * ty)
    by = -(iA10 * tx + iA11 * ty)
    theta = torch.stack([
        torch.stack([iA00, iA01, bx], dim=1),
        torch.stack([iA10, iA11, by], dim=1)], dim=1)            # (B,2,3)
    grid = F.affine_grid(theta, imgs.shape, align_corners=False)
    out = F.grid_sample(imgs, grid, align_corners=False, padding_mode="zeros")

    # transform coords by the FORWARD affine (image content moves by A)
    n = coords * 2 - 1
    nx, ny = n[..., 0], n[..., 1]
    fx = A00.unsqueeze(1) * nx + A01.unsqueeze(1) * ny + tx.unsqueeze(1)
    fy = A10.unsqueeze(1) * nx + A11.unsqueeze(1) * ny + ty.unsqueeze(1)
    new = torch.stack([(fx + 1) / 2, (fy + 1) / 2], dim=-1)
    inb = ((new[..., 0] >= 0) & (new[..., 0] <= 1) &
           (new[..., 1] >= 0) & (new[..., 1] <= 1)).float()
    return _photometric(out), new, vis * inb


if __name__ == "__main__":
    # Correctness test: a bright dot drawn at each keypoint must still sit at
    # the TRANSFORMED keypoint after augmentation (image & coords in sync).
    torch.manual_seed(0)
    B, S = 64, 224
    coords = torch.rand(B, 1, 2) * 0.6 + 0.2          # keypoints in [0.2,0.8]
    vis = torch.ones(B, 1)
    imgs = torch.zeros(B, 3, S, S)
    px = (coords[:, 0, 0] * (S - 1)).long()
    py = (coords[:, 0, 1] * (S - 1)).long()
    for b in range(B):
        imgs[b, :, py[b], px[b]] = 1.0                 # bright dot at kp
    # disable photometric for the geometric check
    import training.detectors.landmarks_gpu_augment as M
    _orig = M._photometric; M._photometric = lambda x: x
    out, nc, nv = gpu_augment_batch(imgs, coords.clone(), vis.clone())
    M._photometric = _orig
    errs = []
    for b in range(B):
        if nv[b, 0] < 0.5:
            continue                                   # rotated out of frame
        nz = (out[b].sum(0) > 0.3).nonzero()           # where the dot landed
        if nz.numel() == 0:
            continue
        cy, cx = nz.float().mean(0)                     # blurred by resample
        ex = abs(cx.item() - nc[b, 0, 0].item() * (S - 1))
        ey = abs(cy.item() - nc[b, 0, 1].item() * (S - 1))
        errs.append((ex ** 2 + ey ** 2) ** 0.5)
    import statistics
    print(f"checked {len(errs)} kpts | mean px err = {statistics.mean(errs):.2f} "
          f"| max = {max(errs):.2f}")
    assert statistics.mean(errs) < 2.5, "coords/image out of sync — DO NOT train"
    print("GPU AUGMENT COORD-IMAGE SYNC OK")
