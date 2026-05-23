"""Shared helpers for external-dataset loaders."""

from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(os.environ.get("ASL_REPO_ROOT", str(Path(__file__).resolve().parents[3])))
EXTERNAL_ROOT = Path(os.environ.get("ASL_EXTERNAL_ROOT", str(REPO_ROOT / "data" / "external")))


def repo_rel(p: Path) -> str:
    """Return p as a repo-root-relative path string."""
    p = p.resolve()
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def bbox_from_keypoints(
    kps: list[tuple[float, float, float]], pad_frac: float = 0.15
) -> list[float] | None:
    """Tight axis-aligned bbox around visible keypoints, padded by pad_frac
    of the diagonal in each direction. Returns None if fewer than 4 visible
    keypoints (a bbox from 3 or fewer is too noisy)."""
    visible = [(x, y) for x, y, v in kps if v > 0]
    if len(visible) < 4:
        return None
    xs = [p[0] for p in visible]
    ys = [p[1] for p in visible]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    w, h = x1 - x0, y1 - y0
    pad = pad_frac * max(w, h)
    return [x0 - pad, y0 - pad, x1 + pad, y1 + pad]
