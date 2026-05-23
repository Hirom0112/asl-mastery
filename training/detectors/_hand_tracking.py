"""Temporal hand tracking — keep stable hand identity across frames.

The detector is per-frame and returns hands in score order, and the
downstream slot assignment was purely positional (slot 0 = left of frame).
So when a signer's hands cross the midline (constant in ASL), the two hands'
identities swap — the visible "left point jumps right / they cross" glitch,
AND it scrambles per-hand feature trajectories used for recognition.

This tracker matches each frame's detections to the previous frame's hands by
proximity (greedy nearest-centroid) so a hand keeps its slot through a cross,
and EMA-smooths the keypoints to damp the per-frame orientation jumps ("360").

Pure post-processing — no model, no training. Used by both the live demo and
(for v11) trajectory extraction.
"""
from __future__ import annotations


def _centroid(bbox: list[float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5)


class HandTracker:
    """Assigns up to 2 detections to stable slots [0, 1] across frames.

    update(bboxes, kps, frame_w, frame_h) -> (ordered_bboxes, ordered_kps)
    where index 0/1 is a *temporally consistent* hand identity, not a
    per-frame position. Missing slots are simply absent from the output
    (callers already handle 0/1/2 hands).
    """

    def __init__(self, max_match_frac: float = 0.25, ema_alpha: float = 0.6):
        # max_match_frac: max centroid jump (as fraction of frame diagonal)
        #   still considered the "same" hand. Beyond it, treat as a new hand.
        # ema_alpha: weight on the NEW keypoints (1.0 = no smoothing).
        self.max_match_frac = max_match_frac
        self.ema_alpha = ema_alpha
        self._prev_c: dict[int, tuple[float, float]] = {}   # slot -> centroid
        self._prev_kps: dict[int, list] = {}                # slot -> keypoints

    def reset(self) -> None:
        self._prev_c.clear()
        self._prev_kps.clear()

    def update(self, bboxes, kps, frame_w: int, frame_h: int):
        diag = (frame_w ** 2 + frame_h ** 2) ** 0.5
        max_dist = self.max_match_frac * diag
        dets = [{"bbox": list(b), "kps": k, "c": _centroid(b)}
                for b, k in zip(bboxes, kps)]

        assigned: dict[int, dict] = {}
        used: set[int] = set()
        # 1) match existing slots to nearest unused detection within threshold
        for slot, pc in self._prev_c.items():
            best_i, best_d = None, 1e18
            for i, d in enumerate(dets):
                if i in used:
                    continue
                dx, dy = d["c"][0] - pc[0], d["c"][1] - pc[1]
                dist = (dx * dx + dy * dy) ** 0.5
                if dist < best_d:
                    best_d, best_i = dist, i
            if best_i is not None and best_d <= max_dist:
                assigned[slot] = dets[best_i]
                used.add(best_i)
        # 2) leftover detections take any free slot (0 then 1)
        for i, d in enumerate(dets):
            if i in used:
                continue
            for s in (0, 1):
                if s not in assigned:
                    assigned[s] = d
                    used.add(i)
                    break

        # 3) emit in slot order, EMA-smoothing keypoints against prior slot
        out_bboxes, out_kps = [], []
        new_c, new_kps = {}, {}
        a = self.ema_alpha
        for s in (0, 1):
            if s not in assigned:
                continue
            d = assigned[s]
            k = d["kps"]
            pk = self._prev_kps.get(s)
            if pk is not None and len(pk) == len(k):
                k = [[a * kk[0] + (1 - a) * pp[0],
                      a * kk[1] + (1 - a) * pp[1], *list(kk[2:])]
                     for kk, pp in zip(k, pk)]
            out_bboxes.append(d["bbox"])
            out_kps.append(k)
            new_c[s] = _centroid(d["bbox"])
            new_kps[s] = k
        self._prev_c, self._prev_kps = new_c, new_kps
        return out_bboxes, out_kps
