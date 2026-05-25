"""Temporal hand tracking — keep stable hand identity across frames.

The detector is per-frame and returns hands in score order, and the
downstream slot assignment was purely positional (slot 0 = left of frame).
So when a signer's hands cross the midline (constant in ASL), the two hands'
identities swap — the visible "left point jumps right / they cross" glitch,
AND it scrambles per-hand feature trajectories used for recognition.

This tracker keeps a hand in its slot through a cross by matching on a
**constant-velocity prediction** (where a hand is *heading*, not where it
*was*) and a **globally-optimal 2-hand assignment** (pick the slot↔detection
pairing with the lowest total cost, not greedy first-come). It also **coasts**
a momentarily-missed slot forward for a few frames so identity survives the
brief mutual occlusion at the crossing point, and EMA-smooths keypoints to
damp per-frame orientation jumps ("360").

Pure post-processing — no model, no training. Used by both the live demo and
(for v11) trajectory extraction.
"""
from __future__ import annotations

from itertools import permutations


def _centroid(bbox: list[float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5)


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


class HandTracker:
    """Assigns up to 2 detections to stable slots [0, 1] across frames.

    update(bboxes, kps, frame_w, frame_h) -> (ordered_bboxes, ordered_kps)
    where index 0/1 is a *temporally consistent* hand identity, not a
    per-frame position. Only slots with a real detection THIS frame are
    emitted (callers already handle 0/1/2 hands); coasted-but-unseen slots
    are kept internally for re-acquisition but not emitted.
    """

    def __init__(self, max_match_frac: float = 0.25, ema_alpha: float = 0.6,
                 vel_alpha: float = 0.5, vel_decay: float = 0.8,
                 max_coast: int = 8):
        # max_match_frac: max centroid jump (as fraction of frame diagonal)
        #   from a slot's PREDICTED position still considered the "same" hand.
        # ema_alpha: weight on the NEW keypoints (1.0 = no smoothing).
        # vel_alpha: EMA weight on the new per-frame velocity estimate.
        # vel_decay: per-frame velocity damping while coasting (no detection).
        # max_coast: frames a missed slot is kept alive (advanced by velocity)
        #   before its identity is dropped. Lets identity survive the brief
        #   occlusion when hands cross.
        self.max_match_frac = max_match_frac
        self.ema_alpha = ema_alpha
        self.vel_alpha = vel_alpha
        self.vel_decay = vel_decay
        self.max_coast = max_coast
        # slot -> {"c": centroid, "v": velocity, "kps": keypoints, "miss": n}
        self._slots: dict[int, dict] = {}

    def reset(self) -> None:
        self._slots.clear()

    def _predict(self, slot: int, max_dist: float) -> tuple[float, float]:
        s = self._slots[slot]
        vx, vy = s["v"]
        # clamp the prediction step so a glitchy velocity can't fling the anchor
        mag = (vx * vx + vy * vy) ** 0.5
        if mag > max_dist and mag > 1e-6:
            scale = max_dist / mag
            vx, vy = vx * scale, vy * scale
        return (s["c"][0] + vx, s["c"][1] + vy)

    def update(self, bboxes, kps, frame_w: int, frame_h: int):
        diag = (frame_w ** 2 + frame_h ** 2) ** 0.5
        max_dist = self.max_match_frac * diag
        dets = [{"bbox": list(b), "kps": k, "c": _centroid(b)}
                for b, k in zip(bboxes, kps)]

        # 1) globally-optimal assignment of existing slots -> detections,
        #    matching against each slot's PREDICTED (velocity-led) centroid.
        slots = list(self._slots.keys())
        preds = {s: self._predict(s, max_dist) for s in slots}
        slot_for_det: dict[int, int] = {}   # det index -> slot
        det_for_slot: dict[int, int] = {}   # slot -> det index
        if slots and dets:
            best_cost, best_pair = None, None
            # enumerate slot->det pairings (<=2 each, so trivially small) and
            # keep the lowest total cost; ungated pairs are dropped afterward.
            for det_perm in permutations(range(len(dets)), min(len(slots), len(dets))):
                cost, pairs = 0.0, []
                for si, di in enumerate(det_perm):
                    slot = slots[si]
                    d = _dist(preds[slot], dets[di]["c"])
                    cost += d
                    pairs.append((slot, di, d))
                if best_cost is None or cost < best_cost:
                    best_cost, best_pair = cost, pairs
            for slot, di, d in best_pair:
                if d <= max_dist:               # gate each pairing individually
                    det_for_slot[slot] = di
                    slot_for_det[di] = slot

        # 2) leftover detections take a free slot (0 then 1), preferring a slot
        #    not currently occupied by a live OR coasting identity.
        occupied = set(det_for_slot.keys()) | set(self._slots.keys())
        for i, _ in enumerate(dets):
            if i in slot_for_det:
                continue
            free = next((s for s in (0, 1) if s not in occupied), None)
            if free is None:                    # >2 hands: reuse a coasting slot
                free = next((s for s in (0, 1) if s not in det_for_slot), 0)
            det_for_slot[free] = i
            slot_for_det[i] = free
            occupied.add(free)

        # 3) build next state + emit only slots that got a real detection.
        a = self.ema_alpha
        new_slots: dict[int, dict] = {}
        for slot in (0, 1):
            if slot in det_for_slot:
                d = dets[det_for_slot[slot]]
                new_c = d["c"]
                k = d["kps"]
                prev = self._slots.get(slot)
                if prev is not None:
                    pk = prev["kps"]
                    if pk is not None and len(pk) == len(k):
                        k = [[a * kk[0] + (1 - a) * pp[0],
                              a * kk[1] + (1 - a) * pp[1], *list(kk[2:])]
                             for kk, pp in zip(k, pk)]
                    raw_v = (new_c[0] - prev["c"][0], new_c[1] - prev["c"][1])
                    va = self.vel_alpha
                    v = (va * raw_v[0] + (1 - va) * prev["v"][0],
                         va * raw_v[1] + (1 - va) * prev["v"][1])
                else:
                    v = (0.0, 0.0)
                new_slots[slot] = {"c": new_c, "v": v, "kps": k, "miss": 0}
            elif slot in self._slots:
                # coast: advance by velocity, damp it, count the miss; drop if stale
                prev = self._slots[slot]
                if prev["miss"] + 1 <= self.max_coast:
                    new_slots[slot] = {
                        "c": self._predict(slot, max_dist),
                        "v": (prev["v"][0] * self.vel_decay,
                              prev["v"][1] * self.vel_decay),
                        "kps": prev["kps"], "miss": prev["miss"] + 1,
                    }

        self._slots = new_slots
        out_bboxes, out_kps = [], []
        for slot in (0, 1):
            if slot in det_for_slot:
                d = dets[det_for_slot[slot]]
                out_bboxes.append(d["bbox"])
                out_kps.append(new_slots[slot]["kps"])
        return out_bboxes, out_kps
