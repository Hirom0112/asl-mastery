"""Pick the most EXPRESSIVE clean clip per sign for the avatar (not the medoid).

The medoid clip is the most "average" take, which underplays reach/motion (e.g.
UNDERSTAND's hand barely left the chest). A teaching avatar wants a clear,
full-amplitude performance. For each sign we score clips on the from-scratch
trajectory data (data/trajectories_top80_v2/<sign>/*.json) by:
  - detection quality (fraction of frames with neck + shoulders + active wrist),
  - reach: how high the active wrist reliably gets (robust 10th-pct of yrel),
  - motion: smoothed wrist path length,
filtering noisy clips (bad detection, implausible reach, jittery).

Output: dataset/expressive_picks_top80.json  { sign: {clip_id, remote_webm, reach, path, det} }
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRAJ = ROOT / "data" / "trajectories_top80_v2"
OUT = ROOT / "dataset" / "expressive_picks_top80.json"


def _med3(a: np.ndarray) -> np.ndarray:
    if len(a) < 3:
        return a
    out = a.copy()
    out[1:-1] = np.median(np.stack([a[:-2], a[1:-1], a[2:]]), axis=0)
    return out


def score_clip(path: Path):
    try:
        frames = json.loads(path.read_text()).get("frames", [])
    except Exception:
        return None
    neck, rsh, lsh, rw, lw = [], [], [], [], []
    for f in frames:
        p = f.get("pose")
        if not p or len(p) < 8:
            continue
        def v(k):
            q = p[k]
            return q if (q and not (q[0] == 0 and q[1] == 0)) else None
        neck.append(v(1)); rsh.append(v(2)); lsh.append(v(3)); rw.append(v(6)); lw.append(v(7))
    n = len(neck)
    if n < 8:
        return None
    det = sum(1 for a, b, c in zip(neck, rsh, lsh) if a and b and c) / n
    if det < 0.8:
        return None
    spans = [np.hypot(b[0] - c[0], b[1] - c[1]) for a, b, c in zip(neck, rsh, lsh) if a and b and c]
    span = float(np.median([s for s in spans if s > 1e-3]) or 1.0)
    if span < 1e-3:
        return None

    def wrist_series(wl):
        ys, xs, ok = [], [], 0
        for i in range(n):
            if neck[i] and wl[i]:
                ys.append((wl[i][1] - neck[i][1]) / span)
                xs.append((wl[i][0] - neck[i][0]) / span)
                ok += 1
        return np.array(xs), np.array(ys), ok / n

    best = None
    for wl in (rw, lw):
        xs, ys, frac = wrist_series(wl)
        if frac < 0.6 or len(ys) < 5:
            continue
        ys, xs = _med3(ys), _med3(xs)
        reach = float(-np.percentile(ys, 10))           # high = reaches up
        if reach > 1.3:                                  # implausible → noise
            continue
        path = float(np.hypot(np.diff(xs), np.diff(ys)).sum())
        if path > 30:                                    # jitter explosion
            continue
        cand = (reach, path)
        if best is None or cand > best[0]:
            best = (cand, reach, path)
    if best is None:
        return None
    (_, reach, pathlen) = best
    return {"reach": round(reach, 3), "path": round(pathlen, 2), "det": round(det, 2)}


def pick_for_dir(d: Path):
    scored = []
    for f in sorted(d.glob("*.json")):
        s = score_clip(f)
        if s:
            s["_key"] = s["reach"] + 0.05 * min(s["path"], 10)  # reach primary, motion 2nd
            s["clip_id"] = f.stem
            scored.append(s)
    if not scored:
        return d.name, None
    best = max(scored, key=lambda s: s["_key"])
    return d.name, {
        "clip_id": best["clip_id"],
        "remote_webm": f"datasets/sem_lex_top80/clips/{best['clip_id']}.webm",
        "reach": best["reach"], "path": best["path"], "det": best["det"],
        "n_scored": len(scored),
    }


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=min(8, mp.cpu_count()))
    args = ap.parse_args()
    dirs = sorted(d for d in TRAJ.iterdir() if d.is_dir())
    if args.only:
        keep = set(args.only); dirs = [d for d in dirs if d.name in keep]

    with mp.Pool(args.workers) as pool:
        results = pool.map(pick_for_dir, dirs)

    picks = {}
    for name, p in results:
        if p is None:
            print(f"  {name}: NO valid clip"); continue
        picks[name] = {k: p[k] for k in ("clip_id", "remote_webm", "reach", "path", "det")}
        print(f"  {name}: {p['clip_id']} reach={p['reach']} path={p['path']} (of {p['n_scored']})")
    OUT.write_text(json.dumps(picks, indent=2, sort_keys=True))
    print(f"\nwrote {len(picks)} expressive picks → {OUT}")


if __name__ == "__main__":
    main()
