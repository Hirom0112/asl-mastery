"""Pick a representative (medoid) source clip per sign for reference videos.

For each requested sign we load every per-clip trajectory from
data/trajectories_top80_v2/<sign>/*.json (the same files fit_templates reads),
resample to T frames, and select the medoid clip — the real clip with the
smallest summed NaN-tolerant MSE to all other clips. Using the medoid means the
reference video shown to the learner is the same canonical motion the avatar
plays back (the avatar template is the medoid too), rather than an arbitrary or
outlier clip.

Each trajectory JSON carries a `clip_path` like
  /data/datasets/sem_lex_top80/clips/<clip_id>.webm
so the medoid's clip_id maps straight to a source .webm on the Modal volume.

Output: dataset/reference_picks_top80.json
  { "<sign>": {"clip_id": "...", "remote_webm": "datasets/sem_lex_top80/clips/<clip_id>.webm", "n_clips": N} }
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training" / "detectors"))
from fit_templates import _trajectory_from_file, _resample_trajectory, _clip_distance  # noqa: E402


def _medoid_index(clips: np.ndarray) -> int:
    n = clips.shape[0]
    if n == 1:
        return 0
    total = np.zeros(n, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = _clip_distance(clips[i], clips[j])
            total[i] += d
            total[j] += d
    return int(np.argmin(total))


def pick_for_sign(sign_dir: Path, T: int) -> dict | None:
    files = sorted(sign_dir.glob("*.json"))
    clips: list[np.ndarray] = []
    clip_ids: list[str] = []
    for p in files:
        r = _trajectory_from_file(p)
        if r is None:
            continue
        traj, _ = r
        clips.append(_resample_trajectory(traj, T))
        clip_ids.append(p.stem)
    if not clips:
        return None
    arr = np.stack(clips, axis=0)
    idx = _medoid_index(arr)
    clip_id = clip_ids[idx]
    return {
        "clip_id": clip_id,
        "remote_webm": f"datasets/sem_lex_top80/clips/{clip_id}.webm",
        "n_clips": len(clips),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trajectories-dir", type=Path,
                    default=ROOT / "data" / "trajectories_top80_v2")
    ap.add_argument("--out", type=Path, default=ROOT / "dataset" / "reference_picks_top80.json")
    ap.add_argument("--time-steps", type=int, default=32)
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict to these sign ids (default: all sign dirs)")
    args = ap.parse_args()

    sign_dirs = sorted(d for d in args.trajectories_dir.iterdir() if d.is_dir())
    if args.only:
        keep = set(args.only)
        sign_dirs = [d for d in sign_dirs if d.name in keep]

    picks: dict[str, dict] = {}
    for d in sign_dirs:
        pick = pick_for_sign(d, args.time_steps)
        if pick is None:
            print(f"  {d.name}: NO usable clips")
            continue
        picks[d.name] = pick
        print(f"  {d.name}: medoid={pick['clip_id']} (of {pick['n_clips']})")

    args.out.write_text(json.dumps(picks, indent=2, sort_keys=True))
    print(f"\nwrote {len(picks)} picks → {args.out}")


if __name__ == "__main__":
    main()
