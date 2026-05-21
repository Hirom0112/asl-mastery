"""Extract sample frames from cleaned MP4 clips for hand-bbox labeling.

Per docs/VOCABULARY_TRAINER_ROADMAP.md Phase 1 Slice 1.1:
- Walk a directory of cleaned MP4 clips (output of training/data/clean.py
  or the existing dataset/clean/ tree).
- Sample 1 frame per second per clip (default) using ffmpeg.
- Write JPEGs to data/labeled_frames/source_pool/<task>/ with a
  filename encoding the source clip + frame index, so a labeler can
  retrace any frame to its origin.
- Emit a sidecar manifest data/labeled_frames/source_pool/<task>/index.json
  that lists every extracted frame with its source-clip path and
  timestamp. This is the input pool for the labeling tool (CVAT or
  LabelStudio) — the labeler picks frames from this pool, labels them,
  and the labeling tool's export gets normalized to the trainer's
  manifest schema (see training/detectors/dataset.py docstring).

Usage:
    python scripts/extract_frames.py \\
        --clips-dir dataset/clean/v1 \\
        --out-dir data/labeled_frames/source_pool/hand_bbox \\
        --fps 1 \\
        --max-frames-per-clip 5

Requires ffmpeg on PATH.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def _check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        print("error: ffmpeg not found on PATH", file=sys.stderr)
        sys.exit(2)


def _probe_duration_seconds(clip_path: Path) -> float | None:
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(clip_path),
            ],
            text=True,
        ).strip()
        return float(out) if out else None
    except (subprocess.CalledProcessError, ValueError):
        return None


def _extract_one(
    clip_path: Path,
    out_dir: Path,
    fps: float,
    max_frames: int | None,
) -> list[dict]:
    stem = clip_path.stem
    pattern = out_dir / f"{stem}_f%04d.jpg"
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-i", str(clip_path),
           "-vf", f"fps={fps}",
           "-q:v", "3",  # mid-quality JPEG, ~80%
           str(pattern)]
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print(f"  ffmpeg failed for {clip_path}: {e}", file=sys.stderr)
        return []

    frames = sorted(out_dir.glob(f"{stem}_f*.jpg"))
    if max_frames is not None and len(frames) > max_frames:
        # Keep evenly spaced subset; delete the rest.
        keep_indices = set(
            int(round(i * (len(frames) - 1) / (max_frames - 1)))
            for i in range(max_frames)
        )
        for i, f in enumerate(frames):
            if i not in keep_indices:
                f.unlink(missing_ok=True)
        frames = sorted(out_dir.glob(f"{stem}_f*.jpg"))

    duration = _probe_duration_seconds(clip_path)
    records = []
    for i, f in enumerate(frames, start=1):
        # Approximate timestamp from sample index and fps.
        ts = (i - 1) / fps if fps > 0 else None
        records.append(
            {
                "frame_path": str(f),
                "source_clip": str(clip_path),
                "approx_timestamp_s": ts,
                "clip_duration_s": duration,
            }
        )
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips-dir", type=Path, required=True,
                    help="Directory containing cleaned .mp4 clips (recursive).")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Where to write extracted JPEGs + index.json.")
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--max-frames-per-clip", type=int, default=None,
                    help="If set, cap the number of frames per clip after extraction.")
    ap.add_argument("--glob", default="*.mp4")
    args = ap.parse_args()

    _check_ffmpeg()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    clips = sorted(args.clips_dir.rglob(args.glob))
    if not clips:
        print(f"no clips matched {args.clips_dir}/**/{args.glob}", file=sys.stderr)
        sys.exit(1)
    print(f"found {len(clips)} clips")

    all_records: list[dict] = []
    for i, clip in enumerate(clips, start=1):
        print(f"[{i}/{len(clips)}] {clip}")
        all_records.extend(
            _extract_one(clip, args.out_dir, args.fps, args.max_frames_per_clip)
        )

    index_path = args.out_dir / "index.json"
    index_path.write_text(
        json.dumps(
            {
                "version": 1,
                "task": "frame_extraction",
                "fps": args.fps,
                "clips_dir": str(args.clips_dir),
                "total_frames": len(all_records),
                "frames": all_records,
            },
            indent=2,
        )
    )
    print(f"wrote {len(all_records)} frames + {index_path}")


if __name__ == "__main__":
    main()
