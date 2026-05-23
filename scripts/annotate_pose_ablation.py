"""Hand-annotate 4 pose keypoints (nose, neck, r_shoulder, l_shoulder) on
25 val clips at 5 fps for the pose-ablation experiment.

Per Session 17 night discussion: before retraining pose, measure the
downstream effect of perfect pose on matcher P/R. If labeling shows a big
lift, pose IS the lever. If it barely moves, the matcher itself is.

Workflow per clip:
  1. Script opens an OpenCV window showing one frame.
  2. Click 4 points in this order: nose → neck → r_shoulder → l_shoulder.
     The script draws a numbered dot after each click.
  3. Press SPACE to accept and advance to the next sampled frame.
     Press 'r' to reset (re-click the 4 points on the same frame).
     Press 's' to skip the current clip entirely.
     Press 'q' to save progress and quit (resumable).
  4. After the last frame of a clip, the script auto-advances.

Output: data/ablation/annotations.json
  {<sign>/<clip_fname>: {<frame_idx>: {nose: [x,y], neck: [x,y],
                                       r_shoulder: [x,y], l_shoulder: [x,y]}}}

Sampling: target_fps=5. For a clip at 15 fps and 30 frames (2 s), this
yields ~10 annotated frames per clip. For 25 clips × 10 frames × 4 clicks
= 1000 clicks. ~30-50 min of work at 2-3 s/click.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

KP_NAMES = ["nose", "neck", "r_shoulder", "l_shoulder"]
KP_COLORS = [(0, 255, 255), (0, 255, 0), (255, 100, 100), (100, 100, 255)]


def sample_frame_indices(num_frames: int, src_fps: float, target_fps: float) -> list[int]:
    if num_frames <= 0:
        return []
    step = max(1, int(round(src_fps / target_fps)))
    return list(range(0, num_frames, step))


def annotate_clip(
    video_path: Path,
    trajectory: dict,
    target_fps: float,
    window_name: str,
    existing: dict[int, dict] | None = None,
) -> tuple[dict[int, dict], str]:
    """Returns ({traj_frame_idx: {kp: [x,y]}}, action) where action is
    'done' | 'skip' | 'quit'.

    Annotation keys use trajectory frame_idx (0..num_frames-1 in trajectory
    sample space, NOT source video frames). Display seeks to the matching
    source frame via the source-fps / trajectory-fps stride.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  cannot open {video_path}")
        return ({}, "skip")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    traj_fps = float(trajectory.get("fps", 15.0))
    traj_num_frames = int(trajectory.get("num_frames", len(trajectory.get("frames", []))))
    src_stride = max(1, int(round(src_fps / traj_fps)))
    sample_idxs = sample_frame_indices(traj_num_frames, traj_fps, target_fps)
    print(f"  {video_path.name}: {traj_num_frames} traj-frames @ {traj_fps:.1f} fps "
          f"(src {src_fps:.1f} fps, stride {src_stride}) -> "
          f"sampling {len(sample_idxs)} frames @ {target_fps} fps")

    out = dict(existing or {})

    # Index pose-output frames by frame_idx for showing current v0 estimates
    pose_by_idx = {f["frame_idx"]: f.get("pose") for f in trajectory.get("frames", [])}

    state = {"clicks": [], "frame_bgr": None, "frame_idx": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(state["clicks"]) < 4:
            state["clicks"].append((x, y))

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    for sample_pos, traj_idx in enumerate(sample_idxs):
        if traj_idx in out and len(out[traj_idx]) == 4:
            continue  # already labeled (resume)
        source_frame = traj_idx * src_stride
        cap.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
        ok, frame = cap.read()
        if not ok:
            continue
        state["frame_bgr"] = frame
        state["frame_idx"] = traj_idx
        state["clicks"] = []

        while True:
            display = frame.copy()
            # show v0 pose dots as faint reference
            pose = pose_by_idx.get(traj_idx) or []
            for i in range(min(4, len(pose))):
                px, py = pose[i]
                if px and py:
                    cv2.circle(display, (int(px), int(py)), 4, (60, 60, 60), 1)
            # show clicks
            for i, (cx, cy) in enumerate(state["clicks"]):
                cv2.circle(display, (cx, cy), 6, KP_COLORS[i], -1)
                cv2.putText(display, f"{i+1}:{KP_NAMES[i]}", (cx + 8, cy + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, KP_COLORS[i], 1)
            next_kp = KP_NAMES[len(state["clicks"])] if len(state["clicks"]) < 4 else "(done — SPACE)"
            banner = (f"clip {video_path.name}  traj_frame {traj_idx} "
                      f"({sample_pos + 1}/{len(sample_idxs)})  next: {next_kp}")
            cv2.putText(display, banner, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 2)
            cv2.putText(display, "SPACE=next  r=reset  s=skip clip  q=save+quit",
                        (8, display.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            cv2.imshow(window_name, display)
            key = cv2.waitKey(20) & 0xFF
            if key == ord("r"):
                state["clicks"] = []
            elif key == ord("s"):
                cap.release()
                return (out, "skip")
            elif key == ord("q"):
                cap.release()
                return (out, "quit")
            elif key == ord(" ") and len(state["clicks"]) == 4:
                out[traj_idx] = {
                    KP_NAMES[i]: [float(state["clicks"][i][0]),
                                  float(state["clicks"][i][1])]
                    for i in range(4)
                }
                break

    cap.release()
    return (out, "done")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selected", type=Path,
                    default=Path("data/ablation/selected_clips.json"))
    ap.add_argument("--val-dir", type=Path,
                    default=Path("data/trajectories_v6_val"))
    ap.add_argument("--clips-dir", type=Path,
                    default=Path("data/ablation/clips"))
    ap.add_argument("--out", type=Path,
                    default=Path("data/ablation/annotations.json"))
    ap.add_argument("--target-fps", type=float, default=5.0)
    args = ap.parse_args()

    selected = json.loads(args.selected.read_text())
    annotations = {}
    if args.out.exists():
        annotations = json.loads(args.out.read_text())
        print(f"resumed: {len(annotations)} clips already (partially) annotated")

    for i, (sign, fname) in enumerate(selected):
        clip_key = f"{sign}/{fname}"
        traj_path = args.val_dir / sign / fname
        if not traj_path.exists():
            print(f"[{i+1}/{len(selected)}] {clip_key}: trajectory missing, skip")
            continue
        traj = json.loads(traj_path.read_text())
        video_name = Path(traj["clip_path"]).name
        video_path = args.clips_dir / video_name
        if not video_path.exists():
            print(f"[{i+1}/{len(selected)}] {clip_key}: video missing ({video_path}), skip")
            continue
        print(f"[{i+1}/{len(selected)}] {clip_key}")
        existing = {int(k): v for k, v in annotations.get(clip_key, {}).items()}
        result, action = annotate_clip(video_path, traj, args.target_fps,
                                       window_name="pose-ablation", existing=existing)
        annotations[clip_key] = {str(k): v for k, v in result.items()}
        # Persist after each clip
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(annotations, indent=2))
        print(f"  saved -> {args.out}  ({sum(len(v) for v in annotations.values())} frames labeled total)")
        if action == "quit":
            print("quit pressed — progress saved, exiting.")
            cv2.destroyAllWindows()
            return 0

    cv2.destroyAllWindows()
    print(f"\nDone. {sum(len(v) for v in annotations.values())} frames across "
          f"{len(annotations)} clips.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
