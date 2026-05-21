"""Gomez-Donoso Multiview Hand Pose → project-internal manifest.

The dataset ships per-frame:
  - 4 JPGs (one per camera view)
  - per-frame 3D joint TXT (21 lines, "x y z" in meters from Leap Motion)
  - per-camera calibration pickles (rvec.pkl, tvec.pkl) for projecting 3D → 2D

For our use we treat each view-image as an independent training sample
(the model never knows which is which), projecting the shared 3D joints
through that camera's rvec/tvec/intrinsics.

Layout inside the V1 release (subject to verification post-extract):
  multiview_hand_pose_dataset_release/
    annotated_frames/   data_*/<seq>/<frame>_joints.txt
                                       _webcam_1.jpg .._webcam_4.jpg
    calibrations/       webcam_<N>/rvec.pkl, tvec.pkl, intrinsics.txt
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from training.detectors.external_loaders._util import (
    EXTERNAL_ROOT,
    bbox_from_keypoints,
    repo_rel,
)

DATASET_DIR = EXTERNAL_ROOT / "multiview_hand_pose"


def _load_calibrations(calib_root: Path) -> dict[int, dict]:
    import pickle
    calibs = {}
    for cam_dir in sorted(calib_root.glob("webcam_*")):
        try:
            cam_id = int(cam_dir.name.split("_")[1])
        except (ValueError, IndexError):
            continue
        try:
            rvec = pickle.loads((cam_dir / "rvec.pkl").read_bytes())
            tvec = pickle.loads((cam_dir / "tvec.pkl").read_bytes())
        except Exception as e:
            print(f"[multiview] calib read failed for {cam_dir}: {e}")
            continue
        intr_path = cam_dir / "intrinsics.txt"
        K = None
        if intr_path.exists():
            try:
                K = np.loadtxt(str(intr_path)).reshape(3, 3)
            except Exception:
                K = None
        calibs[cam_id] = {"rvec": np.asarray(rvec).flatten(), "tvec": np.asarray(tvec).flatten(), "K": K}
    return calibs


def _project_3d_to_2d(joints_3d_m: np.ndarray, cam: dict) -> np.ndarray | None:
    """Rodrigues + camera projection. Returns (21, 2) pixel coords or None."""
    try:
        import cv2
    except ImportError:
        return None
    if cam.get("K") is None:
        return None
    rvec = cam["rvec"].astype(np.float32).reshape(3, 1)
    tvec = cam["tvec"].astype(np.float32).reshape(3, 1)
    K = cam["K"].astype(np.float32)
    dist = np.zeros(5, dtype=np.float32)
    pts, _ = cv2.projectPoints(joints_3d_m.astype(np.float32), rvec, tvec, K, dist)
    return pts.reshape(-1, 2)


def load_hand_keypoints() -> list[dict]:
    if not DATASET_DIR.exists():
        print(f"[multiview] dataset dir missing: {DATASET_DIR} — skipping")
        return []
    calib_root = DATASET_DIR / "calibrations"
    frames_root = DATASET_DIR / "annotated_frames"
    if not frames_root.exists():
        # The release may unpack under a wrapping subdir
        candidates = list(DATASET_DIR.glob("*/annotated_frames"))
        if candidates:
            frames_root = candidates[0]
            calib_root = frames_root.parent / "calibrations"
        else:
            print(f"[multiview] annotated_frames not found under {DATASET_DIR} — skipping")
            return []

    calibs = _load_calibrations(calib_root)
    if not calibs:
        print("[multiview] no calibrations loaded — skipping")
        return []

    items: list[dict] = []
    for joints_path in frames_root.rglob("*_joints.txt"):
        try:
            joints = np.loadtxt(str(joints_path))
        except Exception:
            continue
        if joints.shape != (21, 3):
            continue

        frame_stem = joints_path.stem.replace("_joints", "")
        seq_dir = joints_path.parent
        for cam_id, cam in calibs.items():
            img_path = seq_dir / f"{frame_stem}_webcam_{cam_id}.jpg"
            if not img_path.exists():
                continue
            pts2d = _project_3d_to_2d(joints, cam)
            if pts2d is None:
                continue
            kps = [(float(x), float(y), 2.0) for x, y in pts2d]
            bbox = bbox_from_keypoints(kps)
            if bbox is None:
                continue
            items.append(
                {
                    "image_path": repo_rel(img_path),
                    "width": 0,  # filled at load time via PIL if needed
                    "height": 0,
                    "hands": [
                        {
                            "bbox": bbox,
                            "keypoints": kps,
                            "source": f"multiview_hand_pose_cam{cam_id}",
                        }
                    ],
                }
            )
    print(f"[multiview] emitted {len(items)} hand_keypoint items (×4 views per frame)")
    return items
