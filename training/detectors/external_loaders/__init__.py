"""Per-external-dataset annotation normalizers.

Each loader reads its dataset's native annotation format and emits items
in the project-internal manifest schema:

  - hand_bbox: { image_path, width, height, bboxes: [[x0,y0,x1,y1], ...] }
  - hand_keypoints: { image_path, width, height, hands: [{
        bbox: [x0,y0,x1,y1],
        keypoints: [[x,y,visibility], ...]  # length 21, canonical topology
    }, ...] }
  - face_bbox: { image_path, width, height, bboxes: [[x0,y0,x1,y1], ...] }
  - pose_keypoints: { image_path, width, height, persons: [{
        bbox: [x0,y0,x1,y1],
        keypoints: [[x,y,visibility], ...]  # length 8 in our topology
    }, ...] }

The canonical 21-hand-keypoint topology (matches FreiHAND, CMU, OpenPose,
MediaPipe convention; the universal mapping in the field):

  0: wrist
  1-4:   thumb  (CMC, MCP, IP, tip)
  5-8:   index  (MCP, PIP, DIP, tip)
  9-12:  middle (MCP, PIP, DIP, tip)
  13-16: ring   (MCP, PIP, DIP, tip)
  17-20: pinky  (MCP, PIP, DIP, tip)

The canonical 8-pose-keypoint topology (per ADR 0011 Phase 3 Slice 3.1):

  0: nose, 1: neck, 2: r_shoulder, 3: l_shoulder,
  4: r_elbow, 5: l_elbow, 6: r_wrist, 7: l_wrist.

All loaders must be importable without their target dataset on disk —
they may return an empty list with a warning if the dataset directory
is missing. This lets normalize_external.py run partial conversions.
"""

from __future__ import annotations
