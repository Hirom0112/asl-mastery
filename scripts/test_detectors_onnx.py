"""Test the shipped from-scratch detectors in a terminal — no checkpoints needed.

Runs the exact ONNX models the browser ships (public/models/) on a webcam or a
single image, and draws the hand boxes, 21 hand keypoints, and 8 pose keypoints.
This is the zero-setup way to see the detectors work: it needs only the
committed ONNX files plus `onnxruntime`, `opencv-python`, and `numpy`.

Usage:
  python -m scripts.test_detectors_onnx                      # webcam 0
  python -m scripts.test_detectors_onnx --camera 1           # a different camera
  python -m scripts.test_detectors_onnx --image path/to.jpg  # annotate one image → out.png

Hotkeys (webcam): q quit · m toggle mirror.

The decode mirrors lib/inference/keypoints.ts (the browser pipeline) so the
overlays match what the app sees. No pretrained models are used — every weight
is from this project (see docs/model_cards/).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

MODELS_DIR = Path(__file__).resolve().parent.parent / "public" / "models"

DET_SIZE, DET_STRIDE = 320, 4
LM_SIZE = 224
POSE_SIZE = 256
DET_THRESHOLD = 0.02          # detector is undercalibrated; very low bar (matches extraction)
SECOND_HAND_THRESHOLD = 0.15  # only keep a 2nd hand if it clears this
MIN_BBOX_PX, MAX_ASPECT = 8, 3.5
LM_PAD_FRAC = 0.2

# 21-keypoint hand skeleton (wrist=0, finger MCP→tip chains).
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20), # pinky
]


def load_models() -> dict[str, ort.InferenceSession]:
    def sess(name: str) -> ort.InferenceSession:
        p = MODELS_DIR / name
        if not p.exists():
            raise SystemExit(f"missing model: {p} (run from the repo root)")
        return ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])

    return {
        "detector": sess("hand_detector_v2.onnx"),
        "landmarks": sess("hand_landmarks_v2_combined.onnx"),
        "pose": sess("pose_detector_v0.onnx"),
    }


def region_to_chw(img: np.ndarray, x0: int, y0: int, w: int, h: int, size: int) -> np.ndarray:
    """Crop (x0,y0,w,h) from a BGR frame → (1,3,size,size) float32 RGB in [0,1]."""
    crop = img[y0 : y0 + h, x0 : x0 + w]
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.transpose(rgb, (2, 0, 1))[None]  # (1,3,H,W)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def detect_hands(m: dict, frame: np.ndarray) -> list[tuple[float, list[int]]]:
    h, w = frame.shape[:2]
    inp = region_to_chw(frame, 0, 0, w, h, DET_SIZE)
    out = m["detector"].run(None, {m["detector"].get_inputs()[0].name: inp})
    # Identify heatmap (1,1,G,G) vs size (1,2,G,G) by channel count.
    hm, sz = (out[0], out[1]) if out[0].shape[1] == 1 else (out[1], out[0])
    G = hm.shape[-1]
    prob = _sigmoid(hm[0, 0])               # (G,G)
    sx, sy = w / DET_SIZE, h / DET_SIZE
    cand = []
    for cy in range(G):
        for cx in range(G):
            p = prob[cy, cx]
            if p < DET_THRESHOLD:
                continue
            y0, y1 = max(0, cy - 1), min(G, cy + 2)
            x0, x1 = max(0, cx - 1), min(G, cx + 2)
            if p < prob[y0:y1, x0:x1].max():
                continue  # not a 3×3 peak
            bw, bh = sz[0, 0, cy, cx], sz[0, 1, cy, cx]
            xc, yc = (cx + 0.5) * DET_STRIDE, (cy + 0.5) * DET_STRIDE
            box = [
                int((xc - bw / 2) * sx), int((yc - bh / 2) * sy),
                int((xc + bw / 2) * sx), int((yc + bh / 2) * sy),
            ]
            cand.append((float(p), box))
    cand.sort(key=lambda c: -c[0])
    valid = []
    for s, b in cand[:2]:
        bw, bh = b[2] - b[0], b[3] - b[1]
        if bw < MIN_BBOX_PX or bh < MIN_BBOX_PX:
            continue
        if max(bw, bh) / max(min(bw, bh), 1) > MAX_ASPECT:
            continue
        valid.append((s, b))
    kept = []
    if valid:
        kept.append(valid[0])
        if len(valid) > 1 and valid[1][0] >= SECOND_HAND_THRESHOLD:
            kept.append(valid[1])
    return kept


def landmarks_for(m: dict, frame: np.ndarray, box: list[int]) -> np.ndarray:
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = box
    pad = LM_PAD_FRAC * max(x1 - x0, y1 - y0)
    cx0, cy0 = max(0, int(x0 - pad)), max(0, int(y0 - pad))
    cx1, cy1 = min(w, int(x1 + pad)), min(h, int(y1 + pad))
    if cx1 <= cx0 or cy1 <= cy0:
        cx0, cy0, cx1, cy1 = 0, 0, w, h
    cw, ch = cx1 - cx0, cy1 - cy0
    inp = region_to_chw(frame, cx0, cy0, cw, ch, LM_SIZE)
    coords = m["landmarks"].run(None, {m["landmarks"].get_inputs()[0].name: inp})[0].reshape(-1, 2)
    return np.stack([cx0 + coords[:, 0] * cw, cy0 + coords[:, 1] * ch], axis=1)


def pose_for(m: dict, frame: np.ndarray, boxes: list[list[int]]) -> np.ndarray | None:
    if not boxes:
        return None
    h, w = frame.shape[:2]
    x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes); y1 = max(b[3] for b in boxes)
    hw = max(b[2] - b[0] for b in boxes); hh = max(b[3] - b[1] for b in boxes)
    x0 -= 1.5 * hw; x1 += 1.5 * hw; y0 -= 3.5 * hh; y1 += 0.5 * hh
    side = 0.4 * min(w, h)
    if x1 - x0 < side:
        cx = (x0 + x1) / 2; x0, x1 = cx - side / 2, cx + side / 2
    if y1 - y0 < side:
        cy = (y0 + y1) / 2; y0, y1 = cy - side / 2, cy + side / 2
    bx0, by0 = max(0, int(x0)), max(0, int(y0))
    bx1, by1 = min(w, int(x1)), min(h, int(y1))
    if bx1 <= bx0 or by1 <= by0:
        return None
    cw, ch = bx1 - bx0, by1 - by0
    inp = region_to_chw(frame, bx0, by0, cw, ch, POSE_SIZE)
    coords = m["pose"].run(None, {m["pose"].get_inputs()[0].name: inp})[0].reshape(-1, 2)
    return np.stack([bx0 + coords[:, 0] * cw, by0 + coords[:, 1] * ch], axis=1)


def draw(frame: np.ndarray, m: dict) -> np.ndarray:
    dets = detect_hands(m, frame)
    boxes = [b for _, b in dets]
    for score, b in dets:
        cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (40, 220, 80), 2)
        cv2.putText(frame, f"{score:.2f}", (b[0], max(0, b[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 220, 80), 1, cv2.LINE_AA)
        kps = landmarks_for(m, frame, b)
        for a, c in HAND_EDGES:
            cv2.line(frame, tuple(kps[a].astype(int)), tuple(kps[c].astype(int)), (0, 200, 255), 1, cv2.LINE_AA)
        for x, y in kps.astype(int):
            cv2.circle(frame, (x, y), 3, (0, 140, 255), -1, cv2.LINE_AA)
    pose = pose_for(m, frame, boxes)
    if pose is not None:
        for x, y in pose.astype(int):
            cv2.circle(frame, (x, y), 4, (255, 180, 60), -1, cv2.LINE_AA)
    cv2.putText(frame, f"hands: {len(boxes)}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--camera", type=int, default=0, help="webcam index")
    ap.add_argument("--image", type=Path, default=None, help="annotate one image instead of the webcam")
    ap.add_argument("--out", type=Path, default=Path("out.png"), help="output path for --image")
    args = ap.parse_args()

    m = load_models()
    print(f"loaded 3 from-scratch ONNX detectors from {MODELS_DIR}")

    if args.image is not None:
        frame = cv2.imread(str(args.image))
        if frame is None:
            raise SystemExit(f"could not read image: {args.image}")
        cv2.imwrite(str(args.out), draw(frame, m))
        print(f"wrote {args.out}")
        return 0

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {args.camera} (try --image instead)")
    mirror = True
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if mirror:
            frame = cv2.flip(frame, 1)
        cv2.imshow("from-scratch detectors (q quit, m mirror)", draw(frame, m))
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("m"):
            mirror = not mirror
    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
