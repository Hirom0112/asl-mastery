"""From-scratch face detector (single-class CenterNet-style).

Identical architecture pattern to HandDetector — just retrained on
WIDER FACE rather than hand bbox data. Same input size, same head,
same loss. Aliased so downstream code (training, ONNX export,
Modal entrypoints) treats the two detectors symmetrically.

Per ADR 0011 the face detector is only used for the framing-bracket
UI affordance + onboarding calibration; it does not participate in
sign recognition.
"""

from __future__ import annotations

from training.detectors.hand_detector import HandDetector


class FaceDetector(HandDetector):
    """Alias of HandDetector. Trained on WIDER FACE bbox manifests via the
    same training/detectors/train.py entrypoint with a face_bbox manifest.
    """
    pass


if __name__ == "__main__":
    import torch
    m = FaceDetector()
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    out = m(torch.randn(2, 3, 320, 320))
    assert out["heatmap"].shape == (2, 1, 80, 80)
    print(f"FaceDetector OK. parameters={n:,} (~{n/1e6:.2f}M)")
