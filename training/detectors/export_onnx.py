"""Export trained detector checkpoints to ONNX for browser inference (Phase 5).

Each detector exports independently. The ONNX export is a faithful
trace of the from-scratch PyTorch graph — no foreign-weight injection,
no architecture surgery. Output: one .onnx per detector + a parity log
showing max |torch - onnxruntime| diff on a random input.

Usage:

    # Export one detector:
    python -m training.detectors.export_onnx \
        --detector hand_detector \
        --checkpoint runs/hand_det/hand_det_v0_a100_resume_20260521_145244Z/best.pt \
        --out artifacts/onnx/hand_detector_v0.onnx

    # Export all three from a runs/ tree (looks for best.pt under each subdir):
    python -m training.detectors.export_onnx --all \
        --hand-det-run    runs/hand_det/hand_det_v0_a100_resume_20260521_145244Z \
        --hand-lm-run     runs/hand_landmarks/hand_landmarks_v0_20260521_105603Z \
        --face-run        runs/face/face_det_v0_20260521_105723Z \
        --out-dir         artifacts/onnx
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from training.detectors.hand_detector import HandDetector
from training.detectors.hand_landmarks import HandLandmarkRegressor
from training.detectors.pose_detector import PoseRegressor


DETECTOR_SPECS = {
    "hand_detector": {
        "cls": HandDetector,
        "input_shape": (1, 3, 320, 320),
        "input_names": ["image"],
        "output_names": ["heatmap", "size"],
    },
    "face_detector": {
        # face_detector is an alias of HandDetector with independent weights.
        "cls": HandDetector,
        "input_shape": (1, 3, 320, 320),
        "input_names": ["image"],
        "output_names": ["heatmap", "size"],
    },
    "hand_landmarks": {
        "cls": HandLandmarkRegressor,
        "input_shape": (1, 3, 224, 224),
        "input_names": ["hand_crop"],
        "output_names": ["coords", "visibility"],
    },
    "pose_detector": {
        "cls": PoseRegressor,
        "input_shape": (1, 3, 256, 256),
        "input_names": ["person_crop"],
        "output_names": ["coords", "visibility"],
    },
}


class _HandDetectorTraceable(torch.nn.Module):
    """ONNX-friendly wrapper that returns a tuple instead of a dict."""

    def __init__(self, m: HandDetector) -> None:
        super().__init__()
        self.m = m

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.m(x)
        return out["heatmap"], out["size"]


class _RegressorTraceable(torch.nn.Module):
    def __init__(self, m: torch.nn.Module) -> None:
        super().__init__()
        self.m = m

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.m(x)
        return out["coords"], out["visibility"]


def _load(detector: str, checkpoint: Path) -> torch.nn.Module:
    spec = DETECTOR_SPECS[detector]
    model = spec["cls"]()
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(sd)
    model.eval()
    return model


def _wrap(detector: str, model: torch.nn.Module) -> torch.nn.Module:
    if detector in ("hand_detector", "face_detector"):
        return _HandDetectorTraceable(model)
    return _RegressorTraceable(model)


def export_one(detector: str, checkpoint: Path, out_path: Path,
               opset: int = 17, parity_atol: float = 5e-4) -> dict:
    spec = DETECTOR_SPECS[detector]
    model = _load(detector, checkpoint)
    wrapped = _wrap(detector, model)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.randn(*spec["input_shape"])

    with torch.no_grad():
        torch_out = wrapped(dummy)

    torch.onnx.export(
        wrapped,
        dummy,
        str(out_path),
        input_names=spec["input_names"],
        output_names=spec["output_names"],
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes={spec["input_names"][0]: {0: "batch"}},
    )

    parity = _check_parity(out_path, dummy, torch_out, atol=parity_atol)
    report = {
        "detector": detector,
        "checkpoint": str(checkpoint),
        "onnx": str(out_path),
        "input_shape": list(spec["input_shape"]),
        "opset": opset,
        "parity": parity,
    }
    print(json.dumps(report, indent=2))
    return report


def _check_parity(onnx_path: Path, dummy: torch.Tensor,
                  torch_out: tuple[torch.Tensor, ...], atol: float) -> dict:
    try:
        import onnxruntime as ort
    except ImportError:
        return {"status": "skipped — onnxruntime not installed", "atol": atol}

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    ort_out = sess.run(None, {in_name: dummy.numpy()})

    max_diffs = []
    for i, (t, o) in enumerate(zip(torch_out, ort_out)):
        d = float((torch.from_numpy(o) - t).abs().max().item())
        max_diffs.append({"index": i, "max_abs_diff": d, "ok": d < atol})
    return {"atol": atol, "outputs": max_diffs,
            "ok": all(d["ok"] for d in max_diffs)}


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", choices=list(DETECTOR_SPECS), default=None)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--hand-det-run", type=Path)
    ap.add_argument("--hand-lm-run", type=Path)
    ap.add_argument("--face-run", type=Path)
    ap.add_argument("--pose-run", type=Path, default=None,
                    help="Optional — pose checkpoint dir; skipped if not provided.")
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/onnx"))
    ap.add_argument("--opset", type=int, default=17)
    return ap.parse_args()


def main() -> int:
    args = _parse_args()

    if args.all:
        plan = [
            ("hand_detector", args.hand_det_run, "hand_detector_v0.onnx"),
            ("hand_landmarks", args.hand_lm_run, "hand_landmarks_v0.onnx"),
            ("face_detector", args.face_run, "face_detector_v0.onnx"),
        ]
        if args.pose_run is not None:
            plan.append(("pose_detector", args.pose_run, "pose_detector_v0.onnx"))

        reports = []
        for det, run_dir, fname in plan:
            if run_dir is None:
                print(f"[skip] {det}: no --{det.replace('_', '-')}-run path")
                continue
            ckpt = run_dir / "best.pt"
            if not ckpt.exists():
                print(f"[skip] {det}: no best.pt at {ckpt}")
                continue
            reports.append(export_one(det, ckpt, args.out_dir / fname, opset=args.opset))

        (args.out_dir / "_parity.json").write_text(json.dumps(reports, indent=2))
        return 0

    if args.detector is None or args.checkpoint is None or args.out is None:
        raise SystemExit("provide --detector / --checkpoint / --out, or use --all")
    export_one(args.detector, args.checkpoint, args.out, opset=args.opset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
