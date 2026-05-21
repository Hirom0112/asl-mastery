"""ONNX export per docs/MODEL.md §6 and docs/ARCHITECTURE.md §2.5 step 8.

Exports the classifier with dynamic batch and temporal axes,
verifies float32 ONNX output matches PyTorch within tolerance, and
writes the artifact bundle the browser inference path consumes:

  artifacts/v<NNN>/
    classifier.onnx        — the model
    config.json            — { classes, temperature, per_sign_thresholds, mediapipe_version, ... }
    validation.json        — symlinked / copied from validate.py output
    manifest.json          — { artifact_version, training_run, dataset_version, git_sha, sha256s }

Per docs/MODEL.md §6 quantization is optional under ADR 0006 and
applied only if Phase 4 measurement shows a meaningful win.

Usage:

    python -m training.classifier.export \\
        --checkpoint runs/v1-001/best.pt \\
        --validation runs/v1-001/validation.json \\
        --output artifacts/v1.0.0/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.classifier.model import (  # noqa: E402
    BiLSTMClassifier,
    TransformerClassifier,
)
from training.keypoints import TEMPORAL_LENGTH, TOTAL_COORDS  # noqa: E402

log = logging.getLogger("export")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export(args: argparse.Namespace) -> None:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    classes: list[str] = ckpt["classes"]
    name: str = ckpt["model_name"]

    if name == "bilstm":
        model = BiLSTMClassifier(input_dim=TOTAL_COORDS, num_classes=len(classes))
    elif name == "transformer":
        model = TransformerClassifier(input_dim=TOTAL_COORDS, num_classes=len(classes))
    else:
        raise ValueError(name)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    onnx_path = output_dir / "classifier.onnx"
    dummy = torch.randn(1, TEMPORAL_LENGTH, TOTAL_COORDS)
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["keypoints"],
        output_names=["logits"],
        dynamic_axes={"keypoints": {0: "batch", 1: "time"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    log.info("wrote ONNX → %s", onnx_path)

    # Parity check.
    import onnxruntime as ort  # type: ignore

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"keypoints": dummy.numpy()})[0]
    with torch.no_grad():
        torch_out = model(dummy).numpy()
    max_diff = float(np.abs(onnx_out - torch_out).max())
    log.info("ONNX↔PyTorch max abs diff: %.6f", max_diff)
    if max_diff > 1e-4:
        log.warning(
            "ONNX↔PyTorch divergence above tolerance; investigate before promoting (docs/MODEL.md §6)."
        )

    # Validation report (optional).
    validation_data: dict = {}
    if args.validation and args.validation.exists():
        with args.validation.open() as f:
            validation_data = json.load(f)
        shutil.copy(args.validation, output_dir / "validation.json")

    config = {
        "classes": classes,
        "temporal_length": TEMPORAL_LENGTH,
        "input_dim": TOTAL_COORDS,
        "temperature": validation_data.get("temperature", 1.0),
        "per_sign_thresholds": validation_data.get("per_sign_confidence_thresholds", {}),
        "mediapipe_version_target": validation_data.get("mediapipe_version_target"),
    }
    config_path = output_dir / "config.json"
    with config_path.open("w") as f:
        json.dump(config, f, indent=2)
    log.info("wrote %s", config_path)

    manifest = {
        "artifact_version": args.version,
        "checkpoint": str(args.checkpoint),
        "validation": str(args.validation) if args.validation else None,
        "onnx_path": str(onnx_path),
        "onnx_sha256": _sha256(onnx_path),
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "model_architecture": name,
        "num_classes": len(classes),
        "input_shape": [None, TEMPORAL_LENGTH, TOTAL_COORDS],
        "max_abs_parity_diff": max_diff,
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    log.info("wrote %s", manifest_path)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", type=str, default="v1.0.0")
    args = parser.parse_args()
    export(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
