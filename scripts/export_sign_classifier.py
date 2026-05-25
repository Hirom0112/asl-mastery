"""Export a trained SignClassifier checkpoint to ONNX + config.json for the
browser (lib/inference/keypoint-predict.ts).

The checkpoint (saved by measure_norm_ab --save-model) holds
{model, num_features, num_classes, feature_version, classes}. The classifier
was trained with hidden=256, num_blocks=5 (the measure_norm_ab default), so we
instantiate with those to match the state_dict.

Usage:
    python -m scripts.export_sign_classifier \
        --checkpoint data/ckpts_new/sign_classifier_v4_best.pt \
        --out public/models/sign_classifier_v4.onnx \
        --config-out public/models/sign_classifier_v4.config.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from training.detectors.sign_classifier import SignClassifier

TIME_STEPS = 32


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--config-out", type=Path, required=True)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--num-blocks", type=int, default=5)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    classes = ck["classes"]
    num_features = ck["num_features"]
    num_classes = ck["num_classes"]
    feature_version = ck.get("feature_version", "hand_v2_108d")

    model = SignClassifier(num_features=num_features, num_classes=num_classes,
                           hidden=args.hidden, num_blocks=args.num_blocks)
    model.load_state_dict(ck["model"])
    model.eval()

    dummy = torch.randn(1, TIME_STEPS, num_features)
    with torch.no_grad():
        torch_out = model(dummy)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, str(args.out),
        input_names=["trajectory"], output_names=["logits"],
        opset_version=args.opset, do_constant_folding=True,
        dynamic_axes={"trajectory": {0: "batch"}, "logits": {0: "batch"}},
    )

    # parity check vs onnxruntime
    import onnxruntime as ort
    sess = ort.InferenceSession(str(args.out), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"trajectory": dummy.numpy()})[0]
    diff = float(np.abs(ort_out - torch_out.numpy()).max())

    cfg = {
        "classes": classes,
        "feature_version": feature_version,
        "num_features": num_features,
        "time_steps": TIME_STEPS,
        "norm": "hand",
        "temperature": 1.0,
        "perSignThresholds": {},
    }
    args.config_out.write_text(json.dumps(cfg, indent=1))

    print(json.dumps({
        "onnx": str(args.out),
        "config": str(args.config_out),
        "classes": len(classes),
        "num_features": num_features,
        "parity_max_abs_diff": diff,
        "ok": diff < 5e-4,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
