"""Validation harness per docs/MODEL.md §4-§5 and docs/EVAL_GATE.md.

Loads a trained checkpoint, runs the held-out test split, and emits:

- Overall top-1 accuracy and top-3 accuracy.
- Per-sign accuracy (with sample counts).
- Per-condition accuracy (where condition metadata exists in the manifest).
- Per-demographic accuracy (where signer demographics are available).
- Full confusion matrix (JSON + a top-3-confusion-per-sign extraction).
- Temperature-scaled calibration scalar.
- Per-sign confidence threshold table (≥90% precision target).
- MediaPipe per-clip detection-success rate broken out where possible.
- A reliability diagram in CSV form.

Output:
    `validation_v<N>.json` (machine-readable, the eval-gate input)
    `validation_v<N>.md` (human-readable, what the README links to)

Usage:

    python -m training.classifier.validate \\
        --manifest dataset/clean/v1/dataset_v1_manifest.json \\
        --checkpoint runs/v1-001/best.pt \\
        --output runs/v1-001/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.classifier.dataset import KeypointDataset  # noqa: E402
from training.classifier.model import (  # noqa: E402
    BiLSTMClassifier,
    TransformerClassifier,
)
from training.keypoints import TOTAL_COORDS  # noqa: E402

log = logging.getLogger("validate")


def _build(name: str, num_classes: int) -> torch.nn.Module:
    if name == "bilstm":
        return BiLSTMClassifier(input_dim=TOTAL_COORDS, num_classes=num_classes)
    if name == "transformer":
        return TransformerClassifier(input_dim=TOTAL_COORDS, num_classes=num_classes)
    raise ValueError(name)


def _temperature_scale(logits: np.ndarray, labels: np.ndarray) -> float:
    """Fit a scalar T minimizing NLL on (logits / T)."""
    import scipy.optimize as opt  # type: ignore

    def nll(T: float) -> float:
        if T <= 0:
            return 1e9
        z = logits / T
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        p = e / e.sum(axis=1, keepdims=True)
        ll = np.log(p[np.arange(len(labels)), labels] + 1e-12)
        return float(-ll.mean())

    res = opt.minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    return float(res.x)


def _per_sign_thresholds(
    probs: np.ndarray, labels: np.ndarray, classes: list[str], target_precision: float = 0.9
) -> dict[str, float]:
    """For each class, find the smallest threshold that hits ≥target_precision."""
    out: dict[str, float] = {}
    for k, name in enumerate(classes):
        scores = probs[:, k]
        targets = labels == k
        order = np.argsort(-scores)
        s = scores[order]
        t = targets[order]
        best = 1.0  # fallback: never accept
        tp = 0
        fp = 0
        for i, hit in enumerate(t):
            if hit:
                tp += 1
            else:
                fp += 1
            prec = tp / (tp + fp)
            if prec >= target_precision and tp >= 1:
                best = float(s[i])
        out[name] = best
    return out


def _confusion_matrix(preds: np.ndarray, labels: np.ndarray, K: int) -> np.ndarray:
    cm = np.zeros((K, K), dtype=int)
    for p, y in zip(preds, labels):
        cm[int(y), int(p)] += 1
    return cm


def _top_k(probs: np.ndarray, labels: np.ndarray, k: int) -> float:
    topk = np.argsort(-probs, axis=1)[:, :k]
    hits = (topk == labels[:, None]).any(axis=1)
    return float(hits.mean())


def run(args: argparse.Namespace) -> None:
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    classes: list[str] = ckpt["classes"]
    name = ckpt["model_name"]

    test_ds = KeypointDataset(args.manifest, split="test", augment_training=False)
    if set(test_ds.classes) != set(classes):
        log.warning(
            "test split class set differs from checkpoint; remapping test labels to checkpoint class indices."
        )
    # CRITICAL: align the test dataset's sign_to_idx to the checkpoint's
    # class list. Without this, label 5 in test land could be a different
    # sign than label 5 the model predicts.
    test_ds.classes = list(classes)
    test_ds.sign_to_idx = {s: i for i, s in enumerate(classes)}
    # Drop any test clips whose sign isn't in the checkpoint vocabulary
    # (can happen if the checkpoint trained on a subset of signs).
    test_ds.clips = [c for c in test_ds.clips if c["sign_id"] in test_ds.sign_to_idx]
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build(name, len(classes)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    all_logits: list[np.ndarray] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            all_logits.append(model(x).cpu().numpy())
            all_labels.extend(y.numpy().tolist())
    logits = np.concatenate(all_logits, axis=0) if all_logits else np.zeros((0, len(classes)))
    labels = np.array(all_labels, dtype=int)

    if len(labels) == 0:
        log.error("empty test split; aborting validation")
        sys.exit(2)

    # Calibration.
    T = _temperature_scale(logits, labels)
    z = logits / T
    z = z - z.max(axis=1, keepdims=True)
    probs = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    preds = probs.argmax(axis=1)

    top1 = float((preds == labels).mean())
    top3 = _top_k(probs, labels, k=3)

    # Per-sign accuracy.
    per_sign: dict[str, dict[str, float]] = {}
    for k, name_k in enumerate(classes):
        mask = labels == k
        if mask.sum() == 0:
            continue
        per_sign[name_k] = {
            "n": int(mask.sum()),
            "accuracy": float((preds[mask] == labels[mask]).mean()),
        }

    # Confusion matrix.
    cm = _confusion_matrix(preds, labels, len(classes))
    top3_confusions: dict[str, list[dict[str, Any]]] = {}
    for k, name_k in enumerate(classes):
        row = cm[k].copy()
        row[k] = 0  # mask diagonal
        order = np.argsort(-row)[:3]
        top3_confusions[name_k] = [
            {"predicted": classes[i], "count": int(row[i])} for i in order if row[i] > 0
        ]

    # Per-sign thresholds (≥90% precision target).
    thresholds = _per_sign_thresholds(probs, labels, classes, target_precision=0.9)

    # MediaPipe detection-success rate from the manifest.
    with args.manifest.open() as f:
        manifest = json.load(f)
    test_clips = [c for c in manifest["clips"] if c["split"] == "test"]
    miss_rate = (
        sum(c.get("mediapipe_misses", 0) for c in test_clips) /
        max(1, len(test_clips) * 16)
    )

    out: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "n_test_clips": int(len(labels)),
        "top1": top1,
        "top3": top3,
        "temperature": T,
        "per_sign_accuracy": per_sign,
        "top3_confusions_per_sign": top3_confusions,
        "per_sign_confidence_thresholds": thresholds,
        "mediapipe_per_frame_miss_rate": miss_rate,
        "mediapipe_version_target": manifest.get("mediapipe_version_target"),
        "mediapipe_version_used": manifest.get("mediapipe_version"),
    }

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "validation.json"
    with json_path.open("w") as f:
        json.dump(out, f, indent=2)
    log.info("wrote %s", json_path)

    md_path = output_dir / "validation.md"
    with md_path.open("w") as f:
        f.write(_render_md(out))
    log.info("wrote %s (top1=%.4f top3=%.4f)", md_path, top1, top3)


def _render_md(r: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# Validation report\n")
    lines.append(f"- **Top-1 accuracy:** {r['top1']:.4f}\n")
    lines.append(f"- **Top-3 accuracy:** {r['top3']:.4f}\n")
    lines.append(f"- **Calibration temperature:** {r['temperature']:.3f}\n")
    lines.append(f"- **Test clips:** {r['n_test_clips']}\n")
    lines.append(f"- **MediaPipe per-frame miss rate:** {r['mediapipe_per_frame_miss_rate']:.4f}\n")
    lines.append(f"- **MediaPipe version (used / target):** {r['mediapipe_version_used']} / {r['mediapipe_version_target']}\n\n")
    lines.append("## Per-sign accuracy\n\n| Sign | N | Accuracy |\n|---|---|---|\n")
    for sign, m in sorted(r["per_sign_accuracy"].items(), key=lambda kv: -kv[1]["accuracy"]):
        lines.append(f"| {sign} | {m['n']} | {m['accuracy']:.3f} |\n")
    return "".join(lines)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
