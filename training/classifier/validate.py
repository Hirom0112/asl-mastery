"""Validation harness per docs/MODEL.md §4-§5 and docs/EVAL_GATE.md.

Adapted from the BiLSTM-on-keypoints validator that shipped under the
now-superseded ADR 0006: loads a `SmallR2Plus1D` checkpoint
(`training/classifier/cnn.py`), runs the held-out test split via
`VideoClipDataset` (`training/classifier/dataset_video.py`), and
emits the same shape of validation report — minus the
MediaPipe-specific per-clip detection-success field, which has no
analogue under Path B.

Output:
    `validation.json` — machine-readable, the eval-gate input
    `validation.md`   — human-readable, what the README links to

Usage:

    python -m training.classifier.validate \\
        --manifest dataset/clean/v3/dataset_v3_manifest.json \\
        --checkpoint runs/v3-001/best.pt \\
        --output runs/v3-001/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.classifier.augment import make_val_transform  # noqa: E402
from training.classifier.cnn import SmallR2Plus1D  # noqa: E402
from training.classifier.dataset_video import VideoClipDataset  # noqa: E402

log = logging.getLogger("validate")


def _temperature_scale(logits: np.ndarray, labels: np.ndarray) -> float:
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
    out: dict[str, float] = {}
    for k, name in enumerate(classes):
        scores = probs[:, k]
        targets = labels == k
        order = np.argsort(-scores)
        s = scores[order]
        t = targets[order]
        best = 1.0
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
    input_h = int(ckpt.get("input_height", 96))
    input_w = int(ckpt.get("input_width", 96))

    val_transform = make_val_transform(input_height=input_h, input_width=input_w)
    test_ds = VideoClipDataset(
        args.manifest,
        split="test",
        augment=lambda frames: val_transform(frames, "_"),
        input_height=input_h,
        input_width=input_w,
    )
    if set(test_ds.classes) != set(classes):
        log.warning(
            "test split class set differs from checkpoint; remapping test labels to checkpoint class indices."
        )
    test_ds.classes = list(classes)
    test_ds.sign_to_idx = {s: i for i, s in enumerate(classes)}
    test_ds.clips = [c for c in test_ds.clips if c["sign_id"] in test_ds.sign_to_idx]
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallR2Plus1D(
        num_classes=len(classes), input_height=input_h, input_width=input_w
    ).to(device)
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

    T = _temperature_scale(logits, labels)
    z = logits / T
    z = z - z.max(axis=1, keepdims=True)
    probs = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    preds = probs.argmax(axis=1)

    top1 = float((preds == labels).mean())
    top3 = _top_k(probs, labels, k=3)

    per_sign: dict[str, dict[str, float]] = {}
    for k, name_k in enumerate(classes):
        mask = labels == k
        if mask.sum() == 0:
            continue
        per_sign[name_k] = {
            "n": int(mask.sum()),
            "accuracy": float((preds[mask] == labels[mask]).mean()),
        }

    cm = _confusion_matrix(preds, labels, len(classes))
    top3_confusions: dict[str, list[dict[str, Any]]] = {}
    for k, name_k in enumerate(classes):
        row = cm[k].copy()
        row[k] = 0
        order = np.argsort(-row)[:3]
        top3_confusions[name_k] = [
            {"predicted": classes[i], "count": int(row[i])} for i in order if row[i] > 0
        ]

    thresholds = _per_sign_thresholds(probs, labels, classes, target_precision=0.9)

    out: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "n_test_clips": int(len(labels)),
        "top1": top1,
        "top3": top3,
        "temperature": T,
        "model_architecture": ckpt.get("model_architecture", "small_r2plus1d"),
        "input_height": input_h,
        "input_width": input_w,
        "per_sign_accuracy": per_sign,
        "top3_confusions_per_sign": top3_confusions,
        "per_sign_confidence_thresholds": thresholds,
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
    lines.append(f"# Validation report\n\n")
    lines.append(f"- **Model:** {r['model_architecture']}\n")
    lines.append(f"- **Top-1 accuracy:** {r['top1']:.4f}\n")
    lines.append(f"- **Top-3 accuracy:** {r['top3']:.4f}\n")
    lines.append(f"- **Calibration temperature:** {r['temperature']:.3f}\n")
    lines.append(f"- **Test clips:** {r['n_test_clips']}\n")
    lines.append(f"- **Input shape:** (T=16, H={r['input_height']}, W={r['input_width']}, 3)\n\n")
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
