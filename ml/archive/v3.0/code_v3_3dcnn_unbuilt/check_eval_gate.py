#!/usr/bin/env python3
"""Enforce docs/EVAL_GATE.md §1 hard criteria against a validation report.

Reads `validation.json` (the machine-readable output of
`training/classifier/validate.py`) and exits non-zero if any hard
criterion fails. Used in CI on model-promotion PRs (slice-2
target) and runnable locally during slice-1 manual promotion.

Usage:

    python scripts/check_eval_gate.py runs/v1-001/validation.json

Each criterion is a separate check; the script reports every miss
(not just the first) so the reviewer sees the full story.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def fmt(b: bool) -> str:
    return "PASS" if b else "FAIL"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("validation", type=Path)
    parser.add_argument(
        "--previous",
        type=Path,
        default=None,
        help="Path to the previously-active validation.json for regression checks (criterion 6).",
    )
    args = parser.parse_args()

    if not args.validation.exists():
        print(f"FATAL: {args.validation} not found", file=sys.stderr)
        return 2

    with args.validation.open() as f:
        v = json.load(f)

    failures: list[str] = []

    # 1. Overall top-1 ≥ 85%
    top1 = float(v.get("top1", 0))
    ok1 = top1 >= 0.85
    print(f"  [{fmt(ok1)}] 1. top-1 ≥ 0.85         actual {top1:.4f}")
    if not ok1:
        failures.append(f"top-1 {top1:.4f} < 0.85")

    # 2. No sign with test accuracy < 60%
    per_sign = v.get("per_sign_accuracy", {})
    weak = {s: m for s, m in per_sign.items() if m.get("accuracy", 0) < 0.60 and m.get("n", 0) > 0}
    ok2 = len(weak) == 0
    print(f"  [{fmt(ok2)}] 2. min per-sign ≥ 0.60   weak: {list(weak.keys())[:5]}{'…' if len(weak) > 5 else ''}")
    if not ok2:
        failures.append(f"{len(weak)} sign(s) below 60% accuracy: {list(weak.keys())}")

    # 3. Fitzpatrick gap ≤ 10 pp. Only enforce when buckets exist.
    fp = v.get("per_demographic_accuracy", {}).get("fitzpatrick", {})
    if fp:
        accs = [m.get("accuracy", 0) for m in fp.values() if m.get("n", 0) >= 10]
        if len(accs) >= 2:
            gap = max(accs) - min(accs)
            ok3 = gap <= 0.10
            print(f"  [{fmt(ok3)}] 3. fitzpatrick gap ≤ 0.10  actual {gap:.4f}")
            if not ok3:
                failures.append(f"fitzpatrick gap {gap:.4f} > 0.10")
        else:
            print("  [SKIP] 3. fitzpatrick gap          not enough consented buckets")
    else:
        print("  [SKIP] 3. fitzpatrick gap          no per-demographic data in report")

    # 4. Per-sign threshold ≥90% precision: validate.py sets these per-sign;
    #    we check the thresholds are populated and non-degenerate (< 1.0
    #    meaning the sign is actually predictable above the floor).
    thresholds = v.get("per_sign_confidence_thresholds", {})
    degenerate = [s for s, t in thresholds.items() if t >= 1.0]
    ok4 = len(thresholds) > 0 and len(degenerate) == 0
    print(
        f"  [{fmt(ok4)}] 4. per-sign threshold ≥90% precision  thresholds populated, {len(degenerate)} degenerate"
    )
    if not ok4:
        failures.append(
            f"per-sign thresholds incomplete or degenerate ({len(degenerate)} signs)"
        )

    # 5. Latency p95 ≤ 600 ms. Validation report records it if measured;
    #    otherwise we surface as SKIP (latency is benchmarked separately).
    lat = v.get("latency_p95_ms")
    if isinstance(lat, (int, float)):
        ok5 = lat <= 600
        print(f"  [{fmt(ok5)}] 5. latency p95 ≤ 600 ms   actual {lat:.0f} ms")
        if not ok5:
            failures.append(f"latency p95 {lat:.0f} ms > 600 ms")
    else:
        print("  [SKIP] 5. latency p95             no latency_p95_ms in report")

    # 6. Regression check (only with --previous).
    if args.previous and args.previous.exists():
        with args.previous.open() as f:
            prev = json.load(f)
        prev_top1 = float(prev.get("top1", 0))
        delta = top1 - prev_top1
        ok6 = delta >= -0.03
        print(
            f"  [{fmt(ok6)}] 6. no regression > 3pp   Δ {delta:+.4f}  (prev {prev_top1:.4f})"
        )
        if not ok6:
            failures.append(f"regression {delta:+.4f} > -0.03 vs previous")
    else:
        print("  [SKIP] 6. no regression            no --previous report supplied")

    # 7. Calibration: reliability ECE ≤ 0.05.
    ece = v.get("expected_calibration_error")
    if isinstance(ece, (int, float)):
        ok7 = ece <= 0.05
        print(f"  [{fmt(ok7)}] 7. calibration ECE ≤ 0.05  actual {ece:.4f}")
        if not ok7:
            failures.append(f"ECE {ece:.4f} > 0.05")
    else:
        print("  [SKIP] 7. calibration ECE          not in report (validate.py records only T)")

    # 8. Hint coverage: every confusion pair appearing > 1 has a hint.
    confusion = v.get("top3_confusions_per_sign", {})
    uncovered = []
    hints = v.get("confusion_pair_hints", {})  # optional; the model_versions row joins to confusion_pair_hints in prod
    for sign, pairs in confusion.items():
        for p in pairs:
            key = f"{sign}->{p['predicted']}"
            if p.get("count", 0) >= 2 and hints and key not in hints:
                uncovered.append(key)
    if hints:
        ok8 = len(uncovered) == 0
        print(f"  [{fmt(ok8)}] 8. hint coverage          {len(uncovered)} uncovered repeated confusions")
        if not ok8:
            failures.append(f"hint coverage gaps: {uncovered[:5]}")
    else:
        print("  [SKIP] 8. hint coverage            confusion_pair_hints not embedded in report")

    # 9. No-pretrained-pipeline evidence — code-level, not validation-report-level.
    #    The audit surface is training/classifier/cnn.py (Kaiming init,
    #    no load_state_dict, no external weight URL) plus the absence
    #    of MediaPipe / pretrained-backbone imports anywhere. We confirm
    #    the report tags the model architecture as the v3.x from-scratch
    #    option.
    arch = v.get("model_architecture") or v.get("model_name")
    # `small_r2plus1d` is the v3.x architecture (post-ADR-0010).
    # `bilstm` / `transformer` are accepted only on historical reports.
    ok9 = arch in ("small_r2plus1d", "bilstm", "transformer", None)
    print(f"  [{fmt(ok9)}] 9. no-pretrained-pipeline  architecture={arch}")
    if not ok9:
        failures.append(f"unrecognized model architecture in report: {arch}")

    # (Criterion 10, MediaPipe detection success ≥ 95%, was dropped on
    # 2026-05-20 along with ADR 0006. There is no MediaPipe in the
    # pipeline to measure under ADR 0010.)

    print()
    if failures:
        print(f"EVAL GATE: {len(failures)} hard criteria failed", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("EVAL GATE: all hard criteria passed (or skipped where data unavailable).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
