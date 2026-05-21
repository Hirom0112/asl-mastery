"""Apply ADR 0008's per-sign clip-count floor.

Reads ingestion manifests (WLASL, MS-ASL), counts per-sign
downloadable clips combined, drops signs below the floor while
keeping the slice-1 vocabulary count ≥ 75 (Brief Requirement 2). If
the floor would push the count under 75, the floor is reduced and
the reduction is recorded in the output for the validation report.

Usage:

    python -m training.data.filter_vocabulary \\
        --manifests dataset/raw/wlasl_manifest.json dataset/raw/msasl_manifest.json \\
        --floor 15 \\
        --output dataset/slice1_vocabulary.json

Output schema:

    {
      "applied_floor": int,
      "requested_floor": int,
      "floor_was_reduced": bool,
      "kept_signs": [
        {"sign_id": "...", "gloss": "...", "downloadable_count": int, "sources": {"wlasl": int, "msasl": int}},
        ...
      ],
      "dropped_signs": [
        {"sign_id": "...", "gloss": "...", "downloadable_count": int, "reason": "below_floor"}
      ],
      "kept_count": int
    }
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.data.vocabulary import get_all  # noqa: E402

log = logging.getLogger("filter_vocabulary")

MIN_VOCAB = 75  # Brief Requirement 2


def _load_manifest_counts(path: Path) -> dict[str, int]:
    with path.open() as f:
        manifest = json.load(f)
    return dict(manifest.get("per_sign_downloadable", {}))


def filter_vocab(
    manifests: list[Path],
    requested_floor: int,
) -> dict:
    items = get_all()

    # Combined per-sign downloadable count, plus per-source breakdown.
    combined: dict[str, int] = defaultdict(int)
    per_source: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for m_path in manifests:
        if not m_path.exists():
            log.warning("manifest not found: %s (skipping)", m_path)
            continue
        with m_path.open() as f:
            data = json.load(f)
        src = data.get("source", m_path.stem)
        for sid, n in data.get("per_sign_downloadable", {}).items():
            combined[sid] += n
            per_source[sid][src] += n

    def _drop_below(floor: int) -> tuple[list, list]:
        kept = []
        dropped = []
        for item in items:
            n = combined.get(item.sign_id, 0)
            row = {
                "sign_id": item.sign_id,
                "gloss": item.gloss,
                "downloadable_count": n,
                "sources": dict(per_source[item.sign_id]),
            }
            if n >= floor:
                kept.append(row)
            else:
                dropped.append({**row, "reason": "below_floor"})
        return kept, dropped

    applied_floor = requested_floor
    kept, dropped = _drop_below(applied_floor)
    floor_was_reduced = False
    while len(kept) < MIN_VOCAB and applied_floor > 1:
        applied_floor -= 1
        floor_was_reduced = True
        kept, dropped = _drop_below(applied_floor)
        log.warning(
            "kept count %d below MIN_VOCAB=%d; reducing floor to %d",
            len(kept),
            MIN_VOCAB,
            applied_floor,
        )

    return {
        "applied_floor": applied_floor,
        "requested_floor": requested_floor,
        "floor_was_reduced": floor_was_reduced,
        "kept_signs": kept,
        "dropped_signs": dropped,
        "kept_count": len(kept),
        "min_vocab_constraint": MIN_VOCAB,
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--floor", type=int, default=15)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = filter_vocab(args.manifests, args.floor)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(result, f, indent=2)

    log.info(
        "applied floor=%d (requested=%d, reduced=%s); kept %d, dropped %d",
        result["applied_floor"],
        result["requested_floor"],
        result["floor_was_reduced"],
        result["kept_count"],
        len(result["dropped_signs"]),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
