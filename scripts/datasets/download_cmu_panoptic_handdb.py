"""Download CMU Panoptic HandDB (Simon et al., CVPR 2017).

Audit entry: docs/data/external_datasets_audit.md § B.
License: CMU Panoptic — research-only, non-commercial. No redistribution.

Per ADR 0015, this script will not run without --confirm.

Three subsets, each downloadable separately:
- Manual (MPII+NZSL): human-labeled, ~140 MB
- Multiview-bootstrapped (CMU Panoptic dome): triangulated from project-trained detector, ~17 GB
- Synthetic: rendered, ~3 GB
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    banner_plan,
    download_file,
    ensure_dir,
    parse_common_args,
    target_dir,
    write_provenance,
)


DATASET = "cmu_panoptic_handdb"
PROJECT_PAGE = "http://domedb.perception.cs.cmu.edu/handdb.html"

# Per the project page, the subset archives are linked there. URLs change
# occasionally — verify on the page before --confirm. The names below are
# the conventional filenames.
ARCHIVES = [
    (
        "hand_labels.zip",
        "http://domedb.perception.cs.cmu.edu/panopticDB/hands/hand_labels.zip",
        None,
    ),
    (
        "hand_labels_synth.zip",
        "http://domedb.perception.cs.cmu.edu/panopticDB/hands/hand_labels_synth.zip",
        None,
    ),
    (
        "hand143_panopticdb.tar",
        "http://domedb.perception.cs.cmu.edu/panopticDB/hands/hand143_panopticdb.tar",
        None,
    ),
]


def main() -> int:
    args = parse_common_args(DATASET)
    out = target_dir(DATASET, args.out_dir)

    plan = [
        f"target: {out}",
        "subsets:",
        "  - hand_labels.zip — manual subset (MPII+NZSL), ~140 MB",
        "  - hand_labels_synth.zip — synthetic subset, ~3 GB",
        "  - hand143_panopticdb.tar — multiview-bootstrapped subset, ~17 GB",
        "license: research-only, non-commercial, no redistribution",
        "primary source: " + PROJECT_PAGE,
        "before --confirm: verify each URL is still live on the project page",
    ]
    banner_plan(DATASET, plan)

    if not (args.confirm or args.dry_run):
        return 0

    ensure_dir(out, dry_run=args.dry_run)
    digests: dict[str, str] = {}
    for fname, url, expected in ARCHIVES:
        digest = download_file(url, out / fname, expected_sha256=expected, dry_run=args.dry_run)
        if digest:
            digests[fname] = digest

    write_provenance(
        out,
        dataset_name=DATASET,
        source_urls=[PROJECT_PAGE, *(url for _, url, _ in ARCHIVES)],
        archives=digests,
        license_summary=(
            "Research-only, non-commercial. From the CMU Panoptic license: "
            '"shared only for research purposes and cannot be used for any commercial purposes. '
            'The dataset or its modified version cannot be redistributed without permission from '
            'dataset organizers."'
        ),
        license_url=PROJECT_PAGE,
        originating_paper=(
            "Simon, Joo, Matthews, Sheikh. "
            '"Hand Keypoint Detection in Single Images Using Multiview Bootstrapping." '
            "CVPR 2017. arXiv:1704.07809."
        ),
        audit_entry="B. CMU Panoptic HandDB",
        notes=(
            "Three subsets with distinct provenance: "
            "(i) hand_labels.zip — PURE HUMAN annotation (MPII + NZSL images). "
            "(ii) hand_labels_synth.zip — synthetic renders; annotations from renderer geometry. "
            "(iii) hand143_panopticdb.tar — multiview-bootstrapped: produced by the authors' OWN "
            "keypoint detector first trained on subset (i), then triangulated across the 31-camera "
            "Panoptic Studio dome with multi-view consistency filtering. No third-party pretrained "
            "CV detector in any subset's annotation chain."
        ),
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
