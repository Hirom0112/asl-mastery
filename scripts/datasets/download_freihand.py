"""Download FreiHAND (Zimmermann et al., ICCV 2019).

Audit entry: docs/data/external_datasets_audit.md § A.
License: research-only, no commercial use.

Per ADR 0015, this script will not run without --confirm.
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


DATASET = "freihand"
PROJECT_PAGE = "https://lmb.informatik.uni-freiburg.de/projects/freihand/"
ARCHIVES = [
    # Primary archive (training images + annotations). URL is documented on
    # the project page; mirror is occasionally updated — verify before --confirm.
    (
        "FreiHAND_pub_v2.zip",
        "https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2.zip",
        None,  # No published SHA256; this script records its own digest in PROVENANCE.md.
    ),
    (
        "FreiHAND_pub_v2_eval.zip",
        "https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2_eval.zip",
        None,
    ),
]


def main() -> int:
    args = parse_common_args(DATASET)
    out = target_dir(DATASET, args.out_dir)

    plan = [
        f"target: {out}",
        f"archives: {len(ARCHIVES)}",
        "  - FreiHAND_pub_v2.zip (~9 GB, training images + annotations)",
        "  - FreiHAND_pub_v2_eval.zip (~1 GB, eval split)",
        "license: research-only, no commercial use",
        "primary source: " + PROJECT_PAGE,
        "checksums: not published upstream — script records its own SHA256",
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
            "Research-only. Verbatim from the project page: "
            '"This dataset is provided for research purposes only and without any warranty. '
            'Any commercial use is prohibited."'
        ),
        license_url=PROJECT_PAGE,
        originating_paper=(
            "Zimmermann, Ceylan, Yang, Russell, Argus, Brox. "
            '"FreiHAND: A Dataset for Markerless Capture of Hand Pose and Shape from Single RGB Images." '
            "ICCV 2019. arXiv:1909.04349."
        ),
        audit_entry="A. FreiHAND",
        notes=(
            "Annotation methodology: semi-automated human-in-the-loop. Initial sparse 2D keypoints "
            "and segmentation masks are HUMAN-annotated. MANO hand-model fitting + project-trained "
            "iterative network produce dense 3D pose + shape from 8-camera multi-view captures. "
            "No third-party pretrained CV detector is in the annotation pipeline."
        ),
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
