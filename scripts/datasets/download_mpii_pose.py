"""Download MPII Human Pose (Andriluka et al., CVPR 2014).

Audit entry: docs/data/external_datasets_audit.md § I.
License: annotations BSD; images research-only (YouTube source rights).

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


DATASET = "mpii_pose"
PROJECT_PAGE = "http://human-pose.mpi-inf.mpg.de/"

ARCHIVES = [
    (
        "mpii_human_pose_v1.tar.gz",
        "https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz",
        None,
    ),
    (
        "mpii_human_pose_v1_u12_2.zip",
        "https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1_u12_2.zip",
        None,
    ),
]


def main() -> int:
    args = parse_common_args(DATASET)
    out = target_dir(DATASET, args.out_dir)

    plan = [
        f"target: {out}",
        "archives:",
        "  - mpii_human_pose_v1.tar.gz (~12 GB, images)",
        "  - mpii_human_pose_v1_u12_2.zip (~100 MB, annotations in MAT)",
        "license: annotations BSD; images research-only (YouTube source rights)",
        f"primary source: {PROJECT_PAGE}",
    ]
    banner_plan(DATASET, plan)

    if not (args.confirm or args.dry_run):
        return 0

    ensure_dir(out, dry_run=args.dry_run)
    digests: dict[str, str] = {}
    for fname, url, expected in ARCHIVES:
        d = download_file(url, out / fname, expected_sha256=expected, dry_run=args.dry_run)
        if d:
            digests[fname] = d

    write_provenance(
        out,
        dataset_name=DATASET,
        source_urls=[PROJECT_PAGE, *(url for _, url, _ in ARCHIVES)],
        archives=digests,
        license_summary=(
            "Annotations: Simplified BSD License — freely available for research purposes. "
            "Images: extracted from YouTube videos; commercial use is not allowed as the "
            "dataset authors do not own image copyright."
        ),
        license_url=PROJECT_PAGE,
        originating_paper=(
            "Andriluka, Pishchulin, Gehler, Schiele. "
            '"2D Human Pose Estimation: New Benchmark and State of the Art Analysis." '
            "CVPR 2014."
        ),
        audit_entry="I. MPII Human Pose",
        notes=(
            "Crowd-sourced human annotation via Amazon Mechanical Turk. 16 keypoints per "
            "person; maps cleanly to our 8-keypoint upper-body head (shoulders, elbows, "
            "wrists, neck, nose subset)."
        ),
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
