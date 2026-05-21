"""Download COCO 2017 images + COCO-WholeBody annotations (Jin et al., ECCV 2020).

Audit entry: docs/data/external_datasets_audit.md § E.
License: annotations CC BY 4.0; images under Flickr Terms.

Per ADR 0015, this script will not run without --confirm.

Two layers:
- COCO 2017 train + val images (~19 GB) — base imagery
- COCO-WholeBody annotation JSON (~150 MB) — adds 133 keypoints per person
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


DATASET = "coco_wholebody"
ANNOTATIONS_REPO = "https://github.com/jin-s13/COCO-WholeBody"

# COCO 2017 images + standard COCO annotations
COCO_TRAIN_IMAGES = "http://images.cocodataset.org/zips/train2017.zip"
COCO_VAL_IMAGES = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANNOTATIONS = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

# COCO-WholeBody extra annotations (released on Google Drive / OneDrive per the
# repo README — the JSON file is not on a stable HTTPS endpoint, so we treat
# it like InterHand2.6M and require manual placement).
WHOLEBODY_ANNOT_NAME = "coco_wholebody_train_v1.0.json"


def main() -> int:
    args = parse_common_args(DATASET)
    out = target_dir(DATASET, args.out_dir)

    plan = [
        f"target: {out}",
        "automatic downloads:",
        f"  - train2017.zip (~18 GB) from {COCO_TRAIN_IMAGES}",
        f"  - val2017.zip (~1 GB) from {COCO_VAL_IMAGES}",
        f"  - annotations_trainval2017.zip (~250 MB) from {COCO_ANNOTATIONS}",
        "manual step:",
        f"  - download {WHOLEBODY_ANNOT_NAME} (~150 MB) + corresponding val JSON",
        f"    from the README of {ANNOTATIONS_REPO}",
        f"  - place at {out}/{WHOLEBODY_ANNOT_NAME}",
        "  - then re-run with --confirm to record SHA256",
        "license: annotations CC BY 4.0 (whole-body) and CC BY 4.0 (COCO); images Flickr Terms",
    ]
    banner_plan(DATASET, plan)

    if not (args.confirm or args.dry_run):
        return 0

    ensure_dir(out, dry_run=args.dry_run)
    digests: dict[str, str] = {}

    for fname, url in [
        ("train2017.zip", COCO_TRAIN_IMAGES),
        ("val2017.zip", COCO_VAL_IMAGES),
        ("annotations_trainval2017.zip", COCO_ANNOTATIONS),
    ]:
        d = download_file(url, out / fname, expected_sha256=None, dry_run=args.dry_run)
        if d:
            digests[fname] = d

    if not args.dry_run:
        from _common import sha256_file
        for extra in out.glob("coco_wholebody_*.json"):
            d = sha256_file(extra)
            digests[extra.name] = d
            print(f"recorded SHA256 for {extra.name} ({d[:16]}…)")

    write_provenance(
        out,
        dataset_name=DATASET,
        source_urls=[
            ANNOTATIONS_REPO,
            COCO_TRAIN_IMAGES,
            COCO_VAL_IMAGES,
            COCO_ANNOTATIONS,
        ],
        archives=digests,
        license_summary=(
            "Annotations: CC BY 4.0 (both COCO base and COCO-WholeBody extension). "
            "Images: Flickr Terms of Use — academic use OK; commercial use of the dataset "
            "as a whole is restricted by individual image rights. Slice-2 commercial cliff "
            "applies (see ADR 0015 § Consequences)."
        ),
        license_url="https://creativecommons.org/licenses/by/4.0/",
        originating_paper=(
            "Jin, Xu, Xu, Wang, Liu, Qian, Ouyang, Luo. "
            '"Whole-Body Human Pose Estimation in the Wild." '
            "ECCV 2020. arXiv:2007.11858."
        ),
        audit_entry="E. COCO-WholeBody",
        notes=(
            "First whole-body benchmark with fully manual annotations (per paper abstract). "
            "21-per-hand topology; validity flag filters out unclear hands. ~40K hand instances "
            "with validity=True. Hand keypoint topology may differ slightly from FreiHAND / CMU; "
            "verify mapping before joint training."
        ),
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
