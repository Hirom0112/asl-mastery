"""Download Multiview Hand Pose Dataset (Gomez-Donoso, Orts-Escolano, Cazorla).

Audit entry: docs/data/external_datasets_audit.md § C.
License: BSD.

Per ADR 0015, this script will not run without --confirm.

V2 is hosted on Google Drive (manual click-through). V1 is a direct ZIP.
This script downloads V1 automatically and prints instructions for V2.
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


DATASET = "multiview_hand_pose"
PROJECT_PAGE = "https://www.rovit.ua.es/dataset/mhpdataset/"

V1_URL = "http://www.rovit.ua.es/dataset/mhpdataset_v1/multiview_hand_pose_dataset_release.zip"
V2_URL = "https://drive.google.com/file/d/1KWG4c4ieT_4K9Rd7EYdXqH27Py1wRiNk/view?usp=sharing"


def main() -> int:
    args = parse_common_args(DATASET)
    out = target_dir(DATASET, args.out_dir)

    plan = [
        f"target: {out}",
        f"V1 (auto-download): {V1_URL}",
        f"V2 (Google Drive, manual): {V2_URL}",
        "  After downloading V2 manually, place the ZIP at:",
        f"    {out}/mhpdataset_v2.zip",
        "  then re-run this script with --confirm to record its SHA256 in PROVENANCE.md.",
        "license: BSD",
    ]
    banner_plan(DATASET, plan)

    if not (args.confirm or args.dry_run):
        return 0

    ensure_dir(out, dry_run=args.dry_run)
    digests: dict[str, str] = {}

    v1_path = out / "multiview_hand_pose_dataset_release.zip"
    digest = download_file(V1_URL, v1_path, expected_sha256=None, dry_run=args.dry_run)
    if digest:
        digests[v1_path.name] = digest

    v2_path = out / "mhpdataset_v2.zip"
    if v2_path.exists() and not args.dry_run:
        from _common import sha256_file
        v2_digest = sha256_file(v2_path)
        digests[v2_path.name] = v2_digest
        print(f"V2 archive present, SHA256 recorded ({v2_digest[:16]}…)")
    elif args.dry_run:
        print(f"[dry-run] would record V2 SHA256 if {v2_path} exists")
    else:
        print(f"NOTE: {v2_path} not found. Download V2 from {V2_URL} and re-run --confirm.")

    write_provenance(
        out,
        dataset_name=DATASET,
        source_urls=[PROJECT_PAGE, V1_URL, V2_URL],
        archives=digests,
        license_summary='BSD License. Verbatim from project page: "This dataset is publicly available under the BSD License."',
        license_url=PROJECT_PAGE,
        originating_paper=(
            "Gomez-Donoso, Orts-Escolano, Cazorla. "
            '"Large-scale Multiview 3D Hand Pose Dataset." '
            "arXiv:1707.03742 (2017); IVC 2018."
        ),
        audit_entry="C. Multiview Hand Pose Dataset (Gomez-Donoso, Orts-Escolano, Cazorla)",
        notes=(
            "Annotation methodology: 3D joint positions from a Leap Motion Controller "
            "(physical sensor). 2D projections + bounding boxes derived computationally from "
            "supplied multi-camera calibration matrices. NO learned model in the annotation "
            "pipeline — pure physical sensor + projective geometry."
        ),
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
