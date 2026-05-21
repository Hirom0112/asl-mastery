"""Download WIDER FACE (Yang et al., CVPR 2016).

Audit entry: docs/data/external_datasets_audit.md § K.
License: CC BY-NC-ND 4.0.

Per ADR 0015, this script will not run without --confirm.

WIDER FACE images are hosted on Google Drive / Tencent — direct
HTTPS endpoints rotate. This script downloads the annotation ZIP
(stable HTTPS) and prints manual instructions for the image splits.
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


DATASET = "wider_face"
PROJECT_PAGE = "http://shuoyang1213.me/WIDERFACE/"

ANNOTATIONS_URL = "http://shuoyang1213.me/WIDERFACE/support/bbx_annotation/wider_face_split.zip"


def main() -> int:
    args = parse_common_args(DATASET)
    out = target_dir(DATASET, args.out_dir)

    plan = [
        f"target: {out}",
        "automatic:",
        f"  - wider_face_split.zip (annotations, small) from {ANNOTATIONS_URL}",
        "manual (Google Drive / Tencent links from project page):",
        "  - WIDER_train.zip (~1.4 GB)",
        "  - WIDER_val.zip (~360 MB)",
        "  - WIDER_test.zip (~1.8 GB; not strictly needed for our use)",
        f"  download from {PROJECT_PAGE}, place under {out}/, re-run --confirm",
        "license: CC BY-NC-ND 4.0",
    ]
    banner_plan(DATASET, plan)

    if not (args.confirm or args.dry_run):
        return 0

    ensure_dir(out, dry_run=args.dry_run)
    digests: dict[str, str] = {}

    d = download_file(ANNOTATIONS_URL, out / "wider_face_split.zip", expected_sha256=None, dry_run=args.dry_run)
    if d:
        digests["wider_face_split.zip"] = d

    if not args.dry_run:
        from _common import sha256_file
        for archive in sorted(out.glob("WIDER_*.zip")):
            dd = sha256_file(archive)
            digests[archive.name] = dd
            print(f"recorded SHA256 for {archive.name} ({dd[:16]}…)")

    write_provenance(
        out,
        dataset_name=DATASET,
        source_urls=[PROJECT_PAGE, ANNOTATIONS_URL],
        archives=digests,
        license_summary=(
            "CC BY-NC-ND 4.0 (per the HuggingFace mirror's license field). "
            "Non-commercial, no-derivatives. Acceptable for pilot research use; "
            "flag for slice-2 commercial-cliff review."
        ),
        license_url="https://creativecommons.org/licenses/by-nc-nd/4.0/",
        originating_paper=(
            "Yang, Luo, Loy, Tang. "
            '"WIDER FACE: A Face Detection Benchmark." CVPR 2016.'
        ),
        audit_entry="K. WIDER FACE",
        notes=(
            "Manually labeled bounding boxes by dataset curators (per paper §3.1). "
            "Per-face attributes: pose, occlusion level (partial/heavy), blur, expression, "
            "illumination, makeup, event category. 32,203 images / 393,703 faces."
        ),
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
