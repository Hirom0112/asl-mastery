"""Download InterHand2.6M — human_annot subset (Moon et al., ECCV 2020).

Audit entry: docs/data/external_datasets_audit.md § D.
License: CC-BY-NC 4.0.

Per ADR 0015 default-conservative rule, we download only the human_annot
subset until the machine_annot bootstrap detector's training source is
confirmed.

Per ADR 0015, this script will not run without --confirm. The 5fps split
is recommended (~32 GB images + ~3 GB annotations) unless you have very
large disk budget for the 30fps full split (~700+ GB).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    banner_plan,
    ensure_dir,
    parse_common_args,
    target_dir,
    write_provenance,
)


DATASET = "interhand26m"
PROJECT_PAGE = "https://mks0601.github.io/InterHand2.6M/"


def main() -> int:
    args = parse_common_args(DATASET)
    out = target_dir(DATASET, args.out_dir)

    # InterHand2.6M is distributed via the project page with Google Drive /
    # Dropbox links that rotate and require accepting the LICENSE first.
    # This script will not auto-download — it produces the directory + a
    # README explaining the per-file manual steps + records SHA256 once
    # archives land on disk.

    plan = [
        f"target: {out}",
        "release URLs are on the project page and require accepting the LICENSE first:",
        f"  {PROJECT_PAGE}",
        "what to download:",
        "  - 5fps_batch1.tar.gz + 5fps_batch2.tar.gz + ... (images, ~32 GB total)",
        "  - InterHand2.6M.images.5.fps.v1.0.tar.gz (alternative single-archive form)",
        "  - InterHand2.6M.annotations.5.fps.v1.0.tar.gz (~3 GB annotations)",
        "place every downloaded archive at:",
        f"  {out}/<archive_name>",
        "then re-run with --confirm to record SHA256s in PROVENANCE.md.",
        "USE ONLY human_annot subset per ADR 0015 (audit memo § D notes)",
        "license: CC-BY-NC 4.0",
    ]
    banner_plan(DATASET, plan)

    if not (args.confirm or args.dry_run):
        return 0

    ensure_dir(out, dry_run=args.dry_run)

    digests: dict[str, str] = {}
    if not args.dry_run:
        from _common import sha256_file
        for archive in sorted(out.glob("*.tar.gz")) + sorted(out.glob("*.tar")):
            d = sha256_file(archive)
            digests[archive.name] = d
            print(f"recorded SHA256 for {archive.name} ({d[:16]}…)")
        if not digests:
            print(
                f"NOTE: no archives found in {out}. Download from {PROJECT_PAGE} "
                "(LICENSE accept required), place them in this directory, and re-run --confirm."
            )

    write_provenance(
        out,
        dataset_name=DATASET,
        source_urls=[PROJECT_PAGE],
        archives=digests,
        license_summary=(
            "CC-BY-NC 4.0 (per facebookresearch/InterHand2.6M LICENSE file). "
            "Non-commercial use only; attribution required; can adapt and remix."
        ),
        license_url="https://creativecommons.org/licenses/by-nc/4.0/",
        originating_paper=(
            "Moon, Yu, Wen, Shiratori, Lee. "
            '"InterHand2.6M: A Dataset and Baseline for 3D Interacting Hand Pose Estimation '
            'from a Single RGB Image." ECCV 2020. arXiv:2008.09309.'
        ),
        audit_entry="D. InterHand2.6M (Moon et al., ECCV 2020 — Meta / Facebook Research)",
        notes=(
            "USE ONLY the human_annot/ subset per ADR 0015. The machine_annot/ subset was "
            "produced by an automated detector the authors trained; until that detector's "
            "training source is confirmed to be only human_annot data, the default-conservative "
            "rule applies. "
            "Recommend the 5fps split unless disk budget supports 30fps (~700+ GB). "
            "After download, the dataset loader should filter for human_annot in the per-frame "
            "annotation key."
        ),
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
