"""Ingest MS-ASL clips for the 96 glosses in docs/VOCABULARY.md.

Phase 3d secondary slice-1 source. Currently a stub: MS-ASL class
manifests are gated behind a Microsoft Research license-acceptance
flow that must be completed manually before this script can run.

Once the license is accepted and the class manifest JSONs
(`MSASL_train.json`, `MSASL_val.json`, `MSASL_test.json`) are
downloaded, this script mirrors the structure of
`ingest_wlasl.py` — yt-dlp the URLs, log 404/blocked,
emit a manifest with per-sign downloadable counts.

Usage (once the license JSONs are in hand):

    python -m training.data.ingest_msasl \\
        --msasl-dir path/to/msasl_jsons/ \\
        --output dataset/raw/msasl \\
        --manifest dataset/raw/msasl_manifest.json

Until then, running this prints instructions and exits non-zero.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

log = logging.getLogger("ingest_msasl")


LICENSE_INSTRUCTIONS = """
MS-ASL ingestion requires a Microsoft Research license. To run:

  1. Visit https://www.microsoft.com/en-us/research/project/ms-asl/
     and accept the license terms.
  2. Download MSASL_train.json, MSASL_val.json, MSASL_test.json into
     a local directory.
  3. Re-run this command with --msasl-dir pointing at that
     directory.

The license terms must be recorded in docs/DATASET.md §6 once
acceptance is complete; until that happens, ingestion is gated.
"""


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--msasl-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("dataset/raw/msasl"))
    parser.add_argument("--manifest", type=Path, default=Path("dataset/raw/msasl_manifest.json"))
    args = parser.parse_args()

    if not args.msasl_dir or not args.msasl_dir.exists():
        log.error(LICENSE_INSTRUCTIONS)
        return 2

    # TODO(phase 3d): mirror ingest_wlasl.py's structure once the
    # license JSONs are committed. The MS-ASL JSON schema differs
    # slightly from WLASL (per-clip `url`, `start_time`,
    # `end_time`, `signer` fields). For now we exit so the manifest
    # stays absent and filter_vocabulary.py degrades gracefully.
    log.error("MS-ASL ingestion not yet implemented; pending manifest schema parsing.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
