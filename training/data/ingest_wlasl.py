"""Ingest WLASL clips for the 96 glosses in docs/VOCABULARY.md.

Phase 3d of docs/ROADMAP.md, primary slice-1 data source under
ADR 0008. Downloads each WLASL `instances[].url` (YouTube clips
mostly), skips ones that 404 or are blocked, records the actual
downloadable count per sign in a JSON manifest. That manifest
is the input to `filter_vocabulary.py`, which applies ADR 0008's
per-sign clip-count floor.

Usage:

    python -m training.data.ingest_wlasl \\
        --wlasl-json path/to/WLASL_v0.3.json \\
        --output dataset/raw/wlasl \\
        --manifest dataset/raw/wlasl_manifest.json

WLASL source:
    https://github.com/dxli94/WLASL (Li, Rodriguez Opazo, Yu, Li
    2020 WACV). The `WLASL_v0.3.json` we use is committed to the
    upstream repo's `start_kit/` directory.

Provenance recorded per clip in the manifest:
    - source: "wlasl"
    - wlasl_video_id (the entry's `video_id`)
    - original_url (`url` field)
    - source_signer_id (`signer_id` field if present; many WLASL
      entries lack this, in which case we leave it null and the
      cleaning pipeline's signer-disjoint split treats each clip
      as its own signer — conservative)
    - source_split (WLASL's own `split` field, kept for reference;
      our slice-1 split is reassigned in clean.py)
    - frame_start, frame_end (the `frame_start`/`frame_end` fields
      that annotate the sign window within the source video)

This script does NOT extract MediaPipe keypoints. That happens in
`training/data/clean.py` (Phase 3f) after the per-sign filter.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Local import; only used here for the gloss list.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.data.vocabulary import SIGN_IDS, load_vocabulary  # noqa: E402

log = logging.getLogger("ingest_wlasl")


@dataclass
class ClipRecord:
    sign_id: str
    gloss: str
    source: str  # "wlasl"
    wlasl_video_id: str
    original_url: str
    source_signer_id: str | None
    source_split: str | None
    frame_start: int | None
    frame_end: int | None
    local_path: str | None  # null if download failed
    download_status: str  # "ok" | "404" | "blocked" | "skipped" | "other-error"


def _slug(gloss: str) -> str:
    """Match the slug rule used in supabase/migrations/...seed_vocabulary.sql."""
    return gloss.lower().replace("-", "_")


def _wlasl_gloss_to_our_sign_id(wlasl_gloss: str, alias_map: dict[str, str]) -> str | None:
    """Map a WLASL gloss (lowercase, sometimes hyphenated) to our sign_id.

    Most of our glosses are direct: WLASL "thank you" → "thank_you".
    A few use ASL-LEX aliases — those live in alias_map and were
    captured during vocabulary build.
    """
    g = wlasl_gloss.strip().lower()
    # Direct match path: WLASL gloss as snake_case.
    direct = re.sub(r"[^a-z0-9]+", "_", g).strip("_")
    if direct in SIGN_IDS:
        return direct
    aliased = alias_map.get(direct)
    if aliased and aliased in SIGN_IDS:
        return aliased
    return None


# WLASL glosses that differ in surface form from our slugs.
ALIAS_MAP: dict[str, str] = {
    "thanks": "thank_you",
    "thank_you": "thank_you",
    "mother": "mom",
    "father": "dad",
    "grandmother": "grandma",
    "what_1": "what",
}


def _try_yt_dlp_download(url: str, out_dir: Path, video_id: str) -> tuple[str, Path | None]:
    """Attempt to download via yt-dlp. Returns (status, local_path)."""
    out_path = out_dir / f"{video_id}.mp4"
    if out_path.exists():
        return "ok", out_path

    cmd = [
        "yt-dlp",
        "--quiet",
        "--no-warnings",
        "--no-playlist",
        "--format",
        "bestvideo[ext=mp4][height<=720]+bestaudio/best[ext=mp4]/best",
        "--merge-output-format",
        "mp4",
        "-o",
        str(out_path),
        url,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except FileNotFoundError:
        log.error("yt-dlp not installed. Run `pip install yt-dlp` (or `pipx install yt-dlp`).")
        sys.exit(2)
    except subprocess.TimeoutExpired:
        return "other-error", None

    if proc.returncode == 0 and out_path.exists():
        return "ok", out_path

    stderr = (proc.stderr or "").lower()
    if "video unavailable" in stderr or "removed by the user" in stderr or "404" in stderr:
        return "404", None
    if "private" in stderr or "blocked" in stderr or "age" in stderr:
        return "blocked", None
    return "other-error", None


def ingest(
    wlasl_json: Path,
    output_dir: Path,
    manifest_path: Path,
    dry_run: bool,
) -> None:
    log.info("loading WLASL JSON from %s", wlasl_json)
    with wlasl_json.open() as f:
        entries: list[dict[str, Any]] = json.load(f)
    log.info("WLASL has %d glosses; filtering to our 96", len(entries))

    output_dir.mkdir(parents=True, exist_ok=True)

    load_vocabulary()  # populates GLOSS_TO_ID if not already

    records: list[ClipRecord] = []
    per_sign_attempts: dict[str, int] = defaultdict(int)
    per_sign_ok: dict[str, int] = defaultdict(int)

    for entry in entries:
        wlasl_gloss = entry.get("gloss", "")
        sign_id = _wlasl_gloss_to_our_sign_id(wlasl_gloss, ALIAS_MAP)
        if not sign_id:
            continue
        instances = entry.get("instances", [])
        for inst in instances:
            video_id = str(inst.get("video_id", ""))
            url = inst.get("url", "")
            if not video_id or not url:
                continue
            per_sign_attempts[sign_id] += 1

            if dry_run:
                status, local = "skipped", None
            else:
                status, local = _try_yt_dlp_download(url, output_dir, video_id)
                if status == "ok":
                    per_sign_ok[sign_id] += 1

            records.append(
                ClipRecord(
                    sign_id=sign_id,
                    gloss=wlasl_gloss,
                    source="wlasl",
                    wlasl_video_id=video_id,
                    original_url=url,
                    source_signer_id=str(inst.get("signer_id")) if inst.get("signer_id") else None,
                    source_split=inst.get("split"),
                    frame_start=inst.get("frame_start"),
                    frame_end=inst.get("frame_end"),
                    local_path=str(local) if local else None,
                    download_status=status,
                )
            )

    # Sanity check: every gloss in our vocab should have appeared at
    # least once in the WLASL filter. If not, we either missed an
    # alias or WLASL itself lacks the sign — surface it.
    missing = [sid for sid in GLOSS_TO_ID.values() if per_sign_attempts[sid] == 0]
    if missing:
        log.warning("%d signs have zero WLASL attempts: %s", len(missing), missing)

    manifest = {
        "source": "wlasl",
        "wlasl_json": str(wlasl_json),
        "dry_run": dry_run,
        "per_sign_attempts": dict(per_sign_attempts),
        "per_sign_downloadable": dict(per_sign_ok),
        "records": [asdict(r) for r in records],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    log.info(
        "wrote manifest with %d records (%d downloaded, %d attempted)",
        len(records),
        sum(per_sign_ok.values()),
        sum(per_sign_attempts.values()),
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wlasl-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("dataset/raw/wlasl"))
    parser.add_argument("--manifest", type=Path, default=Path("dataset/raw/wlasl_manifest.json"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip downloads; only build the manifest of attempts. Useful to enumerate per-sign counts before committing to a long yt-dlp run.",
    )
    args = parser.parse_args()
    ingest(args.wlasl_json, args.output, args.manifest, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
