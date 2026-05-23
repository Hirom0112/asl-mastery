"""Phase 4.9 — scrape additional clips for the 22 vocab signs with <50 clips.

Targets the bottom of the imbalance: bad, blue, happy, woman (currently 5
each), how (17), sister/computer/brother/yesterday/sign (~25), have/pizza/dad/
think/your/mom/bathroom/sorry/grandma/go/tomorrow/my (30-46). Goal: bring
each to ≥100.

This is a SKELETON. It outlines the sources, manifest format, and audit
checkpoints. The actual scraping for each source has its own quirks
(WLASL is a JSON+URL list, MSASL needs Azure auth, ASL Citizen is HF data,
YouTube is youtube-dl + caption filtering). Run each source's section
manually, audit, then ingest.

Sources, ranked by ease:
  1. WLASL 2000 (https://github.com/dxli94/WLASL) — JSON manifest of
     YouTube URLs + start/end times for ~2000 signs. Most of our 22 are
     in there. License: research-only, see WLASL repo.
  2. ASL Citizen (https://huggingface.co/datasets/asl-citizen) — public on
     HF, multi-signer, ~80k clips, 2700 signs.
  3. Lifeprint scrape continuation (existing infra in training/data/
     ingest_handspeak.py). Quality but limited per-sign.
  4. YouTube search continuation (ingest_youtube_search.py). Open-ended,
     rate-limited.

ADR 0015 audit checklist for each source:
  - Public download URL + clear license
  - Signer-disjoint train/test possible (or single-source acceptable)
  - Gloss alignment to our sign_id (use slice1_vocabulary.json gloss field)
  - No fingerspelling-only clips for non-fingerspelled signs
  - Frame trimming consistent (we use 15fps, 32-frame windows)

Output: per-source delta manifest at
  dataset/raw/<source>_thin_signs_manifest.json

Then re-run rebuild_unified_manifest.py to fold these in.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def signs_needing_data(target_per_sign: int = 100,
                       v2_manifest: Path = Path("data/labeled_frames/unified_clip_manifest_modal_v2.json")
                       ) -> dict[str, int]:
    """Return {sign_id: clips_needed} for signs currently below target."""
    m = json.loads(v2_manifest.read_text())
    counts = Counter(c["sign_id"] for c in m["clips"])
    return {sign: max(0, target_per_sign - n)
            for sign, n in counts.items()
            if n < target_per_sign}


def wlasl_scrape_plan(needed: dict[str, int]) -> dict:
    """Match needed signs against WLASL 2000 vocab.

    NOTE: this function returns the PLAN (URLs + counts). The actual
    download is a separate runner because YouTube fetches need cookies/
    rate limits and shouldn't run unattended.

    Steps the runner would take:
      1. git clone https://github.com/dxli94/WLASL.git (or fetch WLASL_v0.3.json)
      2. For each entry, match `gloss` to our sign_id (lowercase, strip _N)
      3. For each match, write to dataset/raw/wlasl/clips/<wlasl_id>.mp4 via
         yt-dlp with the URL + frame_start/end fields
      4. Build a wlasl_manifest.json with {sign_id, gloss, source, url,
         frame_start, frame_end, download_status, local_path}
      5. Run training/data/clean.py on it
      6. Re-run rebuild_unified_manifest.py to fold into the unified manifest

    This function returns the matching plan as a JSON-able dict so the
    runner can be invoked / re-tried per-sign.
    """
    return {
        "source": "wlasl",
        "url": "https://github.com/dxli94/WLASL",
        "next_steps": [
            "1. clone WLASL repo or download WLASL_v0.3.json",
            "2. for each `gloss` in WLASL_v0.3.json:",
            "     normalize: gloss.lower()",
            "     if matches a sign_id in `needed`:",
            "       fetch up to needed[sign_id] clips for that sign",
            "3. download YouTube URLs via yt-dlp (cookie-free works for most)",
            "4. build delta manifest at dataset/raw/wlasl_thin_signs_manifest.json",
            "5. run training/data/clean.py on it",
            "6. re-run scripts/rebuild_unified_manifest.py",
        ],
        "target_signs": needed,
    }


def asl_citizen_plan(needed: dict[str, int]) -> dict:
    """ASL Citizen is on Hugging Face — easier to bulk-fetch."""
    return {
        "source": "asl_citizen",
        "url": "https://huggingface.co/datasets/asl-citizen",
        "next_steps": [
            "1. pip install datasets",
            "2. ds = load_dataset('asl-citizen', split='train')",
            "3. for each row in ds:",
            "     if row['gloss'].lower() matches a sign_id in `needed`:",
            "       save row['video'] to dataset/raw/asl_citizen/clips/<id>.mp4",
            "4. write dataset/raw/asl_citizen_thin_signs_manifest.json",
            "5. run training/data/clean.py + rebuild_unified_manifest.py",
        ],
        "target_signs": needed,
        "estimated_yield": "~50-200 per sign for common signs; ASL Citizen has ~30 clips/sign avg",
    }


def youtube_search_plan(needed: dict[str, int]) -> dict:
    """Continuation of the v2-yt ytsearch scrape that produced
    dataset/clean/v2-yt/normalized_videos. Existing infra in
    training/data/ingest_youtube_search.py."""
    return {
        "source": "youtube_search",
        "next_steps": [
            "1. use training/data/ingest_youtube_search.py with target signs",
            "2. expand search queries per sign: include 'asl', 'sign language', 'sign for'",
            "3. filter results by length (<10s) and presence of single signer",
            "4. captions or video title containing target sign as confirmation",
            "5. write dataset/raw/ytsearch_thin_signs_manifest.json",
        ],
        "target_signs": needed,
        "estimated_yield": "varies; 5-30 per sign before manual review needed",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=100,
                    help="Target clips per sign (default 100)")
    ap.add_argument("--out", type=Path,
                    default=Path("data/phase49_scrape_plan.json"))
    args = ap.parse_args()

    needed = signs_needing_data(target_per_sign=args.target)
    print(f"Signs below {args.target} clips: {len(needed)}")
    print(f"Total clips needed across all thin signs: {sum(needed.values())}")
    print()
    print(f"{'sign':<14} {'needed':>8}")
    print('-' * 24)
    for sign, n in sorted(needed.items(), key=lambda x: -x[1]):
        print(f"  {sign:<12} {n:>8}")

    plan = {
        "target_per_sign": args.target,
        "signs_below_target": needed,
        "total_clips_needed": sum(needed.values()),
        "sources": {
            "wlasl": wlasl_scrape_plan(needed),
            "asl_citizen": asl_citizen_plan(needed),
            "youtube_search": youtube_search_plan(needed),
        },
        "ordering": [
            "1. ASL Citizen (easiest — HF dataset)",
            "2. WLASL 2000 (most coverage)",
            "3. YouTube ytsearch (gap-filler)",
        ],
        "audit_note": "Run each source's ingestion separately, audit under ADR 0015, then merge.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plan, indent=2))
    print(f"\n[plan] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
