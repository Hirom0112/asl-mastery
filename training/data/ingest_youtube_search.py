"""Supplement WLASL/Lifeprint by searching YouTube for per-sign tutorial videos.

Used to top up signs that came out thin after the curated sources. We
query yt-dlp's `ytsearch:` provider for `"ASL <sign> Bill Vicars"`
(plus a generic `"ASL sign for <word>"` fallback) and download the
top N matches under 60 seconds.

The query quality bar is modest — these are not curated. The
cleaning pipeline's MediaPipe extraction filters out clips that
don't have detected hands, which removes the worst false positives.
Per-sign filtering with the ADR-0008 floor then handles the rest.

Usage:

    python -m training.data.ingest_youtube_search \\
        --output dataset/raw/ytsearch \\
        --manifest dataset/raw/ytsearch_manifest.json \\
        --cookies-from-browser chrome \\
        --per-sign 5 \\
        --only-thin-signs dataset/raw/wlasl_manifest.json:dataset/raw/lifeprint_manifest.json
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.data.vocabulary import get_all  # noqa: E402

log = logging.getLogger("ingest_youtube_search")

# Each entry: (query template, max-results requested from YouTube). The
# Bill Vicars query is tried first because his videos are short,
# tightly framed, and high-quality.
QUERY_TEMPLATES = [
    "how to sign {gloss} in ASL",
    "ASL sign for {gloss}",
    "ASL {gloss} tutorial",
]


@dataclass
class ClipRecord:
    sign_id: str
    gloss: str
    source: str
    wlasl_video_id: str  # we reuse this column to store the YT video id
    original_url: str
    source_signer_id: str | None
    source_split: str | None
    frame_start: int | None
    frame_end: int | None
    local_path: str | None
    download_status: str
    search_query: str = ""


def _yt_search(query: str, n: int, cookies_from_browser: str | None) -> list[dict]:
    """Return up to n entries from yt-dlp's ytsearch provider for the query.

    Each entry: {"id": "...", "title": "...", "duration": float}.
    """
    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--quiet",
        "--no-warnings",
        "--dump-json",
        "--flat-playlist",
        "--default-search",
        "ytsearch",
        f"ytsearch{n}:{query}",
    ]
    if cookies_from_browser:
        cmd[3:3] = ["--cookies-from-browser", cookies_from_browser]
        cmd[3:3] = ["--remote-components", "ejs:github"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return []
    if proc.returncode != 0:
        return []
    entries: list[dict] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            j = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "id" in j:
            entries.append(j)
    return entries


def _yt_download(
    video_id: str, out_path: Path, cookies_from_browser: str | None
) -> bool:
    if out_path.exists():
        return True
    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--quiet",
        "--no-warnings",
        "--no-playlist",
        "--format",
        "bestvideo[ext=mp4][height<=720]+bestaudio/best[ext=mp4]/best",
        "--merge-output-format",
        "mp4",
        "-o",
        str(out_path),
        f"https://www.youtube.com/watch?v={video_id}",
    ]
    if cookies_from_browser:
        cmd[3:3] = ["--cookies-from-browser", cookies_from_browser]
        cmd[3:3] = ["--remote-components", "ejs:github"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        return proc.returncode == 0 and out_path.exists()
    except Exception:
        return False


def _load_thin_signs(manifest_paths: list[Path], min_count: int) -> set[str]:
    """Return signs whose combined per-source downloadable count is below
    `min_count`. Empty set if no manifests supplied (treat every sign as thin)."""
    if not manifest_paths:
        return {item.sign_id for item in get_all()}
    counts: dict[str, int] = defaultdict(int)
    for p in manifest_paths:
        if not p.exists():
            continue
        with p.open() as f:
            m = json.load(f)
        for sid, n in m.get("per_sign_downloadable", {}).items():
            counts[sid] += n
    thin: set[str] = set()
    for item in get_all():
        if counts.get(item.sign_id, 0) < min_count:
            thin.add(item.sign_id)
    return thin


def ingest(
    output_dir: Path,
    manifest_path: Path,
    cookies_from_browser: str | None,
    per_sign: int,
    only_thin: list[Path],
    thin_threshold: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    items = get_all()

    thin_signs = _load_thin_signs(only_thin, thin_threshold) if only_thin else set(
        item.sign_id for item in items
    )
    log.info("targeting %d thin signs (threshold %d)", len(thin_signs), thin_threshold)

    records: list[ClipRecord] = []
    per_sign_attempts: dict[str, int] = defaultdict(int)
    per_sign_ok: dict[str, int] = defaultdict(int)

    for item in items:
        if item.sign_id not in thin_signs:
            continue
        downloaded = 0
        seen_ids: set[str] = set()
        for q_tmpl in QUERY_TEMPLATES:
            if downloaded >= per_sign:
                break
            q = q_tmpl.format(gloss=item.gloss.replace("-", " ").lower())
            log.info("[%s] querying: %s", item.sign_id, q)
            entries = _yt_search(q, per_sign * 2, cookies_from_browser)
            for entry in entries:
                if downloaded >= per_sign:
                    break
                vid = entry.get("id")
                if not vid or vid in seen_ids:
                    continue
                seen_ids.add(vid)
                per_sign_attempts[item.sign_id] += 1
                duration = entry.get("duration")
                if duration and duration > 60:
                    continue  # skip long videos — too noisy for one-sign training
                local = output_dir / f"{item.sign_id}__ytsearch__{vid}.mp4"
                ok = _yt_download(vid, local, cookies_from_browser)
                url = f"https://www.youtube.com/watch?v={vid}"
                if ok:
                    downloaded += 1
                    per_sign_ok[item.sign_id] += 1
                    status = "ok"
                else:
                    status = "other-error"
                records.append(
                    ClipRecord(
                        sign_id=item.sign_id,
                        gloss=item.gloss,
                        source="ytsearch",
                        wlasl_video_id=vid,
                        original_url=url,
                        source_signer_id=None,
                        source_split=None,
                        frame_start=None,
                        frame_end=None,
                        local_path=str(local) if ok else None,
                        download_status=status,
                        search_query=q,
                    )
                )

    manifest = {
        "source": "ytsearch",
        "per_sign_attempts": dict(per_sign_attempts),
        "per_sign_downloadable": dict(per_sign_ok),
        "records": [asdict(r) for r in records],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    log.info(
        "ytsearch: %d records, %d downloaded across %d signs",
        len(records),
        sum(per_sign_ok.values()),
        len([s for s, n in per_sign_ok.items() if n > 0]),
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("dataset/raw/ytsearch"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("dataset/raw/ytsearch_manifest.json")
    )
    parser.add_argument("--cookies-from-browser", type=str, default=None)
    parser.add_argument("--per-sign", type=int, default=5)
    parser.add_argument("--thin-threshold", type=int, default=15)
    parser.add_argument(
        "--only-thin-signs",
        type=str,
        default="",
        help="Colon-separated list of manifest paths whose per_sign_downloadable counts are summed; only signs below thin-threshold are queried.",
    )
    args = parser.parse_args()
    only_thin = [Path(p) for p in args.only_thin_signs.split(":") if p]
    ingest(
        args.output,
        args.manifest,
        args.cookies_from_browser,
        args.per_sign,
        only_thin,
        args.thin_threshold,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
