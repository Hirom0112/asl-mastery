"""Scrape canonical per-sign videos from Lifeprint.com (and HandSpeak fallback).

Lifeprint is the curriculum we aligned our 96-sign vocabulary to
(per ADR 0004), so its per-sign pages are the highest-quality
*curated* reference clips we can get without a Deaf-instructor
engagement. Each Lifeprint page now embeds a single YouTube video
(Bill Vicars's channel); the script extracts that YouTube ID and
downloads via yt-dlp the same way ingest_wlasl.py does.

The script is intended to **augment** WLASL after it finishes. Each
Lifeprint clip is exactly one Deaf-authored canonical reference per
sign, so it raises per-sign clip count by 1 across most of the
vocabulary. Combined with WLASL it pushes the average clips/sign
meaningfully above the ADR-0008 floor.

Usage:

    python -m training.data.ingest_handspeak \\
        --output dataset/raw/handspeak \\
        --manifest dataset/raw/handspeak_manifest.json

The manifest format matches ingest_wlasl.py's so filter_vocabulary.py
sums per-sign counts across sources without code changes.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.data.vocabulary import get_all  # noqa: E402

log = logging.getLogger("ingest_handspeak")

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.0 Safari/605.1.15"
)
THROTTLE_SEC = 1.0


@dataclass
class ClipRecord:
    sign_id: str
    gloss: str
    source: str  # "handspeak" or "lifeprint"
    source_url: str
    original_url: str  # the video URL itself
    source_signer_id: str | None
    source_split: str | None
    frame_start: int | None
    frame_end: int | None
    local_path: str | None
    download_status: str  # "ok" | "404" | "no-video-found" | "other-error"
    wlasl_video_id: str = ""  # kept for manifest shape compat with WLASL


def _fetch(url: str, timeout: float = 20.0) -> str | None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        log.debug("http %s on %s", e.code, url)
        return None
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        log.debug("fetch failed %s: %s", url, e)
        return None


def _download(url: str, out_path: Path, timeout: float = 60.0) -> bool:
    if out_path.exists() and out_path.stat().st_size > 0:
        return True
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
            if not data:
                return False
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = out_path.with_suffix(out_path.suffix + ".tmp")
            with tmp.open("wb") as f:
                f.write(data)
            tmp.replace(out_path)
            return True
    except Exception as e:
        log.debug("download failed %s: %s", url, e)
        return False


YOUTUBE_EMBED_RE = re.compile(
    r"""youtube\.com/embed/([A-Za-z0-9_-]{11})""", re.IGNORECASE
)


def _extract_youtube_ids(html: str) -> list[str]:
    """Find every YouTube embed video ID on the page."""
    return list(dict.fromkeys(YOUTUBE_EMBED_RE.findall(html)))


def _lifeprint_url_candidates(sign_id: str, gloss: str, lesson: str) -> list[str]:
    """Lifeprint has per-sign pages under /pages-signs/<letter>/<word>.htm.

    Slug rules are inconsistent across Lifeprint's history; we try a
    few variants per sign before falling back to the lesson page.
    """
    out: list[str] = []
    first = gloss[0].lower()
    base_glosses = {gloss.lower(), sign_id, gloss.lower().replace("-", "")}
    # Handle vocabulary aliases — words where our slug differs from
    # Lifeprint's URL (commonly the canonical full word vs informal).
    aliases = {
        "mom": ["mother"],
        "dad": ["father"],
        "grandma": ["grandmother"],
        "thank_you": ["thankyou", "thank-you"],
        "they": ["theythem", "they-them"],
        "his": ["hers", "his-her"],
        "our": ["our-ours"],
    }
    if sign_id in aliases:
        base_glosses.update(aliases[sign_id])
    for slug in base_glosses:
        out.append(f"https://www.lifeprint.com/asl101/pages-signs/{first}/{slug}.htm")
    return out


def _record(
    sign_id: str,
    gloss: str,
    source: str,
    source_url: str,
    original_url: str,
    local_path: Path | None,
    status: str,
) -> ClipRecord:
    return ClipRecord(
        sign_id=sign_id,
        gloss=gloss,
        source=source,
        source_url=source_url,
        original_url=original_url,
        source_signer_id=None,
        source_split=None,
        frame_start=None,
        frame_end=None,
        local_path=str(local_path) if local_path else None,
        download_status=status,
    )


def _yt_dlp_download(video_id: str, out_path: Path, cookies_from_browser: str | None) -> bool:
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
    import subprocess  # local import to keep top-level deps minimal

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        return proc.returncode == 0 and out_path.exists()
    except Exception:
        return False


def ingest(
    output_dir: Path,
    manifest_path: Path,
    cookies_from_browser: str | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    items = get_all()
    records: list[ClipRecord] = []
    per_sign_attempts: dict[str, int] = defaultdict(int)
    per_sign_ok: dict[str, int] = defaultdict(int)

    for idx, item in enumerate(items, 1):
        log.info("(%d/%d) %s", idx, len(items), item.gloss)
        urls_to_try = _lifeprint_url_candidates(item.sign_id, item.gloss, item.lifeprint_lesson)

        # Find the first Lifeprint page that exists for this sign, then
        # collect its YouTube embeds. Lifeprint pages typically embed
        # 1-2 YouTube videos per sign (a primary and sometimes a slow-
        # motion variant).
        page_url_found: str | None = None
        youtube_ids: list[str] = []
        for page_url in urls_to_try:
            time.sleep(THROTTLE_SEC)
            per_sign_attempts[item.sign_id] += 1
            html = _fetch(page_url)
            if html is None:
                records.append(
                    _record(item.sign_id, item.gloss, "lifeprint", page_url, "", None, "404")
                )
                continue
            page_url_found = page_url
            youtube_ids = _extract_youtube_ids(html)
            break  # found the page; stop trying other slug variants

        if not page_url_found:
            continue  # all candidates were 404; logged already

        if not youtube_ids:
            records.append(
                _record(
                    item.sign_id,
                    item.gloss,
                    "lifeprint",
                    page_url_found,
                    "",
                    None,
                    "no-video-found",
                )
            )
            continue

        # Download each YouTube embed via yt-dlp.
        for vid in youtube_ids[:3]:  # cap at 3 per sign
            per_sign_attempts[item.sign_id] += 1
            local = output_dir / f"{item.sign_id}__lifeprint__{vid}.mp4"
            ok = _yt_dlp_download(vid, local, cookies_from_browser)
            video_url = f"https://www.youtube.com/watch?v={vid}"
            if ok:
                per_sign_ok[item.sign_id] += 1
                records.append(
                    _record(
                        item.sign_id,
                        item.gloss,
                        "lifeprint",
                        page_url_found,
                        video_url,
                        local,
                        "ok",
                    )
                )
            else:
                records.append(
                    _record(
                        item.sign_id,
                        item.gloss,
                        "lifeprint",
                        page_url_found,
                        video_url,
                        None,
                        "other-error",
                    )
                )

    manifest = {
        "source": "lifeprint",
        "per_sign_attempts": dict(per_sign_attempts),
        "per_sign_downloadable": dict(per_sign_ok),
        "records": [asdict(r) for r in records],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    log.info(
        "manifest: %d records, %d downloaded across %d signs",
        len(records),
        sum(per_sign_ok.values()),
        len([s for s, n in per_sign_ok.items() if n > 0]),
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("dataset/raw/handspeak"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("dataset/raw/handspeak_manifest.json")
    )
    parser.add_argument(
        "--cookies-from-browser",
        type=str,
        default=None,
        help="Pass through to yt-dlp for the YouTube downloads.",
    )
    args = parser.parse_args()
    ingest(args.output, args.manifest, args.cookies_from_browser)
    return 0


if __name__ == "__main__":
    sys.exit(main())
