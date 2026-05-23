"""Targeted ytsearch for the still-thin slice1 signs (post v3+msasl+lifeprint).

Sidesteps the existing training/data/ingest_youtube_search.py which iterates
the 96-sign vocabulary; here we read the slice1 80-sign vocab + the live
aggregate counts and only scrape signs that are still below `--target`.

Per-query: yt-dlp's `ytsearch<N>:` provider with a small set of high-quality
queries per sign ("ASL <SIGN> Bill Vicars", "ASL <SIGN> sign language",
"how to sign <SIGN> in ASL"). Downloads clips under 60s. Cookies-from-browser
required to avoid YouTube bot challenge.

Usage:
    training/.venv/bin/python -m scripts.ytsearch_thin_signs \\
        --target 105 --per-sign 12 --cookies-from-browser chrome
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


log = logging.getLogger("ytsearch_thin")

QUERIES_TEMPLATES = [
    'ASL {sign} Bill Vicars',
    'ASL {sign} sign language',
    'how to sign {sign} in ASL',
    '{sign} ASL dictionary',
]


def _ytsearch_list(query: str, n: int, cookies_browser: str | None) -> list[dict]:
    """Return up to n entries from yt-dlp's ytsearch provider."""
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--quiet", "--no-warnings", "--no-playlist",
        "--dump-json", "--flat-playlist",
        "--match-filter", "duration < 60",
        f"ytsearch{n}:{query}",
    ]
    if cookies_browser:
        cmd[3:3] = ["--cookies-from-browser", cookies_browser]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return []
    entries = []
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def _download(url: str, sign: str, video_id: str, out_dir: Path,
              cookies_browser: str | None) -> tuple[str, Path | None]:
    out_path = out_dir / f"{sign}__ytsearch__{video_id}.mp4"
    if out_path.exists() and out_path.stat().st_size > 1024:
        return "ok", out_path
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--quiet", "--no-warnings", "--no-playlist",
        "--format", "bestvideo[ext=mp4][height<=720]+bestaudio/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "--match-filter", "duration < 60",
        "-o", str(out_path),
    ]
    if cookies_browser:
        cmd[3:3] = ["--cookies-from-browser", cookies_browser]
    cmd.append(url)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        return "timeout", None
    if proc.returncode == 0 and out_path.exists() and out_path.stat().st_size > 1024:
        return "ok", out_path
    stderr = (proc.stderr or "").lower()
    if "video unavailable" in stderr or "removed by the user" in stderr or "404" in stderr:
        return "404", None
    if "sign in to confirm" in stderr or "not a bot" in stderr:
        return "bot-challenge", None
    if "private" in stderr or "blocked" in stderr or "age" in stderr:
        return "blocked", None
    if "filters do not match" in stderr or "skipping" in stderr:
        return "filter-skipped", None
    return "other-error", None


def _scrape_sign(sign: str, needed: int, per_sign_cap: int, out_dir: Path,
                 cookies_browser: str | None) -> dict:
    """Scrape up to `min(needed, per_sign_cap)` clips for this sign."""
    target = min(needed, per_sign_cap)
    log.info("[%s] target=%d clips", sign, target)
    ok_records: list[dict] = []
    other_records: list[dict] = []
    seen_ids: set[str] = set()
    for template in QUERIES_TEMPLATES:
        if len(ok_records) >= target:
            break
        query = template.format(sign=sign.upper())
        log.info("  [%s] query: %s", sign, query)
        entries = _ytsearch_list(query, target * 2, cookies_browser)
        for ent in entries:
            if len(ok_records) >= target:
                break
            vid = ent.get("id")
            if not vid or vid in seen_ids:
                continue
            seen_ids.add(vid)
            url = ent.get("url") or f"https://www.youtube.com/watch?v={vid}"
            status, local = _download(url, sign, vid, out_dir, cookies_browser)
            rec = {
                "sign_id": sign, "source": "ytsearch_v2", "wlasl_video_id": vid,
                "original_url": url, "source_signer_id": None,
                "source_split": None, "frame_start": None, "frame_end": None,
                "local_path": str(local) if local else None,
                "download_status": status,
                "query": query,
            }
            if status == "ok":
                ok_records.append(rec)
            else:
                other_records.append(rec)
    log.info("[%s] done: %d ok, %d failed", sign, len(ok_records), len(other_records))
    return {"sign": sign, "ok": ok_records, "other": other_records}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--v3-manifest", type=Path,
                    default=Path("data/labeled_frames/unified_clip_manifest_modal_v3.json"))
    ap.add_argument("--msasl-manifest", type=Path,
                    default=Path("dataset/raw/msasl_thin_signs_manifest.json"))
    ap.add_argument("--lifeprint-manifest", type=Path,
                    default=Path("dataset/raw/lifeprint_manifest_v2.json"))
    ap.add_argument("--vocab-json", type=Path,
                    default=Path("dataset/slice1_vocabulary.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("dataset/raw/ytsearch_v2"))
    ap.add_argument("--manifest", type=Path,
                    default=Path("dataset/raw/ytsearch_v2_manifest.json"))
    ap.add_argument("--target", type=int, default=105,
                    help="Per-sign target clip count (only scrape signs below this).")
    ap.add_argument("--per-sign", type=int, default=15,
                    help="Max clips to download per sign in this run.")
    ap.add_argument("--cookies-from-browser", type=str, default="chrome")
    ap.add_argument("--workers", type=int, default=4,
                    help="Sign-level parallelism (each sign serializes its yt-dlp calls).")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Build aggregate per-sign counts from v3 + msasl + lifeprint
    v3 = json.loads(args.v3_manifest.read_text())
    v3c = Counter(c["sign_id"] for c in v3["clips"])
    msc = Counter(r["sign_id"] for r in json.loads(args.msasl_manifest.read_text())["records"]
                  if r.get("download_status") == "ok")
    lpc = Counter(r["sign_id"] for r in json.loads(args.lifeprint_manifest.read_text())["records"]
                  if r.get("download_status") == "ok")
    vocab = json.loads(args.vocab_json.read_text())
    sign_ids = {s["sign_id"] for s in vocab["kept_signs"]}
    aggregate = {s: v3c.get(s, 0) + msc.get(s, 0) + lpc.get(s, 0) for s in sign_ids}
    needed_by_sign = {s: max(0, args.target - n) for s, n in aggregate.items() if n < args.target}
    log.info("aggregate < target=%d: %d signs need %d additional clips total",
             args.target, len(needed_by_sign), sum(needed_by_sign.values()))
    if not needed_by_sign:
        log.info("no scrape needed; all signs already ≥ target")
        return 0
    for s, n in sorted(needed_by_sign.items(), key=lambda x: -x[1]):
        log.info("  %s = %d  (need +%d)", s, aggregate[s], n)

    t0 = time.time()
    all_records: list[dict] = []
    summary_by_sign: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_scrape_sign, s, n, args.per_sign,
                          args.out_dir, args.cookies_from_browser): s
                for s, n in needed_by_sign.items()}
        for fut in as_completed(futs):
            res = fut.result()
            all_records.extend(res["ok"] + res["other"])
            summary_by_sign[res["sign"]] = {"ok": len(res["ok"]), "other": len(res["other"])}
            log.info("DONE %s: ok=%d other=%d (running %ds)",
                     res["sign"], len(res["ok"]), len(res["other"]),
                     int(time.time() - t0))

    by_status = Counter(r.get("download_status") for r in all_records)
    by_sign_ok = Counter(r["sign_id"] for r in all_records if r.get("download_status") == "ok")
    manifest = {
        "source": "ytsearch_v2",
        "license": "Per-clip third-party URL (YouTube); follow upstream uploader license",
        "queries_templates": QUERIES_TEMPLATES,
        "n_targets_attempted": sum(s["ok"] + s["other"] for s in summary_by_sign.values()),
        "n_ok": sum(v["ok"] for v in summary_by_sign.values()),
        "by_status": dict(by_status),
        "by_sign": dict(by_sign_ok),
        "per_sign_downloadable": dict(by_sign_ok),
        "summary_by_sign": summary_by_sign,
        "records": all_records,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2))
    log.info("done: ok=%d, by_sign=%s, %ds total",
             manifest["n_ok"], dict(by_sign_ok), int(time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
