"""Local MSAsl thin-sign fetcher — runs yt-dlp from your machine + browser cookies.

Modal-side scrape is blocked by YouTube's bot detection on datacenter IPs;
local home IPs work fine when authenticated via --cookies-from-browser.

Usage:
    training/.venv/bin/python -m scripts.fetch_msasl_local \
        --msasl-dir /Users/hirom/Downloads/MS-ASL \
        --out-dir dataset/raw/msasl \
        --manifest dataset/raw/msasl_thin_signs_manifest.json \
        --cookies-from-browser chrome

Prereqs:
    - yt-dlp installed in the venv (training/.venv has it).
    - Chrome / Safari / Firefox logged into YouTube. The cookies are read
      directly from the browser profile by yt-dlp.
    - ffmpeg on PATH (for --download-sections re-encode).

Expected: ~613 candidate clips, ~30-60 min wall-clock with 8 parallel workers
on a typical home connection. Failures (videos removed since 2019) are normal.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ALIASES = {"mom": ["mother"], "dad": ["father"], "grandma": ["grandmother"]}


def _build_targets(msasl_dir: Path, still_thin: dict[str, int], target_per_sign: int) -> list[dict]:
    synonyms = json.loads((msasl_dir / "MSASL_synonym.json").read_text())
    sign_for_gloss: dict[str, str] = {}
    for sid in still_thin:
        sign_for_gloss[sid] = sid
        for a in ALIASES.get(sid, []):
            sign_for_gloss[a] = sid
    for group in synonyms:
        lowered = [g.lower() for g in group]
        owner = next((sign_for_gloss[g] for g in lowered if g in sign_for_gloss), None)
        if owner:
            for g in lowered:
                sign_for_gloss.setdefault(g, owner)

    needed = {s: target_per_sign - n for s, n in still_thin.items()}
    targets: list[dict] = []
    got: dict[str, int] = {s: 0 for s in still_thin}
    for fname in ("MSASL_train.json", "MSASL_val.json", "MSASL_test.json"):
        split = fname.split("_")[1].split(".")[0]
        for r in json.loads((msasl_dir / fname).read_text()):
            gloss = (r.get("clean_text") or "").lower()
            sid = sign_for_gloss.get(gloss)
            if not sid or got[sid] >= needed[sid]:
                continue
            targets.append({
                "sign_id": sid,
                "msasl_gloss": gloss,
                "msasl_label": r.get("label"),
                "msasl_split": split,
                "url": r["url"],
                "start_time": float(r.get("start_time", 0.0)),
                "end_time": float(r["end_time"]) if r.get("end_time") is not None else None,
                "signer_id": r.get("signer_id"),
                "box": r.get("box"),
                "fps": r.get("fps"),
                "width": r.get("width"),
                "height": r.get("height"),
            })
            got[sid] += 1
    return targets


def _download_one(rec: dict, clips_dir: Path, cookies_browser: str | None,
                  timeout_sec: int = 240) -> dict:
    url = rec["url"]
    video_id = url.rsplit("v=", 1)[-1].split("&")[0]
    start = rec.get("start_time", 0.0)
    end = rec.get("end_time")
    slug = f"{rec['sign_id']}__msasl__{video_id}_{int(start * 1000)}_{int((end or 0) * 1000)}"
    out_path = clips_dir / f"{slug}.mp4"
    result = dict(rec)
    result["local_path"] = str(out_path)
    if out_path.exists() and out_path.stat().st_size > 1024:
        result["download_status"] = "ok"
        return result
    section = f"*{start:.3f}-{end:.3f}" if end is not None else f"*{start:.3f}-inf"
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--quiet", "--no-warnings", "--no-playlist",
        "--format", "bestvideo[ext=mp4][height<=720]+bestaudio/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "--download-sections", section,
        "--force-keyframes-at-cuts",
        "-o", str(out_path),
    ]
    if cookies_browser:
        cmd += ["--cookies-from-browser", cookies_browser]
    cmd.append(url)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        result["download_status"] = "timeout"
        return result
    if proc.returncode == 0 and out_path.exists() and out_path.stat().st_size > 1024:
        result["download_status"] = "ok"
        return result
    stderr = (proc.stderr or "").lower()
    if "video unavailable" in stderr or "removed by the user" in stderr or "404" in stderr:
        result["download_status"] = "404"
    elif "private" in stderr or "blocked" in stderr or "age" in stderr:
        result["download_status"] = "blocked"
    elif "sign in to confirm" in stderr or "not a bot" in stderr:
        result["download_status"] = "bot-challenge"
        result["stderr_tail"] = stderr[-300:]
    else:
        result["download_status"] = "other-error"
        result["stderr_tail"] = stderr[-300:] if stderr else ""
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--msasl-dir", type=Path, default=Path("/Users/hirom/Downloads/MS-ASL"))
    ap.add_argument("--v3-manifest", type=Path,
                    default=Path("data/labeled_frames/unified_clip_manifest_modal_v3.json"))
    ap.add_argument("--v2-manifest", type=Path,
                    default=Path("data/labeled_frames/unified_clip_manifest_modal_v2.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("dataset/raw/msasl"))
    ap.add_argument("--manifest", type=Path,
                    default=Path("dataset/raw/msasl_thin_signs_manifest.json"))
    ap.add_argument("--target-per-sign", type=int, default=100)
    ap.add_argument("--cookies-from-browser", type=str, default="chrome",
                    help="chrome | safari | firefox | brave | edge")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    v3c = Counter(c["sign_id"] for c in json.loads(args.v3_manifest.read_text())["clips"])
    v2c = Counter(c["sign_id"] for c in json.loads(args.v2_manifest.read_text())["clips"])
    thin = {s for s, n in v2c.items() if n < args.target_per_sign}
    still_thin = {s: v3c.get(s, 0) for s in thin if v3c.get(s, 0) < args.target_per_sign}

    targets = _build_targets(args.msasl_dir, still_thin, args.target_per_sign)
    print(f"▶ {len(targets)} MSAsl targets across {len(set(t['sign_id'] for t in targets))} signs")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = args.out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_download_one, r, clips_dir, args.cookies_from_browser)
                   for r in targets]
        for i, fut in enumerate(as_completed(futures), 1):
            results.append(fut.result())
            if i % 25 == 0 or i == len(targets):
                ok = sum(1 for r in results if r.get("download_status") == "ok")
                bot = sum(1 for r in results if r.get("download_status") == "bot-challenge")
                rate = i / max(time.time() - t0, 0.1)
                print(f"  {i}/{len(targets)}  ok={ok}  bot-challenge={bot}  "
                      f"({rate:.1f}/s, {time.time() - t0:.0f}s elapsed)")
                if bot >= 5 and ok == 0:
                    print(f"\n!! >5 bot-challenge failures with 0 ok — your browser is likely "
                          f"not logged into YouTube, or yt-dlp can't read cookies from "
                          f"--cookies-from-browser={args.cookies_from_browser}. Stop "
                          f"this run, log into YouTube in that browser, retry.")

    by_status = Counter(r.get("download_status") for r in results)
    ok = [r for r in results if r.get("download_status") == "ok"]
    by_sign = Counter(r["sign_id"] for r in ok)
    print(f"\n▶ done: {dict(by_status)} ({time.time() - t0:.0f}s)")
    print(f"▶ ok-per-sign: {dict(by_sign)}")

    manifest = {
        "source": "msasl",
        "license": "C-UDA (Computational Use of Data Agreement); non-commercial research",
        "citation": "Vaezi Joze, Koller. MS-ASL. BMVC 2019",
        "n_targets": len(targets),
        "n_ok": len(ok),
        "by_status": dict(by_status),
        "by_sign": dict(by_sign),
        "records": results,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2))
    print(f"▶ wrote {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
