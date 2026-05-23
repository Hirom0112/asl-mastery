"""Rebuild unified_clip_manifest_modal.json to recover ~750 clips that the
prior builder silently dropped due to a strict gloss-matching filter.

Original builder filtered Sem-Lex records by gloss == sign_id, which excluded
ALL clips whose gloss has a `_N` suffix (eat_1, eat_2, deaf_1, fine_1, live_2,
...). For affected signs this dropped 95-99% of available training data.

This script regenerates the manifest using sign_id directly (which is already
the unsuffixed gloss). Output is compatible with extract_trajectories_v2.

Usage:
    python -m scripts.rebuild_unified_manifest \
        --out data/labeled_frames/unified_clip_manifest_modal_v2.json

Then upload to Modal (see ASCII recipe at the bottom of this file).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


SEM_LEX_RAW = Path("dataset/raw/sem_lex_manifest.json")
PROJECT_CLEAN_DIR = Path("dataset/clean/v2-yt/normalized_videos")
ASL_CITIZEN_RAW = Path("dataset/raw/asl_citizen_manifest.json")
WLASL_RAW = Path("dataset/raw/wlasl_manifest.json")
MSASL_RAW = Path("dataset/raw/msasl_thin_signs_manifest.json")
LIFEPRINT_RAW = Path("dataset/raw/lifeprint_manifest_v2.json")
YTSEARCH_V2_RAW = Path("dataset/raw/ytsearch_v2_manifest.json")
VOCAB_JSON = Path("dataset/slice1_vocabulary.json")

# Modal volume mounts; matches what extract_trajectories_v2 expects.
SEM_LEX_MODAL_PREFIX = "/data/datasets/sem_lex_clips/clips"
PROJECT_CLEAN_MODAL_PREFIX = "/data/datasets/asl_clips/v2-yt/normalized_videos"
ASL_CITIZEN_MODAL_PREFIX = "/data/datasets/asl_citizen/clips"
WLASL_MODAL_PREFIX = "/data/datasets/wlasl"
MSASL_MODAL_PREFIX = "/data/datasets/msasl/clips"
LIFEPRINT_MODAL_PREFIX = "/data/datasets/lifeprint"
YTSEARCH_V2_MODAL_PREFIX = "/data/datasets/ytsearch_v2"


def load_vocab_sign_ids() -> set[str]:
    v = json.loads(VOCAB_JSON.read_text())
    return {s["sign_id"] for s in v["kept_signs"]}


def sem_lex_clips(allowed_signs: set[str]) -> list[dict]:
    raw = json.loads(SEM_LEX_RAW.read_text())
    out = []
    gloss_kept = Counter()
    gloss_dropped = Counter()
    for r in raw["records"]:
        if r.get("download_status") != "ok":
            continue
        sid = r["sign_id"]
        if sid not in allowed_signs:
            gloss_dropped[("not_in_vocab", sid)] += 1
            continue
        local = Path(r["local_path"])
        modal_path = f"{SEM_LEX_MODAL_PREFIX}/{local.name}"
        out.append({
            "clip_path": modal_path,
            "sign_id": sid,
            "source": "sem_lex",
            "ext": local.suffix.lstrip("."),
            "sem_lex_label": r.get("gloss"),  # keep variant info for traceability
            "split": r.get("source_split", "unknown"),
            "signer_id": r.get("source_signer_id"),
        })
        gloss_kept[r.get("gloss")] += 1
    return out, gloss_kept, gloss_dropped


def asl_citizen_clips(allowed_signs: set[str]) -> list[dict]:
    """Read dataset/raw/asl_citizen_manifest.json and project to the unified
    clip schema. ADR-0015: license = MSR-LA (research-only, no redistribution).
    Signer IDs preserved for signer-disjoint splits downstream.
    """
    if not ASL_CITIZEN_RAW.exists():
        return []
    raw = json.loads(ASL_CITIZEN_RAW.read_text())
    out = []
    for r in raw.get("records", []):
        if r.get("download_status") != "ok":
            continue
        sid = r.get("sign_id")
        if sid not in allowed_signs:
            continue
        local = r.get("local_path")
        if not local:
            continue
        basename = Path(local).name
        out.append({
            "clip_path": f"{ASL_CITIZEN_MODAL_PREFIX}/{basename}",
            "sign_id": sid,
            "source": "asl_citizen",
            "ext": Path(basename).suffix.lstrip("."),
            "asl_citizen_gloss": r.get("gloss"),
            "split": r.get("source_split"),
            "signer_id": r.get("source_signer_id"),
        })
    return out


def wlasl_clips(allowed_signs: set[str]) -> list[dict]:
    """Read dataset/raw/wlasl_manifest.json and project to the unified clip
    schema. ADR-0015: source clips were downloaded from third-party URLs
    (YouTube etc.) per WLASL's `instances[].url` field — license follows the
    upstream source per-clip; the WLASL annotations are research-only.
    """
    if not WLASL_RAW.exists():
        return []
    raw = json.loads(WLASL_RAW.read_text())
    out = []
    for r in raw.get("records", []):
        if r.get("download_status") != "ok":
            continue
        sid = r.get("sign_id")
        if sid not in allowed_signs:
            continue
        local = r.get("local_path")
        if not local:
            continue
        basename = Path(local).name
        out.append({
            "clip_path": f"{WLASL_MODAL_PREFIX}/{basename}",
            "sign_id": sid,
            "source": "wlasl",
            "ext": Path(basename).suffix.lstrip("."),
            "wlasl_video_id": r.get("wlasl_video_id"),
            "split": r.get("source_split"),
            "signer_id": r.get("source_signer_id"),
            "frame_start": r.get("frame_start"),
            "frame_end": r.get("frame_end"),
        })
    return out


def msasl_clips(allowed_signs: set[str]) -> list[dict]:
    """Read MSAsl thin-sign manifest (C-UDA license; Vaezi Joze & Koller 2019).
    Records include signer_id, msasl_split, source URL for traceability.
    """
    if not MSASL_RAW.exists():
        return []
    raw = json.loads(MSASL_RAW.read_text())
    out = []
    for r in raw.get("records", []):
        if r.get("download_status") != "ok":
            continue
        sid = r.get("sign_id")
        if sid not in allowed_signs:
            continue
        local = r.get("local_path")
        if not local:
            continue
        basename = Path(local).name
        out.append({
            "clip_path": f"{MSASL_MODAL_PREFIX}/{basename}",
            "sign_id": sid,
            "source": "msasl",
            "ext": Path(basename).suffix.lstrip("."),
            "msasl_gloss": r.get("msasl_gloss"),
            "split": r.get("msasl_split"),
            "signer_id": r.get("signer_id"),
            "fps": r.get("fps"),
            "frame_start_sec": r.get("start_time"),
            "frame_end_sec": r.get("end_time"),
        })
    return out


def lifeprint_clips(allowed_signs: set[str]) -> list[dict]:
    """Read lifeprint manifest (Bill Vicars / lifeprint.com — single curated
    canonical reference per sign). Manifest schema mirrors WLASL.
    """
    if not LIFEPRINT_RAW.exists():
        return []
    raw = json.loads(LIFEPRINT_RAW.read_text())
    out = []
    for r in raw.get("records", []):
        if r.get("download_status") != "ok":
            continue
        sid = r.get("sign_id")
        if sid not in allowed_signs:
            continue
        local = r.get("local_path")
        if not local:
            continue
        basename = Path(local).name
        out.append({
            "clip_path": f"{LIFEPRINT_MODAL_PREFIX}/{basename}",
            "sign_id": sid,
            "source": "lifeprint",
            "ext": Path(basename).suffix.lstrip("."),
            "original_url": r.get("original_url"),
        })
    return out


def ytsearch_v2_clips(allowed_signs: set[str]) -> list[dict]:
    """Read ytsearch_v2 manifest (post-MSAsl gap-closer, Phase 4.9).
    Source clips were YouTube hits via 'ASL <SIGN> Bill Vicars' style queries.
    """
    if not YTSEARCH_V2_RAW.exists():
        return []
    raw = json.loads(YTSEARCH_V2_RAW.read_text())
    out = []
    for r in raw.get("records", []):
        if r.get("download_status") != "ok":
            continue
        sid = r.get("sign_id")
        if sid not in allowed_signs:
            continue
        local = r.get("local_path")
        if not local:
            continue
        basename = Path(local).name
        out.append({
            "clip_path": f"{YTSEARCH_V2_MODAL_PREFIX}/{basename}",
            "sign_id": sid,
            "source": "ytsearch_v2",
            "ext": Path(basename).suffix.lstrip("."),
            "original_url": r.get("original_url"),
            "query": r.get("query"),
        })
    return out


def project_clean_clips(allowed_signs: set[str]) -> list[dict]:
    out = []
    if not PROJECT_CLEAN_DIR.exists():
        return out
    for mp4 in sorted(PROJECT_CLEAN_DIR.glob("*.mp4")):
        # filename pattern: <sign>__ytsearch__<id>.mp4
        sign_id = mp4.stem.split("__")[0]
        if sign_id not in allowed_signs:
            continue
        modal_path = f"{PROJECT_CLEAN_MODAL_PREFIX}/{mp4.name}"
        out.append({
            "clip_path": modal_path,
            "sign_id": sign_id,
            "source": "project_clean",
            "ext": "mp4",
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path("data/labeled_frames/unified_clip_manifest_modal_v2.json"))
    args = ap.parse_args()

    allowed = load_vocab_sign_ids()
    print(f"[vocab]   {len(allowed)} allowed signs")

    sl_clips, gloss_kept, gloss_dropped = sem_lex_clips(allowed)
    pc_clips = project_clean_clips(allowed)
    ac_clips = asl_citizen_clips(allowed)
    wl_clips = wlasl_clips(allowed)
    ms_clips = msasl_clips(allowed)
    lp_clips = lifeprint_clips(allowed)
    yt_clips = ytsearch_v2_clips(allowed)

    print(f"[sem_lex]       kept {len(sl_clips)} clips across {len(gloss_kept)} glosses")
    print(f"                top 10 gloss variants now included:")
    for g, n in gloss_kept.most_common(10):
        print(f"                  {g}: {n}")
    print(f"[project_clean] {len(pc_clips)} clips")
    print(f"[asl_citizen]   {len(ac_clips)} clips  (MSR-LA, signer-disjoint splits available)")
    print(f"[wlasl]         {len(wl_clips)} clips  (third-party URL sources; signer IDs sparse)")
    print(f"[msasl]         {len(ms_clips)} clips  (C-UDA; per-clip start/end_sec for sign-window slicing)")
    print(f"[lifeprint]     {len(lp_clips)} clips  (Bill Vicars curated single-canonical-per-sign)")
    print(f"[ytsearch_v2]   {len(yt_clips)} clips  (Phase 4.9 gap-closer; per-clip third-party URL)")

    all_clips = sl_clips + pc_clips + ac_clips + wl_clips + ms_clips + lp_clips + yt_clips
    covered = set(c["sign_id"] for c in all_clips)
    per_source = Counter(c["source"] for c in all_clips)
    per_sign = Counter(c["sign_id"] for c in all_clips)

    manifest = {
        "version": 2,
        "vocab_size": len(allowed),
        "covered_signs": len(covered),
        "total_clips": len(all_clips),
        "per_source": dict(per_source),
        "rebuild_note": "v2 fixes prior strict gloss filter — includes _N variants",
        "clips": all_clips,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2))
    print(f"\n[out]     {args.out}  total_clips={len(all_clips)}, covered={len(covered)}/{len(allowed)}")

    # Diff vs the v1 unified manifest
    v1_path = Path("data/labeled_frames/unified_clip_manifest_modal.json")
    if v1_path.exists():
        v1 = json.loads(v1_path.read_text())
        v1_per_sign = Counter(c["sign_id"] for c in v1["clips"])
        gains = []
        for sid in sorted(allowed):
            new = per_sign.get(sid, 0)
            old = v1_per_sign.get(sid, 0)
            if new - old > 0:
                gains.append((sid, old, new, new - old))
        gains.sort(key=lambda x: -x[3])
        print(f"\n[diff]    Signs with NEW clips (top 15):")
        print(f"  {'sign':<14} {'old':>6} {'new':>6} {'+':>6}")
        for sid, old, new, delta in gains[:15]:
            print(f"  {sid:<14} {old:>6} {new:>6} {delta:>+6}")
        total_new = sum(g[3] for g in gains)
        print(f"  Total recoverable clips: +{total_new}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
