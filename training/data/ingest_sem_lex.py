"""Ingest Sem-Lex clips for our 75 slice-1 glosses.

Phase 9 extension — Sem-Lex (Kezar et al. 2023, ASSETS '23) is an
isolated-sign ASL dataset of 91,148 clips across 3,149 signs from 44
deaf signers, aligned with ASL-LEX 2.0. Disjoint from ASL Citizen
(per the dataset card's "may be combined for 174k total" wording).

**License: CC BY-NC-SA 4.0.** Same non-commercial slice-2 cliff as
ASL Citizen / ADR 0009. Plus ethical-use commitments (DHH community
accessibility, attribution, etc.) which align with ADR 0004's
existing slice-2 plan.

Usage:

    # 1. Place metadata CSV + the 6 video chunks on disk (manually
    #    downloaded after accepting the project's ToU at
    #    https://github.com/leekezar/SemLex).
    #
    #    dataset/raw/sem_lex/semlex_metadata.csv
    #    dataset/raw/sem_lex/chunk1.bin  (and chunk2..6)
    #
    # 2. Run the ingestion:
    python -m training.data.ingest_sem_lex \\
        --metadata dataset/raw/sem_lex/semlex_metadata.csv \\
        --chunks dataset/raw/sem_lex/chunk1.bin dataset/raw/sem_lex/chunk2.bin ... \\
        --filter dataset/slice1b_vocabulary.json \\
        --output dataset/raw/sem_lex \\
        --manifest dataset/raw/sem_lex_manifest.json

The chunk files are auto-detected (ZIP vs tar.gz vs concatenated
mp4 stream). For each, we walk the contained files and selectively
extract video_ids matching our filter — same selective-fetch strategy
as ASL Citizen, just operating on local archive files rather than
HTTP Range.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import tarfile
import zipfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.data.vocabulary import SIGN_IDS, load_vocabulary  # noqa: E402

log = logging.getLogger("ingest_sem_lex")


SIGN_ID_TO_ALIASES: dict[str, tuple[str, ...]] = {
    "mom": ("mother",),
    "dad": ("father",),
    "grandma": ("grandmother",),
}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())


def _build_sid_index(kept_signs: set[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for sid in kept_signs:
        out[_norm(sid)] = sid
        for a in SIGN_ID_TO_ALIASES.get(sid, ()):
            out[_norm(a)] = sid
    return out


@dataclass
class ClipRecord:
    sign_id: str
    gloss: str
    source: str  # "sem_lex"
    wlasl_video_id: str  # parity with manifest schema; here = sem_lex video_id
    original_url: str  # "sem_lex://<video_id>"
    source_signer_id: str | None
    source_split: str | None
    frame_start: int | None  # always None — pre-trimmed clips
    frame_end: int | None
    local_path: str | None
    download_status: str  # ok | missing | extract-error


def _detect_archive_kind(path: Path) -> str:
    """Detect whether a chunk file is a ZIP, tarball, or unknown."""
    with path.open("rb") as f:
        magic = f.read(8)
    if magic[:4] == b"PK\x03\x04":
        return "zip"
    if magic[:5] == b"\x1f\x8b\x08\x00\x00" or magic[:2] == b"\x1f\x8b":
        return "gzip"
    if magic[257:262] == b"ustar" if len(magic) >= 262 else False:
        return "tar"
    # Fallback: tarfile.is_tarfile can read any tar variant
    try:
        if tarfile.is_tarfile(str(path)):
            return "tar"
    except Exception:
        pass
    return "unknown"


def _extract_from_archive(
    chunk: Path,
    wanted_ids: set[str],
    dst_dir: Path,
) -> dict[str, Path]:
    """Walk the archive; extract every member whose basename stem matches a
    wanted video_id. Returns dict {video_id: extracted path}."""
    kind = _detect_archive_kind(chunk)
    extracted: dict[str, Path] = {}
    log.info("opening %s (kind=%s)", chunk, kind)
    if kind == "zip":
        with zipfile.ZipFile(chunk) as zf:
            for info in zf.infolist():
                name = Path(info.filename).name
                stem = name.rsplit(".", 1)[0]
                if stem in wanted_ids:
                    dst = dst_dir / name
                    if not dst.exists():
                        with zf.open(info) as src, dst.open("wb") as out:
                            import shutil

                            shutil.copyfileobj(src, out)
                    extracted[stem] = dst
    elif kind in ("tar", "gzip"):
        mode = "r:gz" if kind == "gzip" else "r"
        with tarfile.open(chunk, mode) as tf:
            for info in tf:
                if not info.isfile():
                    continue
                name = Path(info.name).name
                stem = name.rsplit(".", 1)[0]
                if stem in wanted_ids:
                    dst = dst_dir / name
                    if not dst.exists():
                        src = tf.extractfile(info)
                        if src is None:
                            continue
                        with dst.open("wb") as out:
                            import shutil

                            shutil.copyfileobj(src, out)
                    extracted[stem] = dst
    else:
        log.error("could not detect archive format for %s", chunk)
    log.info("extracted %d clips from %s", len(extracted), chunk.name)
    return extracted


def ingest(
    metadata_path: Path,
    chunks: list[Path],
    filter_path: Path,
    output_dir: Path,
    manifest_path: Path,
) -> None:
    load_vocabulary()
    with filter_path.open() as f:
        filt = json.load(f)
    kept_signs = {s["sign_id"] for s in filt["kept_signs"]}
    log.info("kept_signs from filter: %d", len(kept_signs))

    sid_by_norm = _build_sid_index(kept_signs)

    # 1. Scan metadata for wanted rows. Same logic as the recon script.
    with metadata_path.open(newline="", encoding="utf-8", errors="replace") as f:
        all_rows = list(csv.DictReader(f))
    log.info("metadata: %d rows", len(all_rows))

    wanted: dict[str, dict[str, Any]] = {}  # video_id → row info
    for r in all_rows:
        if r.get("label_type") == "freetext":
            continue
        label = (r.get("label") or "").strip()
        if not label or "." in label:
            continue
        nm = _norm(label)
        sid = sid_by_norm.get(nm) or sid_by_norm.get(re.sub(r"\d+$", "", nm))
        if not sid:
            continue
        vid = r["video_id"]
        # If a video has both asllex and signbank labels, prefer asllex
        # because that's the canonical naming we built our lookup against.
        if vid in wanted and wanted[vid].get("label_type") == "asllex":
            continue
        wanted[vid] = {
            "sign_id": sid,
            "label": label,
            "signer_id": r.get("signer_id") or None,
            "split": r.get("split") or None,
            "label_type": r.get("label_type"),
        }

    log.info("filtered to %d unique video_ids matching our 75 signs", len(wanted))
    per_sign = defaultdict(int)
    for w in wanted.values():
        per_sign[w["sign_id"]] += 1
    log.info("per-sign min/median/max: %d / %d / %d",
             min(per_sign.values()), sorted(per_sign.values())[len(per_sign)//2], max(per_sign.values()))

    output_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = output_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    # 2. For each chunk, selectively extract.
    found: dict[str, Path] = {}
    for chunk in chunks:
        if not chunk.exists():
            log.error("chunk missing: %s", chunk)
            continue
        extracted = _extract_from_archive(chunk, set(wanted), clips_dir)
        found.update(extracted)
        log.info("running total extracted: %d / %d wanted", len(found), len(wanted))

    # 3. Build records.
    records: list[ClipRecord] = []
    for vid, info in wanted.items():
        local = found.get(vid)
        rec = ClipRecord(
            sign_id=info["sign_id"],
            gloss=info["label"],
            source="sem_lex",
            wlasl_video_id=vid,
            original_url=f"sem_lex://{vid}",
            source_signer_id=str(info["signer_id"]) if info["signer_id"] else None,
            source_split=info["split"],
            frame_start=None,
            frame_end=None,
            local_path=str(local) if local else None,
            download_status="ok" if local else "missing",
        )
        records.append(rec)

    per_sign_ok = defaultdict(int)
    for r in records:
        if r.download_status == "ok":
            per_sign_ok[r.sign_id] += 1

    manifest = {
        "source": "sem_lex",
        "license": "CC BY-NC-SA 4.0",
        "citation": "Kezar et al. 2023, ASSETS '23 (arXiv:2310.00196)",
        "ethical_use_commitments": [
            "Treat dataset subjects with dignity (CC BY-NC-SA + project commitments)",
            "Make research accessible to DHH communities",
            "Use culturally appropriate terminology",
        ],
        "per_sign_attempts": {sid: per_sign[sid] for sid in per_sign},
        "per_sign_downloadable": dict(per_sign_ok),
        "records": [asdict(r) for r in records],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest_path.with_suffix(".tmp.json")
    with tmp.open("w") as f:
        json.dump(manifest, f, indent=2)
    tmp.replace(manifest_path)
    log.info(
        "wrote %s with %d records (%d extracted, %d missing)",
        manifest_path, len(records), sum(per_sign_ok.values()),
        sum(1 for r in records if r.download_status == "missing"),
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, nargs="+", required=True)
    parser.add_argument("--filter", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("dataset/raw/sem_lex"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("dataset/raw/sem_lex_manifest.json")
    )
    args = parser.parse_args()
    ingest(args.metadata, args.chunks, args.filter, args.output, args.manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
