"""Ingest ASL Citizen clips for the 80 slice-1 glosses.

Phase 9b.3 (ROADMAP / TODO Phase 9). Secondary slice-1 data source on
top of WLASL + Lifeprint + ytsearch. ASL Citizen is hosted by Microsoft
Research under the **MSR-LA license** (Microsoft Research License
Agreement) — non-commercial research use only, no redistribution. See
9b.1 verification notes. Commercial / slice-2 deployment requires a
separate license; this script's output must not be redistributed.

Dataset stats (per Desai et al. 2023, arXiv:2304.05934):
  - 83,912 clips
  - 2,731 signs (from ASL-LEX 2.0)
  - 52 signers
  - Signer-independent train/val/test splits

Vocabulary overlap with our slice-1 80 signs is 100% (verified in 9b.2);
8 of our sign_ids need alias resolution (e.g. our "mom" → ASL-LEX
"mother"; "fine" matches both "fine_1" and "fine_2" variants).

Usage:

  # 1. Click through the EULA at
  #    https://www.microsoft.com/en-us/download/details.aspx?id=105253
  #    and copy the time-limited ASL_Citizen.zip download URL.
  # 2. Inspect CSVs only (cheap; the script downloads + extracts only
  #    the splits files first):
  python -m training.data.ingest_asl_citizen \\
      --zip-url "https://download.microsoft.com/.../ASL_Citizen.zip" \\
      --output dataset/raw/asl_citizen \\
      --manifest dataset/raw/asl_citizen_manifest.json \\
      --inspect-only

  # 3. Full ingestion (selective extraction; only clips for our 80
  #    signs are kept; the rest of the 42.8 GB ZIP is discarded after
  #    extraction):
  python -m training.data.ingest_asl_citizen \\
      --zip-url "https://download.microsoft.com/.../ASL_Citizen.zip" \\
      --output dataset/raw/asl_citizen \\
      --manifest dataset/raw/asl_citizen_manifest.json

  # 4. If you've already downloaded the ZIP locally, point at it:
  python -m training.data.ingest_asl_citizen \\
      --zip-path /path/to/ASL_Citizen.zip \\
      --output dataset/raw/asl_citizen \\
      --manifest dataset/raw/asl_citizen_manifest.json

Manifest schema mirrors `wlasl_manifest.json` so `clean.py` can consume
WLASL + ASL Citizen records as a single concatenated stream.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import shutil
import subprocess
import sys
import zipfile
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.data.vocabulary import SIGN_IDS, load_vocabulary  # noqa: E402

log = logging.getLogger("ingest_asl_citizen")


# English-gloss aliases between our sign_ids and ASL Citizen's gloss
# tokens. ASL Citizen normalizes glosses to UPPERCASE-NO-SEPARATOR (e.g.
# our "thank_you" → ASL Citizen's "THANKYOU", our "mom" → "MOTHER1").
# Variants of the same English gloss carry a trailing digit (PIZZA1,
# PIZZA2). The match function below normalizes both sides to alpha-only
# lowercase + strips trailing digits, so most signs map without an
# explicit alias entry. This table covers the small set where our
# sign_id and the canonical English gloss differ in surface form.
SIGN_ID_TO_GLOSS_ALIASES: dict[str, tuple[str, ...]] = {
    "mom": ("mother",),
    "dad": ("father",),
    "grandma": ("grandmother",),
}


def _norm_gloss(s: str) -> str:
    """Normalize a gloss for matching: lowercase, alpha-only."""
    return re.sub(r"[^a-z]", "", s.lower())


def _norm_strip_variant(s: str) -> str:
    """Same as _norm_gloss, then strip a trailing variant digit run.

    ASL Citizen variant convention: WANT1, WANT2 = two distinct signs
    sharing the gloss "want". Both map to our "want" sign_id; the
    classifier learns the gloss target either way and the extra data
    is welcome.
    """
    return re.sub(r"\d+$", "", _norm_gloss(s))


def _build_gloss_lookup(citizen_glosses: Iterable[str]) -> dict[str, str]:
    """Map each observed ASL Citizen gloss → our sign_id.

    Strategy:
      1. Build a normalized-form → sign_id index using SIGN_IDS plus the
         explicit alias table (mom↔mother, dad↔father, grandma↔grandmother).
      2. For every observed ASL Citizen gloss, try the normalized form
         directly first; if that misses, try with the trailing variant
         digit stripped. The two-step keeps false matches like
         "CATEGORY"→"cat" out (normalized "category" ≠ "cat") while
         picking up "CAT1"→"cat" via the digit-strip.
    """
    sid_by_norm: dict[str, str] = {}
    for sid in SIGN_IDS:
        sid_by_norm[_norm_gloss(sid)] = sid
        for alias in SIGN_ID_TO_GLOSS_ALIASES.get(sid, ()):
            sid_by_norm[_norm_gloss(alias)] = sid

    lookup: dict[str, str] = {}
    for g in citizen_glosses:
        # Exclude fingerspelled variants (notation: letters separated by
        # periods, e.g. "W.H.A.T" — the fingerspelled form of "what"
        # rather than the natural ASL sign WHAT). They're a structurally
        # different sign and would confuse the classifier if labeled the
        # same as the natural form.
        if "." in g:
            continue
        n1 = _norm_gloss(g)
        if n1 in sid_by_norm:
            lookup[g] = sid_by_norm[n1]
            continue
        n2 = _norm_strip_variant(g)
        if n2 in sid_by_norm:
            lookup[g] = sid_by_norm[n2]
    return lookup


# Candidate CSV column names — the actual ASL Citizen CSVs use one of
# these for each conceptual field. First match wins. Verified column
# names will be filled in after the first --inspect-only run.
CSV_COL_CANDIDATES: dict[str, tuple[str, ...]] = {
    "gloss": ("Gloss", "gloss", "Sign", "sign", "Label", "label", "EntryID", "Entry ID"),
    "video": ("Video file", "Video", "Filename", "filename", "video_file", "Clip", "clip"),
    "participant": (
        "Participant ID",
        "participant_id",
        "ParticipantID",
        "Signer",
        "signer",
        "Signer ID",
        "signer_id",
    ),
}


@dataclass
class ClipRecord:
    sign_id: str
    gloss: str  # the ASL-LEX EntryID, e.g. "mother" or "fine_1"
    source: str  # "asl_citizen"
    wlasl_video_id: str  # keep the field name for manifest-schema parity (= clip basename)
    original_url: str  # "asl_citizen://<basename>" — clips not hosted on a URL
    source_signer_id: str | None
    source_split: str | None  # ASL Citizen's own train/val/test (overridden by clean.py 9a.6)
    frame_start: int | None  # always None — clips already sign-windowed
    frame_end: int | None
    local_path: str | None
    download_status: str  # "ok" | "missing" | "extract-error" | "skipped"


def _resolve_columns(header: list[str]) -> dict[str, str]:
    """Pick the actual column names from the CSV header."""
    resolved: dict[str, str] = {}
    lowered = {h.lower(): h for h in header}
    for field, candidates in CSV_COL_CANDIDATES.items():
        chosen: str | None = None
        for c in candidates:
            if c in header:
                chosen = c
                break
            if c.lower() in lowered:
                chosen = lowered[c.lower()]
                break
        if chosen is None:
            raise RuntimeError(
                f"could not find a CSV column for field '{field}' in header {header}. "
                f"Tried: {candidates}. Update CSV_COL_CANDIDATES."
            )
        resolved[field] = chosen
    return resolved


def _download_zip(zip_url: str, out_path: Path) -> None:
    """Stream-download the ZIP with curl --continue-at - for resume."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading %s → %s (resume-capable)", zip_url, out_path)
    cmd = [
        "curl",
        "-L",
        "--fail",
        "--show-error",
        "--continue-at",
        "-",
        "-o",
        str(out_path),
        zip_url,
    ]
    subprocess.run(cmd, check=True)


@contextmanager
def _open_zip(zip_url: str | None, zip_path: Path | None):
    """Open the archive either locally (ZipFile) or remotely via HTTP Range
    (remotezip.RemoteZip). Both objects expose the same ZipFile-style API
    we use below: ``.infolist()``, ``.open(name_or_info)`` for streaming
    a member's bytes.

    Remote mode (preferred when given a URL): only the central directory
    and the byte ranges of the members we actually extract are fetched
    from the CDN. For our use case (~4,500 of 84,000 clips + 3 CSVs)
    that's ~2.3 GB out of 42.8 GB — a ~95% download save.

    Falls back to local download + ZipFile if remotezip raises (CDN
    doesn't support Range, URL signature too tight to allow multiple
    requests, etc.).
    """
    if zip_path is not None:
        with zipfile.ZipFile(zip_path) as zf:
            yield zf, "local"
        return

    assert zip_url is not None
    try:
        from remotezip import RemoteZip  # type: ignore
    except ImportError as e:
        raise SystemExit(
            "remotezip not installed; `pip install remotezip` or use --zip-path"
        ) from e

    log.info("opening remote ZIP via HTTP Range: %s", zip_url[:80] + "...")
    try:
        with RemoteZip(zip_url) as rz:
            yield rz, "remote"
    except Exception as e:  # noqa: BLE001 — fall back to download on any remote error
        log.warning(
            "remote ZIP open failed (%s); falling back to full download",
            e.__class__.__name__,
        )
        raise


def _iter_zip_csv_members(zf: zipfile.ZipFile) -> Iterable[tuple[str, zipfile.ZipInfo]]:
    """Yield (split_name, zipinfo) for train/val/test CSVs inside the archive."""
    for info in zf.infolist():
        name = info.filename
        base = Path(name).name.lower()
        if base in ("train.csv", "val.csv", "test.csv"):
            split = base.removesuffix(".csv")
            yield split, info


def _video_path_in_zip(zf: zipfile.ZipFile, video_basename: str) -> str | None:
    """Locate a clip filename inside the archive (paths may have a prefix
    like ``ASL_Citizen/videos/``). Linear scan once, cached by caller."""
    target = video_basename.lower()
    for info in zf.infolist():
        if Path(info.filename).name.lower() == target:
            return info.filename
    return None


def _index_zip_videos(zf: zipfile.ZipFile) -> dict[str, str]:
    """Build basename → archive-path index in one pass (42 GB ZIP has tens
    of thousands of members; we don't want O(N*M) scans)."""
    idx: dict[str, str] = {}
    for info in zf.infolist():
        n = Path(info.filename).name
        if n.lower().endswith((".mp4", ".mov", ".webm")):
            idx[n.lower()] = info.filename
    return idx


def ingest(
    zip_url: str | None,
    zip_path: Path | None,
    output_dir: Path,
    manifest_path: Path,
    inspect_only: bool,
    keep_zip: bool,
) -> None:
    load_vocabulary()
    if not zip_url and not zip_path:
        raise SystemExit("either --zip-url or --zip-path required")

    output_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = output_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    log.info("scanning ASL Citizen CSVs against %d of our sign_ids", len(SIGN_IDS))

    try:
        zip_ctx = _open_zip(zip_url, zip_path)
    except Exception:
        # Fallback: full download then local open.
        if zip_url is None:
            raise
        fallback_path = output_dir / "_download" / "ASL_Citizen.zip"
        _download_zip(zip_url, fallback_path)
        zip_ctx = _open_zip(None, fallback_path)
        zip_path = fallback_path

    with zip_ctx as (zf, mode):
        log.info("zip mode: %s", mode)
        # 1. Resolve CSVs (header inspection).
        csv_members = list(_iter_zip_csv_members(zf))
        if not csv_members:
            raise SystemExit("no train.csv / val.csv / test.csv found in archive")
        log.info("found split CSVs: %s", [s for s, _ in csv_members])

        # 2. Parse all CSVs, filter to glosses we want.
        records: list[ClipRecord] = []
        per_sign_attempts: dict[str, int] = defaultdict(int)
        per_sign_ok: dict[str, int] = defaultdict(int)
        col_resolved: dict[str, str] | None = None

        # First pass: read every row, accumulate the full gloss inventory.
        all_rows: list[tuple[str, str, str, str | None]] = []  # split, gloss, video, pid
        for split, info in csv_members:
            with zf.open(info) as f:
                text = f.read().decode("utf-8", errors="replace")
            reader = csv.DictReader(text.splitlines())
            if reader.fieldnames is None:
                continue
            if col_resolved is None:
                col_resolved = _resolve_columns(list(reader.fieldnames))
                log.info("resolved CSV columns: %s", col_resolved)
            for row in reader:
                g = (row.get(col_resolved["gloss"]) or "").strip()
                if not g:
                    continue
                video = (row.get(col_resolved["video"]) or "").strip()
                pid = (row.get(col_resolved["participant"]) or "").strip() or None
                all_rows.append((split, g, video, pid))

        # Build a data-driven gloss lookup from the actual observed glosses.
        # This handles ASL Citizen's UPPERCASE/no-separator/digit-variant
        # convention (WANT1, THANKYOU, MOTHER2) automatically, without us
        # having to enumerate every variant up front.
        observed_glosses = {g for _, g, _, _ in all_rows}
        gloss_lookup = _build_gloss_lookup(observed_glosses)
        log.info(
            "matched %d ASL Citizen glosses → %d of our sign_ids",
            len(gloss_lookup),
            len(set(gloss_lookup.values())),
        )

        wanted_rows: list[tuple[str, str, str, str | None]] = []
        for split, g, video, pid in all_rows:
            sid = gloss_lookup.get(g)
            if not sid:
                continue
            per_sign_attempts[sid] += 1
            wanted_rows.append((split, g, video, pid))

        log.info("CSV scan: %d rows match our signs across all splits", len(wanted_rows))
        missing_sids = sorted(set(SIGN_IDS) - set(gloss_lookup.values()))
        for sid in sorted(SIGN_IDS):
            n = per_sign_attempts.get(sid, 0)
            marker = "  " if n > 0 else " *"
            log.info("%s %-15s -> %d candidate clips", marker, sid, n)
        if missing_sids:
            log.warning("%d sign_ids with NO ASL Citizen matches: %s", len(missing_sids), missing_sids)

        if inspect_only:
            log.info("--inspect-only: stopping before video extraction.")
            return

        # 3. Build basename → archive-path index for selective extraction.
        log.info("indexing video members of the archive (one full pass)...")
        video_idx = _index_zip_videos(zf)
        log.info("archive contains %d video members", len(video_idx))

        # 4. Selectively extract videos for the rows we care about.
        for i, (split, lex, video_basename, pid) in enumerate(wanted_rows):
            sid = gloss_lookup[lex]
            arc_path = video_idx.get(video_basename.lower())
            if not arc_path:
                records.append(
                    ClipRecord(
                        sign_id=sid, gloss=lex, source="asl_citizen",
                        wlasl_video_id=video_basename, original_url=f"asl_citizen://{video_basename}",
                        source_signer_id=pid, source_split=split,
                        frame_start=None, frame_end=None,
                        local_path=None, download_status="missing",
                    )
                )
                continue
            dst = clips_dir / Path(video_basename).name
            if not dst.exists():
                try:
                    with zf.open(arc_path) as src, dst.open("wb") as out:
                        shutil.copyfileobj(src, out)
                except Exception as e:  # noqa: BLE001 — robustness over precision; mark and continue
                    log.warning("extract failed %s: %s", video_basename, e)
                    records.append(
                        ClipRecord(
                            sign_id=sid, gloss=lex, source="asl_citizen",
                            wlasl_video_id=video_basename,
                            original_url=f"asl_citizen://{video_basename}",
                            source_signer_id=pid, source_split=split,
                            frame_start=None, frame_end=None,
                            local_path=None, download_status="extract-error",
                        )
                    )
                    continue
            per_sign_ok[sid] += 1
            records.append(
                ClipRecord(
                    sign_id=sid, gloss=lex, source="asl_citizen",
                    wlasl_video_id=video_basename, original_url=f"asl_citizen://{video_basename}",
                    source_signer_id=pid, source_split=split,
                    frame_start=None, frame_end=None,
                    local_path=str(dst), download_status="ok",
                )
            )

            if (i + 1) % 200 == 0:
                _write_manifest(manifest_path, zip_path, per_sign_attempts, per_sign_ok, records)
                log.info("checkpoint: %d/%d clips extracted (%d ok)", i + 1, len(wanted_rows), sum(per_sign_ok.values()))

    _write_manifest(manifest_path, zip_path, per_sign_attempts, per_sign_ok, records)
    log.info(
        "wrote %s with %d records (%d extracted)",
        manifest_path,
        len(records),
        sum(per_sign_ok.values()),
    )

    if zip_url and zip_path and not keep_zip:
        # We're done with the 42.8 GB archive; reclaim the disk.
        zip_path.unlink(missing_ok=True)
        log.info("removed %s (--keep-zip to retain)", zip_path)


def _write_manifest(
    manifest_path: Path,
    zip_path: Path,
    per_sign_attempts: dict[str, int],
    per_sign_ok: dict[str, int],
    records: list[ClipRecord],
) -> None:
    manifest = {
        "source": "asl_citizen",
        "license": "MSR-LA (Microsoft Research License Agreement); non-commercial research use only; no redistribution",
        "citation": "Desai et al. 2023, arXiv:2304.05934",
        "zip_path": str(zip_path),
        "per_sign_attempts": dict(per_sign_attempts),
        "per_sign_downloadable": dict(per_sign_ok),
        "records": [asdict(r) for r in records],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest_path.with_suffix(".tmp.json")
    with tmp.open("w") as f:
        json.dump(manifest, f, indent=2)
    tmp.replace(manifest_path)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--zip-url", type=str, help="EULA-gated URL from Microsoft Download Center.")
    grp.add_argument("--zip-path", type=Path, help="Path to a locally-staged ASL_Citizen.zip.")
    parser.add_argument("--output", type=Path, default=Path("dataset/raw/asl_citizen"))
    parser.add_argument("--manifest", type=Path, default=Path("dataset/raw/asl_citizen_manifest.json"))
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Read the CSV splits only, print per-sign candidate counts, then exit before extracting videos.",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Don't delete the 42.8 GB ZIP after extraction. Default deletes when --zip-url was used.",
    )
    args = parser.parse_args()
    ingest(args.zip_url, args.zip_path, args.output, args.manifest, args.inspect_only, args.keep_zip)
    return 0


if __name__ == "__main__":
    sys.exit(main())
