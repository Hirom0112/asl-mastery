"""Shared helpers for external-dataset download scripts.

Per ADR 0015 + docs/data/external_datasets_audit.md, every download
script must:
- target /data/external/<dataset_name>/
- write a PROVENANCE.md recording source URL, download date, license,
  and SHA256 of every downloaded archive
- verify checksums when the dataset ships them
- never auto-execute — the calling script must require an explicit
  --confirm flag so a `bash script.py` accident does not start tens
  of GB of downloads

This module ships no networking by itself. It provides the file-system
and bookkeeping primitives the per-dataset scripts compose.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_ROOT = REPO_ROOT / "data" / "external"


def parse_common_args(dataset_name: str) -> argparse.Namespace:
    """Standard CLI surface every dataset script accepts."""
    parser = argparse.ArgumentParser(
        prog=f"download_{dataset_name}.py",
        description=textwrap.dedent(
            f"""\
            Download the {dataset_name} dataset under ADR 0015.

            By default this script is a NO-OP and prints the plan.
            Pass --confirm to actually download. Pass --dry-run to
            print every step that would happen without touching the
            network or disk.
            """
        ).rstrip(),
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Required to actually download. Without it the script prints the plan and exits 0.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print every step without touching network or disk. Overrides --confirm.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=f"Override target directory. Default: {EXTERNAL_ROOT / dataset_name}",
    )
    return parser.parse_args()


def target_dir(dataset_name: str, override: Path | None) -> Path:
    return override.resolve() if override else (EXTERNAL_ROOT / dataset_name)


def ensure_dir(path: Path, *, dry_run: bool) -> None:
    if dry_run:
        print(f"[dry-run] mkdir -p {path}")
        return
    path.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def have_cmd(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def run_or_print(cmd: list[str], *, dry_run: bool) -> int:
    if dry_run:
        print(f"[dry-run] $ {' '.join(cmd)}")
        return 0
    print(f"$ {' '.join(cmd)}")
    return subprocess.call(cmd)


def download_file(
    url: str,
    out_path: Path,
    *,
    expected_sha256: str | None = None,
    dry_run: bool,
) -> str:
    """Curl-or-wget download with mid-stream resume and SHA256 verification.

    Returns the SHA256 of the on-disk file (empty string in dry-run).
    """
    if dry_run:
        print(f"[dry-run] download {url} -> {out_path}")
        return ""

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        print(f"already on disk: {out_path}")
    else:
        if have_cmd("curl"):
            rc = subprocess.call(
                ["curl", "-L", "-C", "-", "--fail", "--output", str(out_path), url]
            )
        elif have_cmd("wget"):
            rc = subprocess.call(["wget", "-c", "-O", str(out_path), url])
        else:
            print("ERROR: neither curl nor wget on PATH", file=sys.stderr)
            sys.exit(2)
        if rc != 0:
            print(f"ERROR: download failed with exit {rc}", file=sys.stderr)
            sys.exit(rc)

    digest = sha256_file(out_path)
    if expected_sha256:
        if digest.lower() != expected_sha256.lower():
            print(
                f"ERROR: SHA256 mismatch for {out_path.name}\n"
                f"  expected: {expected_sha256}\n"
                f"  got:      {digest}",
                file=sys.stderr,
            )
            sys.exit(3)
        print(f"  SHA256 OK ({digest[:16]}…) — matches expected")
    else:
        print(f"  SHA256 recorded ({digest[:16]}…) — no expected hash supplied")
    return digest


def write_provenance(
    target: Path,
    *,
    dataset_name: str,
    source_urls: list[str],
    archives: dict[str, str],
    license_summary: str,
    license_url: str | None,
    originating_paper: str,
    audit_entry: str,
    notes: str | None = None,
    dry_run: bool,
) -> None:
    """Write data/external/<dataset>/PROVENANCE.md.

    archives: mapping of on-disk archive filename -> SHA256.
    """
    body_lines = [
        f"# PROVENANCE — {dataset_name}",
        "",
        f"**Downloaded:** {datetime.now(timezone.utc).isoformat()}",
        f"**Audit entry:** `docs/data/external_datasets_audit.md` § {audit_entry}",
        f"**Governing ADR:** [ADR 0015](../../../docs/decisions/0015-external-cv-datasets-provenance.md)",
        "",
        "## Source URLs",
        "",
    ]
    for u in source_urls:
        body_lines.append(f"- {u}")
    body_lines += ["", "## Originating paper", "", originating_paper, ""]
    body_lines += ["## License", "", license_summary]
    if license_url:
        body_lines.append("")
        body_lines.append(f"License text: {license_url}")
    body_lines += ["", "## Archive SHA256 checksums", ""]
    if archives:
        for fname, digest in archives.items():
            body_lines.append(f"- `{fname}` — `{digest}`")
    else:
        body_lines.append("- (no archives recorded — dry-run or no archives downloaded)")
    if notes:
        body_lines += ["", "## Notes", "", notes]

    out = target / "PROVENANCE.md"
    if dry_run:
        print(f"[dry-run] would write {out}:")
        print(textwrap.indent("\n".join(body_lines), "    "))
        return
    out.write_text("\n".join(body_lines) + "\n")
    print(f"wrote {out}")


def banner_plan(dataset_name: str, plan_lines: list[str]) -> None:
    """Print the dry-plan banner shown when --confirm is missing."""
    bar = "─" * 60
    print(bar)
    print(f"  DOWNLOAD PLAN — {dataset_name}")
    print(bar)
    for line in plan_lines:
        print(f"  {line}")
    print(bar)
    print("  No download performed. Pass --confirm to execute, or")
    print("  --dry-run to see every command without touching disk.")
    print(bar)
