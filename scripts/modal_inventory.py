"""One-shot Modal CPU job: count inodes + bytes per top-level dir on the
asl-mastery-data volume. Read-only; tiny CPU spend (~$0.001).

Usage:
    modal run scripts/modal_inventory.py
"""

import json
import subprocess
from pathlib import Path

import modal

app = modal.App("asl-inventory")
volume = modal.Volume.from_name("asl-mastery-data")
image = modal.Image.debian_slim().apt_install("findutils", "coreutils")


@app.function(image=image, volumes={"/data": volume}, timeout=600)
def inventory():
    root = Path("/data")
    results = []
    for child in sorted(root.iterdir()):
        if not child.exists():
            continue
        # inode count
        if child.is_dir():
            n = int(subprocess.check_output(
                f"find {child} -mindepth 1 | wc -l", shell=True
            ).decode().strip())
            try:
                size_bytes = int(subprocess.check_output(
                    f"du -sb {child}", shell=True
                ).decode().split()[0])
            except Exception:
                size_bytes = -1
        else:
            n = 1
            size_bytes = child.stat().st_size
        results.append({
            "path": str(child.relative_to(root)),
            "inodes": n,
            "size_gb": round(size_bytes / 1e9, 2) if size_bytes >= 0 else None,
            "is_dir": child.is_dir(),
        })
    total_inodes = sum(r["inodes"] for r in results)
    print(f"\n{'PATH':<35}{'INODES':>12}{'GB':>10}")
    print("-" * 57)
    for r in sorted(results, key=lambda x: -x["inodes"]):
        gb = f"{r['size_gb']:.2f}" if r["size_gb"] is not None else "?"
        print(f"{r['path']:<35}{r['inodes']:>12,}{gb:>10}")
    print("-" * 57)
    print(f"{'TOTAL':<35}{total_inodes:>12,}")
    return results


@app.local_entrypoint()
def main():
    results = inventory.remote()
    Path("data/_modal_inventory.json").parent.mkdir(parents=True, exist_ok=True)
    Path("data/_modal_inventory.json").write_text(json.dumps(results, indent=2))
    print("\nwrote data/_modal_inventory.json")
