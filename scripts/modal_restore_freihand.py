"""Restore /external/freihand on the Modal volume (deleted in the 2026-05-21
inode cleanup; needed for hand_det training in Job A).

Single-purpose: downloads + extracts the two FreiHAND archives from the
upstream Freiburg URLs into /external/freihand/. ~10 min CPU job, ~$0.05.
"""

import subprocess
from pathlib import Path

import modal

app = modal.App("asl-restore-freihand")
volume = modal.Volume.from_name("asl-mastery-data")
image = modal.Image.debian_slim().apt_install("aria2", "unzip", "curl")

URLS = [
    ("FreiHAND_pub_v2.zip",
     "https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2.zip"),
    ("FreiHAND_pub_v2_eval.zip",
     "https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2_eval.zip"),
]


@app.function(
    image=image,
    volumes={"/data": volume},
    cpu=4.0,
    memory=4096,
    timeout=60 * 30,
)
def restore():
    root = Path("/data/external/freihand")
    root.mkdir(parents=True, exist_ok=True)
    for fname, url in URLS:
        out = root / fname
        if out.exists() and out.stat().st_size > 1_000_000:
            print(f"  [skip] {fname} already present")
            continue
        print(f"  downloading {fname}...")
        # Same flags as the successful setup_external_datasets() run.
        rc = subprocess.call([
            "aria2c", "-x", "16", "-s", "16",
            "--check-certificate=false",
            "--file-allocation=none",
            "--auto-file-renaming=false",
            "--allow-overwrite=true",
            "--connect-timeout=60",
            "--timeout=120",
            "--max-tries=5",
            "--retry-wait=10",
            "-c",
            "-d", str(root), "-o", fname, url,
        ])
        if rc != 0:
            # Fallback: try curl with long timeout
            print(f"    aria2c rc={rc}, falling back to curl...")
            subprocess.check_call([
                "curl", "-L", "-C", "-", "--fail",
                "--connect-timeout", "60",
                "--max-time", "1800",
                "--retry", "5", "--retry-delay", "10",
                "-o", str(out), url,
            ])
        print(f"    {out.stat().st_size / 1e9:.2f} GB")
    for fname, _ in URLS:
        zp = root / fname
        print(f"  extracting {fname}...")
        subprocess.check_call(["unzip", "-q", "-o", str(zp), "-d", str(root)])
    volume.commit()
    # Quick sanity check
    n = sum(1 for _ in root.rglob("*.jpg"))
    print(f"\nfreihand restored: {n:,} jpgs at {root}")


@app.local_entrypoint()
def main():
    restore.remote()
