"""Tar superseded trajectory dirs on the volume and drill into /external.

Phase 1: tar `/trajectories`, `/trajectories_v2`, `/trajectories_v3`,
         `/trajectories_v4` into `/_archive/<name>.tar`, then remove the
         original directory (tar preserves every byte).
Phase 2: list `/external` contents + per-subdir inode count, so we can
         pick the next tar target.

CPU-only, ~$0.005 total.
"""

import subprocess
from pathlib import Path

import modal

app = modal.App("asl-tar-old-traj")
volume = modal.Volume.from_name("asl-mastery-data")
image = modal.Image.debian_slim().apt_install("tar", "findutils", "coreutils")

OLD_TRAJ_DIRS = ["trajectories", "trajectories_v2", "trajectories_v3", "trajectories_v4"]


@app.function(image=image, volumes={"/data": volume}, timeout=1800)
def cleanup():
    root = Path("/data")
    archive_dir = root / "_archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    # ----- Phase 1: tar + rm superseded trajectory dirs -----
    print("=" * 60)
    print("PHASE 1: tar + rm superseded trajectory dirs")
    print("=" * 60)
    total_freed = 0
    for name in OLD_TRAJ_DIRS:
        d = root / name
        if not d.exists():
            print(f"  [skip] {name}: not present")
            continue
        n_before = int(subprocess.check_output(
            f"find {d} -mindepth 1 | wc -l", shell=True).decode().strip())
        tar_path = archive_dir / f"{name}.tar"
        if tar_path.exists():
            print(f"  [skip] {tar_path} already exists")
            continue
        print(f"  taring {name} ({n_before:,} inodes)...")
        subprocess.check_call(
            f"tar -cf {tar_path} -C {root} {name}", shell=True)
        tar_size = tar_path.stat().st_size
        print(f"    → {tar_path} ({tar_size / 1e6:.1f} MB)")
        print(f"    removing original tree...")
        subprocess.check_call(f"rm -rf {d}", shell=True)
        total_freed += n_before
        # 1 inode for the tarball, but tarball lives in _archive
        print(f"    freed ~{n_before - 1:,} inodes")

    # Commit the volume changes (so the rm + new tarballs are persisted)
    volume.commit()
    print(f"\nphase 1 freed ~{total_freed - len(OLD_TRAJ_DIRS):,} inodes "
          f"(minus {len(OLD_TRAJ_DIRS)} tarballs)")

    # ----- Phase 2: inventory /external -----
    print()
    print("=" * 60)
    print("PHASE 2: /external breakdown")
    print("=" * 60)
    ext = root / "external"
    if ext.exists():
        rows = []
        for child in sorted(ext.iterdir()):
            try:
                n = int(subprocess.check_output(
                    f"find {child} -mindepth 1 2>/dev/null | wc -l",
                    shell=True).decode().strip())
            except Exception:
                n = -1
            try:
                gb = int(subprocess.check_output(
                    f"du -sb {child} 2>/dev/null", shell=True
                ).decode().split()[0]) / 1e9
            except Exception:
                gb = -1.0
            rows.append((str(child.relative_to(root)), n, gb))
        rows.sort(key=lambda r: -r[1])
        print(f"\n{'PATH':<45}{'INODES':>12}{'GB':>10}")
        print("-" * 67)
        for path, n, gb in rows:
            print(f"{path:<45}{n:>12,}{gb:>10.2f}")
    else:
        print("  /external not present")


@app.local_entrypoint()
def main():
    cleanup.remote()
