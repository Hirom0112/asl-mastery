"""Modal entry points for the slice-1 training pipeline.

Per docs/ROADMAP.md Phase 4 and docs/MODEL.md §2: training, validation,
and ONNX export run on GPU in a reproducible container. The yt-dlp
ingestion (Phase 3d) and the MP4 cleaning (Phase 3f) run locally on
your laptop because they are CPU-bound and benefit from a residential
IP for YouTube; the cleaned dataset is then uploaded to the Modal
Volume below and the training functions read from it.

Under ADR 0010 (reversal of ADR 0006) the cleaning pipeline produces
per-clip MP4s, not MediaPipe keypoint tensors. The training-time
dataset loader (written in T4) reads MP4s with torchvision.io.

Usage:

    # 0. One-time: authenticate Modal CLI.
    modal token new

    # 1. Push cleaned dataset to the Modal Volume.
    modal volume create asl-mastery-data    # idempotent
    modal volume put asl-mastery-data dataset/clean/v1 /datasets/v1

    # 2. Train on a GPU (SmallR2Plus1D 3D CNN under ADR 0010).
    modal run training/modal_app.py::train \\
        --manifest /datasets/v3/dataset_v3_manifest.json \\
        --run-id v3-001 \\
        --epochs 60

    # 3. Validate.
    modal run training/modal_app.py::validate \\
        --manifest /datasets/v1/dataset_v1_manifest.json \\
        --run-id v1-001

    # 4. Export ONNX artifact.
    modal run training/modal_app.py::export \\
        --run-id v1-001 \\
        --artifact-version v1.0.0

    # 5. Pull artifacts back.
    modal volume get asl-mastery-data /runs/v1-001 ./runs/v1-001
    modal volume get asl-mastery-data /artifacts/v1.0.0 ./artifacts/v1.0.0

The image is built from training/requirements.txt; the assertion in
training/classifier/init.py that there is no load_state_dict on the
classifier still fires in this container.
"""

from __future__ import annotations

from pathlib import Path

import modal

_REQS = str(Path(__file__).parent / "requirements.txt")

# Image: Debian slim + Python 3.12 + ffmpeg (for any optional video
# re-encodes the GPU container might run) + our pinned requirements.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install_from_requirements(_REQS)
    .add_local_python_source("training")
)

app = modal.App("asl-mastery-training", image=image)

# Persistent volume for datasets, run checkpoints, and exported artifacts.
volume = modal.Volume.from_name("asl-mastery-data", create_if_missing=True)
VOLUME_PATH = "/data"

# GPU choice: L4 is the cheapest "real" GPU on Modal and is plenty for
# the ~200K-param BiLSTM. Swap to "A10G" if Phase 4 measurement shows
# meaningful per-epoch differences.
GPU = "L4"
TIMEOUT_SEC = 60 * 60  # 1 hour per call; the BiLSTM trains in minutes.


@app.function(
    cpu=2.0,
    memory=2048,
    volumes={VOLUME_PATH: volume},
    timeout=60 * 60,  # 1 hr budget — Modal's bandwidth should land 42 GB in well under that
)
def download_drive(file_specs: str) -> dict:
    """Download large files from Google Drive directly into the Modal volume.

    Bypasses Drive's per-IP quota that throttles laptop downloads — Modal's
    egress is on a different IP and typically a much fatter pipe. Handles
    Drive's "virus scan warning" interstitial by parsing the uuid token out
    and re-requesting via drive.usercontent.google.com.

    ``file_specs`` is a semicolon-delimited list of ``<drive_id>=<volume_path>``
    pairs, e.g. ``"1abc...=raw/sem_lex/train.tar.gz;1def...=raw/sem_lex/val.tar.gz"``.
    """
    import os
    import re
    import urllib.request
    import urllib.error

    base = Path(VOLUME_PATH)
    os.chdir(base)

    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s drive: %(message)s", force=True)
    log = logging.getLogger("drive")

    results: list[dict] = []
    for spec in file_specs.split(";"):
        spec = spec.strip()
        if not spec or "=" not in spec:
            continue
        file_id, rel_path = spec.split("=", 1)
        file_id = file_id.strip()
        rel_path = rel_path.strip().lstrip("/")
        dst = base / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)

        try:
            log.info("→ %s as %s", file_id[:16] + "...", rel_path)
            # 1. Hit the warning page to extract the uuid confirmation token.
            warning_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            req = urllib.request.Request(
                warning_url, headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                html = resp.read().decode("utf-8", errors="ignore")
            log.info("  warning page %d bytes; title: %s", len(html),
                     (re.search(r'<title>([^<]+)</title>', html) or ['?', '?'])[1] if re.search(r'<title>([^<]+)</title>', html) else 'none')
            uuid_match = re.search(r'name="uuid" value="([^"]+)"', html)
            if uuid_match:
                uuid = uuid_match.group(1)
                download_url = (
                    "https://drive.usercontent.google.com/download"
                    f"?id={file_id}&export=download&confirm=t&uuid={uuid}"
                )
            else:
                download_url = warning_url  # small file, no confirmation needed

            # 2. Stream to disk.
            req = urllib.request.Request(
                download_url, headers={"User-Agent": "Mozilla/5.0"}
            )
            total = 0
            with urllib.request.urlopen(req, timeout=600) as resp:
                log.info("  GET %d %s, cl=%s", resp.status, resp.headers.get("Content-Type"), resp.headers.get("Content-Length"))
                with dst.open("wb") as f:
                    last_log = 0
                    while True:
                        chunk = resp.read(1024 * 1024)  # 1 MiB chunks
                        if not chunk:
                            break
                        f.write(chunk)
                        total += len(chunk)
                        if total - last_log > 100 * 1024 * 1024:
                            log.info("    %.1f GB downloaded", total / 1e9)
                            last_log = total

            # 3. Sanity: Drive serves a 2 KB HTML "Quota exceeded" page when blocked.
            if total < 100_000 and dst.suffix in (".tar", ".tar.gz", ".gz", ".zip"):
                head = dst.read_bytes()[:1024].decode("utf-8", errors="ignore")
                if "<title>" in head.lower() or "quota" in head.lower():
                    results.append(
                        {"id": file_id, "path": rel_path, "bytes": total, "status": "drive-quota-block"}
                    )
                    dst.unlink(missing_ok=True)
                    continue
            results.append(
                {"id": file_id, "path": rel_path, "bytes": total, "status": "ok"}
            )
        except Exception as e:  # noqa: BLE001 — best-effort; report and continue
            results.append({"id": file_id, "path": rel_path, "error": str(e), "status": "exception"})

    volume.commit()
    return {"downloads": results}


@app.function(
    cpu=8.0,
    memory=16384,
    volumes={VOLUME_PATH: volume},
    timeout=60 * 120,  # 2 hr budget (with periodic volume.commit, restarts are cheap)
)
def clean(
    raw_manifests: str,
    filter_path: str,
    output_subdir: str,
    version: str,
    seed: int = 42,
    workers: int = 8,
) -> dict:
    """Run the cleaning pipeline (ffmpeg normalize + dedup + split
    assignment + manifest write) over a set of raw manifests that
    already live on the Modal volume.

    Under ADR 0010 the cleaning pipeline outputs per-clip MP4s only
    (no MediaPipe keypoint stage). The `max_miss_rate` parameter that
    rejected clips with high MediaPipe per-frame miss rates under
    ADR 0006 is removed; the v3.x training-time dataset loader sees
    every cleaned clip.

    All paths in arguments are *relative to the volume root* (e.g.
    ``raw/wlasl_manifest.json``); we resolve them to absolute container
    paths here.
    """
    import logging
    import os
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", force=True)

    from training.data.clean import clean as _clean

    base = Path(VOLUME_PATH)
    # Cwd to volume root so the relative `local_path` entries inside the
    # raw manifests (e.g. "dataset/raw/asl_citizen/clips/x.mp4") resolve
    # against the mirrored upload layout at `/data/dataset/raw/...`.
    os.chdir(base)

    raw_paths = [base / p.strip().lstrip("/") for p in raw_manifests.split(",") if p.strip()]
    filt = base / filter_path.lstrip("/")
    out = base / output_subdir.lstrip("/")

    # Periodically commit MP4 writes to the volume so an unexpected
    # timeout doesn't lose all progress. A background thread snapshots
    # the volume every 60 s; the idempotent path in `clean()` finds
    # already-normalized MP4s on a retry and skips ffmpeg for them.
    import threading

    stop_evt = threading.Event()

    def _periodic_commit() -> None:
        while not stop_evt.wait(60):
            try:
                volume.commit()
            except Exception:  # noqa: BLE001
                pass

    committer = threading.Thread(target=_periodic_commit, daemon=True)
    committer.start()
    try:
        _clean(
            raw_paths,
            filt,
            out,
            version=version,
            seed=seed,
            workers=workers,
        )
    finally:
        stop_evt.set()
        committer.join(timeout=2)
    volume.commit()
    return {"version": version, "output": str(out)}


@app.function(
    gpu=GPU,
    volumes={VOLUME_PATH: volume},
    timeout=TIMEOUT_SEC,
)
def train(
    manifest: str,
    run_id: str,
    epochs: int = 60,
    batch_size: int = 32,
    lr: float = 1e-3,
    seed: int = 42,
    input_size: int = 96,
    num_workers: int = 2,
    background_bank: str = "",
    bg_swap_prob: float = 0.5,
) -> dict:
    """Train the v3.x SmallR2Plus1D 3D CNN. `manifest` is a path inside
    the volume (e.g. `/datasets/v3/dataset_v3_manifest.json`). Outputs
    land at `/runs/<run_id>/best.pt` + `run.json`.

    Under ADR 0010 the only architecture is the from-scratch
    R(2+1)D-style 3D CNN — the `--model bilstm/transformer` flag from
    the superseded ADR 0006 era is gone.
    """
    import argparse
    from pathlib import Path

    from training.classifier.train import train as _train

    output = Path(f"{VOLUME_PATH}/runs/{run_id}")
    output.mkdir(parents=True, exist_ok=True)

    bg_path = (
        Path(f"{VOLUME_PATH}{background_bank}") if background_bank.startswith("/")
        else (Path(background_bank) if background_bank else None)
    )

    args = argparse.Namespace(
        manifest=Path(f"{VOLUME_PATH}{manifest}") if manifest.startswith("/") else Path(manifest),
        output=output,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        early_stop_patience=8,
        input_size=input_size,
        num_workers=num_workers,
        background_bank=bg_path,
        bg_swap_prob=bg_swap_prob,
    )
    _train(args)
    volume.commit()
    return {"run_id": run_id, "output": str(output)}


@app.function(
    gpu=GPU,
    volumes={VOLUME_PATH: volume},
    timeout=TIMEOUT_SEC,
)
def validate(manifest: str, run_id: str) -> dict:
    """Run validation against the held-out test split. Outputs
    `validation.json` + `validation.md` next to the checkpoint.
    """
    import argparse
    from pathlib import Path

    from training.classifier.validate import run as _run

    output = Path(f"{VOLUME_PATH}/runs/{run_id}")
    args = argparse.Namespace(
        manifest=Path(f"{VOLUME_PATH}{manifest}") if manifest.startswith("/") else Path(manifest),
        checkpoint=output / "best.pt",
        output=output,
    )
    _run(args)
    volume.commit()
    return {"run_id": run_id, "validation": str(output / "validation.json")}


@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=TIMEOUT_SEC,
)
def export(run_id: str, artifact_version: str = "v1.0.0") -> dict:
    """Export the trained classifier to ONNX, run a PyTorch parity
    check, and write the artifact bundle to
    `/artifacts/<artifact_version>/`. No GPU required.
    """
    import argparse
    from pathlib import Path

    from training.classifier.export import export as _export

    run_dir = Path(f"{VOLUME_PATH}/runs/{run_id}")
    out_dir = Path(f"{VOLUME_PATH}/artifacts/{artifact_version}")
    out_dir.mkdir(parents=True, exist_ok=True)

    args = argparse.Namespace(
        checkpoint=run_dir / "best.pt",
        validation=run_dir / "validation.json",
        output=out_dir,
        version=artifact_version,
    )
    _export(args)
    volume.commit()
    return {"artifact_version": artifact_version, "output": str(out_dir)}


@app.local_entrypoint()
def main(
    manifest: str = "/datasets/v3/dataset_v3_manifest.json",
    run_id: str = "v3-001",
    artifact_version: str = "v3.0.0",
    epochs: int = 60,
):
    """Convenience: run train → validate → export end-to-end in one call.

    Usage:
        modal run training/modal_app.py \\
            --manifest /datasets/v3/dataset_v3_manifest.json \\
            --run-id v3-001 \\
            --artifact-version v3.0.0
    """
    print(f"▶ training {run_id}…")
    train.remote(manifest=manifest, run_id=run_id, epochs=epochs)
    print(f"▶ validating {run_id}…")
    validate.remote(manifest=manifest, run_id=run_id)
    print(f"▶ exporting → {artifact_version}…")
    export.remote(run_id=run_id, artifact_version=artifact_version)
    print(f"✓ done. Pull artifacts with: modal volume get asl-mastery-data /artifacts/{artifact_version} ./artifacts/{artifact_version}")
