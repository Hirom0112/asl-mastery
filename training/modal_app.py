"""Modal entry points for the slice-1 training pipeline.

Per docs/ROADMAP.md Phase 4 and docs/MODEL.md §2: training, validation,
and ONNX export run on GPU in a reproducible container. The yt-dlp
ingestion (Phase 3d) and the MediaPipe cleaning (Phase 3f) run locally
on your laptop because they are CPU-bound and benefit from a
residential IP for YouTube; the cleaned dataset is then uploaded to
the Modal Volume below and the training functions read from it.

Usage:

    # 0. One-time: authenticate Modal CLI.
    modal token new

    # 1. Push cleaned dataset to the Modal Volume.
    modal volume create asl-mastery-data    # idempotent
    modal volume put asl-mastery-data dataset/clean/v1 /datasets/v1

    # 2. Train on a GPU.
    modal run training/modal_app.py::train \\
        --manifest /datasets/v1/dataset_v1_manifest.json \\
        --model bilstm \\
        --run-id v1-001

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

import modal

# Image: Debian slim + Python 3.12 + ffmpeg (for any optional video
# re-encodes the GPU container might run) + our pinned requirements.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install_from_requirements("requirements.txt")
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
    gpu=GPU,
    volumes={VOLUME_PATH: volume},
    timeout=TIMEOUT_SEC,
)
def train(
    manifest: str,
    run_id: str,
    model: str = "bilstm",
    epochs: int = 60,
    batch_size: int = 128,
    lr: float = 1e-3,
    seed: int = 42,
) -> dict:
    """Train a classifier. `manifest` is a path inside the volume
    (e.g. `/datasets/v1/dataset_v1_manifest.json`). Outputs land at
    `/runs/<run_id>/best.pt` + `run.json`.
    """
    import argparse
    from pathlib import Path

    from training.classifier.train import train as _train

    output = Path(f"{VOLUME_PATH}/runs/{run_id}")
    output.mkdir(parents=True, exist_ok=True)

    args = argparse.Namespace(
        manifest=Path(f"{VOLUME_PATH}{manifest}") if manifest.startswith("/") else Path(manifest),
        output=output,
        model=model,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        early_stop_patience=8,
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
    manifest: str = "/datasets/v1/dataset_v1_manifest.json",
    run_id: str = "v1-001",
    artifact_version: str = "v1.0.0",
    model: str = "bilstm",
    epochs: int = 60,
):
    """Convenience: run train → validate → export end-to-end in one call.

    Usage:
        modal run training/modal_app.py \\
            --manifest /datasets/v1/dataset_v1_manifest.json \\
            --run-id v1-001 \\
            --artifact-version v1.0.0
    """
    print(f"▶ training {run_id}…")
    train.remote(manifest=manifest, run_id=run_id, model=model, epochs=epochs)
    print(f"▶ validating {run_id}…")
    validate.remote(manifest=manifest, run_id=run_id)
    print(f"▶ exporting → {artifact_version}…")
    export.remote(run_id=run_id, artifact_version=artifact_version)
    print(f"✓ done. Pull artifacts with: modal volume get asl-mastery-data /artifacts/{artifact_version} ./artifacts/{artifact_version}")
