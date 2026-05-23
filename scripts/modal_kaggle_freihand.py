"""Download FreiHAND from Kaggle mirror directly into the Modal volume.

Uses the `kaggle-api` Modal Secret (KAGGLE_USERNAME + KAGGLE_KEY).
Mirror: danieldelro/freihand (~3.9 GB).

CPU-only, ~$0.05. Wall-clock ~1–2 min (Modal → Kaggle GCS is fast).
"""

import os
import subprocess
from pathlib import Path

import modal

app = modal.App("asl-kaggle-freihand")
volume = modal.Volume.from_name("asl-mastery-data")
image = (
    modal.Image.debian_slim()
    .apt_install("unzip")
    .pip_install("kaggle")
)


@app.function(
    image=image,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("kaggle-api")],
    cpu=4.0,
    memory=4096,
    timeout=60 * 20,
)
def download():
    root = Path("/data/external/freihand")
    root.mkdir(parents=True, exist_ok=True)

    # The kaggle CLI reads ~/.kaggle/kaggle.json or env vars.
    kdir = Path.home() / ".kaggle"
    kdir.mkdir(exist_ok=True)
    kjson = kdir / "kaggle.json"
    kjson.write_text(
        f'{{"username":"{os.environ["KAGGLE_USERNAME"]}",'
        f'"key":"{os.environ["KAGGLE_KEY"]}"}}'
    )
    kjson.chmod(0o600)

    print("downloading danieldelro/freihand → /data/external/freihand/ ...")
    rc = subprocess.call([
        "kaggle", "datasets", "download",
        "-d", "danieldelro/freihand",
        "-p", str(root),
        "--unzip",
    ])
    if rc != 0:
        raise RuntimeError(f"kaggle download failed rc={rc}")

    volume.commit()

    n_train = sum(1 for _ in (root / "training" / "rgb").glob("*.jpg")) if (root / "training" / "rgb").exists() else 0
    n_eval = sum(1 for _ in (root / "evaluation" / "rgb").glob("*.jpg")) if (root / "evaluation" / "rgb").exists() else 0
    has_json = list(root.glob("*.json"))
    print(f"\ntrain JPGs: {n_train:,}  eval JPGs: {n_eval:,}  JSONs: {[j.name for j in has_json]}")


@app.local_entrypoint()
def main():
    download.remote()
