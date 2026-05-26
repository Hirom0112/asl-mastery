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
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0", "aria2", "curl", "unzip")
    .pip_install_from_requirements(_REQS)
    # remotezip: HTTP-Range based partial extraction from a remote .zip
    # without downloading the whole file. (Used in early HaGRID download
    # attempt; superseded by HuggingFace mirror path.)
    .pip_install("remotezip>=0.12")
    # huggingface_hub + datasets: HaGRID via cj-mills/hagrid-sample-500k-384p
    # mirror lives on HuggingFace Cloudflare CDN, ~30× faster from US Modal
    # than Sbercloud Moscow. 13.4 GB total instead of 119 GB.
    .pip_install("huggingface_hub>=0.30", "datasets>=3.0", "pyarrow>=15.0")
    .add_local_python_source("training")
    .add_local_python_source("scripts")
)

app = modal.App("asl-mastery-training", image=image)

# Persistent volume for datasets, run checkpoints, and exported artifacts.
volume = modal.Volume.from_name("asl-mastery-data", create_if_missing=True)
VOLUME_PATH = "/data"

# GPU choice: L4 is the cheapest "real" GPU on Modal and is plenty for
# the ~200K-param BiLSTM. Swap to "A10G" if Phase 4 measurement shows
# meaningful per-epoch differences.
GPU = "L4"
# 12 hours per call. The from-scratch detectors at 60 epochs over
# 60K+ labeled hand samples on an L4 land around ~10 hours wall-time
# per detector (first epoch ~30 min for data caching, subsequent
# epochs ~10 min). The pre-pivot BiLSTM that this constant used to be
# sized for trained in minutes; the new architecture needs the runway.
TIMEOUT_SEC = 60 * 60 * 12


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
    timeout=60 * 60,
)
def download_and_prep_wflw() -> dict:
    """Download WFLW (98 human-annotated face landmarks; ADR-0015 compliant),
    extract, and build the face_keypoints train/val manifests. Images via
    Google Drive (724 MB), annotations via direct HTTP (16 MB)."""
    import json
    import re
    import tarfile
    import urllib.request
    from pathlib import Path

    root = Path(f"{VOLUME_PATH}/external/wflw")
    root.mkdir(parents=True, exist_ok=True)

    def _gdrive(file_id: str, dst: Path) -> None:
        if dst.exists() and dst.stat().st_size > 1e8:
            print(f"  {dst.name} already present"); return
        warn = f"https://drive.google.com/uc?export=download&id={file_id}"
        req = urllib.request.Request(warn, headers={"User-Agent": "Mozilla/5.0"})
        html = urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "ignore")
        m = re.search(r'name="uuid" value="([^"]+)"', html)
        url = (f"https://drive.usercontent.google.com/download?id={file_id}"
               f"&export=download&confirm=t&uuid={m.group(1)}") if m else warn
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        total = 0
        with urllib.request.urlopen(req, timeout=1800) as resp, dst.open("wb") as f:
            while True:
                c = resp.read(1 << 20)
                if not c:
                    break
                f.write(c); total += len(c)
        print(f"  downloaded {dst.name}: {total/1e9:.2f} GB")

    def _http(url: str, dst: Path) -> None:
        if dst.exists():
            print(f"  {dst.name} already present"); return
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=300) as resp, dst.open("wb") as f:
            f.write(resp.read())
        print(f"  downloaded {dst.name}: {dst.stat().st_size/1e6:.1f} MB")

    img_tar = root / "WFLW_images.tar.gz"
    ann_tar = root / "WFLW_annotations.tar.gz"
    _gdrive("1hzBd48JIdWTJSsATBEB_eFVvPL1bx6UC", img_tar)
    _http("https://wywu.github.io/projects/LAB/support/WFLW_annotations.tar.gz", ann_tar)
    for tar, marker in ((img_tar, root / "WFLW_images"),
                        (ann_tar, root / "WFLW_annotations")):
        if not marker.exists():
            print(f"  extracting {tar.name} ...")
            with tarfile.open(tar) as tf:
                tf.extractall(root)
    volume.commit()

    from training.detectors.external_loaders import wflw
    tr, va = wflw.load_face_keypoints(root)
    out = Path(f"{VOLUME_PATH}/labeled_frames/face_keypoints")
    out.mkdir(parents=True, exist_ok=True)
    (out / "train.json").write_text(json.dumps({"version": 1, "task": "face_keypoints", "items": tr}))
    (out / "val.json").write_text(json.dumps({"version": 1, "task": "face_keypoints", "items": va}))
    volume.commit()
    return {"train": len(tr), "val": len(va),
            "train_manifest": "/labeled_frames/face_keypoints/train.json",
            "val_manifest": "/labeled_frames/face_keypoints/val.json"}


@app.function(
    gpu="H100",
    volumes={VOLUME_PATH: volume},
    timeout=TIMEOUT_SEC,
)
def train_face_landmarks(
    run_id: str,
    train_manifest: str = "/labeled_frames/face_keypoints/train.json",
    val_manifest: str = "/labeled_frames/face_keypoints/val.json",
    epochs: int = 120,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    num_workers: int = 8,
) -> dict:
    """Train the from-scratch 98-keypoint FACE landmark regressor on WFLW.
    Reuses the keypoint-agnostic train_landmarks loop (num_keypoints=98,
    instance_key='faces', 224² crops). Small dataset (7.5k imgs) → fast."""
    from pathlib import Path
    from training.detectors.train_landmarks import train as _train
    run_dir = Path(f"{VOLUME_PATH}/runs/{run_id}")
    result = _train(
        train_manifest=_resolve_volume(train_manifest),
        val_manifest=_resolve_volume(val_manifest),
        run_dir=run_dir, epochs=epochs, batch_size=batch_size, lr=lr,
        weight_decay=weight_decay, num_workers=num_workers,
        num_keypoints=98, instance_key="faces", input_size=224,
    )
    volume.commit()
    return {"run_id": run_id, **result}


@app.function(
    gpu="L4",
    volumes={VOLUME_PATH: volume},
    timeout=60 * 60,
    cpu=16.0,
    memory=32 * 1024,
)
def measure_v10(traj_dir: str = "/trajectories_v10",
                sources: str = "sem_lex", epochs: int = 80) -> dict:
    """Run the measurement ON the volume (no local pull): coverage diagnostic
    + the sem_lex signer-disjoint 100D classifier baseline on v10."""
    import sys
    traj = str(_resolve_volume(traj_dir))
    man = str(_resolve_volume("/labeled_frames/unified_clip_manifest_modal_v4.json"))
    vocab = str(_resolve_volume("/vocabulary/slice1b_vocabulary.json"))
    argv0 = sys.argv[:]

    print("=" * 60 + "\nCOVERAGE (p1_landmark_diag) on v10\n" + "=" * 60, flush=True)
    from scripts import p1_landmark_diag
    sys.argv = ["p1", "--traj-root", traj, "--manifest", man,
                "--out", str(_resolve_volume("/runs/p1_v10_coverage.json"))]
    try:
        p1_landmark_diag.main()
    finally:
        sys.argv = argv0

    print("\n" + "=" * 60 + "\nCLASSIFIER: sem_lex, signer-disjoint, 100D, natural "
          "(vs 11.1% v8 baseline)\n" + "=" * 60, flush=True)
    from scripts import p0_signer_baseline
    sys.argv = ["p0", "--traj-root", traj, "--manifest", man, "--vocab", vocab,
                "--sources", sources, "--feat-dim", "100",
                "--run-dir", str(_resolve_volume("/runs/p0_v10_semlex_100d")),
                "--epochs", str(epochs)]
    try:
        p0_signer_baseline.main()
    finally:
        sys.argv = argv0

    volume.commit()
    return {"done": True, "run_dir": "/runs/p0_v10_semlex_100d"}


@app.function(
    gpu="L4",
    volumes={VOLUME_PATH: volume},
    timeout=60 * 90,
    cpu=16.0,
    memory=48 * 1024,
)
def measure_norm_ab(traj_dir: str = "/trajectories_v10",
                    sources: str = "sem_lex", epochs: int = 80,
                    confusion_signs: str = "", seed: int = 42,
                    save_model: bool = False, only: str = "",
                    manifest: str = "/labeled_frames/unified_clip_manifest_modal_v4.json",
                    vocab: str = "/vocabulary/slice1b_vocabulary.json") -> dict:
    """v2 A/B: in ONE job, train the sem_lex signer-disjoint classifier for
    BOTH norm="body" (v1, 100D) and norm="hand" (v2, 108D) on the SAME loaded
    frames, reporting top-1/top-5 (+ confusion sub-matrix if --confusion-signs
    is set). Reads the ~12k trajectory JSONs ONCE; builds both feature versions
    from the same in-memory frames (no double read). Writes a summary JSON to
    /runs/measure_norm_ab/summary.json.

    Mirrors measure_v10's structure. DO NOT confuse with measure_v10, which is
    body-only and shells out to p0_signer_baseline.main().
    """
    import json
    import time
    from collections import Counter

    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    from scripts.p0_signer_baseline import (
        DS, build_clip_to_signer, evaluate, print_confusion_submatrix,
        signer_disjoint_split, TIME_STEPS,
    )
    from training.detectors.sign_classifier import SignClassifier, count_parameters
    from training.detectors.sign_matcher import trajectory_from_frames
    from training.detectors.train_classifier import _load_vocab

    traj_root = _resolve_volume(traj_dir)
    man = _resolve_volume(manifest)
    vocab_path = _resolve_volume(vocab)
    out_dir = _resolve_volume("/runs/measure_norm_ab")
    out_dir.mkdir(parents=True, exist_ok=True)
    src_set = {s for s in sources.split(",") if s} if sources else set()
    conf_signs = [s for s in confusion_signs.split(",") if s]

    print("=" * 60 + "\nNORM A/B (body v1 100D vs hand v2 108D) — single load\n"
          + "=" * 60, flush=True)
    clip_meta = build_clip_to_signer(man)
    print(f"[ab] manifest clips: {len(clip_meta)}", flush=True)

    # ---- load every clip's frames ONCE; build BOTH feature versions -------
    # samples_<norm>: list of (sign, signer_key, source, feats_TF). Both lists
    # are built from the same frame dicts in lockstep so a clip either lands in
    # both or neither (drop a clip if EITHER norm yields an all-NaN trajectory).
    t_load = time.time()
    samples_body: list[tuple] = []
    samples_hand: list[tuple] = []
    not_in_manifest = 0
    n_seen = 0
    # Gather (sign, json_path) pairs, then read them in PARALLEL. The
    # network-volume per-file read latency dominates (13k cold reads ≈ 60 min
    # single-threaded); a thread pool cuts the load to a few minutes. Processing
    # logic below is unchanged — only the read_text() is parallelized.
    from concurrent.futures import ThreadPoolExecutor
    pairs: list[tuple] = []
    for sign_dir in sorted(traj_root.iterdir()):
        if not sign_dir.is_dir():
            continue
        for j in sorted(sign_dir.glob("*.json")):
            pairs.append((sign_dir.name, j))
    print(f"[ab] reading {len(pairs)} trajectory files (32 threads)…", flush=True)

    def _read_one(item):
        sign, j = item
        try:
            return sign, j, j.read_text()
        except Exception:
            return sign, j, None

    with ThreadPoolExecutor(max_workers=32) as _ex:
        loaded = list(_ex.map(_read_one, pairs))

    for sign, j, txt in loaded:
        if txt is None:
            continue
        try:
            t = json.loads(txt)
        except Exception:
            continue
        cp = t.get("clip_path")
        meta = clip_meta.get(cp)
        if meta is None:
            not_in_manifest += 1
            continue
        src = meta.get("source")
        if src_set and src not in src_set:
            continue
        frames = t.get("frames", [])
        if len(frames) < 2:
            continue
        # v2 has no embeddings; strip so the body (100D) path stays 100D.
        for f in frames:
            for h in (f.get("hands") or []):
                h.pop("embedding", None)
        feats_body = trajectory_from_frames(frames, TIME_STEPS, norm="body")
        feats_hand = trajectory_from_frames(frames, TIME_STEPS, norm="hand")
        if not (np.isfinite(feats_body).any() and np.isfinite(feats_hand).any()):
            continue
        sid = meta.get("signer_id")
        signer_key = f"{src}:{sid}" if sid is not None else f"{src}:clip:{j.stem}"
        samples_body.append((sign, signer_key, src, feats_body.astype(np.float32)))
        samples_hand.append((sign, signer_key, src, feats_hand.astype(np.float32)))
        n_seen += 1
    print(f"[ab] loaded {n_seen} clips ({not_in_manifest} not_in_manifest) "
          f"in {time.time() - t_load:.1f}s", flush=True)

    vocab = _load_vocab(vocab_path)
    present = sorted({r[0] for r in samples_body} & set(vocab))
    sign_to_idx = {s: i for i, s in enumerate(present)}
    num_classes = len(present)
    print(f"[ab] classes: {num_classes}", flush=True)
    print(f"[ab] available signs: {present}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[ab] device={device}", flush=True)

    def _run_one(samples: list[tuple], norm: str, feat_dim: int,
                 jitter_std: float, save_path=None) -> dict:
        # Signer-disjoint split with a FIXED seed so body & hand see the SAME
        # train/val partition (the two sample lists are index-aligned).
        train, val, val_signers, train_signers = signer_disjoint_split(
            samples, val_frac=0.2, seed=seed)
        train = [r for r in train if r[0] in sign_to_idx]
        val = [r for r in val if r[0] in sign_to_idx]
        print(f"\n[ab/{norm}] {len(train)} train / {len(val)} val | "
              f"signer overlap={len(val_signers & train_signers)} (must be 0)",
              flush=True)

        tl = DataLoader(DS(train, sign_to_idx, augment=True, jitter_std=jitter_std),
                        batch_size=256, shuffle=True, drop_last=True)
        vl = DataLoader(DS(val, sign_to_idx, augment=False),
                        batch_size=256, shuffle=False)
        model = SignClassifier(num_features=feat_dim, num_classes=num_classes,
                               hidden=256, num_blocks=5).to(device)
        print(f"[ab/{norm}] params: {count_parameters(model):,}", flush=True)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        crit = nn.CrossEntropyLoss(label_smoothing=0.05)

        best = {"top1": -1.0}
        best_state = None
        patience = 0
        for ep in range(1, epochs + 1):
            model.train()
            t0 = time.time()
            for x, y in tl:
                x = x.to(device); y = y.to(device)
                loss = crit(model(x), y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            sched.step()
            m = evaluate(model, vl, device, num_classes)
            print(f"  [ab/{norm}] ep {ep:03d}/{epochs} top1={m['top1']:.3f} "
                  f"top5={m['top5']:.3f} distinct={m['distinct_predicted']}/{num_classes} "
                  f"{time.time() - t0:.1f}s", flush=True)
            if m["top1"] > best["top1"] + 1e-4:
                best = {**m, "epoch": ep}; patience = 0
                if save_path is not None:
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= 12:
                    print(f"  [ab/{norm}] early stop (best top1={best['top1']:.3f} "
                          f"@ ep {best['epoch']})", flush=True)
                    break

        if save_path is not None and best_state is not None:
            torch.save({"model": best_state, "num_features": feat_dim,
                        "num_classes": num_classes,
                        "feature_version": f"{norm}_v2_{feat_dim}d",
                        "classes": present}, save_path)
            (save_path.parent / "sign_classifier_v2_classes.json").write_text(
                json.dumps(present, indent=2))
            print(f"  [ab/{norm}] saved deployable model -> {save_path} "
                  f"(best top1={best['top1']:.3f} @ ep {best.get('epoch')})", flush=True)

        result = {"norm": norm, "feat_dim": feat_dim,
                  "n_train": len(train), "n_val": len(val),
                  "top1": best["top1"], "top5": best["top5"],
                  "best_epoch": best.get("epoch")}
        if conf_signs:
            final = evaluate(model, vl, device, num_classes, collect_confusion=True)
            idx_to_sign = {i: s for s, i in sign_to_idx.items()}
            result["confusion_signs"] = print_confusion_submatrix(
                final, conf_signs, idx_to_sign, sign_to_idx)
        return result

    if only != "hand":
        body_res = _run_one(samples_body, "body", 100, jitter_std=5.0)
    else:
        body_res = {"skipped": "only=hand"}
    hand_save = (out_dir / "sign_classifier_v2_best.pt") if save_model else None
    hand_res = _run_one(samples_hand, "hand", 108, jitter_std=0.05, save_path=hand_save)

    summary = {
        "phase": "P2-norm-ab",
        "traj_dir": traj_dir,
        "sources": sorted(src_set) or "ALL",
        "epochs": epochs,
        "n_classes": num_classes,
        "confusion_signs_requested": conf_signs,
        "body_v1_100d": body_res,
        "hand_v2_108d": hand_res,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n[ab] === NORM A/B SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    volume.commit()
    return {"done": True, "run_dir": "/runs/measure_norm_ab",
            "body_top1": body_res.get("top1"), "hand_top1": hand_res["top1"]}


@app.function(
    gpu="L4",
    volumes={VOLUME_PATH: volume},
    timeout=60 * 30,
    cpu=8.0,
    memory=16384,
)
def zcheck_corpus(
    manifest: str = "/labeled_frames/unified_clip_manifest_sem_lex_top80.json",
    model_3d: str = "/models/hand_landmarks_3d_v1/best.pt",
    hand_det_ckpt: str = "/runs/hand_det_p3_facenegs_20260523/last.pt",
    per_sign: int = 5,
    signs: str = "eat,drink,water,money,help,write,school,good,animal,clean",
    fps: float = 15.0,
    pad_frac: float = 0.20,
) -> dict:
    """CHEAP z-sanity-check: run the 3D landmark model over a SAMPLE of real
    sem_lex corpus clips and measure whether the predicted depth is usable
    SIGNAL or NOISE — BEFORE spending on the full v3 extract+retrain.

    Metrics per sign (z in palm-length units, root-relative):
      - valid_frac: fraction of hand-frames with finite z
      - z_spread:  mean over frames of std(z) across the 21 keypoints
                   (structure WITHIN a hand — ~0 means collapsed/garbage)
      - step:      mean frame-to-frame |Δz| per keypoint (jitter; lower=smoother)
      - ratio:     step / z_spread  (<~0.5 = stable signal; >~1 = noise-dominated)
    FreiHAND-val reference depth err was ~0.095, z_spread typically ~0.4-0.7.
    """
    import json
    import numpy as np
    import torch
    from training.detectors.extract_trajectories_v2 import (
        _decode_clip, _detect_hands_batched, _crop_for_landmarks,
    )
    from training.detectors.hand_detector import HandDetector
    from training.detectors.hand_landmarks import HandLandmarkRegressor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    det = HandDetector().to(device).eval()
    det.load_state_dict(torch.load(_resolve_volume(hand_det_ckpt),
                                   map_location=device, weights_only=False)["model"])
    lm = HandLandmarkRegressor(predict_z=True).to(device).eval()
    lm.load_state_dict(torch.load(_resolve_volume(model_3d),
                                  map_location=device, weights_only=False)["model"])

    clips = json.loads(_resolve_volume(manifest).read_text())["clips"]
    want = [s for s in signs.split(",") if s]
    by_sign: dict[str, list] = {s: [] for s in want}
    for c in clips:
        s = c.get("sign_id")
        if s in by_sign and len(by_sign[s]) < per_sign:
            by_sign[s].append(c["clip_path"])

    report = {}
    INPUT = HandLandmarkRegressor.INPUT_SIZE
    for sign, paths in by_sign.items():
        spreads, steps, valid_tot, frame_tot = [], [], 0, 0
        for cp in paths:
            try:
                frames = _decode_clip(__import__("pathlib").Path(cp), fps)
            except Exception:
                continue
            if frames.numel() == 0:
                continue
            bboxes = _detect_hands_batched(det, frames, device)
            # top hand per frame → one crop per frame
            zs = []  # (frame) -> (21,) z, only frames with a hand
            crops, fidx = [], []
            for fi, bbs in enumerate(bboxes):
                if not bbs:
                    continue
                crops.append(_crop_for_landmarks(frames[fi], bbs[0], size=INPUT, pad_frac=pad_frac))
                fidx.append(fi)
            frame_tot += frames.shape[0]
            if not crops:
                continue
            with torch.no_grad():
                out = lm(torch.stack(crops, 0).to(device))
            z = out["depth"].cpu().numpy()  # (F_hand, 21)
            finite = np.isfinite(z).all(axis=1)
            valid_tot += int(finite.sum())
            z = z[finite]
            if z.shape[0] < 2:
                continue
            spreads.append(float(np.mean(np.std(z, axis=1))))         # within-hand structure
            steps.append(float(np.mean(np.abs(np.diff(z, axis=0)))))  # frame-to-frame jitter
        if spreads:
            sp = float(np.mean(spreads)); st = float(np.mean(steps))
            report[sign] = {
                "n_clips": len(paths), "valid_frac": round(valid_tot / max(frame_tot, 1), 3),
                "z_spread": round(sp, 3), "step": round(st, 3),
                "ratio_step_over_spread": round(st / sp, 3) if sp > 1e-6 else None,
            }

    # aggregate verdict
    sps = [r["z_spread"] for r in report.values()]
    rts = [r["ratio_step_over_spread"] for r in report.values()
           if r["ratio_step_over_spread"] is not None]
    vfs = [r["valid_frac"] for r in report.values()]
    agg = {
        "mean_z_spread": round(float(np.mean(sps)), 3) if sps else None,
        "mean_ratio": round(float(np.mean(rts)), 3) if rts else None,
        "mean_valid_frac": round(float(np.mean(vfs)), 3) if vfs else None,
    }
    agg["verdict_usable"] = bool(
        agg["mean_z_spread"] and agg["mean_z_spread"] > 0.2
        and agg["mean_ratio"] is not None and agg["mean_ratio"] < 0.6
        and agg["mean_valid_frac"] and agg["mean_valid_frac"] > 0.6)
    result = {"per_sign": report, "aggregate": agg}
    out_dir = _resolve_volume("/runs/zcheck"); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "zcheck_corpus.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    print(json.dumps(result, indent=2), flush=True)
    return result


@app.function(
    gpu="L4",
    volumes={VOLUME_PATH: volume},
    timeout=60 * 30,
    cpu=16.0,
    memory=48 * 1024,
)
def eval_per_sign_v2(
    traj_dir: str = "/trajectories_top80_v2",
    sources: str = "sem_lex_top80",
    seed: int = 42,
    model_path: str = "/runs/measure_norm_ab/sign_classifier_v2_best.pt",
    manifest: str = "/labeled_frames/unified_clip_manifest_sem_lex_top80.json",
    vocab: str = "/vocabulary/sem_lex_top80_vocabulary.json",
) -> dict:
    """EVAL-ONLY (no training): reproduce measure_norm_ab's exact signer-disjoint
    val split (seed=42, val_frac=0.2, norm=hand 108D) and score the SAVED v2
    model on it, dumping per-sign top-1 for ALL classes. Built-in correctness
    check: overall top1 must reproduce the deployed 0.7252 (else the split/model
    didn't match and the per-sign numbers are invalid)."""
    import json
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from scripts.p0_signer_baseline import (
        DS, build_clip_to_signer, evaluate, signer_disjoint_split, TIME_STEPS,
    )
    from training.detectors.sign_classifier import SignClassifier
    from training.detectors.sign_matcher import trajectory_from_frames
    from training.detectors.train_classifier import _load_vocab

    traj_root = _resolve_volume(traj_dir)
    clip_meta = build_clip_to_signer(_resolve_volume(manifest))
    src_set = {s for s in sources.split(",") if s} if sources else set()

    # Build hand (108D) samples EXACTLY as measure_norm_ab did. Read the JSONs in
    # PARALLEL (32 threads) — the network-volume per-file latency dominates (~13k
    # cold reads ≈ 60 min single-threaded); the pool cuts it to a few minutes.
    from concurrent.futures import ThreadPoolExecutor
    pairs: list[tuple] = []
    for sign_dir in sorted(traj_root.iterdir()):
        if not sign_dir.is_dir():
            continue
        for j in sorted(sign_dir.glob("*.json")):
            pairs.append((sign_dir.name, j))
    print(f"[eval] reading {len(pairs)} trajectory files (32 threads)…", flush=True)

    def _read_one(item):
        sign, j = item
        try:
            return sign, j, j.read_text()
        except Exception:
            return sign, j, None

    with ThreadPoolExecutor(max_workers=32) as _ex:
        loaded = list(_ex.map(_read_one, pairs))

    samples_hand: list[tuple] = []
    for sign, j, txt in loaded:
        if txt is None:
            continue
        try:
            t = json.loads(txt)
        except Exception:
            continue
        meta = clip_meta.get(t.get("clip_path"))
        if meta is None:
            continue
        src = meta.get("source")
        if src_set and src not in src_set:
            continue
        frames = t.get("frames", [])
        if len(frames) < 2:
            continue
        for f in frames:
            for h in (f.get("hands") or []):
                h.pop("embedding", None)
        feats_hand = trajectory_from_frames(frames, TIME_STEPS, norm="hand")
        if not np.isfinite(feats_hand).any():
            continue
        sid = meta.get("signer_id")
        signer_key = f"{src}:{sid}" if sid is not None else f"{src}:clip:{j.stem}"
        samples_hand.append((sign, signer_key, src, feats_hand.astype(np.float32)))

    ckpt = torch.load(_resolve_volume(model_path), map_location="cpu", weights_only=False)
    present = ckpt["classes"]  # use the SAVED class order (must match training)
    sign_to_idx = {s: i for i, s in enumerate(present)}
    num_classes = len(present)
    vocab_signs = set(_load_vocab(_resolve_volume(vocab)))
    assert present == sorted(set(present) & vocab_signs), "class set/order mismatch vs vocab"

    _, val, val_signers, train_signers = signer_disjoint_split(
        samples_hand, val_frac=0.2, seed=seed)
    val = [r for r in val if r[0] in sign_to_idx]
    assert len(val_signers & train_signers) == 0, "signer overlap!"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SignClassifier(num_features=ckpt["num_features"], num_classes=num_classes,
                           hidden=256, num_blocks=5).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    vl = DataLoader(DS(val, sign_to_idx, augment=False), batch_size=256, shuffle=False)
    out = evaluate(model, vl, device, num_classes, collect_confusion=True)

    confusion = out["confusion"]
    per_class_total = out["per_class_total"]
    per_sign = {}
    for s, i in sign_to_idx.items():
        tot = per_class_total.get(i, 0)
        per_sign[s] = {"n": tot,
                       "top1": (confusion.get((i, i), 0) / tot) if tot else None}
    ranked = sorted((v["top1"], k, v["n"]) for k, v in per_sign.items()
                    if v["top1"] is not None)

    # Off-diagonal confusions per sign (for the validation report §7): which
    # WRONG sign each true sign was most often predicted as. Serialized so the
    # committed artifact carries the full failure structure, not just per-sign
    # top-1. Also dump the signer->split assignment (deterministic from seed).
    idx_to_sign = {i: s for s, i in sign_to_idx.items()}
    top_confusions = {}
    for s, i in sign_to_idx.items():
        offdiag = sorted(
            ((cnt, idx_to_sign[j]) for (ti, j), cnt in confusion.items()
             if ti == i and j != i and cnt > 0),
            reverse=True,
        )
        if offdiag:
            top_confusions[s] = [{"predicted": pj, "count": int(c)} for c, pj in offdiag[:5]]

    result = {
        "overall_top1": out["top1"], "overall_top5": out["top5"],
        "n_val": len(val), "n_classes": num_classes,
        "reproduced_0.7252": abs(out["top1"] - 0.7252) < 0.01,
        "per_sign": per_sign,
        "worst_10": [{"sign": k, "top1": round(t, 3), "n": n} for t, k, n in ranked[:10]],
        "best_10": [{"sign": k, "top1": round(t, 3), "n": n} for t, k, n in ranked[-10:]],
        "top_confusions_per_sign": top_confusions,
        "val_signers": sorted(val_signers),
        "train_signers": sorted(train_signers),
        "n_val_signers": len(val_signers), "n_train_signers": len(train_signers),
    }
    out_dir = _resolve_volume("/runs/measure_norm_ab")
    (out_dir / "per_sign_accuracy.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    print(json.dumps({k: result[k] for k in
                      ("overall_top1", "overall_top5", "n_val", "reproduced_0.7252",
                       "worst_10", "best_10")}, indent=2), flush=True)
    return result


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


@app.function(
    gpu="A100",
    volumes={VOLUME_PATH: volume},
    timeout=TIMEOUT_SEC,
)
def train_hand_detector(
    train_manifest: str,
    val_manifest: str,
    run_id: str,
    epochs: int = 60,
    batch_size: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    num_workers: int = 4,
    resume_from: str = "",
    cache_in_memory: bool = False,
    bf16: bool = False,
    use_compile: bool = False,
    channels_last: bool = False,
    warmup_epochs: int = 0,
    use_ema: bool = False,
    ema_decay: float = 0.999,
    early_stop_patience: int = 0,
    early_stop_delta: float = 0.005,
) -> dict:
    """Train the from-scratch hand detector per ADR 0011 Phase 1.

    Manifests are paths inside the Modal volume (e.g.
    `/datasets/hand_bbox/train.json`). Outputs land at
    `/runs/<run_id>/best.pt` + `history.json`.

    Architecture: training/detectors/hand_detector.HandDetector
    (~1-3M params, CenterNet-style anchor-free). No pretrained weights.
    """
    from pathlib import Path

    from training.detectors.train import train as _train

    run_dir = Path(f"{VOLUME_PATH}/runs/{run_id}")
    train_manifest_path = (
        Path(f"{VOLUME_PATH}{train_manifest}") if train_manifest.startswith("/")
        else Path(train_manifest)
    )
    val_manifest_path = (
        Path(f"{VOLUME_PATH}{val_manifest}") if val_manifest.startswith("/")
        else Path(val_manifest)
    )

    resume_path = None
    if resume_from:
        resume_path = (
            Path(f"{VOLUME_PATH}{resume_from}") if resume_from.startswith("/")
            else Path(resume_from)
        )

    result = _train(
        train_manifest=train_manifest_path,
        val_manifest=val_manifest_path,
        run_dir=run_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        num_workers=num_workers,
        resume_from=resume_path,
        cache_in_memory=cache_in_memory,
        use_bf16=bf16,
        use_compile=use_compile,
        use_channels_last=channels_last,
        warmup_epochs=warmup_epochs,
        use_ema=use_ema,
        ema_decay=ema_decay,
        early_stop_patience=early_stop_patience,
        early_stop_delta=early_stop_delta,
    )
    volume.commit()
    return {"run_id": run_id, **result}


@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=TIMEOUT_SEC,
    cpu=32.0,
    memory=48 * 1024,
)
def pack_hand_dataset(manifest: str, out_base: str, workers: int = 32) -> dict:
    """One-time CPU job: pre-decode a hand_bbox manifest into a uint8 memmap
    on the volume so detector training skips per-epoch JPEG decode. Cheap
    (~10–20 min on a CPU container) and reused by every subsequent run."""
    from pathlib import Path
    from scripts.pack_hand_dataset import pack
    res = pack(_resolve_volume(manifest), Path(_resolve_volume(out_base)), workers)
    volume.commit()
    return res


@app.function(
    cpu=32.0,
    memory=64 * 1024,
    volumes={VOLUME_PATH: volume},
    timeout=60 * 60 * 3,
)
def pack_hand_dataset_combined(
    base_manifest: str = "/labeled_frames/hand_bbox/external_plus_hagrid_train_with_face_negatives.json",
    coco_split: str = "train",
    out_base: str = "/labeled_frames/hand_bbox/packed/train_coco_combined",
    workers: int = 32,
) -> dict:
    """Build a COMBINED hand_bbox packed set: the base manifest (FreiHAND + CMU +
    HaGRID + face-negatives) PLUS COCO-WholeBody hand BOXES (human-annotated,
    in-the-wild — the same source that won for landmarks). COCO images are
    extracted from the volume zips to the container's /tmp (ZERO volume inodes),
    then everything is packed via the proven `scripts.pack_hand_dataset.pack`
    (uint8 320² memmap, bboxes scaled to 320² space). One clean repack — no
    incremental concat. Re-decodes the base too (~20-25 min) for correctness."""
    import json, os, re, zipfile
    from pathlib import Path
    from scripts.pack_hand_dataset import pack

    os.environ["ASL_REPO_ROOT"] = VOLUME_PATH
    os.environ["ASL_EXTERNAL_ROOT"] = f"{VOLUME_PATH}/external"
    from training.detectors.external_loaders import coco_wholebody_hands

    def _norm(p: str) -> str:
        # canonical /__modal/volumes/<id>/... → the /data mount; absolutize rest
        p = re.sub(r"^/__modal/volumes/[^/]+", VOLUME_PATH, str(p))
        return p if p.startswith("/") else f"{VOLUME_PATH}/{p}"

    # 1) base items — keep boxes + negatives verbatim, just normalize paths
    base = json.loads(_resolve_volume(base_manifest).read_text())
    assert base["task"] == "hand_bbox", base["task"]
    items = [{"image_path": _norm(it["image_path"]), "width": it["width"],
              "height": it["height"], "bboxes": it["bboxes"]}
             for it in base["items"]]
    n_base = len(items)
    n_base_neg = sum(1 for it in items if not it["bboxes"])

    # 2) COCO hand-box items (one manifest item per image, all its hands' boxes)
    coco_items = coco_wholebody_hands.load_hand_keypoints(coco_split)
    needed: set[str] = set()
    coco_recs = []
    tmp_coco = Path("/tmp/coco_imgs"); tmp_coco.mkdir(parents=True, exist_ok=True)
    for it in coco_items:
        boxes = [h["bbox"] for h in it.get("hands", []) if h.get("bbox")]
        if not boxes:
            continue
        name = Path(it["image_path"]).name
        needed.add(name)
        coco_recs.append({"image_path": str(tmp_coco / name),
                          "width": int(it.get("width", 0)),
                          "height": int(it.get("height", 0)), "bboxes": boxes})

    # 3) extract ONLY the referenced COCO images from the zip(s) → /tmp
    coco_root = _resolve_volume("/external/coco_wholebody")
    extracted = 0
    for split in ("train2017", "val2017"):
        zpath = coco_root / f"{split}.zip"
        if not zpath.exists():
            continue
        with zipfile.ZipFile(zpath) as zf:
            for nm in zf.namelist():
                b = Path(nm).name
                if b in needed and not (tmp_coco / b).exists():
                    with zf.open(nm) as src, open(tmp_coco / b, "wb") as dst:
                        dst.write(src.read())
                    extracted += 1
    items.extend(coco_recs)
    n_coco = len(coco_recs)
    print(f"[combined] base={n_base} (neg={n_base_neg}) coco_imgs={n_coco} "
          f"extracted={extracted} total={len(items)}", flush=True)

    # 4) write the combined manifest, PERSIST it to the volume (train needs a
    #    manifest whose load_manifest() length matches the packed N — when
    #    packed, the dataset uses records only for __len__), then pack.
    manifest_obj = {"version": 1, "task": "hand_bbox", "items": items}
    vol_manifest = Path(str(_resolve_volume(out_base)) + ".manifest.json")
    vol_manifest.parent.mkdir(parents=True, exist_ok=True)
    vol_manifest.write_text(json.dumps(manifest_obj))
    combined = Path("/tmp/combined_hand_bbox.json")
    combined.write_text(json.dumps(manifest_obj))
    res = pack(combined, Path(_resolve_volume(out_base)), workers)
    volume.commit()
    return {"n_base": n_base, "n_base_neg": n_base_neg, "n_coco": n_coco,
            "extracted": extracted, "manifest": str(vol_manifest), **res}


@app.function(
    gpu="A100",
    volumes={VOLUME_PATH: volume},
    timeout=TIMEOUT_SEC,
    cpu=16.0,
    memory=64 * 1024,
)
def train_hand_detector_packed(
    run_id: str = "hand_det_v3_coco_combined",
    train_manifest: str = "/labeled_frames/hand_bbox/packed/train_coco_combined.manifest.json",
    packed_train: str = "/labeled_frames/hand_bbox/packed/train_coco_combined.meta.json",
    val_manifest: str = "/labeled_frames/hand_bbox/packed/val_coco_combined.manifest.json",
    packed_val: str = "/labeled_frames/hand_bbox/packed/val_coco_combined.meta.json",
    epochs: int = 45,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    warmup_epochs: int = 2,
    num_workers: int = 12,
    early_stop_patience: int = 6,
    early_stop_delta: float = 0.002,
    localize: bool = True,
) -> dict:
    """FROM-SCRATCH single-A100 hand-detector train on the COMBINED packed set
    (FreiHAND + CMU + HaGRID + face-negs + COCO in-the-wild boxes). Uses the
    fast pipeline: uint8 memmap + GPU augmentation + uint8→float ON GPU +
    early-stop + EMA. No resume (random init). The DDP entrypoint wires the
    same pipeline for 8×H100; this is the cost-effective single-GPU version."""
    import json as _json
    import shutil, time
    from pathlib import Path
    from training.detectors.train import train as _train

    def _localize(meta_p: str) -> Path:
        """Copy the packed .dat to /tmp local SSD (per-epoch random reads from
        the network volume are slow); rewrite meta's 'dat' to the local copy."""
        vol_meta = _resolve_volume(meta_p)
        if not localize:
            return vol_meta
        meta = _json.loads(vol_meta.read_text())
        dst_dir = Path("/tmp/packed_det"); dst_dir.mkdir(parents=True, exist_ok=True)
        dst_dat = dst_dir / meta["dat"]
        t0 = time.time()
        if not dst_dat.exists():
            shutil.copy(vol_meta.parent / meta["dat"], dst_dat)
        dst_meta = dst_dir / Path(meta_p).name
        dst_meta.write_text(_json.dumps(meta))  # meta['dat'] is a basename
        print(f"[det] localized {meta['dat']} ({dst_dat.stat().st_size/1e9:.1f}GB) "
              f"in {time.time()-t0:.0f}s", flush=True)
        return dst_meta

    packed_train_p = _localize(packed_train)
    packed_val_p = _localize(packed_val)

    run_dir = Path(f"{VOLUME_PATH}/runs/{run_id}")
    result = _train(
        train_manifest=_resolve_volume(train_manifest),
        val_manifest=_resolve_volume(val_manifest),
        run_dir=run_dir,
        epochs=epochs, batch_size=batch_size, lr=lr,
        weight_decay=weight_decay, num_workers=num_workers,
        warmup_epochs=warmup_epochs,
        gpu_aug=True,
        packed_train_path=packed_train_p,
        packed_val_path=packed_val_p,
        use_ema=True,
        early_stop_patience=early_stop_patience,
        early_stop_delta=early_stop_delta,
    )
    volume.commit()
    return {"run_id": run_id, **result}


@app.function(
    gpu="H100:8",
    volumes={VOLUME_PATH: volume},
    timeout=60 * 60,            # 1-HOUR HARD CAP — a hung 8×H100 job burns ~$32/hr
    cpu=64.0,
    memory=128 * 1024,
)
def train_hand_det_ddp(
    run_id: str,
    train_manifest: str = "/labeled_frames/hand_bbox/external_plus_hagrid_train_with_negatives.json",
    val_manifest: str = "/labeled_frames/hand_bbox/external_val.json",
    packed_train: str = "/labeled_frames/hand_bbox/packed/train.meta.json",
    packed_val: str = "/labeled_frames/hand_bbox/packed/val.meta.json",
    resume_from: str = "/runs/hand_det_v2_hagrid_fromscratch/best.pt",
    world_size: int = 8,
    epochs: int = 30,
    per_gpu_batch: int = 64,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    warmup_epochs: int = 2,
    num_workers: int = 6,
) -> dict:
    """Option B: DDP fine-tune the hand detector on `world_size` GPUs in one
    container. Localizes packed data to /tmp ONCE (single process, before
    spawn), then runs one process per GPU. NOT launched by default."""
    import json as _json
    import shutil as _shutil
    import time as _time
    from pathlib import Path
    import torch as _torch

    from training.detectors import train_ddp

    ngpu = _torch.cuda.device_count()
    if ngpu < world_size:
        raise RuntimeError(f"requested world_size={world_size} but container "
                           f"has only {ngpu} GPUs")

    def _localize(meta_path: Path) -> str:
        meta = _json.loads(Path(meta_path).read_text())
        src = Path(meta_path).parent / meta["dat"]
        ldir = Path("/tmp/packed"); ldir.mkdir(parents=True, exist_ok=True)
        ldat = ldir / meta["dat"]; t0 = _time.time()
        _shutil.copyfile(src, ldat)
        lmeta = ldir / Path(meta_path).name
        lmeta.write_text(_json.dumps(meta))
        print(f"localized {src.name} ({ldat.stat().st_size/1e9:.1f} GB "
              f"in {_time.time()-t0:.0f}s)")
        return str(lmeta)

    cfg = {
        "train_manifest": str(_resolve_volume(train_manifest)),
        "val_manifest": str(_resolve_volume(val_manifest)),
        "packed_train_local": _localize(_resolve_volume(packed_train)),
        "packed_val_local": _localize(_resolve_volume(packed_val)),
        "resume_from": str(_resolve_volume(resume_from)) if resume_from else "",
        "run_dir": f"{VOLUME_PATH}/runs/{run_id}",
        "epochs": epochs, "per_gpu_batch": per_gpu_batch, "lr": lr,
        "weight_decay": weight_decay, "warmup_epochs": warmup_epochs,
        "num_workers": num_workers, "master_port": "29512",
    }
    train_ddp.launch(cfg, world_size)
    volume.commit()
    return {"run_id": run_id, "run_dir": cfg["run_dir"], "world_size": world_size,
            "epochs": epochs}


@app.function(
    gpu="H100",
    volumes={VOLUME_PATH: volume},
    timeout=TIMEOUT_SEC,
    cpu=32.0,
    memory=64 * 1024,
)
def train_and_extract_h100(
    train_manifest: str,
    val_manifest: str,
    run_id: str,
    extract_manifest: str,
    extract_out_dir: str,
    hand_landmarks_ckpt: str,
    pose_ckpt: str,
    epochs: int = 100,
    batch_size: int = 512,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    warmup_epochs: int = 5,
    fps: float = 15.0,
    decode_workers: int = 12,
    packed_train: str = "",
    packed_val: str = "",
    gpu_aug: bool = True,
    num_workers: int = 16,
    skip_extract: bool = False,
    resume_from: str = "",
) -> dict:
    """Fused job: H100 — train hand_det then re-extract trajectories with
    the new checkpoint, in one container (no second cold-start).

    P3-optimized path (target ~30 min vs ~2 h): pre-decoded uint8 memmap
    (packed_train/packed_val) instead of per-epoch JPEG decode + the
    share_memory_() SIGBUS trap; GPU augmentation; batch 512; 16 workers.
    Falls back to RAM-cache + CPU aug when packed paths are empty.
    """
    from pathlib import Path

    from training.detectors.train import train as _train
    from training.detectors.extract_trajectories_v2 import main as _extract_main
    import sys

    run_dir = Path(f"{VOLUME_PATH}/runs/{run_id}")
    train_p = _resolve_volume(train_manifest)
    val_p = _resolve_volume(val_manifest)
    packed_train_path = _resolve_volume(packed_train) if packed_train else None
    packed_val_path = _resolve_volume(packed_val) if packed_val else None

    # CRITICAL for speed: the packed .dat lives on the network-backed volume.
    # Random per-batch reads over the network cap throughput at ~800 img/s.
    # Copy each memmap to the container's LOCAL disk once (sequential, fast),
    # then train against the local copy → full H100 throughput.
    def _localize(meta_path):
        import json as _json, shutil as _shutil, time as _t
        meta = _json.loads(Path(meta_path).read_text())
        src = Path(meta_path).parent / meta["dat"]
        ldir = Path("/tmp/packed"); ldir.mkdir(parents=True, exist_ok=True)
        ldat = ldir / meta["dat"]
        t0 = _t.time()
        _shutil.copyfile(src, ldat)
        lmeta = ldir / Path(meta_path).name
        lmeta.write_text(_json.dumps(meta))
        print(f"  localized {src} → {ldat} "
              f"({ldat.stat().st_size/1e9:.1f} GB in {_t.time()-t0:.0f}s)")
        return lmeta
    if packed_train_path is not None:
        packed_train_path = _localize(packed_train_path)
    if packed_val_path is not None:
        packed_val_path = _localize(packed_val_path)

    print("=" * 60)
    print(f"PHASE 1: train hand_det → {run_dir}  "
          f"(packed={'yes' if packed_train_path else 'no'}, gpu_aug={gpu_aug})")
    print("=" * 60)
    train_result = _train(
        train_manifest=train_p,
        val_manifest=val_p,
        run_dir=run_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        num_workers=num_workers,
        cache_in_memory=packed_train_path is None,
        use_bf16=True,
        use_compile=True,
        use_channels_last=True,
        warmup_epochs=warmup_epochs,
        gpu_aug=gpu_aug,
        packed_train_path=packed_train_path,
        packed_val_path=packed_val_path,
        resume_from=(_resolve_volume(resume_from) if resume_from else None),
    )
    volume.commit()
    best_pt = run_dir / "best.pt"
    if not best_pt.exists():
        raise RuntimeError(f"no best.pt produced at {best_pt}")

    if skip_extract:
        print("\nskip_extract=True → stopping after training. "
              "Gate on det_recall/conf@gt in history.json, then extract separately.")
        return {"run_id": run_id, "phase": "train_only", **train_result}

    print("\n" + "=" * 60)
    print(f"PHASE 2: re-extract trajectories → {extract_out_dir}")
    print("=" * 60)
    argv_backup = sys.argv[:]
    sys.argv = [
        "extract_trajectories_v2",
        "--manifest", str(_resolve_volume(extract_manifest)),
        "--out-dir", str(_resolve_volume(extract_out_dir)),
        "--hand-detector-ckpt", str(best_pt),
        "--hand-landmarks-ckpt", str(_resolve_volume(hand_landmarks_ckpt)),
        "--pose-ckpt", str(_resolve_volume(pose_ckpt)),
        "--fps", str(fps),
        "--decode-workers", str(decode_workers),
        "--repo-root", VOLUME_PATH,
    ]
    try:
        rc = _extract_main()
    finally:
        sys.argv = argv_backup
    volume.commit()
    return {
        "run_id": run_id,
        "train": train_result,
        "extract_rc": rc,
        "extract_out_dir": extract_out_dir,
    }


@app.function(
    cpu=4.0,
    memory=4096,
    volumes={VOLUME_PATH: volume},
    timeout=60 * 60 * 2,  # 2 hours
)
def setup_external_datasets() -> dict:
    """Download + extract every recommended external CV dataset directly
    into the Modal volume. Replaces the user-side push of ~94 GB with a
    Modal-side fetch from the same upstream URLs at Modal's bandwidth.

    Layout written to volume:
      /external/freihand/                <- FreiHAND_pub_v2.zip + eval, extracted
      /external/cmu_panoptic_handdb/     <- 3 subset archives, extracted
      /external/coco_wholebody/          <- train2017.zip + val2017.zip + annotations, extracted
      /external/mpii_pose/               <- images tar + annotations zip, extracted
      /external/wider_face/              <- annotations only (image splits manual)
    Multiview Hand Pose intentionally skipped per audit memo § "Multiview drop".
    """
    import subprocess
    from pathlib import Path

    ROOT = Path(f"{VOLUME_PATH}/external")
    ROOT.mkdir(parents=True, exist_ok=True)

    JOBS = [
        # (subdir, [(filename, url), ...])
        ("freihand", [
            ("FreiHAND_pub_v2.zip",
             "https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2.zip"),
            ("FreiHAND_pub_v2_eval.zip",
             "https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2_eval.zip"),
        ]),
        ("cmu_panoptic_handdb", [
            ("hand_labels.zip",
             "http://domedb.perception.cs.cmu.edu/panopticDB/hands/hand_labels.zip"),
            ("hand_labels_synth.zip",
             "http://domedb.perception.cs.cmu.edu/panopticDB/hands/hand_labels_synth.zip"),
            ("hand143_panopticdb.tar",
             "http://domedb.perception.cs.cmu.edu/panopticDB/hands/hand143_panopticdb.tar"),
        ]),
        ("coco_wholebody", [
            ("train2017.zip", "http://images.cocodataset.org/zips/train2017.zip"),
            ("val2017.zip", "http://images.cocodataset.org/zips/val2017.zip"),
            ("annotations_trainval2017.zip",
             "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"),
        ]),
        ("mpii_pose", [
            ("mpii_human_pose_v1.tar.gz",
             "https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz"),
            ("mpii_human_pose_v1_u12_2.zip",
             "https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1_u12_2.zip"),
        ]),
        ("wider_face", [
            ("wider_face_split.zip",
             "http://shuoyang1213.me/WIDERFACE/support/bbx_annotation/wider_face_split.zip"),
        ]),
    ]

    from concurrent.futures import ThreadPoolExecutor, as_completed

    has_aria = subprocess.call(["which", "aria2c"], stdout=subprocess.DEVNULL) == 0

    def _download_one(subdir: str, fname: str, url: str) -> tuple[str, str, int]:
        d = ROOT / subdir
        d.mkdir(parents=True, exist_ok=True)
        target = d / fname
        # aria2c with -c resumes if a partial exists; harmless if file already
        # complete (server returns 416, we treat as success).
        print(f"  [{subdir}] {fname} ← {url}", flush=True)
        if has_aria:
            rc = subprocess.call([
                "aria2c", "-x", "16", "-s", "16", "--check-certificate=false",
                "--file-allocation=none", "--auto-file-renaming=false",
                "--allow-overwrite=true", "-c", "-d", str(d), "-o", fname, url
            ])
        else:
            rc = subprocess.call(["curl", "-L", "-C", "-", "--fail", "-o", str(target), url])
        return (subdir, fname, rc)

    def _extract_subset(subdir: str) -> dict:
        d = ROOT / subdir
        per_archive: dict[str, str] = {}
        for archive in sorted(list(d.glob("*.zip")) + list(d.glob("*.tar.gz")) + list(d.glob("*.tar"))):
            print(f"  [{subdir}] extracting {archive.name}", flush=True)
            if archive.suffix == ".zip":
                rc = subprocess.call(["unzip", "-qq", "-n", str(archive), "-d", str(d)])
            else:
                rc = subprocess.call(["tar", "-xf", str(archive), "-C", str(d)])
            per_archive[archive.name] = "ok" if rc == 0 else f"rc{rc}"
        return per_archive

    # Step 1: flatten all (subdir, fname, url) tasks, run in parallel.
    tasks = [(s, fn, u) for s, files in JOBS for fn, u in files]
    download_results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=len(tasks)) as ex:
        futures = [ex.submit(_download_one, s, fn, u) for s, fn, u in tasks]
        for fut in as_completed(futures):
            subdir, fname, rc = fut.result()
            download_results.setdefault(subdir, {})[fname] = "ok" if rc == 0 else f"rc{rc}"
    print(f"DOWNLOADS DONE: {download_results}", flush=True)

    # Step 2: extract per-subset in parallel (one worker per subset, each
    # worker processes its subset's archives sequentially since unzip is
    # already disk-bound).
    extract_results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=len(JOBS)) as ex:
        futures = {ex.submit(_extract_subset, s): s for s, _ in JOBS}
        for fut in as_completed(futures):
            subdir = futures[fut]
            extract_results[subdir] = fut.result()
    print(f"EXTRACTIONS DONE: {extract_results}", flush=True)

    volume.commit()
    return {"downloads": download_results, "extracts": extract_results}


@app.function(
    cpu=32.0,
    memory=64 * 1024,
    volumes={VOLUME_PATH: volume},
    timeout=60 * 60 * 3,
)
def pack_landmark_dataset(
    out_base: str = "/labeled_frames/hand_keypoints/packed_combined",
    input_size: int = 224,
    pad_frac: float = 0.20,
    workers: int = 32,
) -> dict:
    """Pack FreiHAND + CMU + COCO-WholeBody hands into ONE uint8 crop memmap
    for fast GPU-aug landmark training. Optimized: (1) COCO's hand-bearing
    images are extracted to the container's /tmp (ephemeral — ZERO volume
    inodes, dodging the 500k cap); (2) per-hand 224² crops are pre-computed in
    parallel so training has no per-epoch JPEG decode. Writes <out_base>.dat
    (N,3,224,224 uint8) + <out_base>.meta.json (keypoints [N,21,2] crop-norm,
    visibility [N,21], source [N])."""
    import json, os, zipfile
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path
    import numpy as np
    from PIL import Image

    os.environ["ASL_REPO_ROOT"] = VOLUME_PATH
    os.environ["ASL_EXTERNAL_ROOT"] = f"{VOLUME_PATH}/external"
    from training.detectors.external_loaders import freihand, cmu_handdb, coco_wholebody_hands

    # ---- 1) gather hand instances from all three sources --------------------
    # each instance: (kind, ref, bbox, kps[list[(x,y,v)]])  kind in {file,coco}
    insts: list[tuple] = []
    def _add_items(items, kind):
        for it in items:
            p = it["image_path"]
            for h in it.get("hands", []):
                if h.get("bbox") and h.get("keypoints") and len(h["keypoints"]) == 21:
                    insts.append((kind, p, h["bbox"], h["keypoints"]))
    _add_items(freihand.load_hand_keypoints(), "file")
    n_fh = len(insts)
    # CMU: only the 'manual' subset (human-labeled MPII+NZSL signing hands;
    # loads fast). synth/multiview do rglob + per-file exists() over ~47k files
    # → pathologically slow on the network volume, so skip them. COCO carries
    # the in-the-wild signal anyway.
    _add_items(cmu_handdb.load_hand_keypoints(include=("manual",)), "file")
    n_cmu = len(insts) - n_fh
    coco_items = coco_wholebody_hands.load_hand_keypoints("train") + \
        coco_wholebody_hands.load_hand_keypoints("val")
    _add_items(coco_items, "coco")
    n_coco = len(insts) - n_fh - n_cmu
    print(f"[pack] instances: freihand={n_fh} cmu={n_cmu} coco={n_coco} total={len(insts)}", flush=True)

    # ---- 2) extract ONLY referenced COCO images to /tmp (ephemeral) ---------
    coco_root = _resolve_volume("/external/coco_wholebody")
    tmp_coco = Path("/tmp/coco_imgs"); tmp_coco.mkdir(parents=True, exist_ok=True)
    needed = {Path(ref).name for (k, ref, _, _) in insts if k == "coco"}
    for split in ("train2017", "val2017"):
        zpath = coco_root / f"{split}.zip"
        if not zpath.exists():
            continue
        with zipfile.ZipFile(zpath) as zf:
            for nm in zf.namelist():
                base = Path(nm).name
                if base in needed and not (tmp_coco / base).exists():
                    with zf.open(nm) as src, open(tmp_coco / base, "wb") as dst:
                        dst.write(src.read())
    print(f"[pack] extracted {len(list(tmp_coco.iterdir()))} COCO imgs to /tmp", flush=True)

    # ---- 3) crop + resize each hand → memmap (parallel) ---------------------
    n = len(insts)
    dat_path = _resolve_volume(out_base + ".dat")
    dat_path.parent.mkdir(parents=True, exist_ok=True)
    mm = np.memmap(dat_path, dtype=np.uint8, mode="w+", shape=(n, 3, input_size, input_size))
    kps_out = np.zeros((n, 21, 2), np.float32)
    vis_out = np.zeros((n, 21), np.float32)
    src_out: list[str] = [""] * n

    def _resolve_img(kind, ref):
        if kind == "coco":
            return tmp_coco / Path(ref).name
        p = Path(ref)
        return p if p.is_absolute() else (Path(VOLUME_PATH) / ref)

    def _one(i):
        kind, ref, bbox, kps = insts[i]
        try:
            img = Image.open(_resolve_img(kind, ref)).convert("RGB")
        except Exception:
            return i, None, None, None, kind
        W, H = img.size
        x0, y0, x1, y1 = bbox
        pad = pad_frac * max(x1 - x0, y1 - y0)
        cx0, cy0 = max(0, int(x0 - pad)), max(0, int(y0 - pad))
        cx1, cy1 = min(W, int(x1 + pad)), min(H, int(y1 + pad))
        if cx1 <= cx0 or cy1 <= cy0:
            cx0, cy0, cx1, cy1 = 0, 0, W, H
        crop = img.crop((cx0, cy0, cx1, cy1)).resize((input_size, input_size), Image.BILINEAR)
        arr = np.asarray(crop, np.uint8).transpose(2, 0, 1)  # (3,H,W)
        cw, ch = max(cx1 - cx0, 1), max(cy1 - cy0, 1)
        k = np.array([[(x - cx0) / cw, (y - cy0) / ch] for x, y, v in kps], np.float32)
        v = np.array([1.0 if vv > 0 else 0.0 for _, _, vv in kps], np.float32)
        return i, arr, k, v, kind

    ok = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, arr, k, v, kind in ex.map(_one, range(n)):
            if arr is None:
                continue
            mm[i] = arr; kps_out[i] = k; vis_out[i] = v; src_out[i] = kind
            ok += 1
    mm.flush()
    # keypoints/visibility as .npy sidecars (NOT in JSON — 160k×21 would be a
    # ~100MB text blob). Dataset reads .dat + .kps.npy + .vis.npy + meta.
    np.save(str(_resolve_volume(out_base + ".kps.npy")), kps_out)
    np.save(str(_resolve_volume(out_base + ".vis.npy")), vis_out)
    meta = {"n": n, "ok": ok, "input_size": input_size, "shape": [n, 3, input_size, input_size],
            "dtype": "uint8", "source": src_out,
            "counts": {"freihand": n_fh, "cmu": n_cmu, "coco": n_coco}}
    _resolve_volume(out_base + ".meta.json").write_text(json.dumps(meta))
    volume.commit()
    gb = dat_path.stat().st_size / 1e9
    print(f"[pack] wrote {ok}/{n} crops → {dat_path} ({gb:.1f} GB)", flush=True)
    return {"n": n, "ok": ok, "gb": round(gb, 1), "counts": meta["counts"]}


@app.function(
    cpu=8.0,
    memory=16384,
    volumes={VOLUME_PATH: volume},
    timeout=60 * 60 * 2,
)
def download_coco_for_hands() -> dict:
    """Focused download for the landmark in-the-wild retrain: COCO 2017 images
    (direct-HTTPS CDN) + COCO-WholeBody HAND annotations (GDrive via gdown).

    Optimized for the inode cap: we KEEP the image ZIPs (1 inode each) and do
    NOT extract the 118k images here — selective extraction of only the
    ~40k hand-bearing images into a packed memmap happens at pack time.
    Surfaces per-file ok/fail so a GDrive quota block is caught immediately.
    """
    import subprocess
    from pathlib import Path

    base = _resolve_volume("/external/coco_wholebody")
    (base / "annotations").mkdir(parents=True, exist_ok=True)
    has_aria = subprocess.call(["which", "aria2c"], stdout=subprocess.DEVNULL) == 0
    results = {}

    def _fetch_http(url: str, target: Path) -> str:
        if target.exists() and target.stat().st_size > 1_000_000:
            return f"exists ({target.stat().st_size/1e9:.1f}GB)"
        if has_aria:
            rc = subprocess.call(["aria2c", "-x", "16", "-s", "16",
                                  "--check-certificate=false", "-d", str(target.parent),
                                  "-o", target.name, url])
        else:
            rc = subprocess.call(["curl", "-L", "-C", "-", "--fail", "-o", str(target), url])
        return "ok" if rc == 0 and target.exists() else f"FAIL rc={rc}"

    # 1) images — cocodataset.org CDN (direct HTTPS, fast on Modal)
    results["train2017.zip"] = _fetch_http(
        "http://images.cocodataset.org/zips/train2017.zip", base / "train2017.zip")
    results["val2017.zip"] = _fetch_http(
        "http://images.cocodataset.org/zips/val2017.zip", base / "val2017.zip")

    # 2) WholeBody hand annotations — GDrive (the only GDrive dependency; small)
    for name, fid in [
        ("coco_wholebody_train_v1.0.json", "1thErEToRbmM9uLNi1JXXfOsaS5VK2FXf"),
        ("coco_wholebody_val_v1.0.json", "1N6VgwKnj8DeyGXCvp1eYgNbRmw6jdfrb"),
    ]:
        out = base / "annotations" / name
        if out.exists() and out.stat().st_size > 1_000_000:
            results[name] = f"exists ({out.stat().st_size/1e6:.0f}MB)"
            continue
        rc = subprocess.call(["gdown", fid, "-O", str(out)])
        results[name] = ("ok" if rc == 0 and out.exists()
                         else f"FAIL rc={rc} — GDrive blocked? fall back to local upload")

    volume.commit()
    sizes = {p.name: p.stat().st_size for p in [
        base / "train2017.zip", base / "val2017.zip",
        base / "annotations" / "coco_wholebody_train_v1.0.json",
        base / "annotations" / "coco_wholebody_val_v1.0.json"] if p.exists()}
    out = {"results": results, "sizes_bytes": sizes}
    print(out, flush=True)
    return out


@app.function(
    cpu=2.0,
    memory=2048,
    volumes={VOLUME_PATH: volume},
    timeout=60 * 30,
)
def fetch_wider_face_images() -> dict:
    """Pull WIDER FACE train+val image splits from Google Drive into the
    volume. Google Drive's large-file flow needs a consent token that
    plain curl/aria2c can't supply; we use gdown which handles that.

    Layout written:
      /external/wider_face/WIDER_train.zip   (~1.4 GB)
      /external/wider_face/WIDER_val.zip     (~370 MB)
      then extracted in place → WIDER_train/images/<event>/*.jpg etc.
    """
    import subprocess
    from pathlib import Path

    d = Path(f"{VOLUME_PATH}/external/wider_face")
    d.mkdir(parents=True, exist_ok=True)

    # Google Drive IDs published on the WIDER FACE official project page
    # (shuoyang1213.me/WIDERFACE). IDs are stable.
    DRIVE_IDS = {
        "WIDER_train.zip": "15hGDLhsx8bLgLcIRD5DhYt5iBxnjNF1M",
        "WIDER_val.zip": "1GUCogbp16PMGa39thoMMeWxp7Rp5oM8Q",
    }

    results: dict[str, str] = {}
    for fname, fid in DRIVE_IDS.items():
        target = d / fname
        if target.exists() and target.stat().st_size > 100 * 1024 * 1024:
            print(f"  [wider_face] {fname} already present ({target.stat().st_size} B)")
            results[fname] = "already_present"
            continue
        url = f"https://drive.google.com/uc?id={fid}"
        print(f"  [wider_face] gdown {fname} ← {url}")
        rc = subprocess.call(["gdown", url, "-O", str(target), "--quiet"])
        results[fname] = "ok" if rc == 0 else f"rc{rc}"

    # Extract
    for fname in DRIVE_IDS:
        archive = d / fname
        if not archive.exists():
            continue
        print(f"  [wider_face] extracting {fname}")
        rc = subprocess.call(["unzip", "-qq", "-n", str(archive), "-d", str(d)])
        results[f"{fname}::extract"] = "ok" if rc == 0 else f"rc{rc}"

    volume.commit()
    return results


@app.function(
    cpu=4.0,
    memory=8192,
    volumes={VOLUME_PATH: volume},
    timeout=60 * 60 * 2,  # 2 hours
)
def build_manifests() -> dict:
    """Run normalize_external on Modal against the volume-resident datasets.

    Writes per-task manifests to /labeled_frames/<task>/external_{train,val,combined,<source>}.json.
    Skips datasets whose extracted directory isn't present on the volume.
    """
    import os
    import sys

    # Steer the loaders at the Modal volume via env vars BEFORE the
    # external_loaders module tree is imported (its DATASET_DIR
    # constants are bound at import time, so a runtime attribute
    # override would not propagate to already-imported loaders).
    os.environ["ASL_REPO_ROOT"] = VOLUME_PATH
    os.environ["ASL_EXTERNAL_ROOT"] = f"{VOLUME_PATH}/external"
    os.chdir(VOLUME_PATH)

    # Now import — _util.py reads the env vars at module load.
    from training.detectors.external_loaders import _util
    from training.detectors import normalize_external as _ne
    _ne.OUT_ROOT = _util.Path(f"{VOLUME_PATH}/labeled_frames")

    # Run for every task
    argv_backup = sys.argv[:]
    sys.argv = ["normalize_external", "--task", "all"]
    try:
        rc = _ne.main()
    finally:
        sys.argv = argv_backup

    # Summary
    summary: dict[str, dict] = {}
    out = _util.Path(f"{VOLUME_PATH}/labeled_frames")
    for task_dir in sorted(out.glob("*")):
        if not task_dir.is_dir():
            continue
        summary[task_dir.name] = {p.name: p.stat().st_size for p in task_dir.glob("*.json")}

    volume.commit()
    return {"rc": rc, "summary": summary}


def _resolve_volume(p: str) -> "Path":
    """Helper: paths starting with '/' are volume-relative; else local."""
    from pathlib import Path
    return Path(f"{VOLUME_PATH}{p}") if p.startswith("/") else Path(p)


@app.function(
    gpu=GPU,
    volumes={VOLUME_PATH: volume},
    timeout=TIMEOUT_SEC,
)
def train_hand_landmarks(
    train_manifest: str,
    val_manifest: str,
    run_id: str,
    epochs: int = 60,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    num_workers: int = 4,
) -> dict:
    """Train the from-scratch 21-keypoint hand landmark regressor (Phase 2)."""
    from pathlib import Path
    from training.detectors.train_landmarks import train as _train
    run_dir = Path(f"{VOLUME_PATH}/runs/{run_id}")
    result = _train(
        train_manifest=_resolve_volume(train_manifest),
        val_manifest=_resolve_volume(val_manifest),
        run_dir=run_dir,
        epochs=epochs, batch_size=batch_size, lr=lr,
        weight_decay=weight_decay, num_workers=num_workers,
    )
    volume.commit()
    return {"run_id": run_id, **result}


@app.function(
    cpu=4.0,
    memory=8192,
    volumes={VOLUME_PATH: volume},
    timeout=60 * 30,
)
def build_freihand_3d_manifest(val_every: int = 20, include_cmu: bool = True) -> dict:
    """Build the 3D (depth-augmented) hand_keypoints manifests for v3.

    Augments the PROVEN 2D manifest (/labeled_frames/hand_keypoints/
    external_freihand.json — correct absolute volume paths, bboxes, 21 2D
    kps) with per-keypoint root-relative scale-normalized depth read from
    FreiHAND's training_xyz.json. Reusing the existing manifest avoids any
    image-path re-derivation. CPU-only — effectively free.

    z_i = (xyz[i].z - xyz[0].z) / ‖xyz[9] - xyz[0]‖   (wrist-relative,
    scaled by the 3D palm bone — matches the v3 feature schema).

    Splits the 32,560 unique FreiHAND samples deterministically: val = idx %
    val_every == 0 (~5%), train = the rest. No augmented-copy leakage
    (external_freihand.json is unique-only).

    include_cmu (default True): also fold CMU HandDB's 2D-only items into
    TRAIN (no keypoints_z → the loss masks depth for them via has_depth, so
    they supervise x,y only). VAL stays FreiHAND-only so the depth metric is
    clean AND the 2D-px number is directly comparable to the FreiHAND-only
    run (13.82 px). CMU ~doubles the 2D supervision — the lever on px error.
    """
    import json
    import re
    import numpy as np

    fh_manifest = _resolve_volume("/labeled_frames/hand_keypoints/external_freihand.json")
    xyz_path = _resolve_volume("/external/freihand/training_xyz.json")
    items = json.loads(fh_manifest.read_text())["items"]
    xyz_all = np.asarray(json.loads(xyz_path.read_text()), dtype=np.float32)  # (N,21,3)

    train_items, val_items, n_skip = [], [], 0
    for it in items:
        m = re.search(r"(\d{8})\.jpg", it["image_path"])
        if not m:
            n_skip += 1
            continue
        idx = int(m.group(1))
        xyz = xyz_all[idx]  # (21,3) mm
        scale = float(np.linalg.norm(xyz[9] - xyz[0]))
        if scale < 1e-3:
            n_skip += 1
            continue
        z_rel = ((xyz[:, 2] - xyz[0, 2]) / scale).tolist()
        for hand in it["hands"]:
            hand["keypoints_z"] = [float(z) for z in z_rel]
            hand["has_depth"] = True
        (val_items if idx % val_every == 0 else train_items).append(it)

    n_fh_train = len(train_items)
    n_cmu = 0
    if include_cmu:
        cmu_path = _resolve_volume("/labeled_frames/hand_keypoints/external_cmu_handdb.json")
        if cmu_path.exists():
            cmu_items = json.loads(cmu_path.read_text())["items"]
            # 2D-only: no keypoints_z attached → dataset emits no depth target,
            # loss masks z. Added to TRAIN only (keeps val FreiHAND-clean).
            train_items.extend(cmu_items)
            n_cmu = len(cmu_items)
        else:
            print(f"[build_3d] CMU manifest missing at {cmu_path} — skipping CMU")

    out_dir = _resolve_volume("/labeled_frames/hand_keypoints")
    train_path = out_dir / "freihand_cmu_3d_train.json"
    val_path = out_dir / "freihand_cmu_3d_val.json"
    train_path.write_text(json.dumps(
        {"version": 1, "task": "hand_keypoints", "items": train_items}))
    val_path.write_text(json.dumps(
        {"version": 1, "task": "hand_keypoints", "items": val_items}))
    volume.commit()
    return {"train_total": len(train_items), "train_freihand": n_fh_train,
            "train_cmu": n_cmu, "val_freihand": len(val_items), "skipped": n_skip,
            "train_manifest": "/labeled_frames/hand_keypoints/freihand_cmu_3d_train.json",
            "val_manifest": "/labeled_frames/hand_keypoints/freihand_cmu_3d_val.json"}


@app.function(
    gpu="A100",
    volumes={VOLUME_PATH: volume},
    timeout=TIMEOUT_SEC,
)
def train_hand_landmarks_3d(
    run_id: str = "hand_landmarks_3d_v1",
    train_manifest: str = "/labeled_frames/hand_keypoints/freihand_cmu_3d_train.json",
    val_manifest: str = "/labeled_frames/hand_keypoints/freihand_cmu_3d_val.json",
    epochs: int = 60,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    num_workers: int = 16,
    z_weight: float = 1.0,
) -> dict:
    """Train the from-scratch 21-keypoint hand landmark regressor WITH a 3D
    depth head (v3). v1 = FreiHAND (3D, depth-supervised) + CMU (2D, depth
    masked) — same config as v0 but ~2× the 2D supervision. A100 + large
    batch + many dataloader workers — the bottleneck for this 4.3M-param CNN
    is volume JPEG I/O, not compute, so workers matter more than GPU tier."""
    from pathlib import Path
    from training.detectors.train_landmarks import train as _train
    run_dir = Path(f"{VOLUME_PATH}/runs/{run_id}")
    result = _train(
        train_manifest=_resolve_volume(train_manifest),
        val_manifest=_resolve_volume(val_manifest),
        run_dir=run_dir,
        epochs=epochs, batch_size=batch_size, lr=lr,
        weight_decay=weight_decay, num_workers=num_workers,
        model_kwargs={"predict_z": True}, z_weight=z_weight,
    )
    volume.commit()
    return {"run_id": run_id, **result}


@app.function(
    gpu="L4",
    cpu=8.0,
    memory=32 * 1024,
    volumes={VOLUME_PATH: volume},
    timeout=60 * 30,
)
def eval_landmark_per_source(
    model_path: str = "/runs/hand_landmarks_v2_combined/best.pt",
    packed_meta: str = "/labeled_frames/hand_keypoints/packed_combined.meta.json",
    seed: int = 42,
    val_frac: float = 0.03,
) -> dict:
    """Eval-only: per-SOURCE mean px error of a landmark model on the packed
    val split (same seed/val_frac as train_hand_landmarks_packed). freihand-px
    is apples-to-apples vs v0's 12.82px; coco-px = in-the-wild quality."""
    import json, random, shutil
    from collections import defaultdict
    from pathlib import Path
    import torch
    from torch.utils.data import DataLoader
    from training.detectors.hand_landmarks import HandLandmarkRegressor
    from training.detectors.landmarks_dataset import PackedLandmarkDataset
    from training.detectors.train_landmarks import _collate

    vol_base = str(_resolve_volume(packed_meta))[: -len(".meta.json")]
    tmp_base = "/tmp/packed_combined"
    for ext in (".dat", ".kps.npy", ".vis.npy", ".meta.json"):
        if not Path(tmp_base + ext).exists():
            shutil.copy(vol_base + ext, tmp_base + ext)
    meta_path = tmp_base + ".meta.json"
    meta = json.loads(Path(meta_path).read_text())
    src = meta["source"]
    INPUT = meta["input_size"]
    ok = [i for i in range(meta["n"]) if src[i]]
    random.Random(seed).shuffle(ok)
    n_val = max(800, int(val_frac * len(ok)))
    val_idx = ok[:n_val]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = HandLandmarkRegressor(num_keypoints=21).to(device).eval()
    m.load_state_dict(torch.load(_resolve_volume(model_path), map_location=device,
                                 weights_only=False)["model"])
    ds = PackedLandmarkDataset(meta_path, augment=None, indices=val_idx)
    dl = DataLoader(ds, batch_size=512, shuffle=False, num_workers=8, collate_fn=_collate)

    acc = defaultdict(lambda: [0.0, 0.0])  # source -> [sum(err*vis), sum(vis)]
    idx_list = ds.indices
    pos = 0
    with torch.no_grad():
        for imgs, tgt in dl:
            imgs = imgs.to(device)
            if imgs.dtype == torch.uint8:
                imgs = imgs.float() / 255.0
            out = m(imgs)
            coords = tgt["coords"].to(device)
            vis = tgt["visibility"].to(device)
            err = ((out["coords"] - coords) ** 2).sum(-1).sqrt() * float(INPUT)  # (B,21)
            ekp = (err * vis).cpu()
            vkp = vis.cpu()
            for b in range(imgs.shape[0]):
                s = src[idx_list[pos + b]]
                acc[s][0] += float(ekp[b].sum())
                acc[s][1] += float(vkp[b].sum())
            pos += imgs.shape[0]

    per_source = {s: {"mean_px": round(v[0] / max(v[1], 1), 2),
                      "n_samples": sum(1 for i in val_idx if src[i] == s)}
                  for s, v in acc.items()}
    tot_e = sum(v[0] for v in acc.values())
    tot_v = sum(v[1] for v in acc.values())
    out = {"model": model_path, "overall_px": round(tot_e / max(tot_v, 1), 2),
           "per_source": per_source, "n_val": len(val_idx),
           "v0_freihand_cmu_ref_px": 12.82}
    (_resolve_volume("/runs/hand_landmarks_v2_combined/per_source_px.json")).write_text(json.dumps(out, indent=2))
    volume.commit()
    print(json.dumps(out, indent=2), flush=True)
    return out


@app.function(
    gpu="A100",
    cpu=16.0,            # option 1: real cores for the dataloader (memmap reads)
    memory=48 * 1024,
    volumes={VOLUME_PATH: volume},
    timeout=TIMEOUT_SEC,
)
def train_hand_landmarks_packed(
    run_id: str = "hand_landmarks_v2_combined",
    packed_meta: str = "/labeled_frames/hand_keypoints/packed_combined.meta.json",
    epochs: int = 60,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    num_workers: int = 16,
    val_frac: float = 0.03,
    seed: int = 42,
) -> dict:
    """Retrain the 2D landmark regressor on the PACKED combined set
    (FreiHAND + CMU + COCO-WholeBody = real in-the-wild hands). No per-epoch
    JPEG decode (memmap) → big batch + many workers on A100. This targets the
    in-the-wild bunched-keypoints failure; z stays off (shelved)."""
    import json, random, shutil, time
    from pathlib import Path
    from training.detectors.train_landmarks import train as _train
    from training.detectors.landmarks_dataset import PackedLandmarkDataset
    from training.detectors.landmarks_augment import default_train_augment

    # Localize the packed set to /tmp (local SSD) — per-epoch random reads from
    # the network volume are slow; the 17.8GB copy is a one-time ~1-2 min cost.
    vol_base = str(_resolve_volume(packed_meta))[: -len(".meta.json")]
    tmp_base = "/tmp/packed_combined"
    t0 = time.time()
    for ext in (".dat", ".kps.npy", ".vis.npy", ".meta.json"):
        if not Path(tmp_base + ext).exists():
            shutil.copy(vol_base + ext, tmp_base + ext)
    print(f"[packed] localized to /tmp in {time.time()-t0:.0f}s", flush=True)
    meta_path = tmp_base + ".meta.json"
    meta = json.loads(Path(meta_path).read_text())
    src = meta["source"]
    ok = [i for i in range(meta["n"]) if src[i]]
    random.Random(seed).shuffle(ok)
    n_val = max(800, int(val_frac * len(ok)))
    val_idx, train_idx = ok[:n_val], ok[n_val:]
    print(f"[packed] train={len(train_idx)} val={len(val_idx)} "
          f"(counts={meta['counts']})", flush=True)

    # option 2: GPU augmentation → dataloader returns RAW crops (augment=None),
    # the train loop augments the whole batch on the A100 (the 7× lever).
    train_ds = PackedLandmarkDataset(meta_path, augment=None, indices=train_idx)
    val_ds = PackedLandmarkDataset(meta_path, augment=None, indices=val_idx)

    run_dir = Path(f"{VOLUME_PATH}/runs/{run_id}")
    result = _train(
        run_dir=run_dir, epochs=epochs, batch_size=batch_size, lr=lr,
        weight_decay=weight_decay, num_workers=num_workers,
        train_ds=train_ds, val_ds=val_ds, gpu_augment=True,
    )
    volume.commit()
    return {"run_id": run_id, **result}


@app.function(
    gpu="L4",
    volumes={VOLUME_PATH: volume},
    timeout=60 * 30,
    cpu=16.0,
    memory=48 * 1024,
)
def eval_landmarks_per_source(
    run_id: str = "hand_landmarks_v2_combined",
    ckpt: str = "best.pt",
    packed_meta: str = "/labeled_frames/hand_keypoints/packed_combined.meta.json",
    val_frac: float = 0.03,
    seed: int = 42,
    batch_size: int = 512,
) -> dict:
    """Per-SOURCE px eval for the combined landmark retrain — the real
    comparison the handoff calls for. Rebuilds the EXACT held-out val split
    used by train_hand_landmarks_packed (same seed/val_frac/`ok` filter),
    then reports mean_keypoint_px_err split by meta['source']:
      - freihand-px = apples-to-apples vs v0's FreiHAND-only 12.82
      - coco-px     = the in-the-wild number that actually matters
    px metric matches train_landmarks exactly (per-keypoint vis-masked
    ||pred-gt||*input_size, averaged over keypoints)."""
    import json, random, shutil, time
    from pathlib import Path
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from training.detectors.landmarks_dataset import PackedLandmarkDataset
    from training.detectors.hand_landmarks import HandLandmarkRegressor

    vol_base = str(_resolve_volume(packed_meta))[: -len(".meta.json")]
    tmp_base = "/tmp/packed_combined"
    t0 = time.time()
    for ext in (".dat", ".kps.npy", ".vis.npy", ".meta.json"):
        if not Path(tmp_base + ext).exists():
            shutil.copy(vol_base + ext, tmp_base + ext)
    print(f"[eval] localized to /tmp in {time.time()-t0:.0f}s", flush=True)
    meta_path = tmp_base + ".meta.json"
    meta = json.loads(Path(meta_path).read_text())
    src = meta["source"]
    input_size = int(meta["input_size"])
    K = HandLandmarkRegressor.NUM_KEYPOINTS

    # Rebuild the SAME val split as the trainer (line-for-line).
    ok = [i for i in range(meta["n"]) if src[i]]
    random.Random(seed).shuffle(ok)
    n_val = max(800, int(val_frac * len(ok)))
    val_idx = ok[:n_val]
    print(f"[eval] val={len(val_idx)} (counts={meta['counts']})", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HandLandmarkRegressor(predict_z=False).to(device).eval()
    sd = torch.load(_resolve_volume(f"/runs/{run_id}/{ckpt}"),
                    map_location=device, weights_only=False)
    model.load_state_dict(sd["model"])
    print(f"[eval] loaded /runs/{run_id}/{ckpt} (epoch={sd.get('epoch')})", flush=True)

    val_ds = PackedLandmarkDataset(meta_path, augment=None, indices=val_idx)
    # source aligned to dataset order (dataset preserves order of valid idxs)
    src_per_item = [src[i] for i in val_ds.indices]
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=8, pin_memory=True)

    # per-source per-keypoint accumulators (match train metric exactly)
    acc: dict[str, list] = {}
    def _bucket(s):
        if s not in acc:
            acc[s] = [torch.zeros(K), torch.zeros(K)]
        return acc[s]

    off = 0
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device, non_blocking=True)
            if imgs.dtype == torch.uint8:
                imgs = imgs.float() / 255.0
            coords_gt = targets["coords"].to(device, non_blocking=True)
            vis = targets["visibility"].to(device, non_blocking=True)
            out = model(imgs)
            err = ((out["coords"] - coords_gt) ** 2).sum(-1).sqrt() * float(input_size)  # (B,K)
            err = (err * vis).cpu()
            visc = vis.cpu()
            b = imgs.shape[0]
            srcs = src_per_item[off:off + b]; off += b
            for j, s in enumerate(srcs):
                ek, nk = _bucket(s)
                ek += err[j]; nk += visc[j]

    def _pkpe(ek, nk):
        return float((ek / nk.clamp(min=1)).mean().item())

    per_source = {s: {"px": round(_pkpe(ek, nk), 3),
                      "n_items": src_per_item.count(s)}
                  for s, (ek, nk) in acc.items()}
    tot_e = sum(ek for ek, _ in acc.values())
    tot_n = sum(nk for _, nk in acc.values())
    overall = round(_pkpe(tot_e, tot_n), 3)
    result = {"run_id": run_id, "ckpt": ckpt, "epoch": sd.get("epoch"),
              "overall_px": overall, "per_source": per_source,
              "n_val": len(val_idx)}
    print(f"[eval] RESULT {json.dumps(result, indent=2)}", flush=True)
    out_dir = _resolve_volume(f"/runs/{run_id}")
    (out_dir / "per_source_eval.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    return result


@app.function(
    gpu="H100",
    volumes={VOLUME_PATH: volume},
    timeout=TIMEOUT_SEC,
)
def train_pose(
    train_manifest: str,
    val_manifest: str,
    run_id: str,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    num_workers: int = 8,
) -> dict:
    """Train the from-scratch 8-keypoint upper-body pose regressor (Phase 3.1).

    Routes through the keypoint-agnostic landmark trainer with the
    PoseRegressor model class and a pose_keypoints manifest. The
    manifest's per-item instance key is "persons" (vs "hands" for
    landmarks); the dataset is parameterized accordingly.
    """
    from pathlib import Path
    from training.detectors.train_landmarks import train as _train
    from training.detectors.pose_detector import PoseRegressor
    run_dir = Path(f"{VOLUME_PATH}/runs/{run_id}")
    result = _train(
        train_manifest=_resolve_volume(train_manifest),
        val_manifest=_resolve_volume(val_manifest),
        run_dir=run_dir,
        epochs=epochs, batch_size=batch_size, lr=lr,
        weight_decay=weight_decay, num_workers=num_workers,
        model_cls=PoseRegressor,
        num_keypoints=PoseRegressor.NUM_KEYPOINTS,
        instance_key="persons",
        input_size=PoseRegressor.INPUT_SIZE,
    )
    volume.commit()
    return {"run_id": run_id, **result}


@app.function(
    gpu=GPU,
    volumes={VOLUME_PATH: volume},
    timeout=TIMEOUT_SEC,
)
def train_face(
    train_manifest: str,
    val_manifest: str,
    run_id: str,
    epochs: int = 60,
    batch_size: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    num_workers: int = 4,
) -> dict:
    """Train the from-scratch face detector (Phase 3.2). Same arch as
    HandDetector, trained on WIDER FACE bbox manifests via the
    face_bbox manifest schema. FaceDetector subclasses HandDetector so
    the training loop in training/detectors/train.py is reused unchanged.
    """
    from pathlib import Path
    from training.detectors.train import train as _train
    from training.detectors import face_detector  # noqa: F401 — registers alias
    run_dir = Path(f"{VOLUME_PATH}/runs/{run_id}")
    result = _train(
        train_manifest=_resolve_volume(train_manifest),
        val_manifest=_resolve_volume(val_manifest),
        run_dir=run_dir,
        epochs=epochs, batch_size=batch_size, lr=lr,
        weight_decay=weight_decay, num_workers=num_workers,
    )
    volume.commit()
    return {"run_id": run_id, **result}


@app.function(
    cpu=4.0,
    memory=8192,
    volumes={VOLUME_PATH: volume},
    timeout=60 * 60 * 3,  # 3 hr — tar of ~13K files on Modal volume takes ~60-90 min due to per-file RTT
)
def tar_and_cleanup_dir(
    src_dir: str,
    tar_out: str,
    delete_after: bool = False,
) -> dict:
    """Tar a directory on the volume, optionally delete the loose tree.

    Use for trajectory-output cleanup: after a Modal extract writes ~12K
    loose JSONs to /trajectories_v8/, this collapses them into a single
    .tar (1 inode) and optionally removes the originals. Keeps the
    volume well under the 500K inode hard limit.

    Call sequence:
      extract_trajectories_v2_l4.remote(...) → writes /trajectories_v8/*
      tar_and_cleanup_dir.remote("/trajectories_v8", "/trajectories_v8.tar",
                                 delete_after=True)

    Trainer/loader downstream untars to /tmp at start of each consumer run.
    """
    import shutil, subprocess, time
    from pathlib import Path

    base = Path(VOLUME_PATH)
    src = base / src_dir.lstrip("/")
    out = base / tar_out.lstrip("/")
    if not src.exists():
        raise FileNotFoundError(f"src dir not found: {src}")

    t0 = time.time()
    n_files = sum(1 for _ in src.rglob("*"))
    print(f"taring {src} ({n_files} entries) → {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "-cf", str(out), "-C", str(src.parent), src.name], check=True)
    tar_size = out.stat().st_size
    print(f"tar done in {time.time() - t0:.0f}s, {tar_size / 1e9:.2f} GB")

    deleted = False
    if delete_after:
        # DATA-SAFETY: never rmtree until the archive PROVABLY contains every
        # source entry. tar exit-0 alone isn't enough — list it back and count.
        listed = subprocess.run(["tar", "-tf", str(out)],
                                capture_output=True, text=True, check=True)
        tar_entries = sum(1 for _ in listed.stdout.splitlines())
        if tar_entries < n_files:
            raise RuntimeError(
                f"tar VERIFY FAILED: archive has {tar_entries} entries < "
                f"{n_files} source entries — NOT deleting {src}")
        print(f"  tar verified: {tar_entries} archived ≥ {n_files} source — safe to delete")
        shutil.rmtree(src)
        deleted = True
        print(f"removed {src} ({n_files} entries reclaimed)")

    volume.commit()
    return {"tar_out": str(out), "n_files_archived": n_files,
            "tar_size_bytes": tar_size, "deleted_src": deleted}


@app.function(
    cpu=16.0,
    memory=32 * 1024,
    volumes={VOLUME_PATH: volume},
    timeout=60 * 60,
)
def untar_clips_to_volume(
    tar_path: str = "/datasets/sem_lex_top80_clips.tar",
    dest_dir: str = "/datasets/sem_lex_top80",
) -> dict:
    """Untar a clips tarball into a volume directory.

    Pairs with extract_trajectories_v2_a100_maxspeed pointing at the same
    dest. Use tar_and_cleanup_dir afterwards to reclaim inodes if needed.
    """
    import subprocess, time
    from pathlib import Path as _P
    src = _P(VOLUME_PATH + tar_path)
    dst = _P(VOLUME_PATH + dest_dir)
    if not src.exists():
        raise FileNotFoundError(f"tar not found: {src}")
    dst.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"untarring {src} ({src.stat().st_size/1e9:.2f} GB) → {dst}")
    subprocess.run(["tar", "-xf", str(src), "-C", str(dst)], check=True)
    n = sum(1 for _ in dst.rglob("*.webm")) + sum(1 for _ in dst.rglob("*.mp4"))
    print(f"untar done in {time.time()-t0:.0f}s, {n} clips at {dst}")
    volume.commit()
    return {"dest": str(dst), "n_clips": n, "elapsed_sec": int(time.time()-t0)}


@app.function(
    gpu="A100",
    volumes={VOLUME_PATH: volume},
    timeout=TIMEOUT_SEC,
    cpu=32.0,
    memory=64 * 1024,
)
def extract_trajectories_v2_a100_maxspeed(
    manifest: str,
    out_dir: str,
    hand_detector_ckpt: str,
    hand_landmarks_ckpt: str,
    pose_ckpt: str,
    handshape_encoder_ckpt: str = "",
    fps: float = 15.0,
    decode_workers: int = 24,
) -> dict:
    """A100 max-speed extract — for Phase 4.6 v4 run with encoder inference.

    Encoder forward (~382K hand crops through 11M-param CNN) dominates total
    time when encoder is enabled. A100 gives ~3-5× encoder throughput over L4
    for $5.13/hr extra. With 12,752-clip v4 manifest the math is:
      L4:   ~60 min × $0.80/hr = $0.80
      A100: ~20 min × $4.10/hr = $1.37
    Slightly more expensive but 3× faster wall-clock. Use when speed matters.
    """
    import sys
    from training.detectors.extract_trajectories_v2 import main as _main
    argv_backup = sys.argv[:]
    sys.argv = [
        "extract_trajectories_v2",
        "--manifest", str(_resolve_volume(manifest)),
        "--out-dir", str(_resolve_volume(out_dir)),
        "--hand-detector-ckpt", str(_resolve_volume(hand_detector_ckpt)),
        "--hand-landmarks-ckpt", str(_resolve_volume(hand_landmarks_ckpt)),
        "--pose-ckpt", str(_resolve_volume(pose_ckpt)),
        "--fps", str(fps),
        "--decode-workers", str(decode_workers),
        "--repo-root", VOLUME_PATH,
    ]
    if handshape_encoder_ckpt:
        sys.argv += ["--handshape-encoder-ckpt", str(_resolve_volume(handshape_encoder_ckpt))]
    try:
        rc = _main()
    finally:
        sys.argv = argv_backup
    volume.commit()
    return {"rc": rc, "out_dir": out_dir}


@app.function(
    gpu="L4",
    volumes={VOLUME_PATH: volume},
    timeout=TIMEOUT_SEC,
    cpu=32.0,            # extract is decode-bound → max CPU on the cheap L4 GPU
    memory=48 * 1024,
)
def extract_trajectories_v2_l4(
    manifest: str,
    out_dir: str,
    hand_detector_ckpt: str,
    hand_landmarks_ckpt: str,
    pose_ckpt: str,
    handshape_encoder_ckpt: str = "",
    fps: float = 15.0,
    decode_workers: int = 24,
    trim_idle: bool = True,   # store idle-trimmed (clean) trajectories
) -> dict:
    """L4 + cpu=16 + decode_workers=12 variant — trajectory extraction is
    CPU-bound (video decode) so L4 ($0.80/hr) at high CPU outperforms A100
    ($4.10/hr) at low CPU. ~$0.20-0.30 for the 7,397-clip job vs ~$2.05.

    Phase 4.6: if handshape_encoder_ckpt is provided, each detected hand bbox
    is also passed through the encoder and a 128D embedding is saved per
    hand in the trajectory JSON (`hand.embedding`). Downstream
    trajectory_from_frames auto-detects and emits 356D features.
    """
    import sys
    from training.detectors.extract_trajectories_v2 import main as _main
    argv_backup = sys.argv[:]
    sys.argv = [
        "extract_trajectories_v2",
        "--manifest", str(_resolve_volume(manifest)),
        "--out-dir", str(_resolve_volume(out_dir)),
        "--hand-detector-ckpt", str(_resolve_volume(hand_detector_ckpt)),
        "--hand-landmarks-ckpt", str(_resolve_volume(hand_landmarks_ckpt)),
        "--pose-ckpt", str(_resolve_volume(pose_ckpt)),
        "--fps", str(fps),
        "--decode-workers", str(decode_workers),
        "--repo-root", VOLUME_PATH,
    ]
    if trim_idle:
        sys.argv += ["--trim-idle"]
    if handshape_encoder_ckpt:
        sys.argv += ["--handshape-encoder-ckpt", str(_resolve_volume(handshape_encoder_ckpt))]
    try:
        rc = _main()
    finally:
        sys.argv = argv_backup
    volume.commit()
    return {"rc": rc, "out_dir": out_dir}


@app.function(
    gpu="L4",
    volumes={VOLUME_PATH: volume},
    timeout=TIMEOUT_SEC,
    cpu=32.0,
    memory=48 * 1024,
)
def extract_trajectories_faceanchored_l4(
    manifest: str,
    out_dir: str,
    hand_detector_ckpt: str,
    hand_landmarks_ckpt: str,
    pose_ckpt: str,
    face_detector_ckpt: str,
    fps: float = 15.0,
    decode_workers: int = 24,
    trim_idle: bool = True,
    limit: int = 0,
) -> dict:
    """FACE-ANCHORED trajectory extraction (the 'clean' terminal pipeline):
    face-detector-anchored pose crop + temporal face smoothing + PoseEMA +
    face-guard. Identical to extract_trajectories_v2_l4 otherwise. Used to test
    whether the face-anchored look trains a better classifier than the
    hand-anchored 75.8 baseline (/trajectories_top80_v3). `limit>0` runs a smoke
    subset.
    """
    import sys
    from training.detectors.extract_trajectories_v2 import main as _main
    argv_backup = sys.argv[:]
    sys.argv = [
        "extract_trajectories_v2",
        "--manifest", str(_resolve_volume(manifest)),
        "--out-dir", str(_resolve_volume(out_dir)),
        "--hand-detector-ckpt", str(_resolve_volume(hand_detector_ckpt)),
        "--hand-landmarks-ckpt", str(_resolve_volume(hand_landmarks_ckpt)),
        "--pose-ckpt", str(_resolve_volume(pose_ckpt)),
        "--face-detector-ckpt", str(_resolve_volume(face_detector_ckpt)),
        "--pose-anchor", "face",
        "--fps", str(fps),
        "--decode-workers", str(decode_workers),
        "--repo-root", VOLUME_PATH,
    ]
    if trim_idle:
        sys.argv += ["--trim-idle"]
    if limit and limit > 0:
        sys.argv += ["--limit", str(limit)]
    try:
        rc = _main()
    finally:
        sys.argv = argv_backup
    volume.commit()
    return {"rc": rc, "out_dir": out_dir}


@app.function(
    gpu="A100",
    volumes={VOLUME_PATH: volume},
    timeout=TIMEOUT_SEC,
    cpu=8.0,
    memory=16384,
)
def extract_trajectories_v2(
    manifest: str,
    out_dir: str,
    hand_detector_ckpt: str,
    hand_landmarks_ckpt: str,
    pose_ckpt: str,
    fps: float = 15.0,
    decode_workers: int = 6,
) -> dict:
    """Phase 4 Slice 4.1 v2 — batched + multi-worker-decode extractor.

    Manifest-driven (unified_clip_manifest.json covering both .mp4 + .webm).
    Reads `clip_path` entries relative to volume root.
    """
    import sys
    from training.detectors.extract_trajectories_v2 import main as _main
    argv_backup = sys.argv[:]
    sys.argv = [
        "extract_trajectories_v2",
        "--manifest", str(_resolve_volume(manifest)),
        "--out-dir", str(_resolve_volume(out_dir)),
        "--hand-detector-ckpt", str(_resolve_volume(hand_detector_ckpt)),
        "--hand-landmarks-ckpt", str(_resolve_volume(hand_landmarks_ckpt)),
        "--pose-ckpt", str(_resolve_volume(pose_ckpt)),
        "--fps", str(fps),
        "--decode-workers", str(decode_workers),
        "--repo-root", VOLUME_PATH,
    ]
    try:
        rc = _main()
    finally:
        sys.argv = argv_backup
    volume.commit()
    return {"rc": rc, "out_dir": out_dir}


@app.local_entrypoint()
def upload_v2yt_clips(
    local_dir: str = "dataset/clean/v2-yt/normalized_videos",
    remote_dir: str = "/datasets/asl_clips/v2-yt/normalized_videos",
) -> None:
    """Upload the 297 project_clean (v2-yt) mp4 clips to the Modal volume so
    extract_trajectories_v2 with the unified_clip_manifest_modal.json can
    decode them. The manifest points at /data/datasets/asl_clips/v2-yt/...
    but the source files were never uploaded (Session 17 path-mismatch bug)."""
    from pathlib import Path
    local = Path(local_dir).resolve()
    if not local.exists():
        raise SystemExit(f"local dir does not exist: {local}")
    mp4s = sorted(local.rglob("*.mp4"))
    print(f"[upload] {len(mp4s)} mp4s from {local}")
    print(f"[upload] → volume asl-mastery-data:{remote_dir}")
    with volume.batch_upload(force=True) as batch:
        batch.put_directory(str(local), remote_dir)
    print(f"[upload] done")


@app.function(
    cpu=4.0,
    memory=4096,
    volumes={VOLUME_PATH: volume},
    timeout=60 * 30,
)
def untar_sem_lex() -> dict:
    """Untar pushed Sem-Lex archive on volume → /datasets/sem_lex_clips/."""
    import subprocess
    from pathlib import Path
    tar_path = Path(f"{VOLUME_PATH}/datasets/sem_lex_clips.tar")
    out = Path(f"{VOLUME_PATH}/datasets/sem_lex_clips")
    if not tar_path.exists():
        return {"error": f"tar not found at {tar_path}"}
    out.mkdir(parents=True, exist_ok=True)
    rc = subprocess.call(["tar", "-xf", str(tar_path), "-C", str(out)])
    volume.commit()
    n = sum(1 for _ in out.rglob("*.webm"))
    return {"rc": rc, "webm_count": n}


@app.function(
    gpu=GPU,
    volumes={VOLUME_PATH: volume},
    timeout=TIMEOUT_SEC,
)
def extract_trajectories(
    clips_dir: str,
    out_dir: str,
    hand_detector_ckpt: str,
    hand_landmarks_ckpt: str,
    pose_ckpt: str,
    fps: float = 15.0,
) -> dict:
    """Run all three trained detectors across the ASL clip corpus and write
    per-clip landmark trajectory JSONs (Phase 4 Slice 4.1).
    """
    import argparse
    from training.detectors.extract_trajectories import main as _main
    import sys
    argv_backup = sys.argv[:]
    sys.argv = [
        "extract_trajectories",
        "--clips-dir", str(_resolve_volume(clips_dir)),
        "--out-dir", str(_resolve_volume(out_dir)),
        "--hand-detector-ckpt", str(_resolve_volume(hand_detector_ckpt)),
        "--hand-landmarks-ckpt", str(_resolve_volume(hand_landmarks_ckpt)),
        "--pose-ckpt", str(_resolve_volume(pose_ckpt)),
        "--fps", str(fps),
    ]
    try:
        rc = _main()
    finally:
        sys.argv = argv_backup
    volume.commit()
    return {"rc": rc, "out_dir": out_dir}


@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=TIMEOUT_SEC,
)
def fit_templates(
    trajectories_dir: str,
    out_dir: str,
    vocab_json: str,
    time_steps: int = 32,
    min_clips: int = 5,
) -> dict:
    """Fit per-sign distribution-of-templates from extracted trajectories
    (Phase 4 Slice 4.2). CPU-only — no GPU needed for stats.
    """
    import sys
    from training.detectors.fit_templates import main as _main
    argv_backup = sys.argv[:]
    sys.argv = [
        "fit_templates",
        "--trajectories-dir", str(_resolve_volume(trajectories_dir)),
        "--out-dir", str(_resolve_volume(out_dir)),
        "--vocab-json", str(_resolve_volume(vocab_json)),
        "--time-steps", str(time_steps),
        "--min-clips", str(min_clips),
    ]
    try:
        rc = _main()
    finally:
        sys.argv = argv_backup
    volume.commit()
    return {"rc": rc, "out_dir": out_dir}


@app.function(
    gpu="A100-80GB",
    volumes={VOLUME_PATH: volume},
    timeout=TIMEOUT_SEC,
    memory=96 * 1024,
)
def train_hand_det_resume_v2(
    run_id: str,
    resume_from: str = "/runs/hand_det_v0_a100_resume_20260521_145244Z/best.pt",
    train_manifest: str = "/labeled_frames/hand_bbox/external_train.json",
    val_manifest: str = "/labeled_frames/hand_bbox/external_val.json",
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    warmup_epochs: int = 3,
    num_workers: int = 8,
    dry_run: bool = False,
    gpu_aug: bool = True,
    from_scratch: bool = False,
    early_stop_patience: int = 3,
    cache_in_memory: bool = True,
) -> dict:
    """Train-only A100 entrypoint for the post-epoch-30 plateau push.

    Bundles the patches diagnosed in the epoch-30 audit (CIoU size loss,
    EMA, early-stop, fixed DataLoader workers, shared-memory uint8 cache,
    no torch.compile). Resumes from the existing 30-epoch checkpoint by
    default and runs up to 50 more epochs with plateau-based early stop.

    No extract phase — train-only. Run extract as a separate cheap job
    after inspecting val_loss curve.

    Outputs to /runs/<run_id>/ on the volume:
      - best.pt (live model, lowest val_loss)
      - best_ema.pt (EMA-averaged copy at same epoch)
      - history.json (cumulative across resumes — never lose visibility again)
      - last.pt
    """
    from pathlib import Path
    import os
    # Force cache into main-process heap (not /dev/shm). Containers default
    # /dev/shm to a tiny size and the 54 GB+ cache for 181k records hits
    # SIGBUS (exit 135). Main-process heap respects the function's memory=
    # request. Pairs with num_workers=0 in train.py (see CACHE_NO_SHM
    # branch there).
    os.environ.setdefault("CACHE_NO_SHM", "1")
    from training.detectors.train import train as _train
    from training.detectors.dataset import load_manifest
    from training.detectors.hand_detector import HandDetector
    from training.detectors.losses import HandDetectorLoss
    import torch

    run_dir = Path(f"{VOLUME_PATH}/runs/{run_id}")
    train_path = Path(f"{VOLUME_PATH}{train_manifest}") if train_manifest.startswith("/") else Path(train_manifest)
    val_path = Path(f"{VOLUME_PATH}{val_manifest}") if val_manifest.startswith("/") else Path(val_manifest)
    resume_path = None if from_scratch else (Path(f"{VOLUME_PATH}{resume_from}") if resume_from.startswith("/") else Path(resume_from))

    if dry_run:
        # Dry-run: verify paths, manifests, model+loss numerics, and resume
        # checkpoint exists. Exits before any GPU training step. Catches the
        # entire failure-mode class from session 16's deep-dive in <30 s.
        print(f"=== DRY RUN ===  from_scratch={from_scratch}")
        print(f"  train_path: {train_path}  exists={train_path.exists()}")
        print(f"  val_path:   {val_path}  exists={val_path.exists()}")
        if resume_path is not None:
            print(f"  resume_path:{resume_path}  exists={resume_path.exists()}")
        check_paths = [train_path, val_path]
        if resume_path is not None:
            check_paths.append(resume_path)
        for p in check_paths:
            if not p.exists():
                raise FileNotFoundError(p)
        # Load manifests (exercises load_manifest, including OOB handling)
        tr_records = load_manifest(train_path)
        va_records = load_manifest(val_path)
        print(f"  train records loaded: {len(tr_records)}")
        print(f"  val records loaded:   {len(va_records)}")
        # 1-batch forward + loss + abort-on-NaN
        from training.detectors.dataset import HandBboxDataset
        ds = HandBboxDataset(tr_records[:4], augment=None, cache_in_memory=False)
        from training.detectors.train import _collate
        batch = _collate([ds[i] for i in range(min(4, len(ds)))])
        images, targets = batch
        model = HandDetector().cuda()
        loss_fn = HandDetectorLoss()
        images = images.cuda()
        targets = {k: v.cuda() for k, v in targets.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(images)
            losses = loss_fn(out, targets)
        for k, v in losses.items():
            ok = torch.isfinite(v).item()
            print(f"  loss[{k}]={v.item():.4f}  finite={ok}")
            if not ok:
                raise RuntimeError(f"non-finite loss in dry-run: {k}")
        volume.commit()
        return {"dry_run": True, "ok": True, "n_train": len(tr_records),
                "n_val": len(va_records)}

    result = _train(
        train_manifest=train_path,
        val_manifest=val_path,
        run_dir=run_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=1e-4,
        num_workers=num_workers,
        resume_from=resume_path,
        cache_in_memory=cache_in_memory,
        use_bf16=True,
        use_compile=False,      # not worth on 2.3M params; adds warmup cost
        use_channels_last=True,
        warmup_epochs=warmup_epochs,
        use_ema=True,
        ema_decay=0.999,
        early_stop_patience=early_stop_patience,
        early_stop_delta=0.005,
        gpu_aug=gpu_aug,
    )
    volume.commit()
    return {"run_id": run_id, **result}


@app.function(
    cpu=8.0,
    memory=16384,
    volumes={VOLUME_PATH: volume},
    timeout=60 * 60 * 2,
)
def download_hagrid_from_hf(
    subsample: int = 120_000,
    seed: int = 42,
    repo_id: str = "cj-mills/hagrid-sample-500k-384p",
) -> dict:
    """Pull HaGRID v1 (384p) from HuggingFace's cj-mills mirror.

    This mirror is materially better than Sbercloud Moscow for our use:
      - 13.4 GB total (vs 119 GB FullHD) — only ~30 GB on disk after extract
      - HuggingFace CDN (Cloudflare) is geographically close to Modal US:
        observed 200-1000+ MiB/s vs Sbercloud's 40 MiB/s per-IP throttle
      - Schema retains only `bboxes` + `labels` + `leading_hand` etc.;
        the disallowed `hand_landmarks` (MediaPipe-derived) and `meta`
        (FairFace/MiVOLO-derived) fields are NOT present in this mirror.
        Same compliance posture as the original Sbercloud release —
        bboxes are still human-drawn via Toloka per HaGRID paper.
      - 509,323 images. License CC-BY-SA-4.0, same as upstream.

    Output:
      - /external/hagrid_v2/images_sampled/<class>/<id>.jpg — 120k JPGs
      - /external/hagrid_v2/hagrid_manifest.json — FrameRecord-format
        manifest with absolute paths + bboxes converted to pixel xyxy.
    """
    import io, json, random, shutil, time, zipfile
    from pathlib import Path
    from huggingface_hub import hf_hub_download
    from PIL import Image
    from training.detectors.external_loaders.hagrid import _check_no_landmark_drift

    base = Path(f"{VOLUME_PATH}/external/hagrid_v2")
    img_root = base / "images_sampled"
    img_root.mkdir(parents=True, exist_ok=True)
    manifest_out = base / "hagrid_manifest.json"

    # The cj-mills mirror ships as a SINGLE 13.4 GB zip in the repo root,
    # not parquet shards. We download it via hf_hub_download (resumable,
    # uses HF's Cloudflare CDN), then iterate entries in-place with
    # zipfile — no full extraction needed, just pull the 120k we sample.
    print(f"downloading {repo_id}/hagrid-sample-500k-384p.zip from HuggingFace Hub...")
    cache_dir = base / "hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    t_dl = time.time()
    zip_local_path = hf_hub_download(
        repo_id=repo_id,
        filename="hagrid-sample-500k-384p.zip",
        repo_type="dataset",
        cache_dir=str(cache_dir),
    )
    dl_elapsed = time.time() - t_dl
    zip_size_gb = Path(zip_local_path).stat().st_size / 1e9
    print(f"  downloaded {zip_size_gb:.1f} GB in {dl_elapsed:.0f}s "
          f"({zip_size_gb * 1024 / max(dl_elapsed, 1):.0f} MB/s)")

    # Open the local zip and probe layout
    print(f"  opening zip to read entry list...")
    zf = zipfile.ZipFile(zip_local_path, "r")
    namelist = zf.namelist()
    print(f"  zip has {len(namelist)} entries; first 10: {namelist[:10]}")

    # Find all image entries (.jpg or .jpeg) and any annotation file
    # (likely train.json or similar). The cj-mills layout is undocumented
    # so we probe it empirically.
    image_entries = [n for n in namelist
                     if n.lower().endswith((".jpg", ".jpeg"))]
    json_entries = [n for n in namelist if n.lower().endswith(".json")]
    print(f"  found {len(image_entries)} image entries, "
          f"{len(json_entries)} json entries")
    if json_entries:
        print(f"  json entries: {json_entries[:5]}")

    # Read annotations to map image_id → (label, bboxes_normalized).
    # We try the obvious filenames first.
    annotations: dict[str, dict] = {}
    for je in json_entries[:20]:
        try:
            with zf.open(je) as f:
                blob = json.loads(f.read().decode("utf-8"))
            if isinstance(blob, dict):
                # Could be flat {image_id: {...}} or {class: {image_id: {...}}}
                # Probe a sample value
                sample_val = next(iter(blob.values()), None)
                if isinstance(sample_val, dict) and "bboxes" in sample_val:
                    # Flat schema
                    annotations.update(blob)
                elif isinstance(sample_val, dict):
                    # Nested schema (class → {image_id → {...}})
                    for cls_dict in blob.values():
                        if isinstance(cls_dict, dict):
                            annotations.update(cls_dict)
            print(f"  loaded {je}: now {len(annotations)} annotations total")
        except Exception as e:
            print(f"  WARN failed to read {je}: {e}")
    if not annotations:
        zf.close()
        return {"ok": False, "error": "no annotations parsed from zip",
                "json_entries": json_entries[:5],
                "image_entries_sample": image_entries[:5]}

    # Build image_id → zip_entry index from image filenames
    print("  building image_id → zip_entry index...")
    id_to_entry: dict[str, str] = {}
    for entry in image_entries:
        # image_id is the filename stem
        last_slash = entry.rfind("/")
        stem = entry[last_slash + 1:]
        if stem.lower().endswith(".jpg"):
            image_id = stem[:-4]
        elif stem.lower().endswith(".jpeg"):
            image_id = stem[:-5]
        else:
            continue
        # Prefer shortest path on collisions for determinism
        prior = id_to_entry.get(image_id)
        if prior is None or len(entry) < len(prior):
            id_to_entry[image_id] = entry
    print(f"  indexed {len(id_to_entry)} image IDs")

    # Intersect annotations with present-in-zip IDs + filter to those
    # with non-empty bboxes
    rng = random.Random(seed)
    candidate_ids = []
    for image_id, ann in annotations.items():
        bboxes = ann.get("bboxes") or []
        if not bboxes:
            continue
        if any(len(b) >= 4 and b[2] > 0 and b[3] > 0 for b in bboxes):
            if image_id in id_to_entry:
                candidate_ids.append(image_id)
    print(f"  {len(candidate_ids)} candidates with bboxes + in zip")
    rng.shuffle(candidate_ids)
    candidate_ids = candidate_ids[:subsample]
    print(f"  selected {len(candidate_ids)} for extract")

    records: list[dict] = []
    n_emitted = 0
    n_no_box = 0
    n_bad = 0
    t0 = time.time()

    # Single loop — extract each selected image + emit FrameRecord
    for image_id in candidate_ids:
            ann = annotations[image_id]
            bboxes_norm = ann.get("bboxes") or []
            if not bboxes_norm:
                n_no_box += 1
                continue
            zip_entry = id_to_entry[image_id]
            try:
                with zf.open(zip_entry) as f:
                    img_bytes = f.read()
                img = Image.open(io.BytesIO(img_bytes))
            except Exception:
                n_bad += 1
                continue
            W, H = img.size
            abs_bboxes = []
            for b in bboxes_norm:
                if len(b) < 4:
                    continue
                nx, ny, nw, nh = float(b[0]), float(b[1]), float(b[2]), float(b[3])
                if nw <= 0 or nh <= 0:
                    continue
                x0 = nx * W
                y0 = ny * H
                x1 = (nx + nw) * W
                y1 = (ny + nh) * H
                if x1 - x0 < 4 or y1 - y0 < 4:
                    continue
                abs_bboxes.append([x0, y0, x1, y1])
            if not abs_bboxes:
                n_bad += 1
                continue

            # Bucket by leading gesture-label to keep dirs manageable.
            labels_field = ann.get("labels") or []
            cls = labels_field[0] if labels_field else "unknown"
            out_dir = img_root / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"hagrid_{n_emitted:07d}.jpg"
            try:
                img.convert("RGB").save(out_path, "JPEG", quality=90)
            except Exception:
                n_bad += 1
                continue

            rec = {
                "image_path": str(out_path),
                "width": W,
                "height": H,
                "bboxes": abs_bboxes,
            }
            # Defense-in-depth: tripwire confirms the emitted record
            # carries no forbidden field (landmark/keypoint/age/race/etc.).
            # The HF mirror's schema already omits these, but if upstream
            # ever widens the schema and we silently propagate it, this
            # catches.
            _check_no_landmark_drift(rec)
            records.append(rec)
            n_emitted += 1
            if n_emitted % 5000 == 0:
                elapsed = time.time() - t0
                rate = n_emitted / max(elapsed, 1e-3)
                print(f"  {n_emitted}/{subsample} | "
                      f"{rate:.0f} img/sec | "
                      f"ETA {(subsample - n_emitted) / max(rate, 0.1):.0f}s",
                      flush=True)

    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(json.dumps({
        "version": 1,
        "task": "hand_bbox",
        "source": f"hagrid_v1_via_{repo_id}",
        "notes": (
            "Downloaded via HuggingFace mirror. ADR-0015 compliant: "
            "only the `bboxes` field is read (human-drawn by Toloka per "
            "HaGRID paper). The mirror schema does not include "
            "`hand_landmarks` (MediaPipe) or `meta` (FairFace/MiVOLO)."
        ),
        "items": records,
    }))
    volume.commit()
    elapsed = time.time() - t0
    print(f"DONE: {n_emitted} records in {elapsed:.0f}s "
          f"({n_no_box} skipped no-bbox, {n_bad} skipped bad)")
    return {
        "n_records": n_emitted,
        "elapsed_s": elapsed,
        "manifest": str(manifest_out),
        "n_skipped_no_bbox": n_no_box,
        "n_skipped_bad": n_bad,
    }


@app.function(
    cpu=16.0,
    memory=16384,
    volumes={VOLUME_PATH: volume},
    timeout=60 * 60 * 2,
)
def hagrid_streaming_subsample(
    subsample: int = 120_000,
    seed: int = 42,
    workers: int = 16,
) -> dict:
    """One-shot: stream-extract a 120k subsample of HaGRID v2 from the
    remote 512px zip WITHOUT downloading the full 119 GB file.

    Process:
      1. Download annotations.zip (tiny) → unpack class JSONs.
      2. Read class JSONs, collect (class, image_id) candidates with
         non-empty `bboxes` (ADR-0012-compliant ingest).
      3. Stratified-sample 120k pairs across classes.
      4. Open the remote 119 GB zip via remotezip (reads only central
         directory — ~few hundred KB).
      5. For each kept pair, HTTP-Range GET that single entry and
         write to /external/hagrid_v2/images_sampled/<cls>/<id>.jpg.
      6. Done — no zip ever lands on the filesystem.

    Total network transfer: ~26 GB (vs 119 GB). At Sbercloud's observed
    ~40 MiB/s aggregate, this is ~10–12 min instead of ~50 min.
    """
    import json, random, shutil, subprocess
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from pathlib import Path
    from remotezip import RemoteZip

    IMG_URL = "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/hagridv2_512.zip"
    ANN_URL = "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/annotations_with_landmarks/annotations.zip"

    base = Path(f"{VOLUME_PATH}/external/hagrid_v2")
    ann_root = base / "annotations"
    img_root = base / "images_sampled"
    ann_root.mkdir(parents=True, exist_ok=True)
    img_root.mkdir(parents=True, exist_ok=True)

    # ── Step 1: annotations.zip (tiny, conventional download + unzip)
    ann_zip = ann_root / "annotations.zip"
    ann_aria_marker = ann_root / "annotations.zip.aria2"
    # aria2's .aria2 sidecar file is its "still in flight" marker — its
    # presence means the zip is incomplete even if a partial file exists.
    if ann_aria_marker.exists():
        for f in (ann_zip, ann_aria_marker):
            if f.exists():
                f.unlink()
    if not ann_zip.exists() or ann_zip.stat().st_size < 1000:
        print(f"  fetching annotations.zip")
        subprocess.check_call([
            "aria2c", "-x", "8", "-s", "8", "--check-certificate=false",
            "--file-allocation=none", "--auto-file-renaming=false",
            "--allow-overwrite=true",
            "-d", str(ann_root), "-o", "annotations.zip", ANN_URL,
        ])
    if not list(ann_root.glob("*.json")):
        subprocess.check_call(["unzip", "-qq", "-o", str(ann_zip), "-d", str(ann_root)])
    json_paths = list(ann_root.rglob("*.json"))
    print(f"  {len(json_paths)} class JSONs")

    # ── Step 2: candidates with valid bboxes
    candidates_by_class: dict[str, list[str]] = {}
    for jp in json_paths:
        cls = jp.stem
        try:
            data = json.loads(jp.read_text())
        except Exception:
            continue
        ids = []
        for image_id, entry in data.items():
            bboxes = entry.get("bboxes") or []
            for b in bboxes:
                if len(b) >= 4 and b[2] > 0 and b[3] > 0:
                    ids.append(image_id)
                    break
        if ids:
            candidates_by_class[cls] = ids
    total_candidates = sum(len(v) for v in candidates_by_class.values())
    print(f"  candidates: {total_candidates} across {len(candidates_by_class)} classes")

    # ── Step 3: stratified-sample proportional to class size
    rng = random.Random(seed)
    selections: dict[str, list[str]] = {}
    if total_candidates <= subsample:
        selections = candidates_by_class
    else:
        for cls, ids in candidates_by_class.items():
            n_take = max(1, int(round(subsample * len(ids) / total_candidates)))
            rng.shuffle(ids)
            selections[cls] = ids[:n_take]
    final_total = sum(len(v) for v in selections.values())
    print(f"  selected {final_total} images for range-extract")

    # ── Step 4: open the remote zip; namelist tells us actual paths
    print(f"  opening remote zip @ {IMG_URL}")
    rzip = RemoteZip(IMG_URL)
    namelist = rzip.namelist()
    namelist_set = set(namelist)
    print(f"  remote zip has {len(namelist_set)} entries")
    # Log a sample so future debugging never has to guess at the layout
    print(f"  first 10 entries: {namelist[:10]}")

    # Build an image_id → full-zip-path index by searching the namelist
    # for entries containing /<image_id>.jpg. Avoids hardcoding any
    # layout prefix — works whether the zip is flat, nested by class,
    # nested by split, or anything else.
    print("  building image_id → zip_path index (one pass over namelist)...")
    id_index: dict[str, str] = {}
    for entry in namelist:
        if not entry.endswith(".jpg"):
            continue
        # image_id is the filename stem
        last_slash = entry.rfind("/")
        image_id = entry[last_slash + 1:-4] if last_slash >= 0 else entry[:-4]
        # If the same id appears in multiple paths, prefer the SHORTEST
        # (closest to the root), so we deterministically pick one.
        prior = id_index.get(image_id)
        if prior is None or len(entry) < len(prior):
            id_index[image_id] = entry
    print(f"  indexed {len(id_index)} unique image IDs from zip")

    # ── Step 5: parallel HTTP-range GET per kept (cls, id)
    tasks: list[tuple[str, str, Path]] = []
    missing_in_zip = 0
    for cls, ids in selections.items():
        out_dir = img_root / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        for image_id in ids:
            zip_entry = id_index.get(image_id)
            if zip_entry is None:
                missing_in_zip += 1
                continue
            target = out_dir / f"{image_id}.jpg"
            if target.exists():
                continue
            tasks.append((zip_entry, image_id, target))
    print(f"  fetching {len(tasks)} entries via {workers} workers "
          f"({missing_in_zip} selected IDs not present in zip)")
    rzip.close()  # close the global one; each worker gets its own RemoteZip

    n_done = 0
    n_failed = 0

    # CRITICAL: each worker keeps ONE RemoteZip open across all its tasks.
    # The earlier per-task RemoteZip opened a fresh HTTP connection + re-
    # fetched the central directory every time → ~3-4 round-trips of
    # overhead per file. With ~120k tasks that was the dominant cost. The
    # task queue + thread-local zip pattern below shares the zip across
    # all pulls in a worker, so the central-directory hit is paid 16
    # times total (once per worker) instead of 120k times.
    import queue as _queue
    import threading
    import time

    task_q: _queue.Queue = _queue.Queue()
    for t in tasks:
        task_q.put(t)

    done_lock = threading.Lock()
    counters = {"done": 0, "failed": 0, "last_log_n": 0}
    t0 = time.time()

    def _worker():
        # One RemoteZip per worker, reused for thousands of pulls.
        try:
            wz = RemoteZip(IMG_URL)
        except Exception as e:
            print(f"  worker init failed: {e}")
            return
        while True:
            try:
                zip_entry, image_id, target = task_q.get_nowait()
            except _queue.Empty:
                break
            try:
                with wz.open(zip_entry) as srcf, open(target, "wb") as dstf:
                    shutil.copyfileobj(srcf, dstf, length=64 * 1024)
                ok = True
            except Exception:
                ok = False
            with done_lock:
                if ok:
                    counters["done"] += 1
                else:
                    counters["failed"] += 1
                total_handled = counters["done"] + counters["failed"]
                if total_handled - counters["last_log_n"] >= 2000:
                    counters["last_log_n"] = total_handled
                    elapsed = time.time() - t0
                    rate = total_handled / max(elapsed, 1e-3)
                    eta = (len(tasks) - total_handled) / max(rate, 0.1)
                    print(f"  {total_handled}/{len(tasks)} done | "
                          f"{rate:.0f} files/sec | ETA {eta:.0f}s", flush=True)
            task_q.task_done()
        try:
            wz.close()
        except Exception:
            pass

    threads = [threading.Thread(target=_worker, daemon=True) for _ in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    n_done = counters["done"]
    n_failed = counters["failed"]

    volume.commit()
    return {
        "ok": True,
        "n_done": n_done,
        "n_failed": n_failed,
        "elapsed_s": time.time() - t0,
        "out_dir": str(img_root),
    }


@app.function(
    cpu=16.0,
    memory=32768,
    volumes={VOLUME_PATH: volume},
    timeout=60 * 60 * 2,
)
def download_hagrid_v2_lite() -> dict:
    """Download HaGRID v2 **512px lite** into the Modal volume.

    Same bbox labels + same compliance status as the FullHD per-class
    download (audit entry I), but 5× less data and a single zip — which
    means a single aria2c stream can saturate its 16-connection split
    against the origin without per-class round-trip overhead.

    Two parallel streams:
      - hagridv2_512.zip   (~119 GB single zip, images)
      - annotations.zip    (~few MB, the label JSONs)

    The 512px image variant is sufficient because our `HandDetector`
    resizes to 320×320 anyway; bbox labels are identical to FullHD.
    """
    import subprocess
    from pathlib import Path
    from concurrent.futures import ThreadPoolExecutor, as_completed

    ROOT = Path(f"{VOLUME_PATH}/external/hagrid_v2")
    ROOT.mkdir(parents=True, exist_ok=True)
    IMG_ROOT = ROOT / "images"
    IMG_ROOT.mkdir(parents=True, exist_ok=True)
    ANN_ROOT = ROOT / "annotations"
    ANN_ROOT.mkdir(parents=True, exist_ok=True)

    IMG_URL = "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/hagridv2_512.zip"
    ANN_URL = "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/annotations_with_landmarks/annotations.zip"

    has_aria = subprocess.call(["which", "aria2c"], stdout=subprocess.DEVNULL) == 0

    def _download(target_dir: Path, fname: str, url: str) -> tuple[str, int]:
        print(f"  [hagrid] {fname} ← {url}", flush=True)
        if has_aria:
            # -x16 -s16 -k4M maxes per-server connections and bumps segment
            # size to 4 MB so we don't burn small-segment overhead on a
            # 119 GB single file. --max-tries 0 = retry forever on resume.
            rc = subprocess.call([
                "aria2c",
                "-x", "16", "-s", "16", "-k", "4M",
                "--check-certificate=false",
                "--file-allocation=none",
                "--auto-file-renaming=false",
                "--allow-overwrite=true",
                "--max-tries=0",
                "-c",
                "-d", str(target_dir), "-o", fname, url,
            ])
        else:
            rc = subprocess.call(["curl", "-L", "-C", "-", "--fail", "-o",
                                  str(target_dir / fname), url])
        return (fname, rc)

    tasks = [
        (IMG_ROOT, "hagridv2_512.zip", IMG_URL),
        (ANN_ROOT, "annotations.zip", ANN_URL),
    ]

    results = {}
    with ThreadPoolExecutor(max_workers=2) as ex:
        futures = {ex.submit(_download, d, fn, url): fn for d, fn, url in tasks}
        for fut in as_completed(futures):
            fn, rc = fut.result()
            results[fn] = "ok" if rc == 0 else f"rc{rc}"
    print(f"  downloads complete: {results}")

    # We deliberately DO NOT extract the giant 119 GB zip here. The
    # follow-on entrypoint extract_hagrid_subsampled() reads it with
    # `zipfile.ZipFile.open()` and pulls just the 120k images we want.
    # The 119 GB zip is then deleted; we save ~119 GB of disk + ~434k
    # of inodes by never landing those images on the filesystem.

    # Annotations are tiny — extract them so the normalizer can read them.
    extract_results = {}
    for z in sorted(ANN_ROOT.glob("*.zip")):
        print(f"  [hagrid] extract {z.name}", flush=True)
        rc = subprocess.call(["unzip", "-qq", "-o", str(z), "-d", str(ANN_ROOT)])
        extract_results[z.name] = "ok" if rc == 0 else f"rc{rc}"

    volume.commit()
    return {"downloads": results, "extracts": extract_results,
            "image_zip": str(IMG_ROOT / "hagridv2_512.zip"),
            "ann_root": str(ANN_ROOT)}


@app.function(
    cpu=8.0,
    memory=8192,
    volumes={VOLUME_PATH: volume},
    timeout=60 * 60,
)
def extract_hagrid_subsampled(
    subsample: int = 120_000,
    seed: int = 42,
    delete_zips_after: bool = True,
) -> dict:
    """Extract ONLY a random `subsample` of HaGRID images from the
    per-class zips, keeping our Modal volume inode budget under control.

    Modal volume has a 500k-inode cap; full HaGRID = 554,800 images =
    554k inodes if extracted naively, over budget. This entrypoint:

      1. Extracts the annotations.zip → 18 per-class JSONs (~few hundred
         inodes total).
      2. Reads each class JSON; collects (class, image_id) pairs whose
         `bboxes` field is non-empty + non-degenerate.
      3. Stratified-samples `subsample` pairs total across classes
         proportional to each class's pool size.
      4. For each kept pair: extracts ONLY that image from its class zip
         into `/external/hagrid_v2/images_sampled/<class>/<id>.jpg`.
      5. Optionally deletes the original 18 class zips to free disk.

    After: ~100k image inodes + 18 annotation JSONs + 18 class dirs +
    bookkeeping ≈ ~100k inodes added. Comfortable headroom vs 500k cap.
    """
    import json, random, shutil, subprocess, zipfile
    from pathlib import Path

    base = Path(f"{VOLUME_PATH}/external/hagrid_v2")
    ann_root = base / "annotations"
    img_root_full = base / "images"
    img_root_sampled = base / "images_sampled"

    # ── Step 1: extract annotations zip if not already done
    ann_zip = ann_root / "annotations.zip"
    if ann_zip.exists() and not list(ann_root.glob("*.json")):
        print(f"  extracting {ann_zip}")
        subprocess.call(["unzip", "-qq", "-o", str(ann_zip), "-d", str(ann_root)])
    # Locate the per-class JSONs. HaGRID v2 puts them in
    # annotations/<split>/<class>.json under the zip; flatten to ann_root.
    json_paths = list(ann_root.rglob("*.json"))
    if not json_paths:
        return {"ok": False, "error": "no annotation JSONs found post-extract"}
    print(f"  found {len(json_paths)} annotation JSON files")

    # ── Step 2: collect candidates with valid bboxes
    candidates_by_class: dict[str, list[str]] = {}
    for jp in json_paths:
        cls = jp.stem
        try:
            data = json.loads(jp.read_text())
        except Exception as e:
            print(f"  WARN skipping {jp.name}: {e}")
            continue
        ids = []
        for image_id, entry in data.items():
            bboxes = entry.get("bboxes") or []
            ok = False
            for b in bboxes:
                if len(b) >= 4 and b[2] > 0 and b[3] > 0:
                    ok = True
                    break
            if ok:
                ids.append(image_id)
        if ids:
            candidates_by_class[cls] = ids
    total_candidates = sum(len(v) for v in candidates_by_class.values())
    print(f"  candidates: {total_candidates} across {len(candidates_by_class)} classes")

    # ── Step 3: stratified sample (proportional to class size)
    rng = random.Random(seed)
    selections: dict[str, list[str]] = {}
    if total_candidates <= subsample:
        # use everything
        selections = candidates_by_class
    else:
        for cls, ids in candidates_by_class.items():
            n_take = max(1, int(round(subsample * len(ids) / total_candidates)))
            rng.shuffle(ids)
            selections[cls] = ids[:n_take]
    final_total = sum(len(v) for v in selections.values())
    print(f"  selected {final_total} images for extraction")

    # ── Step 4: pull just-those images from the SINGLE 512px zip
    # The 512px variant packs everything under <class>/<id>.jpg in one zip.
    # We open it once and iterate the selected (class, id) tuples.
    img_root_sampled.mkdir(parents=True, exist_ok=True)
    big_zip = img_root_full / "hagridv2_512.zip"
    n_extracted = 0
    n_missing = 0
    if big_zip.exists():
        # Probe the namelist for the actual path layout once.
        with zipfile.ZipFile(big_zip, "r") as zf:
            namelist_set = set(zf.namelist())
            for cls, ids in selections.items():
                out_dir = img_root_sampled / cls
                out_dir.mkdir(parents=True, exist_ok=True)
                # Try the most likely layouts in order.
                for image_id in ids:
                    candidates = [
                        f"{cls}/{image_id}.jpg",
                        f"hagrid_512/{cls}/{image_id}.jpg",
                        f"hagrid/{cls}/{image_id}.jpg",
                        f"{image_id}.jpg",
                    ]
                    src = next((c for c in candidates if c in namelist_set), None)
                    if src is None:
                        n_missing += 1
                        continue
                    target = out_dir / f"{image_id}.jpg"
                    if target.exists():
                        n_extracted += 1
                        continue
                    with zf.open(src) as srcf, open(target, "wb") as dstf:
                        shutil.copyfileobj(srcf, dstf)
                    n_extracted += 1
                print(f"  {cls}: kept {len(ids)} samples")
    else:
        # Fall back to per-class zips (legacy FullHD layout)
        for cls, ids in selections.items():
            class_zip = img_root_full / f"{cls}.zip"
            if not class_zip.exists():
                print(f"  WARN missing zip for class {cls}")
                n_missing += len(ids)
                continue
            out_dir = img_root_sampled / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(class_zip, "r") as zf:
                namelist_set = set(zf.namelist())
                for image_id in ids:
                    cand1 = f"{image_id}.jpg"
                    cand2 = f"{cls}/{image_id}.jpg"
                    src = cand1 if cand1 in namelist_set else (
                        cand2 if cand2 in namelist_set else None)
                    if src is None:
                        n_missing += 1
                        continue
                    target = out_dir / f"{image_id}.jpg"
                    if target.exists():
                        n_extracted += 1
                        continue
                    with zf.open(src) as srcf, open(target, "wb") as dstf:
                        shutil.copyfileobj(srcf, dstf)
                    n_extracted += 1
            print(f"  {cls}: extracted {len(ids)} images")

    # ── Step 5: free disk by deleting the source zip(s)
    n_zips_deleted = 0
    if delete_zips_after:
        for z in img_root_full.glob("*.zip"):
            try:
                z.unlink()
                n_zips_deleted += 1
            except Exception:
                pass

    volume.commit()
    return {
        "ok": True,
        "n_extracted": n_extracted,
        "n_missing": n_missing,
        "n_zips_deleted": n_zips_deleted,
        "out_dir": str(img_root_sampled),
    }


@app.function(
    cpu=2.0,
    memory=4096,
    volumes={VOLUME_PATH: volume},
    timeout=60 * 30,
)
def build_hagrid_manifest_and_mix(
    subsample: int = 120_000,
    out_hagrid_manifest: str = "/labeled_frames/hand_bbox/hagrid_v2.json",
    out_mixed_manifest: str = "/labeled_frames/hand_bbox/external_plus_hagrid_train.json",
    base_train_manifest: str = "/labeled_frames/hand_bbox/external_train.json",
    seed: int = 42,
) -> dict:
    """Normalize HaGRID v2 JSONs into our FrameRecord schema (BBOX-ONLY
    per ADR-0012), then build a mixed CMU+FreiHAND+HaGRID-subsample
    training manifest.

    See training/detectors/external_loaders/hagrid.py for the compliance
    rationale — only the human-drawn `bboxes` field crosses into our
    pipeline; `hand_landmarks` (MediaPipe) and `meta` (FairFace/MiVOLO)
    are stripped.
    """
    import json, random
    from pathlib import Path
    from training.detectors.external_loaders.hagrid import build_hagrid_manifest

    base = Path(VOLUME_PATH)
    ann_root = base / "external/hagrid_v2/annotations"
    img_root = base / "external/hagrid_v2/images"
    hagrid_out = base / out_hagrid_manifest.lstrip("/")
    mixed_out = base / out_mixed_manifest.lstrip("/")
    base_train = base / base_train_manifest.lstrip("/")
    existing_manifest = base / "external/hagrid_v2/hagrid_manifest.json"

    print(f"=== HaGRID normalize ===")
    if existing_manifest.exists():
        n = len(json.loads(existing_manifest.read_text())["items"])
        print(f"  reusing existing manifest at {existing_manifest} ({n} records)")
        hagrid_out = existing_manifest
        summary = {"n_records": n}
    else:
        print(f"  ann_root: {ann_root}  exists={ann_root.exists()}")
        print(f"  img_root: {img_root}  exists={img_root.exists()}")
        summary = build_hagrid_manifest(ann_root, img_root, hagrid_out)
        print(f"  wrote {summary['n_records']} HaGRID records → {hagrid_out}")

    # Subsample HaGRID for Path-B (cache fits in RAM)
    hagrid_records = json.loads(hagrid_out.read_text())["items"]
    rng = random.Random(seed)
    if subsample and len(hagrid_records) > subsample:
        rng.shuffle(hagrid_records)
        hagrid_records = hagrid_records[:subsample]
    print(f"  subsampled HaGRID → {len(hagrid_records)} records")

    base_data = json.loads(base_train.read_text())
    mixed_items = list(base_data["items"]) + hagrid_records
    rng.shuffle(mixed_items)
    mixed_payload = {
        "version": 1,
        "task": "hand_bbox",
        "source": "external_plus_hagrid_train",
        "notes": "CMU + FreiHAND (from external_train) + HaGRID v2 bbox-only subsample.",
        "items": mixed_items,
    }
    mixed_out.parent.mkdir(parents=True, exist_ok=True)
    mixed_out.write_text(json.dumps(mixed_payload))
    print(f"  wrote mixed manifest ({len(mixed_items)} records) → {mixed_out}")
    volume.commit()
    return {
        "hagrid_records": summary["n_records"],
        "hagrid_subsampled": len(hagrid_records),
        "mixed_total": len(mixed_items),
        "mixed_manifest": str(mixed_out),
    }


@app.function(
    gpu="A100",
    volumes={VOLUME_PATH: volume},
    timeout=60 * 60 * 2,
)
def generate_pseudo_labels_asl(
    run_id: str = "pseudo_v1",
    checkpoint: str = "/runs/hand_det_v0_a100_resume_20260521_145244Z/best.pt",
    clip_manifest: str = "/datasets/asl_clips/v3/dataset_v3_manifest.json",
    out_frames_dir: str = "/labeled_frames/hand_bbox_pseudo_asl/v1",
    out_manifest: str = "/labeled_frames/hand_bbox/pseudo_asl_v1.json",
    fps: float = 15.0,
    conf_threshold: float = 0.5,
    temporal_tol_px: float = 40.0,
    min_kept_frames: int = 5,
    limit: int = 0,
) -> dict:
    """Job B Phase 1: run hand_det_v0 over all ASL clips, save kept frames
    + emit a manifest in the FrameRecord format for Phase 2 training mix.
    """
    import sys
    from pathlib import Path
    from training.detectors.pseudo_label_asl import main as _main

    ckpt = Path(f"{VOLUME_PATH}{checkpoint}") if checkpoint.startswith("/") else Path(checkpoint)
    cm = Path(f"{VOLUME_PATH}{clip_manifest}") if clip_manifest.startswith("/") else Path(clip_manifest)
    ofd = Path(f"{VOLUME_PATH}{out_frames_dir}") if out_frames_dir.startswith("/") else Path(out_frames_dir)
    om = Path(f"{VOLUME_PATH}{out_manifest}") if out_manifest.startswith("/") else Path(out_manifest)

    argv_backup = sys.argv[:]
    sys.argv = [
        "pseudo_label_asl",
        "--checkpoint", str(ckpt),
        "--clip-manifest", str(cm),
        "--repo-root", VOLUME_PATH,
        "--out-frames-dir", str(ofd),
        "--out-manifest", str(om),
        "--fps", str(fps),
        "--conf-threshold", str(conf_threshold),
        "--temporal-tol-px", str(temporal_tol_px),
        "--min-kept-frames", str(min_kept_frames),
    ]
    if limit > 0:
        sys.argv += ["--limit", str(limit)]
    try:
        rc = _main()
    finally:
        sys.argv = argv_backup
    volume.commit()
    return {"rc": rc, "out_manifest": str(om), "out_frames_dir": str(ofd)}


@app.function(
    gpu="L4",
    volumes={VOLUME_PATH: volume},
    timeout=60 * 60,
    memory=8192,
)
def train_sign_classifier(
    trajectories_dir: str = "/trajectories_v6",
    trajectories_tar: str = "",
    vocab_json: str = "/vocabulary/slice1b_vocabulary.json",
    run_id: str = "sign_classifier_v0",
    epochs: int = 80,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    val_frac: float = 0.20,
    seed: int = 42,
    early_stop_patience: int = 10,
    augment: bool = True,
    balanced_sampling: str = "none",
    max_per_sign: int = 0,
    zero_embeddings: bool = False,
    arch: str = "tcn",
    transformer_d_model: int = 192,
    transformer_layers: int = 3,
    transformer_heads: int = 4,
    transformer_ff: int = 512,
) -> dict:
    """Train the SignClassifier learned head on the volume's trajectories.

    Deterministic split with seed=42 mirrors the local
    `scripts/split_trajectories.py` outcome (5468/1332 on
    /trajectories_v6) — so this is apples-to-apples vs templates_v7.
    """
    import json, random, shutil, subprocess, time
    from pathlib import Path
    from training.detectors.train_classifier import train as _train

    base = Path(VOLUME_PATH)
    vocab_path = base / vocab_json.lstrip("/")
    run_dir = base / f"runs/{run_id}"

    # Tarball mode: untar to /tmp (container-local) to avoid Modal volume
    # per-file RTT during the 10K+ trajectory scan that was timing out the
    # client heartbeat. Falls back to direct volume path when no tar given.
    if trajectories_tar:
        tar_src = base / trajectories_tar.lstrip("/")
        if not tar_src.exists():
            raise FileNotFoundError(f"trajectories tar not found: {tar_src}")
        extract_root = Path("/tmp/trajectories_workspace")
        if extract_root.exists():
            shutil.rmtree(extract_root)
        extract_root.mkdir(parents=True)
        t0 = time.time()
        print(f"untarring {tar_src} ({tar_src.stat().st_size / 1e9:.2f} GB) → {extract_root}")
        subprocess.run(["tar", "-xf", str(tar_src), "-C", str(extract_root)], check=True)
        # Tar may include the wrapping dirname OR start at the contents. Auto-detect.
        children = [p for p in extract_root.iterdir() if p.is_dir()]
        if len(children) == 1 and any(c.is_dir() for c in children[0].iterdir()):
            traj_root = children[0]
        else:
            traj_root = extract_root
        print(f"untar done in {time.time() - t0:.0f}s; trajectories at {traj_root}")
    else:
        traj_root = base / trajectories_dir.lstrip("/")

    # Build train/val split via symlinks in /tmp (matches local split logic)
    tmp = Path("/tmp/sign_classifier_split")
    if tmp.exists():
        shutil.rmtree(tmp)
    train_dir = tmp / "train"
    val_dir = tmp / "val"
    rnd = random.Random(seed)
    n_train = n_val = 0
    for sign_dir in sorted(traj_root.iterdir()):
        if not sign_dir.is_dir():
            continue
        sign = sign_dir.name
        jsons = sorted(sign_dir.glob("*.json"))
        if not jsons:
            continue
        rnd.shuffle(jsons)
        nv = max(1, int(len(jsons) * val_frac)) if len(jsons) >= 2 else 0
        val_jsons = jsons[:nv]
        train_jsons = jsons[nv:]
        (train_dir / sign).mkdir(parents=True, exist_ok=True)
        (val_dir / sign).mkdir(parents=True, exist_ok=True)
        for jp in train_jsons:
            (train_dir / sign / jp.name).symlink_to(jp)
            n_train += 1
        for jp in val_jsons:
            (val_dir / sign / jp.name).symlink_to(jp)
            n_val += 1
    print(f"split: train={n_train} val={n_val}")

    result = _train(
        train_dir=train_dir,
        val_dir=val_dir,
        vocab_json=vocab_path,
        run_dir=run_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        num_workers=0,  # tiny in-RAM dataset
        use_bf16=True,
        augment=augment,
        early_stop_patience=early_stop_patience,
        balanced_sampling=balanced_sampling,
        max_per_sign=max_per_sign,
        zero_embeddings=zero_embeddings,
        arch=arch,
        transformer_d_model=transformer_d_model,
        transformer_layers=transformer_layers,
        transformer_heads=transformer_heads,
        transformer_ff=transformer_ff,
    )
    volume.commit()
    return result


@app.function(
    gpu="H100",
    volumes={VOLUME_PATH: volume},
    timeout=60 * 60 * 2,
    memory=64 * 1024,  # 64 GB host RAM for 12-worker DataLoader prefetch
    cpu=16.0,
)
def train_handshape_encoder(
    crops_tarball: str = "/datasets/hand_crops.tar",
    run_id: str = "handshape_v0",
    epochs: int = 60,
    batch_size: int = 1024,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    temperature: float = 0.1,
    num_workers: int = 12,
    val_frac: float = 0.05,
    seed: int = 42,
    early_stop_patience: int = 8,
    use_compile: bool = True,
) -> dict:
    """Phase 4.6 — contrastive train HandshapeEncoder on the hand-crops tarball.

    Volumes-v1 inode strategy: tarball lives on volume as 1 inode; container
    untars to /tmp (local ephemeral disk, ~50 GB) at start of run. No volume
    inode cost added by 182K crops. Single volume.commit() at end.
    """
    import os, shutil, subprocess, tarfile, time
    from pathlib import Path
    from training.detectors.train_handshape import train as _train

    t0 = time.time()
    tar_path = Path(f"{VOLUME_PATH}{crops_tarball}")
    if not tar_path.exists():
        raise FileNotFoundError(f"crops tarball not found: {tar_path}")

    extract_root = Path("/tmp/crops_workspace")
    extract_root.mkdir(parents=True, exist_ok=True)
    print(f"untarring {tar_path} ({tar_path.stat().st_size / 1e9:.2f} GB) → {extract_root}")
    # Use system tar — faster than tarfile module on large archives
    subprocess.run(["tar", "-xf", str(tar_path), "-C", str(extract_root)], check=True)
    crops_root = extract_root / "hand_crops"
    n_jpgs = sum(1 for _ in crops_root.rglob("*.jpg"))
    print(f"untar done in {time.time() - t0:.0f}s — {n_jpgs} jpgs in {crops_root}")

    # Rewrite index.json's crop_path entries to point at the /tmp extraction
    # (the dumper saved repo-relative paths like data/hand_crops/<sign>/<clip>/<frame>.jpg)
    import json as _json
    idx_path = crops_root / "index.json"
    idx = _json.loads(idx_path.read_text())
    for r in idx["records"]:
        # Convert "data/hand_crops/<sign>/<clip>/<frame>.jpg" → /tmp absolute
        rel = r["crop_path"].split("data/hand_crops/", 1)[-1]
        r["crop_path"] = str(crops_root / rel)
    idx_path.write_text(_json.dumps(idx))
    print(f"rewrote {len(idx['records'])} crop paths to /tmp roots")

    run_dir = Path(f"{VOLUME_PATH}/runs/{run_id}")
    result = _train(
        crops_root=crops_root,
        run_dir=run_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        temperature=temperature,
        num_workers=num_workers,
        val_frac=val_frac,
        seed=seed,
        use_bf16=True,
        use_compile=use_compile,
        use_channels_last=True,
        early_stop_patience=early_stop_patience,
    )
    volume.commit()
    print(f"total wall-clock: {time.time() - t0:.0f}s")
    return result


@app.function(
    gpu="A100",
    volumes={VOLUME_PATH: volume},
    timeout=60 * 60,
    memory=8192,
)
def train_sign_classifier_a100(
    trajectories_dir: str = "/trajectories_v7",
    trajectories_tar: str = "",
    vocab_json: str = "/vocabulary/slice1b_vocabulary.json",
    run_id: str = "sign_classifier_xformer",
    epochs: int = 80,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    val_frac: float = 0.20,
    seed: int = 42,
    early_stop_patience: int = 10,
    augment: bool = True,
    balanced_sampling: str = "none",
    max_per_sign: int = 0,
    zero_embeddings: bool = False,
    arch: str = "transformer",
    transformer_d_model: int = 192,
    transformer_layers: int = 3,
    transformer_heads: int = 4,
    transformer_ff: int = 512,
) -> dict:
    """A100 variant of train_sign_classifier — Phase 4.7 transformer A/B.

    Same split logic as train_sign_classifier but on A100 (~5-7 min wall-
    clock vs L4's ~15) so the parallel `.spawn()` A/B finishes fast.
    """
    return train_sign_classifier.local(
        trajectories_dir=trajectories_dir,
        trajectories_tar=trajectories_tar,
        vocab_json=vocab_json,
        run_id=run_id,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        val_frac=val_frac,
        seed=seed,
        early_stop_patience=early_stop_patience,
        augment=augment,
        balanced_sampling=balanced_sampling,
        max_per_sign=max_per_sign,
        zero_embeddings=zero_embeddings,
        arch=arch,
        transformer_d_model=transformer_d_model,
        transformer_layers=transformer_layers,
        transformer_heads=transformer_heads,
        transformer_ff=transformer_ff,
    )


@app.function(
    cpu=16.0,
    memory=8192,
    volumes={VOLUME_PATH: volume},
    timeout=60 * 60,
)
def download_msasl_clips(targets: list[dict], out_subdir: str = "datasets/msasl") -> dict:
    """Parallel yt-dlp download of MSAsl clips for our thin-sign vocab.

    `targets` is a pre-filtered list of records produced locally from the
    MSASL_train/val/test.json files (license-gated, so Hiromu pre-filters
    and passes only the slice we need — no licensed data crosses the wire
    in bulk). Each record has at minimum:
        {url, start_time, end_time, sign_id, msasl_label, signer_id,
         msasl_split, box, fps, width, height}

    Downloads each clip with --download-sections so we get only the
    sign-window, writes to /datasets/msasl/clips/, builds a manifest.

    16 CPUs × parallel yt-dlp on Modal CDN-grade egress → expect 5-15 min
    wall-clock for ~600 candidate clips. Cost: ~$0.10-0.20 (CPU only).
    """
    import json as _json
    import logging
    import subprocess
    import sys
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from pathlib import Path as _Path
    from collections import Counter as _Counter

    logging.basicConfig(level=logging.INFO, format="%(levelname)s msasl: %(message)s")
    log = logging.getLogger("msasl")

    base = _Path(VOLUME_PATH) / out_subdir
    clips_dir = base / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = base / "msasl_manifest.json"

    log.info("targets received: %d", len(targets))

    def _download(rec: dict) -> dict:
        url = rec["url"]
        # Stable basename: hash of url + start/end so retries hit cache
        video_id = url.rsplit("v=", 1)[-1].split("&")[0]
        start = rec.get("start_time", 0.0)
        end = rec.get("end_time")
        # Slug includes window so we can have many clips per source vid
        slug = f"{rec['sign_id']}__msasl__{video_id}_{int(start * 1000)}_{int((end or 0) * 1000)}"
        out_path = clips_dir / f"{slug}.mp4"
        result = dict(rec)
        result["local_path"] = str(out_path)
        if out_path.exists() and out_path.stat().st_size > 1024:
            result["download_status"] = "ok"
            return result
        # Use --download-sections to slice the sign window
        section = f"*{start:.3f}-{end:.3f}" if end is not None else f"*{start:.3f}-inf"
        cmd = [
            sys.executable, "-m", "yt_dlp",
            "--quiet", "--no-warnings", "--no-playlist",
            "--format", "bestvideo[ext=mp4][height<=720]+bestaudio/best[ext=mp4]/best",
            "--merge-output-format", "mp4",
            "--download-sections", section,
            "--force-keyframes-at-cuts",
            "-o", str(out_path),
            url,
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        except subprocess.TimeoutExpired:
            result["download_status"] = "timeout"
            return result
        if proc.returncode == 0 and out_path.exists():
            result["download_status"] = "ok"
            return result
        stderr = (proc.stderr or "").lower()
        if "video unavailable" in stderr or "removed by the user" in stderr or "404" in stderr:
            result["download_status"] = "404"
        elif "private" in stderr or "blocked" in stderr or "age" in stderr:
            result["download_status"] = "blocked"
        else:
            result["download_status"] = "other-error"
            result["stderr_tail"] = stderr[-400:] if stderr else ""
        return result

    t0 = time.time()
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=16) as ex:
        futures = [ex.submit(_download, r) for r in targets]
        for i, fut in enumerate(as_completed(futures), 1):
            results.append(fut.result())
            if i % 50 == 0:
                ok = sum(1 for r in results if r.get("download_status") == "ok")
                log.info("progress: %d/%d (ok=%d, %.1fs)", i, len(targets), ok, time.time() - t0)

    by_status = _Counter(r.get("download_status") for r in results)
    ok = [r for r in results if r.get("download_status") == "ok"]
    by_sign = _Counter(r["sign_id"] for r in ok)
    log.info("done: %s (%.1fs)", dict(by_status), time.time() - t0)
    log.info("ok-per-sign: %s", dict(by_sign))

    manifest = {
        "source": "msasl",
        "license": "C-UDA (Computational Use of Data Agreement); see C-UDA-0.1_annotated_discussion.pdf",
        "citation": "Vaezi Joze, Koller. MS-ASL. BMVC 2019",
        "n_targets": len(targets),
        "n_ok": len(ok),
        "by_status": dict(by_status),
        "by_sign": dict(by_sign),
        "records": results,
    }
    manifest_path.write_text(_json.dumps(manifest, indent=2))
    volume.commit()
    return {"manifest_path": str(manifest_path), "n_ok": len(ok), "by_sign": dict(by_sign)}


@app.local_entrypoint()
def fetch_msasl_for_thin_signs(
    msasl_dir: str = "/Users/hirom/Downloads/MS-ASL",
    v3_manifest: str = "data/labeled_frames/unified_clip_manifest_modal_v3.json",
    v2_manifest: str = "data/labeled_frames/unified_clip_manifest_modal_v2.json",
    target_per_sign: int = 100,
):
    """Local pre-filter + Modal parallel fetch for MSAsl thin-sign coverage.

    Reads licensed MSAsl JSONs locally (don't ship 25K records over the
    wire), filters down to the records we actually need (matched to v3
    still-thin signs via direct + synonym gloss matching), then hands the
    pre-filtered list to download_msasl_clips() which runs 16-way parallel
    yt-dlp on Modal.
    """
    import json as _json
    from collections import Counter as _Counter
    from pathlib import Path as _Path

    md = _Path(msasl_dir)
    v3c = _Counter(c["sign_id"] for c in _json.loads(_Path(v3_manifest).read_text())["clips"])
    v2c = _Counter(c["sign_id"] for c in _json.loads(_Path(v2_manifest).read_text())["clips"])
    thin = {s for s, n in v2c.items() if n < target_per_sign}
    still_thin = {s: v3c.get(s, 0) for s in thin if v3c.get(s, 0) < target_per_sign}

    # Build gloss → sign_id lookup with synonym expansion
    synonyms = _json.loads((md / "MSASL_synonym.json").read_text())
    aliases = {"mom": ["mother"], "dad": ["father"], "grandma": ["grandmother"]}
    sign_for_gloss: dict[str, str] = {}
    for sid in still_thin:
        sign_for_gloss[sid] = sid
        for a in aliases.get(sid, []):
            sign_for_gloss[a] = sid
    # Expand via synonym groups
    for group in synonyms:
        lowered = [g.lower() for g in group]
        # Does any member of this group already map to one of our thin signs?
        owner = next((sign_for_gloss[g] for g in lowered if g in sign_for_gloss), None)
        if owner:
            for g in lowered:
                sign_for_gloss.setdefault(g, owner)

    # Filter all MSAsl records
    needed_per_sign: dict[str, int] = {s: target_per_sign - n for s, n in still_thin.items()}
    targets: list[dict] = []
    per_sign_collected: dict[str, int] = {s: 0 for s in still_thin}
    for fname in ("MSASL_train.json", "MSASL_val.json", "MSASL_test.json"):
        split = fname.split("_")[1].split(".")[0]
        for r in _json.loads((md / fname).read_text()):
            gloss = (r.get("clean_text") or "").lower()
            sid = sign_for_gloss.get(gloss)
            if not sid:
                continue
            if per_sign_collected[sid] >= needed_per_sign[sid]:
                continue
            targets.append({
                "sign_id": sid,
                "msasl_gloss": gloss,
                "msasl_label": r.get("label"),
                "msasl_split": split,
                "url": r["url"],
                "start_time": float(r.get("start_time", 0.0)),
                "end_time": float(r["end_time"]) if r.get("end_time") is not None else None,
                "signer_id": r.get("signer_id"),
                "box": r.get("box"),
                "fps": r.get("fps"),
                "width": r.get("width"),
                "height": r.get("height"),
            })
            per_sign_collected[sid] += 1

    print(f"▶ filtered MSAsl targets: {len(targets)} clips across {len(set(t['sign_id'] for t in targets))} signs")
    print(f"▶ per-sign collected: {per_sign_collected}")
    result = download_msasl_clips.remote(targets)
    print(f"✓ Modal returned: ok={result['n_ok']}, by_sign={result['by_sign']}")
    print(f"  manifest at {result['manifest_path']}")
    return result


@app.local_entrypoint()
def phase47_transformer_ab(
    trajectories_dir: str = "/trajectories_v7",
    vocab_json: str = "/vocabulary/slice1b_vocabulary.json",
    run_id_small: str = "sign_classifier_xformer_small",
    run_id_large: str = "sign_classifier_xformer_large",
    epochs: int = 80,
    early_stop_patience: int = 10,
):
    """Phase 4.7 — fire fair-comp (192/3/4, ~1M params) and upsized
    (256/4/8, ~2.2M params) transformers in parallel on two A100s.

    Wall-clock ≈ slowest of the two (~5-7 min). Cost ≈ ~$0.95 total
    (2 × ~7/60 × $4.10/hr). Compare top-1 vs Session 18 TCN baseline
    (10.1%) to settle whether attention alone moves the architectural
    ceiling, before investing in handshape (Phase 4.6) or face (4.5).
    """
    small = train_sign_classifier_a100.spawn(
        trajectories_dir=trajectories_dir,
        vocab_json=vocab_json,
        run_id=run_id_small,
        arch="transformer",
        transformer_d_model=192,
        transformer_layers=3,
        transformer_heads=4,
        transformer_ff=384,
        epochs=epochs,
        early_stop_patience=early_stop_patience,
    )
    large = train_sign_classifier_a100.spawn(
        trajectories_dir=trajectories_dir,
        vocab_json=vocab_json,
        run_id=run_id_large,
        arch="transformer",
        transformer_d_model=256,
        transformer_layers=4,
        transformer_heads=8,
        transformer_ff=512,
        epochs=epochs,
        early_stop_patience=early_stop_patience,
    )
    print(f"▶ spawned: small={small.object_id}  large={large.object_id}")
    r_small = small.get()
    print(f"✓ small finished: best_top1={r_small.get('best_top1')}  best_epoch={r_small.get('best_epoch')}")
    r_large = large.get()
    print(f"✓ large finished: best_top1={r_large.get('best_top1')}  best_epoch={r_large.get('best_epoch')}")
    return {"small": r_small, "large": r_large}


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
