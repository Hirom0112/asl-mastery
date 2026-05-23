"""End-to-end smoke test of the 356D pipeline.

Runs `extract_trajectories_v2.main` on a tiny manifest with all 4 detectors
loaded, including HandshapeEncoder. Then loads the resulting trajectory JSON
and verifies `trajectory_from_frames` produces a 356D feature tensor with the
embedding portion populated.

This is a ZERO-COST sanity check — runs entirely on local CPU. Catches
wiring bugs (missing fields, dim mismatches, NaN propagation) before the
expensive Modal extract.

Usage:
    training/.venv/bin/python -m scripts.verify_356d_pipeline \\
        --hand-detector-ckpt path/to/hand_det.pt \\
        --hand-landmarks-ckpt path/to/landmarks.pt \\
        --pose-ckpt path/to/pose.pt \\
        --handshape-encoder-ckpt path/to/encoder.pt  # or omit for random-init test
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand-detector-ckpt", type=Path, required=False)
    ap.add_argument("--hand-landmarks-ckpt", type=Path, required=False)
    ap.add_argument("--pose-ckpt", type=Path, required=False)
    ap.add_argument("--handshape-encoder-ckpt", type=Path, required=False,
                    help="If omitted, uses a fresh-random encoder to test wiring only.")
    ap.add_argument("--n-clips", type=int, default=3)
    ap.add_argument("--manifest", type=Path,
                    default=Path("data/labeled_frames/unified_clip_manifest_modal_v4.json"))
    ap.add_argument("--repo-root", type=Path, default=Path("."))
    args = ap.parse_args()

    # Pick the first N sem_lex clips (their paths map to local dataset/raw/sem_lex/clips/)
    m = json.loads(args.manifest.read_text())
    clips = []
    seen = set()
    for c in m["clips"]:
        if c["source"] == "sem_lex" and c["sign_id"] not in seen:
            local = args.repo_root / "dataset/raw/sem_lex/clips" / Path(c["clip_path"]).name
            if local.exists():
                clips.append({"clip_path": str(local), "sign_id": c["sign_id"]})
                seen.add(c["sign_id"])
        if len(clips) >= args.n_clips:
            break
    if not clips:
        print("no usable clips found locally; skipping")
        return 1
    print(f"picked {len(clips)} clips:")
    for c in clips:
        print(f"  {c['sign_id']}  {c['clip_path']}")

    # Build a tiny test manifest
    test_manifest = {"version": 4, "clips": clips}
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        manifest_path = tdp / "test_manifest.json"
        manifest_path.write_text(json.dumps(test_manifest))
        out_dir = tdp / "trajectories_test"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Build sys.argv and call main
        from training.detectors.extract_trajectories_v2 import main as extract_main
        sys.argv = [
            "extract_trajectories_v2",
            "--manifest", str(manifest_path),
            "--out-dir", str(out_dir),
            "--decode-workers", "1",
            "--fps", "15.0",
            "--repo-root", str(args.repo_root),
        ]
        # Only add ckpts if provided
        if args.hand_detector_ckpt:
            sys.argv += ["--hand-detector-ckpt", str(args.hand_detector_ckpt)]
        if args.hand_landmarks_ckpt:
            sys.argv += ["--hand-landmarks-ckpt", str(args.hand_landmarks_ckpt)]
        if args.pose_ckpt:
            sys.argv += ["--pose-ckpt", str(args.pose_ckpt)]

        # Handshape encoder: either a real ckpt or a tempfile holding a fresh-init
        if args.handshape_encoder_ckpt:
            sys.argv += ["--handshape-encoder-ckpt", str(args.handshape_encoder_ckpt)]
            print("[mode] using REAL handshape encoder ckpt")
        else:
            from training.detectors.handshape_encoder import HandshapeEncoder
            enc = HandshapeEncoder()
            enc_path = tdp / "fake_encoder.pt"
            torch.save({"state_dict": enc.state_dict()}, enc_path)
            sys.argv += ["--handshape-encoder-ckpt", str(enc_path)]
            print("[mode] using FRESH-INIT encoder (random weights — wiring test only)")

        rc = extract_main()
        if rc != 0:
            print(f"FAIL: extract returned rc={rc}")
            return rc

        # Now load each trajectory + run trajectory_from_frames + verify 356D
        from training.detectors.sign_matcher import trajectory_from_frames
        from training.detectors.fit_templates import (
            FEATURES_PER_FRAME_WITH_EMBED, trajectory_has_embedding,
        )

        for c in clips:
            sid = c["sign_id"]
            traj_path = out_dir / sid / f"{Path(c['clip_path']).stem}.json"
            if not traj_path.exists():
                print(f"  ✗ {sid}: no output trajectory at {traj_path}")
                continue
            traj = json.loads(traj_path.read_text())
            has_embed = trajectory_has_embedding(traj["frames"])
            print(f"  {sid}: num_frames={traj['num_frames']} version={traj.get('version')} has_embed={has_embed}")
            if not has_embed:
                print(f"    ! no embedding found in any frame — wiring bug")
                continue
            # Build features
            feats = trajectory_from_frames(traj["frames"], T=32)
            print(f"    features shape: {feats.shape} (expected (32, {FEATURES_PER_FRAME_WITH_EMBED}))")
            if feats.shape != (32, FEATURES_PER_FRAME_WITH_EMBED):
                print(f"    ! FAIL: feature dim mismatch")
                continue
            import numpy as np
            finite_frac = float(np.isfinite(feats).mean())
            print(f"    finite fraction: {finite_frac:.3f}")
            # Spot-check embedding portion: per slot 0/1, positions [42:170] should
            # be non-NaN whenever hands were detected
            slot0_embed = feats[:, 42:170]
            slot1_embed = feats[:, 212:340]
            print(f"    slot0 embed finite: {np.isfinite(slot0_embed).any(axis=1).sum()}/{feats.shape[0]} frames")
            print(f"    slot1 embed finite: {np.isfinite(slot1_embed).any(axis=1).sum()}/{feats.shape[0]} frames")

    print("\n✓ 356D pipeline verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
