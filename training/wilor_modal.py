"""WiLoR crisp-finger extraction on Modal — avatar handshape authoring.

SMPLest-X gives good body/arms but SOFT fingers (its whole-body hand head is weak).
WiLoR (CVPR 2025, SOTA dedicated hand-mesh recovery) gives crisp per-frame MANO
hand pose. We run WiLoR on the SAME clean clip SMPLest-X used (identical `ffmpeg
-vsync 0` frame extraction → frame-aligned), keep SMPLest-X's body+wrist, and graft
WiLoR's finger articulation into `smplx_{l,r}hand_pose` (45-dim axis-angle) locally.

Output per sign → /wilor_out/<sign>.npz with per-frame:
  lhand_pose (F,45)  rhand_pose (F,45)  in MANO/SMPL-X axis-angle order
  lglobal (F,3) rglobal (F,3)          WiLoR global hand orient (kept for reference; we DISCARD it)
  lvalid (F,) rvalid (F,)              hand detected this frame

This is AVATAR CONTENT authoring only; it does not touch the recognition path.

Workflow:
  # 1) clips already on the shared volume under /clips (same as SMPLest-X). If not:
  modal volume put asl-smplestx-assets /tmp/wilor_stage /clips   # 25 <sign>.webm
  # 2) one-time: cache WiLoR + MANO weights on the volume (~1-2 min, CPU)
  modal run training/wilor_modal.py::setup
  # 3) run on a word list (concurrent A10G containers)
  modal run training/wilor_modal.py::run_words --words "animal,eat,baby,..."
  # 4) pull
  modal volume get asl-smplestx-assets /wilor_out ./data/wilor_motion
"""

import modal

GPU = "A10G"  # WiLoR is light; A10G (24GB, cheap) is plenty

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0", "build-essential")
    # torch 2.0.0 default wheel ships cu117 — matches WiLoR's tested config
    .pip_install("torch==2.0.0", "torchvision==0.15.1")
    .pip_install(
        "numpy<2",
        "opencv-python-headless==4.11.0.86",
        "scipy",
        "huggingface_hub",
        "git+https://github.com/warmshao/WiLoR-mini",
    )
)

app = modal.App("asl-wilor")
vol = modal.Volume.from_name("asl-smplestx-assets", create_if_missing=True)
ASSETS = "/assets"
CLIPS = f"{ASSETS}/clips"
WOUT = f"{ASSETS}/wilor_out"
# Persist WiLoR+MANO downloads on the volume so containers don't re-fetch.
WILOR_CACHE = f"{ASSETS}/wilor_pretrained"


def _make_pipe():
    import torch
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
        WiLorHandPose3dEstimationPipeline,
    )

    device = torch.device("cuda")
    dtype = torch.float16
    # WiLoR-mini auto-downloads its checkpoint + MANO_RIGHT.pkl from HF
    # (warmshao/WiLoR-mini) into WILOR_MINI_PATH/pretrained_models on first use.
    pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype)
    return pipe


@app.function(image=image, volumes={ASSETS: vol}, gpu=GPU, timeout=60 * 30)
def setup():
    """Warm the WiLoR + MANO weights onto the volume (downloads on first init)."""
    import os
    os.environ["WILOR_MINI_PATH"] = WILOR_CACHE
    os.makedirs(WILOR_CACHE, exist_ok=True)
    pipe = _make_pipe()
    import numpy as np
    # smoke a blank frame so the full graph (detector + recon + MANO) initializes
    out = pipe.predict(np.zeros((480, 640, 3), dtype=np.uint8))
    print("setup ok; detections on blank:", len(out))
    vol.commit()


@app.cls(image=image, volumes={ASSETS: vol}, gpu=GPU, timeout=60 * 30,
         max_containers=8, retries=1)
class HandExtractor:
    @modal.enter()
    def load(self):
        import os
        os.environ["WILOR_MINI_PATH"] = WILOR_CACHE
        self.pipe = _make_pipe()

    @modal.method()
    def process(self, sign: str) -> dict:
        import os, subprocess, glob
        import numpy as np, cv2

        clip = f"{CLIPS}/{sign}.webm"
        if not os.path.exists(clip):
            return {"sign": sign, "status": "no_clip"}
        fdir = f"/tmp/frames/{sign}"
        os.makedirs(fdir, exist_ok=True)
        # IDENTICAL extraction to smplestx_modal.py so frame indices align 1:1.
        subprocess.run(
            ["ffmpeg", "-y", "-i", clip, "-vsync", "0", "-qscale", "0",
             f"{fdir}/%06d.jpg"],
            check=True, capture_output=True,
        )
        frames = sorted(glob.glob(f"{fdir}/*.jpg"))
        n = len(frames)

        lpose = np.full((n, 45), np.nan, np.float32)
        rpose = np.full((n, 45), np.nan, np.float32)
        lglob = np.full((n, 3), np.nan, np.float32)
        rglob = np.full((n, 3), np.nan, np.float32)
        lvalid = np.zeros(n, bool)
        rvalid = np.zeros(n, bool)

        def as_axis_angle_45(hp):
            a = np.asarray(hp, np.float32).reshape(-1)
            assert a.size == 45, f"hand_pose size {a.size} != 45"
            return a

        for i, fp in enumerate(frames):
            bgr = cv2.imread(fp)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            try:
                outs = self.pipe.predict(rgb)
            except Exception as e:  # robustness: a bad frame shouldn't kill the clip
                outs = []
            # pick the largest-bbox detection per handedness
            best = {0: None, 1: None}  # is_right -> (area, det)
            for det in outs:
                ir = int(det.get("is_right", 1))
                bb = det.get("hand_bbox")
                area = 0.0
                if bb is not None:
                    bb = np.asarray(bb).reshape(-1)
                    if bb.size >= 4:
                        area = abs((bb[2] - bb[0]) * (bb[3] - bb[1]))
                if best[ir] is None or area > best[ir][0]:
                    best[ir] = (area, det)
            for ir, slot in best.items():
                if slot is None:
                    continue
                wp = slot[1]["wilor_preds"]
                hp = as_axis_angle_45(wp["hand_pose"])
                go = np.asarray(wp["global_orient"], np.float32).reshape(-1)[:3]
                if ir == 1:  # right hand
                    rpose[i] = hp; rglob[i] = go; rvalid[i] = True
                else:        # left hand
                    lpose[i] = hp; lglob[i] = go; lvalid[i] = True

        os.makedirs(WOUT, exist_ok=True)
        np.savez_compressed(
            f"{WOUT}/{sign}.npz",
            lhand_pose=lpose, rhand_pose=rpose,
            lglobal=lglob, rglobal=rglob,
            lvalid=lvalid, rvalid=rvalid,
        )
        vol.commit()
        return {"sign": sign, "status": "ok", "frames": n,
                "ldet": int(lvalid.sum()), "rdet": int(rvalid.sum())}


@app.local_entrypoint()
def smoke():
    for r in HandExtractor().process.map(["animal", "eat"]):
        print(r)


@app.local_entrypoint()
def run_words(words: str = ""):
    signs = [w.strip() for w in words.split(",") if w.strip()]
    ok = 0
    for r in HandExtractor().process.map(signs):
        print(r)
        ok += r.get("status") == "ok"
    print(f"\n{ok}/{len(signs)} signs: WiLoR hand pose extracted")
