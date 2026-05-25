"""SMPLest-X whole-body SMPL-X extraction on Modal — avatar motion authoring.

Runs SMPLest-X-Huge (TPAMI 2025, github.com/SMPLCap/SMPLest-X) on each sem_lex
reference clip and dumps **per-frame SMPL-X params** (root/body/left-hand/
right-hand/jaw axis-angle pose + shape/expr/cam) as one .npz per sign. These
feed the X Bot retargeting (avatar arms + fingers) — the same param family
SignAvatars ships, but produced on OUR clips so it covers all 80 signs.

This is AVATAR CONTENT authoring only (reference motion); it does not touch the
strict from-scratch recognition path.

Optimized + parallel: 8.2 GB weights cached on a Modal Volume; the model is
loaded once per container (@enter); the 80 clips are fanned across up to 8
warm A10G containers via .map.

Workflow:
  # 1) one-time, free/cheap (CPU + storage): cache weights + YOLO on the volume
  modal run training/smplestx_modal.py::setup
  # 2) upload SMPL-X models + input clips to the volume (from repo root)
  modal volume put asl-smplestx-assets data/signavatars/human_model_files /human_models/human_model_files
  modal volume put asl-smplestx-assets data/avatar_src_clips /clips
  # 3) GPU smoke on 2 signs (~$0.2), inspect, then the full batch
  modal run training/smplestx_modal.py::smoke
  modal run training/smplestx_modal.py::run_all
  # 4) pull results
  modal volume get asl-smplestx-assets /out ./data/smplx_motion
"""

import modal

REPO = "https://github.com/SMPLCap/SMPLest-X"
HF_REPO = "waanqii/SMPLest-X"          # SMPLest-X-Huge weights (8.2G) + config_base.py
CKPT = "smplest_x_h"
GPU = "A10G"                           # sm_86, 24 GB — matches torch 1.12/cu113, cheap

image = (
    # Slim base: the torch cu113 wheel bundles its own CUDA runtime, and
    # SMPLest-X inference compiles no CUDA extensions, so the heavy
    # nvidia/cuda devel image is unnecessary (it made builds crawl).
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==1.12.0+cu113",
        "torchvision==0.13.0+cu113",
        extra_index_url="https://download.pytorch.org/whl/cu113",
    )
    .pip_install(
        "numpy==1.23.1",        # keeps chumpy 0.70 importable (numpy.bool etc.)
        "smplx==0.1.28",
        "tqdm==4.67.1",
        "opencv-python-headless==4.11.0.86",
        "chumpy==0.70",
        "trimesh==4.6.2",
        "matplotlib==3.7.5",
        "json_tricks==3.17.3",
        "einops==0.8.1",
        "timm==1.0.14",
        "ultralytics==8.3.75",
        "huggingface_hub",
    )
    .run_commands(f"git clone --depth 1 {REPO} /SMPLest-X")
)

app = modal.App("asl-smplestx")
vol = modal.Volume.from_name("asl-smplestx-assets", create_if_missing=True)
ASSETS = "/assets"
PRETRAINED = f"{ASSETS}/pretrained_models"
HUMAN_MODELS = f"{ASSETS}/human_models/human_model_files"
CLIPS = f"{ASSETS}/clips"
OUT = f"{ASSETS}/out"


@app.function(image=image, volumes={ASSETS: vol}, timeout=60 * 60)
def setup():
    """Cache the SMPLest-X-Huge weights + config + YOLO detector on the volume."""
    import os
    from huggingface_hub import snapshot_download

    os.makedirs(f"{PRETRAINED}/{CKPT}", exist_ok=True)
    print(f"downloading {HF_REPO} → {PRETRAINED}/{CKPT}")
    snapshot_download(repo_id=HF_REPO, local_dir=f"{PRETRAINED}/{CKPT}")
    # YOLO person detector (so inference doesn't fetch at runtime)
    import urllib.request
    yolo = f"{PRETRAINED}/yolov8x.pt"
    if not os.path.exists(yolo):
        urllib.request.urlretrieve(
            "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x.pt",
            yolo,
        )
    print("contents:")
    for root, _, files in os.walk(PRETRAINED):
        for f in files:
            p = os.path.join(root, f)
            print(f"  {p}  ({os.path.getsize(p)/1e6:.1f} MB)")
    vol.commit()


def _build_model():
    """Load SMPLest-X + YOLO once (mirrors main/inference.py, no rendering)."""
    import os, sys, glob
    sys.path.insert(0, "/SMPLest-X")
    os.chdir("/SMPLest-X")
    # No symlinks needed: every path below is overridden to an absolute volume
    # location (config, checkpoint, human models, YOLO), so the repo's own
    # relative ./pretrained_models and ./human_models dirs are irrelevant.

    from main.config import Config
    from main.base import Tester
    from human_models.human_models import SMPLX
    from ultralytics import YOLO

    cfg_candidates = [
        f"{PRETRAINED}/{CKPT}/config_base.py",
        "/SMPLest-X/configs/config_smplest_x_h.py",
    ]
    config_path = next(p for p in cfg_candidates if os.path.exists(p))
    cfg = Config.load_config(config_path)
    cfg.update_config({
        "model": {
            "pretrained_model_path": f"{PRETRAINED}/{CKPT}/{CKPT}.pth.tar",
            "human_model_path": HUMAN_MODELS,
        },
        "log": {"exp_name": "infer", "log_dir": "/tmp/log",
                "output_dir": "/tmp/out", "model_dir": "/tmp/md",
                "result_dir": "/tmp/res"},
    })
    cfg.prepare_log()
    smpl_x = SMPLX(cfg.model.human_model_path)
    tester = Tester(cfg)
    tester._make_model()
    detector = YOLO(f"{PRETRAINED}/yolov8x.pt")
    return cfg, tester, detector, smpl_x


PARAM_KEYS = [
    "smplx_root_pose", "smplx_body_pose", "smplx_lhand_pose",
    "smplx_rhand_pose", "smplx_jaw_pose", "smplx_shape", "smplx_expr", "cam_trans",
]


@app.cls(image=image, volumes={ASSETS: vol}, gpu=GPU, timeout=60 * 60,
         max_containers=8, retries=1)
class Extractor:
    @modal.enter()
    def load(self):
        self.cfg, self.tester, self.detector, self.smpl_x = _build_model()

    @modal.method()
    def process(self, sign: str) -> dict:
        import os, subprocess, glob
        import numpy as np, torch, cv2
        import torchvision.transforms as T
        from utils.data_utils import load_img, process_bbox, generate_patch_image

        clip = f"{CLIPS}/{sign}.webm"
        if not os.path.exists(clip):
            return {"sign": sign, "status": "no_clip"}
        fdir = f"/tmp/frames/{sign}"
        os.makedirs(fdir, exist_ok=True)
        # decode the real frames only. These sem_lex webms carry a bogus
        # r_frame_rate=1000/1 (ms-timebase artifact); without -vsync 0 ffmpeg
        # duplicates each frame ~33x to hit 1000fps. passthrough = real frames.
        subprocess.run(
            ["ffmpeg", "-y", "-i", clip, "-vsync", "0", "-qscale", "0",
             f"{fdir}/%06d.jpg"],
            check=True, capture_output=True,
        )
        frames = sorted(glob.glob(f"{fdir}/*.jpg"))
        tf = T.ToTensor()
        cfg = self.cfg
        seq = {k: [] for k in PARAM_KEYS}
        valid = []
        for fp in frames:
            img = load_img(fp)
            h, w = img.shape[:2]
            det = self.detector.predict(
                img, device="cuda", classes=0,
                conf=cfg.inference.detection.conf, save=False, verbose=False,
            )[0].boxes.xyxy.detach().cpu().numpy()
            if len(det) < 1:
                valid.append(False)
                for k in PARAM_KEYS:
                    seq[k].append(None)
                continue
            # largest person box
            areas = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            b = det[int(np.argmax(areas))]
            xywh = np.array([b[0], b[1], abs(b[2] - b[0]), abs(b[3] - b[1])])
            bbox = process_bbox(bbox=xywh, img_width=w, img_height=h,
                                input_img_shape=cfg.model.input_img_shape,
                                ratio=getattr(cfg.data, "bbox_ratio", 1.25))
            patch, _, _ = generate_patch_image(
                cvimg=img, bbox=bbox, scale=1.0, rot=0.0, do_flip=False,
                out_shape=cfg.model.input_img_shape)
            inp = {"img": tf(patch.astype(np.float32) / 255).cuda()[None]}
            with torch.no_grad():
                out = self.tester.model(inp, {}, {}, "test")
            valid.append(True)
            for k in PARAM_KEYS:
                seq[k].append(out[k].detach().cpu().numpy()[0])
        # stack with NaN fill for undetected frames
        n = len(frames)
        arrs = {}
        for k in PARAM_KEYS:
            dim = next((v.shape[0] for v in seq[k] if v is not None), 0)
            a = np.full((n, dim), np.nan, np.float32)
            for i, v in enumerate(seq[k]):
                if v is not None:
                    a[i] = v
            arrs[k] = a
        arrs["valid"] = np.array(valid, bool)
        os.makedirs(OUT, exist_ok=True)
        np.savez_compressed(f"{OUT}/{sign}.npz", **arrs)
        vol.commit()
        ndet = int(np.sum(valid))
        return {"sign": sign, "status": "ok", "frames": n, "detected": ndet}


@app.local_entrypoint()
def smoke():
    signs = ["clean", "eat"]
    for r in Extractor().process.map(signs):
        print(r)


@app.local_entrypoint()
def run_words(words: str = ""):
    """Fan out a comma-separated word list across CONCURRENT A10G containers.
    Each <word>.webm must already be staged on the volume under /clips."""
    signs = [w.strip() for w in words.split(",") if w.strip()]
    ok = 0
    for r in Extractor().process.map(signs):
        print(r)
        ok += r.get("status") == "ok"
    print(f"\n{ok}/{len(signs)} signs extracted")


@app.local_entrypoint()
def run_all():
    import json
    picks = json.load(open("dataset/reference_picks_top80.json"))
    signs = sorted(picks.keys())
    ok = 0
    for r in Extractor().process.map(signs):
        print(r)
        ok += r.get("status") == "ok"
    print(f"\n{ok}/{len(signs)} signs extracted")
