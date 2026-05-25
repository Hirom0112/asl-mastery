"""Bake EXISTING SMPL-X motion (data/smplx_motion/out/<sign>.npz) into a playable
glTF animation grafted onto the FROZEN donor GLB (public/3dlex/<sign>.glb).

This is a $0 CPU baker. It:
  1. Forwards the SMPL-X layer on the stored per-frame params -> world joint
     positions (144 joints incl. finger TIPS, camera-aligned: x right, y DOWN,
     z depth).
  2. Builds, PER FRAME, a SMPL-X torso basis (right = L->R shoulder, up =
     pelvis->neck) from those positions and re-expresses every driven bone's
     parent->child target DIRECTION in that torso frame. This cancels the
     signer's global body lean / camera tilt WITHOUT discarding vertical reach.
  3. Parses the donor GLB, computes the rig's rest-pose forward kinematics to get
     each bone's REST world direction + rest world rotation, and the rig's OWN
     rest torso basis.
  4. CHANGE OF BASIS: maps the torso-frame target direction into the rig's rest
     torso frame via  M = Brig . Bsmplx^T  (a single rotation derived from the
     two rest torso bases). THIS is the fix -- the donor rig's `Armature` node
     carries a +90deg X rotation so the rig's rest WORLD frame is NOT Y-up, and
     the old code's hand-picked up-axis silently bled vertical reach into
     "forward" so the arms never rose. M is computed, not guessed.
  5. Retargets DIRECTION -> local rotation per bone (swing "look-at": rotate each
     bone's rest bone-axis onto M . target_dir). Swing absorbs the donor rig's
     A-pose vs the SMPL-X T-pose rest mismatch automatically. Rotations
     accumulate down the chain so a child is posed in its (already rotated)
     parent frame.
  6. Writes an "Unreal Take" animation (LINEAR quaternion tracks per driven bone
     @ 24fps, matching the existing 3D-LEX clips) into the GLB binary buffer and
     saves over public/3dlex/<sign>.glb.

The donor rig is NOT made world-upright by the clip -- the live renderer
(app/dev/glb-preview) stands the figure upright by rotating the Hips->Neck axis
to world +Y and squaring shoulders to camera. We bake bone-relative pose only;
the Hips track is left at REST (the renderer reorients the whole wrap group), so
the figure stays correctly framed while the limbs move per the sign.

Usage:
  training/.venv/bin/python scripts/smplx_to_glb.py --only animal
  training/.venv/bin/python scripts/smplx_to_glb.py            # all 15

Motion derives from Sem-Lex (SignAvatars/SMPLest-X) => NON-COMMERCIAL use only.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch
import smplx
from smplx.joint_names import JOINT_NAMES
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "smplx_motion" / "out"
GLB = ROOT / "public" / "3dlex"
MODELS = ROOT / "data" / "signavatars" / "human_model_files"

FPS = 24.0
SMOOTH_SIGMA = 1.6          # temporal de-jitter on joint trajectories
DEFAULT_WORDS = ["animal", "big", "bird", "building", "enjoy", "friend", "good",
                 "hot", "learn", "people", "rain", "sun", "swim", "vegetable", "warm"]

# ----------------------------------------------------------------------------
# Bone graph: donor RPM bone -> (SMPL-X parent joint, SMPL-X child joint).
# The bone "points" from parent joint toward child joint in SMPL-X space; we
# rotate the donor bone's rest direction onto that (in the rig's rest torso
# frame). Order matters: parents BEFORE children so the accumulated parent
# rotation is available.
# ----------------------------------------------------------------------------
J = {n: JOINT_NAMES.index(n) for n in [
    "pelvis", "spine1", "spine2", "spine3", "neck", "head",
    "left_collar", "left_shoulder", "left_elbow", "left_wrist",
    "right_collar", "right_shoulder", "right_elbow", "right_wrist",
]}

# Arm driven chain. Each entry: bone_name, (smplx_parent, smplx_child), parent_bone.
# parent_bone is the rig bone whose POSED world rotation roots this bone's local;
# None means root under the (rest) shoulder.
ARM_BONES = [
    ("LeftArm",      ("left_shoulder",  "left_elbow"),  None),
    ("LeftForeArm",  ("left_elbow",     "left_wrist"),  "LeftArm"),
    ("RightArm",     ("right_shoulder", "right_elbow"), None),
    ("RightForeArm", ("right_elbow",    "right_wrist"), "RightArm"),
]

# Finger joints (MANO 4 per finger: j1,j2,j3,tip). Bone i points j_i -> j_{i+1}.
# tips come from the regressed extended joints (66-75).
FINGERS = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
SMPLX_FINGER = {  # side -> finger -> [j1,j2,j3,tip] smplx indices
    "left":  {"Thumb": [37, 38, 39, 66], "Index": [25, 26, 27, 67],
              "Middle": [28, 29, 30, 68], "Ring": [34, 35, 36, 69],
              "Pinky": [31, 32, 33, 70]},
    "right": {"Thumb": [52, 53, 54, 71], "Index": [40, 41, 42, 72],
              "Middle": [43, 44, 45, 73], "Ring": [49, 50, 51, 74],
              "Pinky": [46, 47, 48, 75]},
}
# wrist smplx idx per side; the Hand bone is oriented by wrist->middle1.
WRIST_J = {"left": 20, "right": 21}

# How much of the finger pose to apply (0=rest, 1=full). SMPLest-X hand pose is
# soft, so we damp it; body/arm motion is the priority (avatar.md).
FINGER_GAIN = 0.6


# ===========================================================================
# quaternion / vector helpers (numpy, [x,y,z,w] order to match glTF)
# ===========================================================================
def qmul(a, b):
    ax, ay, az, aw = a; bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def qrot(q, v):
    """Rotate vec3 v by quaternion q [x,y,z,w]."""
    qv = np.array([q[0], q[1], q[2]])
    t = 2.0 * np.cross(qv, v)
    return v + q[3] * t + np.cross(qv, t)


def qnorm(q):
    return q / (np.linalg.norm(q) + 1e-12)


# ----- rotation-matrix helpers (the retarget math runs in matrices, then we
# emit quaternions [x,y,z,w] for the glTF tracks) -----
def mat_to_quat(R):
    """3x3 rotation matrix -> quaternion [x,y,z,w]."""
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    tr = m00 + m11 + m22
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = np.sqrt(1.0 + m22 - m00 - m11) * 2
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    return qnorm(np.array([x, y, z, w]))


def quat_to_mat(q):
    """quaternion [x,y,z,w] -> 3x3 rotation matrix."""
    x, y, z, w = qnorm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def rmat_from_to(a, b):
    """Shortest-arc rotation matrix sending unit vec a -> unit vec b."""
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -0.999999:
        axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0]))
        axis /= np.linalg.norm(axis) + 1e-12
        return _axis_angle_mat(axis, np.pi)
    s = np.linalg.norm(v)
    if s < 1e-9:
        return np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1.0 - c) / (s * s))


def _axis_angle_mat(axis, ang):
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


def damp_rmat(R, g):
    """Scale rotation matrix R toward identity by gain g (g=1 -> R, g=0 -> I)."""
    if g >= 0.999:
        return R
    tr = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    ang = np.arccos(tr)
    if ang < 1e-6:
        return np.eye(3)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    n = np.linalg.norm(axis)
    if n < 1e-9:
        return np.eye(3)
    return _axis_angle_mat(axis / n, ang * g)


def torso_frame(right, up):
    """Orthonormal basis (3x3, columns = right/up/fwd in world) from a right axis
    and an up axis. Gram-Schmidt: keep up, orthogonalize right, fwd = right x up."""
    up = up / (np.linalg.norm(up) + 1e-12)
    right = right - np.dot(right, up) * up
    right = right / (np.linalg.norm(right) + 1e-12)
    fwd = np.cross(right, up)
    return np.column_stack([right, up, fwd])


# ===========================================================================
# GLB binary parse / write (chunk logic mirrors /tmp/makefaceless3.mjs)
# ===========================================================================
def parse_glb(buf: bytes):
    assert buf[:4] == b"glTF"
    off = 12
    js = None
    bin_ = None
    while off < len(buf):
        clen = struct.unpack_from("<I", buf, off)[0]
        ctype = struct.unpack_from("<I", buf, off + 4)[0]
        data = buf[off + 8: off + 8 + clen]
        if ctype == 0x4E4F534A:      # JSON
            js = json.loads(data.decode("utf-8"))
        elif ctype == 0x004E4942:    # BIN
            bin_ = bytearray(data)
        off += 8 + clen
    return js, bin_


def write_glb(js: dict, bin_: bytearray) -> bytes:
    jstr = json.dumps(js, separators=(",", ":")).encode("utf-8")
    jpad = (4 - (len(jstr) % 4)) % 4
    jchunk = jstr + b"\x20" * jpad
    bpad = (4 - (len(bin_) % 4)) % 4
    bchunk = bytes(bin_) + b"\x00" * bpad
    total = 12 + 8 + len(jchunk) + 8 + len(bchunk)
    out = bytearray()
    out += struct.pack("<III", 0x46546C67, 2, total)
    out += struct.pack("<II", len(jchunk), 0x4E4F534A) + jchunk
    out += struct.pack("<II", len(bchunk), 0x004E4942) + bchunk
    return bytes(out)


# ===========================================================================
# Donor-rig rest forward kinematics
# ===========================================================================
def build_rig(js):
    """Return name->node-index, parent map, and node-local TRS dicts."""
    nodes = js["nodes"]
    name2i = {n.get("name"): i for i, n in enumerate(nodes) if n.get("name")}
    parent = {}
    for i, n in enumerate(nodes):
        for c in n.get("children", []):
            parent[c] = i
    return nodes, name2i, parent


def node_local(n):
    t = np.array(n.get("translation", [0, 0, 0]), float)
    r = np.array(n.get("rotation", [0, 0, 0, 1]), float)
    s = np.array(n.get("scale", [1, 1, 1]), float)
    return t, r, s


def rest_world(nodes, parent, idx):
    """World rest (position, rotation-quat) of node idx by walking to root.
    Assumes uniform scale (the rig has S=1 everywhere)."""
    chain = []
    i = idx
    while i is not None:
        chain.append(i)
        i = parent.get(i)
    chain.reverse()
    pos = np.zeros(3)
    rot = np.array([0.0, 0.0, 0.0, 1.0])
    for ci in chain:
        t, r, _ = node_local(nodes[ci])
        pos = pos + qrot(rot, t)
        rot = qnorm(qmul(rot, r))
    return pos, rot


def rest_world_mat(nodes, parent, idx):
    """World rest (position, rotation-MATRIX) of node idx by walking to root."""
    chain = []
    i = idx
    while i is not None:
        chain.append(i)
        i = parent.get(i)
    chain.reverse()
    pos = np.zeros(3)
    R = np.eye(3)
    for ci in chain:
        t, r, _ = node_local(nodes[ci])
        pos = pos + R @ t
        R = R @ quat_to_mat(r)
    return pos, R


# ===========================================================================
# SMPL-X forward
# ===========================================================================
def forward_joints(model, d) -> np.ndarray:
    def t(key):
        return torch.tensor(np.nan_to_num(d[key]), dtype=torch.float32)
    F = d["smplx_root_pose"].shape[0]
    z = torch.zeros(F, 3, dtype=torch.float32)
    out = model(
        global_orient=t("smplx_root_pose"), body_pose=t("smplx_body_pose"),
        left_hand_pose=t("smplx_lhand_pose"), right_hand_pose=t("smplx_rhand_pose"),
        jaw_pose=t("smplx_jaw_pose"), leye_pose=z, reye_pose=z,
        betas=t("smplx_shape"), expression=t("smplx_expr"), return_verts=False,
    )
    return out.joints.detach().numpy()  # (F,144,3) camera-aligned x-right y-DOWN z-depth


def _zero_params(d):
    """A 1-frame all-zero-pose param dict (canonical SMPL-X T-pose) carrying the
    take's body SHAPE, so the rest torso basis matches this signer's proportions.
    Used once to derive the change-of-basis M."""
    return {
        "smplx_root_pose": np.zeros((1, 3), np.float32),
        "smplx_body_pose": np.zeros((1, 63), np.float32),
        "smplx_lhand_pose": np.zeros((1, 45), np.float32),
        "smplx_rhand_pose": np.zeros((1, 45), np.float32),
        "smplx_jaw_pose": np.zeros((1, 3), np.float32),
        "smplx_shape": np.nan_to_num(d["smplx_shape"][:1]),
        "smplx_expr": np.zeros((1, 10), np.float32),
    }


def smplx_torso_frame(W: np.ndarray):
    """SMPL-X torso basis (3x3, cols = right/up/fwd) for ONE frame, in the raw
    model-output frame. right = R->L shoulder, up = pelvis->neck. Re-expressing
    target bone DIRECTIONS in this frame cancels the signer's global body lean /
    camera tilt without discarding vertical reach (the directions stay full-3D in
    a body-attached frame; only the body's global orientation is removed)."""
    right = W[J["left_shoulder"]] - W[J["right_shoulder"]]
    up = W[J["neck"]] - W[J["pelvis"]]
    return torso_frame(right, up)


# ===========================================================================
# Retarget: per frame, solve local rotation for each driven bone.
# ===========================================================================
def retarget_frame(name2i, M, rig_rest_dir, rig_rest_rot, rig_rest_rot_parent,
                   gparent_name, W) -> dict:
    """W: (144,3) SMPL-X world joint positions for this frame (raw model frame).
    Returns name -> local quat [x,y,z,w] for the DRIVEN bones (arms, forearms,
    hands, fingers). Spine/neck/collar are LEFT AT REST so the rig stays vertical
    and the renderer's upright/frame logic (Hips->Neck = +Y) keeps working --
    sign legibility lives in the arms+hands.

    Per bone the target parent->child direction is taken in the SMPL-X torso
    frame, mapped into the rig's rest torso frame via M (the single change-of-
    basis between the two rest torso bases -- this accounts for the donor rig's
    Armature +90deg X so vertical reach lands correctly), then a SWING rotation
    aligns the bone's rest direction onto it. Swing absorbs the rig A-pose vs
    SMPL-X T-pose rest mismatch. Posed rotations accumulate down the chain.
        d_body = Bsmplx^T (W[child]-W[parent])     # de-leaned, full 3D
        d_rig  = M @ d_body                        # into rig rest torso frame
        swing  = align(rig_rest_dir[b] -> d_rig)
        R_world = swing . rig_rest_rot[b]
        local   = posed_parent_world^-1 . R_world
    """
    Sframe = smplx_torso_frame(W)
    St = Sframe.T
    out_local = {}
    posed_world = {}   # bone name -> posed world ROTATION MATRIX

    def tgt_dir(smplx_par, smplx_child):
        d = W[smplx_child] - W[smplx_par]
        if np.linalg.norm(d) < 1e-6:
            return None
        d = M @ (St @ d)                 # de-lean -> rig rest torso frame
        return d / (np.linalg.norm(d) + 1e-9)

    def solve(bone, d_tgt, parent_bone, gain=1.0):
        if d_tgt is None or bone not in name2i:
            return
        swing = rmat_from_to(rig_rest_dir[bone], d_tgt)
        if gain != 1.0:
            swing = damp_rmat(swing, gain)
        R_world = swing @ rig_rest_rot[bone]
        posed_world[bone] = R_world
        if parent_bone is not None and parent_bone in posed_world:
            R_par = posed_world[parent_bone]
        else:
            R_par = rig_rest_rot_parent[bone]
        out_local[bone] = mat_to_quat(R_par.T @ R_world)

    # arms (shoulder/collar stay at rest; Arm hangs under the rest shoulder)
    for bone, (sp, sc), pb in ARM_BONES:
        solve(bone, tgt_dir(J[sp], J[sc]), pb)

    # hand: orient the wrist (Hand bone) by wrist->middle1 so the palm tracks.
    solve("LeftHand",  tgt_dir(WRIST_J["left"],  SMPLX_FINGER["left"]["Middle"][0]),  "LeftForeArm")
    solve("RightHand", tgt_dir(WRIST_J["right"], SMPLX_FINGER["right"]["Middle"][0]), "RightForeArm")

    # fingers (damped; SMPLest-X hand pose is soft)
    for side, pfx in (("left", "Left"), ("right", "Right")):
        for finger in FINGERS:
            chain = SMPLX_FINGER[side][finger]
            prev_bone = f"{pfx}Hand"
            for bi in range(3):
                bone = f"{pfx}Hand{finger}{bi + 1}"
                if bone not in name2i:
                    continue
                solve(bone, tgt_dir(chain[bi], chain[bi + 1]), prev_bone,
                      gain=FINGER_GAIN)
                prev_bone = bone

    return out_local


# ===========================================================================
# Strip any previously-baked animation so re-bakes are IDEMPOTENT.
# Our baker always appends animation data as a contiguous tail of the binary
# buffer (one time bufferView + one VEC4 bufferView per track), so we can drop
# the animations + their accessors/bufferViews and truncate the buffer back to
# the end of the non-animation (mesh/skin) data.
# ===========================================================================
def strip_existing_animations(js, bin_):
    anims = js.get("animations")
    if not anims:
        return bin_
    anim_acc = set()
    for a in anims:
        for s in a.get("samplers", []):
            anim_acc.add(s["input"])
            anim_acc.add(s["output"])
    anim_bv = {js["accessors"][i]["bufferView"] for i in anim_acc
               if "bufferView" in js["accessors"][i]}
    # keep prefix accessors/bufferViews (anything not exclusively for animation)
    keep_bv_end = 0
    for i, bv in enumerate(js["bufferViews"]):
        if i not in anim_bv:
            keep_bv_end = max(keep_bv_end, bv.get("byteOffset", 0) + bv["byteLength"])
    # all animation accessors/bufferViews are the tail (verified): they have the
    # highest indices; drop them and truncate the buffer.
    max_keep_acc = max((i for i in range(len(js["accessors"]))
                        if i not in anim_acc), default=-1)
    max_keep_bv = max((i for i in range(len(js["bufferViews"]))
                       if i not in anim_bv), default=-1)
    js["accessors"] = js["accessors"][:max_keep_acc + 1]
    js["bufferViews"] = js["bufferViews"][:max_keep_bv + 1]
    js["animations"] = []
    new_bin = bin_[:keep_bv_end]
    js["buffers"][0]["byteLength"] = len(new_bin)
    return new_bin


# ===========================================================================
# Append animation to GLB
# ===========================================================================
def append_animation(js, bin_, tracks: dict, n_frames: int):
    """tracks: bone_name -> (n_frames,4) quat array. Adds one bufferView for time
    + one per track, accessors, samplers, channels, named 'Unreal Take'."""
    name2i = {n.get("name"): i for i, n in enumerate(js["nodes"]) if n.get("name")}
    buffer0 = js["buffers"][0]
    # current bin length = end of buffer
    base = len(bin_)
    # pad to 4
    while len(bin_) % 4:
        bin_.append(0)

    accessors = js.setdefault("accessors", [])
    bufferViews = js.setdefault("bufferViews", [])

    times = np.arange(n_frames, dtype=np.float32) / FPS

    def add_bufferview(arr_bytes):
        while len(bin_) % 4:
            bin_.append(0)
        off = len(bin_)
        bin_.extend(arr_bytes)
        bv = {"buffer": 0, "byteOffset": off, "byteLength": len(arr_bytes)}
        bufferViews.append(bv)
        return len(bufferViews) - 1

    # time accessor (shared)
    tbytes = times.astype("<f4").tobytes()
    tbv = add_bufferview(tbytes)
    accessors.append({
        "bufferView": tbv, "componentType": 5126, "count": n_frames,
        "type": "SCALAR", "min": [float(times[0])], "max": [float(times[-1])],
    })
    time_acc = len(accessors) - 1

    samplers = []
    channels = []
    for bone, quats in tracks.items():
        ni = name2i.get(bone)
        if ni is None:
            continue
        q = np.asarray(quats, dtype="<f4")
        qb = q.tobytes()
        qbv = add_bufferview(qb)
        accessors.append({
            "bufferView": qbv, "componentType": 5126, "count": n_frames,
            "type": "VEC4",
        })
        out_acc = len(accessors) - 1
        samplers.append({"input": time_acc, "interpolation": "LINEAR",
                         "output": out_acc})
        channels.append({"sampler": len(samplers) - 1,
                         "target": {"node": ni, "path": "rotation"}})

    anim = {"name": "Unreal Take", "channels": channels, "samplers": samplers}
    js.setdefault("animations", []).append(anim)
    # update buffer length
    buffer0["byteLength"] = len(bin_)
    return len(channels)


# ===========================================================================
# main bake
# ===========================================================================
def bake(model, word: str, verbose=True):
    npz = SRC / f"{word}.npz"
    glb = GLB / f"{word}.glb"
    if not npz.exists():
        print(f"  SKIP {word}: no npz")
        return False
    if not glb.exists():
        print(f"  SKIP {word}: no donor glb")
        return False

    d = np.load(npz)
    Jw = forward_joints(model, d)        # (F,144,3) raw model frame (x R, y DOWN, z depth)
    F = Jw.shape[0]
    if F >= 5:
        Jw = gaussian_filter1d(Jw, SMOOTH_SIGMA, axis=0, mode="nearest")

    js, bin_ = parse_glb(glb.read_bytes())
    bin_ = strip_existing_animations(js, bin_)   # idempotent re-bake
    nodes, name2i, parent = build_rig(js)

    # ---- driven bone list ----
    all_bones = [b for b, _, _ in ARM_BONES]
    for side in ("Left", "Right"):
        all_bones.append(f"{side}Hand")
        for finger in FINGERS:
            for k in (1, 2, 3):
                all_bones.append(f"{side}Hand{finger}{k}")

    # ---- rest-pose FK (rotation MATRICES) for every driven bone ----
    rig_rest_dir = {}         # bone -> rest world bone direction (toward child)
    rig_rest_rot = {}         # bone -> rest world rotation matrix
    rig_rest_rot_parent = {}  # bone -> rest world rotation matrix of glTF parent
    gparent_name = {}
    for bone in all_bones:
        i = name2i.get(bone)
        if i is None:
            continue
        pos_b, R_b = rest_world_mat(nodes, parent, i)
        rig_rest_rot[bone] = R_b
        ch = nodes[i].get("children", [])
        if ch:
            pos_c, _ = rest_world_mat(nodes, parent, ch[0])
            dvec = pos_c - pos_b
            if np.linalg.norm(dvec) > 1e-6:
                rig_rest_dir[bone] = dvec / np.linalg.norm(dvec)
            else:
                rig_rest_dir[bone] = R_b @ np.array([0.0, 1.0, 0.0])
        else:
            rig_rest_dir[bone] = R_b @ np.array([0.0, 1.0, 0.0])
        pi = parent.get(i)
        gparent_name[bone] = nodes[pi].get("name") if pi is not None else None
        if pi is not None:
            _, Rp = rest_world_mat(nodes, parent, pi)
            rig_rest_rot_parent[bone] = Rp
        else:
            rig_rest_rot_parent[bone] = np.eye(3)

    # ---- change-of-basis M between the two REST torso bases ----
    # SMPL-X rest torso (canonical T-pose): from a zero-pose forward.
    rest_J = forward_joints(model, _zero_params(d))[0]   # (144,3)
    Bsmplx = torso_frame(rest_J[J["left_shoulder"]] - rest_J[J["right_shoulder"]],
                         rest_J[J["neck"]] - rest_J[J["pelvis"]])
    # rig rest torso: from rig FK (accounts for the Armature +90deg X).
    hips_p, _ = rest_world_mat(nodes, parent, name2i["Hips"])
    neck_p, _ = rest_world_mat(nodes, parent, name2i["Neck"])
    larm_p, _ = rest_world_mat(nodes, parent, name2i["LeftArm"])
    rarm_p, _ = rest_world_mat(nodes, parent, name2i["RightArm"])
    Brig = torso_frame(larm_p - rarm_p, neck_p - hips_p)
    M = Brig @ Bsmplx.T   # maps a dir in SMPL-X torso frame -> rig rest torso frame

    # ---- per-frame retarget ----
    track_frames = {b: [] for b in all_bones if b in name2i}
    rest_local = {}   # fallback local quat when a bone isn't solved this frame
    for b in all_bones:
        i = name2i.get(b)
        if i is None:
            continue
        rest_local[b] = np.array(nodes[i].get("rotation", [0, 0, 0, 1]), float)

    for f in range(F):
        loc = retarget_frame(name2i, M, rig_rest_dir, rig_rest_rot,
                             rig_rest_rot_parent, gparent_name, Jw[f])
        for b in track_frames:
            track_frames[b].append(loc.get(b, rest_local[b]))

    # quaternion continuity (avoid sign flips that LERP through zero)
    tracks = {}
    for b, frames in track_frames.items():
        arr = np.array(frames, float)
        for k in range(1, len(arr)):
            if np.dot(arr[k], arr[k - 1]) < 0:
                arr[k] = -arr[k]
        tracks[b] = arr

    n_ch = append_animation(js, bin_, tracks, F)
    out = write_glb(js, bin_)
    glb.write_bytes(out)
    if verbose:
        print(f"  baked {word}: {F} frames, {n_ch} bone tracks, "
              f"{len(out)//1024} KB")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--finger-gain", type=float, default=None,
                    help="override FINGER_GAIN (0.6 default damps soft SMPLest-X "
                         "fingers; use 1.0 for crisp WiLoR-grafted fingers)")
    args = ap.parse_args()
    if args.finger_gain is not None:
        globals()["FINGER_GAIN"] = args.finger_gain
        print(f"FINGER_GAIN override -> {args.finger_gain}")
    words = args.only if args.only else DEFAULT_WORDS

    model = smplx.create(str(MODELS), "smplx", gender="neutral", use_pca=False,
                         use_face_contour=True, flat_hand_mean=False, batch_size=1)
    ok = 0
    for w in words:
        if bake(model, w):
            ok += 1
    print(f"\nbaked {ok}/{len(words)} signs -> {GLB}")


if __name__ == "__main__":
    main()
