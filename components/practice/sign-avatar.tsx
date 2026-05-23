"use client";

// Sign-avatar: a rigged Mixamo "X Bot" mannequin (from-scratch art asset per
// ADR-0012) that PERFORMS each sign, driven by the per-frame keypoint
// templates in public/templates/*.json.
//
// Retargeting is "aim"-based and 3D-native: for every driven bone we measure,
// at rest, the world direction toward its child, then each frame rotate it so
// that direction points at the child keypoint's world-mapped position. This
// avoids guessing Mixamo's local bone axes and uses ALL the data — the arm
// chain (shoulder→elbow→wrist) and every finger phalanx.
//
// Coordinates: templates are body-space (origin at neck, unit = shoulder
// span, image-y down). We map (x, y, z) → world via a fixed anchor + scale,
// flipping y. z is read from keypoint[2] when present; today templates are 2D
// so z = 0 (frontal plane). When 3D templates land, z flows straight through.
//
// Signs without a template (a few) fall back to the shipped idle clip.

import { Canvas, useFrame } from "@react-three/fiber";
import { useAnimations, useGLTF } from "@react-three/drei";
import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import {
  Box3,
  Color,
  MeshPhysicalMaterial,
  Quaternion,
  Vector3,
  type AnimationAction,
  type Group,
  type Object3D,
} from "three";
import { clone as cloneSkinned } from "three/examples/jsm/utils/SkeletonUtils.js";

const MODEL_URL = "/avatar/xbot.glb";
const TARGET_HEIGHT = 1.7;
const FPS = 14; // 32 frames over ~2.3 s
const IDLE_CLIP = "idle";

// Framing: a reference avatar should show the signing space (head, torso, and
// the arms working in front of the chest), not the legs. FRAME_ANCHOR is the
// fraction of the model's height placed at the camera's look-at (origin) — the
// higher it is, the more the legs drop out of frame. CAMERA_DIST then zooms so
// the upper body fills the viewport.
const FRAME_ANCHOR = 0.82;
const CAMERA_DIST = 2.0;

// Finger retargeting is off until we have 3D hand data: the current templates
// are 2D, which has no finger curl and makes per-finger aiming twitch. Arms
// (location + movement) are well-defined in 2D, so those stay on.
const ENABLE_FINGERS = false;
// Depth heuristic (shoulder-spans toward the camera) so arms reach forward
// into the signing space instead of collapsing onto the body's center plane.
// Replaced by real per-keypoint z when 3D templates land.
const FORWARD_BIAS = 0.6;
const SMOOTHING = 0.5; // per-frame slerp toward the target pose

// pose keypoint indices (schema pose_order)
const R_SH = 2;
const L_SH = 3;
const R_EL = 4;
const L_EL = 5;
const R_WR = 6;
const L_WR = 7;

// Standard MANO/OpenPose 21-keypoint hand order → Mixamo finger bones.
// Each finger: keypoints [base, pip, dip, tip]; bones <Finger>1..3 aim at the
// next keypoint in turn. (FreiHAND / CMU HandDB confirmed this ordering.)
const FINGERS: { name: string; kps: [number, number, number, number] }[] = [
  { name: "Thumb", kps: [1, 2, 3, 4] },
  { name: "Index", kps: [5, 6, 7, 8] },
  { name: "Middle", kps: [9, 10, 11, 12] },
  { name: "Ring", kps: [13, 14, 15, 16] },
  { name: "Pinky", kps: [17, 18, 19, 20] },
];

type KP = number[]; // [x, y] or [x, y, z]
interface Frame {
  pose: (KP | null)[];
  hand0: (KP | null)[];
  hand1: (KP | null)[];
}
interface Template {
  sign: string;
  frames: Frame[];
  T: number;
  one_handed?: boolean;
}

function makePearlMaterial() {
  return new MeshPhysicalMaterial({
    color: new Color("#dfe6ff"),
    roughness: 0.3,
    metalness: 0.0,
    clearcoat: 1.0,
    clearcoatRoughness: 0.22,
    sheen: 1.0,
    sheenRoughness: 0.4,
    sheenColor: new Color("#9ec8ff"),
    envMapIntensity: 0.9,
  });
}

// One driven bone: which child it points at, plus rest references and a
// per-frame target resolver (reads the world-mapped keypoint arrays).
interface DrivenBone {
  bone: Object3D;
  restQuat: Quaternion;
  restDir: Vector3;
  target: (poseW: Vector3[], handW: Vector3[]) => Vector3 | null;
  // Which arm this bone belongs to, and where its child should point when the
  // arm is resting (used to drop the non-dominant arm on one-handed signs).
  side?: "R" | "L";
  restTarget?: Vector3;
}

// ---- module-scoped scratch (avoid per-frame allocation) ----
const _pos = new Vector3();
const _desired = new Vector3();
const _delta = new Quaternion();
const _newWorld = new Quaternion();
const _parentWorld = new Quaternion();
const _a = new Vector3();
const _b = new Vector3();
const _ZERO = new Vector3();

function bn(side: "Left" | "Right", rest: string) {
  return `mixamorig:${side}${rest}`;
}

function lerpKP(a: KP | null, b: KP | null, t: number, out: Vector3, fallback: Vector3) {
  if (!a || !b) {
    out.copy(fallback);
    return;
  }
  out.set(
    a[0] + (b[0] - a[0]) * t,
    a[1] + (b[1] - a[1]) * t,
    (a[2] ?? 0) + ((b[2] ?? 0) - (a[2] ?? 0)) * t,
  );
}

function Avatar({ signId }: { signId?: string }) {
  const { scene, animations } = useGLTF(MODEL_URL);
  const root = useRef<Group>(null);
  const model = useMemo(() => cloneSkinned(scene) as Group, [scene]);
  const { actions } = useAnimations(animations, model);

  const [tpl, setTpl] = useState<Template | null>(null);

  // Runtime engine state, built lazily on the first frame (when world
  // matrices are valid) and reused across signs.
  const engine = useRef<{
    ready: boolean;
    anchor: Vector3;
    avatarSpan: number;
    depthZ: number;
    chains: DrivenBone[]; // ordered parent→child
    poseW: Vector3[];
    handW0: Vector3[];
    handW1: Vector3[];
    scratch: Frame;
    idle?: AnimationAction | null;
    idlePlaying: boolean;
    tplSpan: number;
    // One-handed handling, resolved once per loaded template.
    restResolvedFor: string | null;
    restingSide: "R" | "L" | null; // arm to drop, or null to drive both
  } | null>(null);

  // material + size normalization (no world reads needed beyond bbox)
  useEffect(() => {
    const pearl = makePearlMaterial();
    model.traverse((o) => {
      const mesh = o as { isMesh?: boolean; material?: unknown; frustumCulled?: boolean };
      if (mesh.isMesh) {
        mesh.material = pearl;
        mesh.frustumCulled = false;
      }
    });
    model.scale.setScalar(1);
    model.position.set(0, 0, 0);
    model.updateMatrixWorld(true);
    const box = new Box3().setFromObject(model);
    const size = new Vector3();
    box.getSize(size);
    const center = new Vector3();
    box.getCenter(center);
    const s = TARGET_HEIGHT / size.y;
    model.scale.setScalar(s);
    model.position.set(-center.x * s, -(box.min.y + size.y * FRAME_ANCHOR) * s, -center.z * s);
    model.updateMatrixWorld(true);
    engine.current = null; // force rebuild against the new transform
  }, [model]);

  // fetch the per-sign template (Avatar is keyed by signId, so it remounts
  // per sign and tpl starts null — no synchronous reset needed here).
  useEffect(() => {
    if (!signId) return;
    let alive = true;
    fetch(`/templates/${signId}.json`)
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (alive) setTpl(data);
      })
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, [signId]);

  function buildEngine() {
    // Map ALL named nodes by a NORMALIZED key. The glTF loader sanitizes
    // names (e.g. strips the ":" in "mixamorig:LeftArm" → "mixamorigLeftArm"),
    // and joints aren't always THREE.Bone instances — so we match loosely.
    const norm = (s: string) => s.replace(/[^a-z0-9]/gi, "").toLowerCase();
    const bones = new Map<string, Object3D>();
    model.traverse((o) => {
      if (o.name) bones.set(norm(o.name), o);
    });
    const find = (name: string) => bones.get(norm(name));

    const rSh = find(bn("Right", "Arm"));
    const lSh = find(bn("Left", "Arm"));
    const neck = find("mixamorig:Neck") ?? find("mixamorig:Spine2");
    const anchor = new Vector3();
    (neck ?? model).getWorldPosition(anchor);
    let avatarSpan = 0.3;
    let shoulderZ = anchor.z;
    if (rSh && lSh) {
      rSh.getWorldPosition(_a);
      lSh.getWorldPosition(_b);
      avatarSpan = _a.distanceTo(_b);
      shoulderZ = (_a.z + _b.z) / 2;
    }
    const depthZ = shoulderZ + FORWARD_BIAS * avatarSpan;

    const chains: DrivenBone[] = [];
    const add = (
      boneName: string,
      childName: string,
      target: DrivenBone["target"],
      side?: "R" | "L",
      restTarget?: Vector3,
    ) => {
      const bone = find(boneName);
      const child = find(childName);
      if (!bone || !child) return;
      const restQuat = bone.getWorldQuaternion(new Quaternion());
      bone.getWorldPosition(_a);
      child.getWorldPosition(_b);
      const restDir = _b.clone().sub(_a).normalize();
      chains.push({ bone, restQuat, restDir, target, side, restTarget });
    };

    (["Right", "Left"] as const).forEach((side) => {
      const isR = side === "Right";
      const elbow = isR ? R_EL : L_EL;
      const wrist = isR ? R_WR : L_WR;
      // Rest targets: where a resting arm's elbow/wrist point — straight down
      // with a slight outward+forward bias so it hangs naturally at the side.
      // Aim only uses direction, so the exact distance doesn't matter.
      const outward = isR ? -1 : 1;
      const shPos = new Vector3();
      const shBone = find(bn(side, "Arm"));
      (shBone ?? model).getWorldPosition(shPos);
      const restElbow = shPos
        .clone()
        .add(new Vector3(outward * 0.12 * avatarSpan, -0.55 * avatarSpan, 0.05 * avatarSpan));
      const restWrist = shPos
        .clone()
        .add(new Vector3(outward * 0.18 * avatarSpan, -1.15 * avatarSpan, 0.1 * avatarSpan));
      const sideTag = isR ? "R" : "L";
      // arm chain (well-defined from 2D)
      add(bn(side, "Arm"), bn(side, "ForeArm"), (p) => p[elbow], sideTag, restElbow);
      add(bn(side, "ForeArm"), bn(side, "Hand"), (p) => p[wrist], sideTag, restWrist);
      if (ENABLE_FINGERS) {
        // hand orientation: aim palm toward the middle-finger base
        add(bn(side, "Hand"), bn(side, "HandMiddle1"), (_p, h) => h[9]);
        for (const f of FINGERS) {
          for (let j = 1; j <= 3; j++) {
            const target = f.kps[j];
            add(
              `mixamorig:${side}Hand${f.name}${j}`,
              `mixamorig:${side}Hand${f.name}${j + 1}`,
              (_p, h) => h[target],
            );
          }
        }
      }
    });

    engine.current = {
      ready: true,
      anchor,
      avatarSpan,
      depthZ,
      chains,
      poseW: Array.from({ length: 8 }, () => new Vector3()),
      handW0: Array.from({ length: 21 }, () => new Vector3()),
      handW1: Array.from({ length: 21 }, () => new Vector3()),
      scratch: {
        pose: Array.from({ length: 8 }, () => [0, 0, 0]),
        hand0: Array.from({ length: 21 }, () => [0, 0, 0]),
        hand1: Array.from({ length: 21 }, () => [0, 0, 0]),
      },
      idle: actions[IDLE_CLIP],
      idlePlaying: false,
      tplSpan: 1,
      restResolvedFor: null,
      restingSide: null,
    };
  }

  // map a body-space point into world (y flipped; z ready for 3D data)
  function mapVec(src: Vector3, out: Vector3) {
    const e = engine.current!;
    const S = e.avatarSpan / e.tplSpan;
    // x,y anchored at the neck (y flipped); z sits on the forward signing
    // plane, plus real per-keypoint depth once 3D templates provide src.z.
    out.set(e.anchor.x + src.x * S, e.anchor.y - src.y * S, e.depthZ + src.z * S);
  }

  function aim(d: DrivenBone, targetWorld: Vector3) {
    const { bone } = d;
    bone.updateWorldMatrix(true, false);
    _pos.setFromMatrixPosition(bone.matrixWorld);
    _desired.copy(targetWorld).sub(_pos);
    if (_desired.lengthSq() < 1e-8) return;
    _desired.normalize();
    _delta.setFromUnitVectors(d.restDir, _desired);
    _newWorld.copy(_delta).multiply(d.restQuat);
    bone.parent?.getWorldQuaternion(_parentWorld);
    _parentWorld.invert().multiply(_newWorld); // _parentWorld is now the local target
    bone.quaternion.slerp(_parentWorld, SMOOTHING);
    bone.updateWorldMatrix(false, false);
  }

  useFrame((state) => {
    if (!engine.current) buildEngine();
    const e = engine.current!;

    // idle fallback for sign with no template
    if (!tpl) {
      if (e.idle && !e.idlePlaying) {
        e.idle.reset().fadeIn(0.3).play();
        e.idlePlaying = true;
      }
      return;
    }
    if (e.idle && e.idlePlaying) {
      e.idle.fadeOut(0.2).stop();
      e.idlePlaying = false;
    }

    // template shoulder span (for scale) — compute once per template
    const f0 = tpl.frames[0];
    if (f0.pose[R_SH] && f0.pose[L_SH]) {
      const dx = f0.pose[R_SH]![0] - f0.pose[L_SH]![0];
      const dy = f0.pose[R_SH]![1] - f0.pose[L_SH]![1];
      e.tplSpan = Math.max(0.3, Math.hypot(dx, dy));
    }

    // Resolve once per template which arm rests. For one-handed signs the
    // non-dominant wrist in the averaged trajectory is washed toward center and
    // reads as a crossed arm; we drop whichever wrist travels less.
    if (e.restResolvedFor !== tpl.sign) {
      e.restResolvedFor = tpl.sign;
      e.restingSide = null;
      if (tpl.one_handed) {
        let pathR = 0;
        let pathL = 0;
        for (let i = 1; i < tpl.frames.length; i++) {
          const pr = tpl.frames[i].pose[R_WR];
          const pr0 = tpl.frames[i - 1].pose[R_WR];
          const pl = tpl.frames[i].pose[L_WR];
          const pl0 = tpl.frames[i - 1].pose[L_WR];
          if (pr && pr0) pathR += Math.hypot(pr[0] - pr0[0], pr[1] - pr0[1]);
          if (pl && pl0) pathL += Math.hypot(pl[0] - pl0[0], pl[1] - pl0[1]);
        }
        e.restingSide = pathR >= pathL ? "L" : "R";
      }
    }

    // interpolate the current frame
    const T = tpl.T;
    const looped = (state.clock.elapsedTime * FPS) % T;
    const i0 = Math.floor(looped);
    const i1 = (i0 + 1) % T;
    const t = looped - i0;
    const fa = tpl.frames[i0];
    const fb = tpl.frames[i1];

    // pose + hands → world targets
    for (let i = 0; i < 8; i++) {
      lerpKP(fa.pose[i], fb.pose[i], t, _a, _ZERO);
      mapVec(_a, e.poseW[i]);
    }
    for (let i = 0; i < 21; i++) {
      lerpKP(fa.hand0[i], fb.hand0[i], t, _a, _ZERO);
      mapVec(_a, e.handW0[i]);
      lerpKP(fa.hand1[i], fb.hand1[i], t, _b, _ZERO);
      mapVec(_b, e.handW1[i]);
    }

    // assign hand slots to sides by nearest wrist
    const c0 = e.handW0[0];
    const dR = c0.distanceTo(e.poseW[R_WR]);
    const dL = c0.distanceTo(e.poseW[L_WR]);
    const rightHand = dL < dR ? e.handW1 : e.handW0;
    const leftHand = dL < dR ? e.handW0 : e.handW1;

    // drive all bones, parent→child. chains are [Right…, Left…]; first half is
    // Right (uses rightHand), second half Left.
    const half = e.chains.length / 2;
    for (let k = 0; k < e.chains.length; k++) {
      const d = e.chains[k];
      // Resting arm on a one-handed sign: aim it straight down to the side.
      if (e.restingSide && d.side === e.restingSide) {
        if (d.restTarget) aim(d, d.restTarget);
        continue;
      }
      const hand = k < half ? rightHand : leftHand;
      const tgt = d.target(e.poseW, hand);
      if (tgt) aim(d, tgt);
    }
  });

  return (
    <group ref={root}>
      <primitive object={model} />
    </group>
  );
}

useGLTF.preload(MODEL_URL);

export function SignAvatar({ className, signId }: { className?: string; signId?: string }) {
  return (
    <div className={className} style={{ position: "relative" }}>
      <Canvas
        camera={{ position: [0.1, 0.0, CAMERA_DIST], fov: 30 }}
        dpr={[1, 2]}
        gl={{ antialias: true, alpha: true }}
        style={{
          width: "100%",
          height: "100%",
          borderRadius: "inherit",
          background: "transparent",
        }}
      >
        <ambientLight intensity={0.8} />
        <directionalLight position={[2.5, 3, 2.5]} intensity={1.6} color="#ffffff" />
        <directionalLight position={[-3, 1.5, -1]} intensity={0.7} color="#b48aff" />
        <pointLight position={[0, 1.2, 2]} intensity={0.4} color="#7b5cff" />

        <Suspense fallback={null}>
          <Avatar key={signId ?? "none"} signId={signId} />
        </Suspense>
      </Canvas>
    </div>
  );
}
