"use client";

// Production practice avatar — 3D-LEX real motion capture (Vicon optical +
// StretchSense finger gloves, CC BY 4.0), played NATIVELY on its own
// Ready-Player-Me rig (correct for every sign), stood upright + framed, pearl
// finish. Ported from the verified prototype app/dev/glb-preview/page.tsx.
//
// Keyed by signId → /3dlex/<signId>.glb. 70/80 signs have mocap; the 10 without
// (see NO_MOCAP) render the shared figure at REST (frozen at t=0).
//
// Replaces the old keypoint/SMPL-X template avatar (a dead end per avatar.md —
// RPM↔Mixamo rest poses are incompatible for retargeting). Visual polish
// (face-contact clipping on a few signs, framing) is a follow-up.

import { useAnimations, useGLTF } from "@react-three/drei";
import { Canvas, useFrame } from "@react-three/fiber";
import { Suspense, useEffect, useMemo, useRef } from "react";
import { clone as skeletonClone } from "three/examples/jsm/utils/SkeletonUtils.js";
import {
  Box3,
  Color,
  LoopRepeat,
  MeshPhysicalMaterial,
  Object3D,
  Quaternion,
  Vector3,
  type Group,
} from "three";

const TARGET_HEIGHT = 1.7;
const FRAME_ANCHOR = 0.74;
const CAMERA_DIST = 2.2;

export const CAM_PRESETS: Record<string, [number, number, number]> = {
  front: [0.05, 0.0, CAMERA_DIST],
  side: [CAMERA_DIST, 0.05, 0.1],
  threequarter: [CAMERA_DIST * 0.72, 0.12, CAMERA_DIST * 0.72],
};

// Any mocap GLB frozen at t=0 is a neutral standing pose (shared body rig).
const REST_REF = "clean";
// All 80 signs now have real motion (the 10 formerly-frozen signs were given
// SMPLest-X body + WiLoR crisp fingers from clean dictionary clips), so nothing
// is force-rested anymore. Kept as an empty escape hatch for any future gap.
const NO_MOCAP = new Set<string>([]);

function pearl() {
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

const findBone = (root: Object3D, names: string[]): Object3D | null => {
  let hit: Object3D | null = null;
  const want = names.map((n) => n.toLowerCase());
  root.traverse((o) => {
    if (!hit && want.includes(o.name.toLowerCase())) hit = o;
  });
  return hit;
};

// Per-sign face-contact correction. On these signs the mocap fingertip lands ON the
// captured signer's face, but the render head sits a few cm proud → fingers bury.
// We lift the signing hand OUT along the outward face normal (mouth/nose ≈ +Z toward
// camera; cheek ≈ ±X) by a hand-tuned amount, ONLY while the hand is near the head
// (gated + ramped), so approach/retract and the other signs are untouched. World-space
// units = framed model space (height 1.7). Whole Hand bone moves → handshape stays rigid.
// Tuned in app/dev/glb-preview (?fix=0 to compare). See avatar.md.
const FACE_FIX: Record<string, { hand: "Left" | "Right"; offset: [number, number, number] }> = {
  water: { hand: "Left", offset: [0, 0, 0.05] }, // W at the mouth/nose (0.02 buried the fingers in the face, 0.07 floated them off; 0.05 = at the lips)
  red: { hand: "Left", offset: [0, 0, 0.03] }, // index at the lips
  kid: { hand: "Left", offset: [0, 0, 0.06] }, // index at the nose (deep)
  sweet: { hand: "Left", offset: [0, -0.005, 0.035] }, // middle at the chin
  fruit: { hand: "Left", offset: [0.045, 0, 0.075] }, // F-hand at the cheek
  drink: { hand: "Left", offset: [0, 0, 0.02] }, // gentle forward lift only (big offset stretched the arm)
  flower: { hand: "Right", offset: [-0.045, 0, 0.03] }, // at the nose/cheek
  hair: { hand: "Left", offset: [0.03, 0, 0.02] }, // at the side of the head
};
const FIX_GATE = 0.2; // fingertip→Head distance (framed space) below which the lift ramps in
const FIX_SMOOTH = 0.3; // per-frame lerp toward the gated target (no pop)

function Mocap({ sign, speed, rest }: { sign: string; speed: number; rest: boolean }) {
  const { scene, animations } = useGLTF(`/3dlex/${sign}.glb`);
  // useGLTF caches & SHARES one scene object; an Object3D has a single parent, so
  // two mounts (or a sign remount: old unmount races the new mount) fight over the
  // same skeleton → collapsed bones → giant dark blob. Clone per instance.
  const model = useMemo(() => skeletonClone(scene) as Group, [scene]);
  const wrap = useRef<Group>(null);
  const { mixer } = useAnimations(animations, model);
  const framed = useRef(false);
  const settle = useRef(0); // skip the first few frames so the skeleton/mixer settle before latching
  const push = useRef(new Vector3()); // smoothed face-contact lift offset (world space)
  const hipsBone = useRef<Object3D | null>(null);
  const hipsRest = useRef(new Vector3()); // bind-pose Hips position; root drift is pinned to this

  useEffect(() => {
    framed.current = false;
    settle.current = 0;
    push.current.set(0, 0, 0);
  }, [sign]);

  useEffect(() => {
    const mat = pearl();
    model.traverse((o) => {
      const m = o as { isMesh?: boolean; material?: unknown; frustumCulled?: boolean };
      if (m.isMesh) {
        m.material = mat;
        m.frustumCulled = false;
      }
    });
  }, [model]);

  // Capture the Hips bind position. Several mocap clips animate the root (Hips)
  // translation, which drifts the figure out of frame over the loop; we pin Hips
  // back to this each frame (limbs still animate). See the pin useFrame below.
  useEffect(() => {
    const h = findBone(model, ["Hips", "mixamorigHips"]);
    hipsBone.current = h;
    if (h) hipsRest.current.copy(h.position);
  }, [model]);

  // Forward repeat loop. (Ping-pong played the sign backward on the return; a
  // two-action crossfade drifted the root off-screen.) One action, LoopRepeat —
  // it stays framed and just repeats.
  useEffect(() => {
    const clip = animations?.[0];
    if (!clip || !mixer) return;
    const a = mixer.clipAction(clip, model);
    a.reset();
    a.setLoop(LoopRepeat, Infinity);
    a.clampWhenFinished = false;
    a.timeScale = speed;
    a.play();
    if (rest) {
      a.paused = true;
      a.time = 0;
    }
    return () => {
      a.stop();
    };
  }, [animations, mixer, model, speed, rest]);

  // Pin the root every frame (after the mixer poses, before framing measures): keep
  // Hips at its bind position so root-translation tracks can't drift the figure out
  // of frame. Limbs animate normally; only the global body position is held.
  useFrame(() => {
    const h = hipsBone.current;
    if (h) h.position.copy(hipsRest.current);
  });

  // Stand upright (hips→neck = +Y) + square shoulders to camera + frame from the
  // posed bone bbox. Runs once per sign (sidesteps each GLB's own up-axis).
  useFrame(() => {
    const g = wrap.current;
    if (!g || framed.current) return;
    if (settle.current++ < 3) return; // let GLTF load + mixer pose + matrices settle
    g.updateMatrixWorld(true);
    const hips = findBone(model, ["Hips", "mixamorigHips"]);
    const neck = findBone(model, ["Neck", "Head", "Spine2"]);
    const lsh = findBone(model, ["LeftArm", "LeftShoulder"]);
    const rsh = findBone(model, ["RightArm", "RightShoulder"]);
    if (!hips || !neck || !lsh || !rsh) return;
    const up = neck.getWorldPosition(new Vector3()).sub(hips.getWorldPosition(new Vector3()));
    if (up.lengthSq() < 1e-6) return;
    g.quaternion.copy(new Quaternion().setFromUnitVectors(up.normalize(), new Vector3(0, 1, 0)));
    g.updateMatrixWorld(true);
    const shoulder = lsh.getWorldPosition(new Vector3()).sub(rsh.getWorldPosition(new Vector3()));
    shoulder.y = 0;
    if (shoulder.lengthSq() > 1e-6) {
      g.quaternion.premultiply(
        new Quaternion().setFromUnitVectors(shoulder.normalize(), new Vector3(1, 0, 0)),
      );
      g.updateMatrixWorld(true);
    }
    const boneBox = () => {
      const b = new Box3();
      const v = new Vector3();
      model.traverse((o) => {
        if ((o as { isBone?: boolean }).isBone) {
          o.getWorldPosition(v);
          b.expandByPoint(v);
        }
      });
      return b;
    };
    let box = boneBox();
    let size = box.getSize(new Vector3());
    if (!(size.y > 0.05)) return; // degenerate bbox (skeleton not settled) → retry, don't latch a huge scale
    const s = TARGET_HEIGHT / Math.max(size.y, 1e-3);
    g.scale.setScalar(s);
    g.updateMatrixWorld(true);
    box = boneBox();
    size = box.getSize(new Vector3());
    const center = box.getCenter(new Vector3());
    g.position.set(-center.x, -(box.min.y + size.y * FRAME_ANCHOR), -center.z);
    g.updateMatrixWorld(true);
    framed.current = true;
  });

  // Face-contact lift: runs every frame AFTER the mixer poses bones + framing latches.
  // Lifts the signing hand out of the face on contact signs (see FACE_FIX); no-op otherwise.
  useFrame(() => {
    if (!framed.current) return;
    const fix = FACE_FIX[sign];
    const target = new Vector3();
    if (fix) {
      const head = findBone(model, ["Head"]);
      const tip =
        findBone(model, [fix.hand + "HandIndex4"]) ?? findBone(model, [fix.hand + "Hand"]);
      if (head && tip) {
        const dist = tip
          .getWorldPosition(new Vector3())
          .distanceTo(head.getWorldPosition(new Vector3()));
        // ramp the lift in as the fingertip nears the head (1 at contact → 0 by the gate)
        const w = Math.max(0, Math.min(1, (FIX_GATE - dist) / FIX_GATE));
        target.set(...fix.offset).multiplyScalar(w);
      }
    }
    push.current.lerp(target, FIX_SMOOTH);
    if (push.current.lengthSq() < 1e-9) return;
    const hand = fix ? findBone(model, [fix.hand + "Hand"]) : null;
    const parent = hand?.parent;
    if (!hand || !parent) return;
    const wp = hand.getWorldPosition(new Vector3());
    const localNow = parent.worldToLocal(wp.clone());
    const localTgt = parent.worldToLocal(wp.add(push.current));
    hand.position.add(localTgt.sub(localNow)); // additive → never fights the next mixer frame
    hand.updateMatrixWorld(true);
  });

  return (
    <group ref={wrap}>
      <primitive object={model} />
    </group>
  );
}

interface SignAvatarProps {
  signId?: string;
  className?: string;
  /** clip playback rate (prototype default 0.5 = half speed, easier to follow). */
  speed?: number;
  /** mirror horizontally for learner follow-along. Prototype/default: off. */
  mirror?: boolean;
  /** force the rest pose (frozen t=0) regardless of mocap availability. */
  rest?: boolean;
  // --- legacy props from the retired SMPL-X preview route; accepted so
  // app/dev/avatar-preview still typechecks. Ignored by the mocap renderer. ---
  templatesBase?: string;
  enableFingers?: boolean;
  camPos?: [number, number, number];
  freezeFrame?: number;
  palmMode?: "data" | "global" | "off";
}

export function SignAvatar({
  signId,
  className,
  speed = 0.5,
  mirror = false,
  rest,
}: SignAvatarProps) {
  const sign = (signId ?? "").toLowerCase();
  const hasMocap = sign.length > 0 && !NO_MOCAP.has(sign);
  const glb = hasMocap ? sign : REST_REF;
  const atRest = rest ?? !hasMocap;

  return (
    <div
      className={className}
      style={{ position: "relative", width: "100%", height: "100%", overflow: "hidden" }}
    >
      <Canvas
        camera={{ position: CAM_PRESETS.front, fov: 30 }}
        dpr={[1, 2]}
        gl={{ antialias: true, alpha: true }}
        style={{
          width: "100%",
          height: "100%",
          borderRadius: "inherit",
          background: "transparent",
          transform: mirror ? "scaleX(-1)" : undefined,
        }}
      >
        <ambientLight intensity={0.8} />
        <directionalLight position={[2.5, 3, 2.5]} intensity={1.6} color="#ffffff" />
        <directionalLight position={[-3, 1.5, -1]} intensity={0.7} color="#b48aff" />
        <pointLight position={[0, 1.2, 2]} intensity={0.4} color="#7b5cff" />
        <Suspense fallback={null}>
          <Mocap key={`${glb}:${atRest}`} sign={glb} speed={speed} rest={atRest} />
        </Suspense>
      </Canvas>
    </div>
  );
}
