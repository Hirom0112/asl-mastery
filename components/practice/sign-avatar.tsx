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
import { Suspense, useEffect, useRef } from "react";
import {
  Box3,
  Color,
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
// The 10 signs with no mocap yet (public/3dlex/_vocab_order.json mocap:false).
// These render at rest and are surfaced in the sidebar's "Pending" category.
const NO_MOCAP = new Set([
  "baby",
  "cool",
  "dark",
  "eat",
  "follow",
  "nothing",
  "numbers",
  "sick",
  "sports",
  "time",
]);

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

function Mocap({ sign, speed, rest }: { sign: string; speed: number; rest: boolean }) {
  const { scene, animations } = useGLTF(`/3dlex/${sign}.glb`);
  const model = scene as Group;
  const wrap = useRef<Group>(null);
  const { actions, names } = useAnimations(animations, model);
  const framed = useRef(false);

  useEffect(() => {
    framed.current = false;
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

  useEffect(() => {
    const a = actions[names[0]];
    if (a) {
      // three.js AnimationAction is configured BY MUTATION (drei's useAnimations
      // hands back live action objects); the immutability lint doesn't apply.
      /* eslint-disable react-hooks/immutability */
      a.reset().play();
      a.timeScale = speed;
      if (rest) {
        a.time = 0;
        a.paused = true;
      }
      /* eslint-enable react-hooks/immutability */
    }
    return () => {
      a?.stop();
    };
  }, [actions, names, speed, rest]);

  // Stand upright (hips→neck = +Y) + square shoulders to camera + frame from the
  // posed bone bbox. Runs once per sign (sidesteps each GLB's own up-axis).
  useFrame(() => {
    const g = wrap.current;
    if (!g || framed.current) return;
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
