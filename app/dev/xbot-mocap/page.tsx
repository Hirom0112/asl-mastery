"use client";

// Drive the faceless X Bot mannequin with 3D-LEX mocap using three's CANONICAL
// cross-rig retargeter (SkeletonUtils.retargetClip) — the standard tool for
// Mixamo<->Ready-Player-Me. Then rigidly stand the X Bot upright + face camera.
// ?sign=clean&cam=front|side|threequarter&t=<freeze>&speed=0.5

import { Canvas, useFrame } from "@react-three/fiber";
import { useGLTF } from "@react-three/drei";
import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import {
  AnimationClip,
  AnimationMixer,
  Box3,
  Color,
  MeshPhysicalMaterial,
  Object3D,
  Quaternion,
  SkinnedMesh,
  Vector3,
  type Group,
} from "three";
import { clone as cloneSkinned, retargetClip } from "three/examples/jsm/utils/SkeletonUtils.js";

const TARGET_HEIGHT = 1.7;
const FRAME_ANCHOR = 0.74;
const CAMERA_DIST = 2.2;
const CAM: Record<string, [number, number, number]> = {
  front: [0.05, 0.0, CAMERA_DIST],
  side: [CAMERA_DIST, 0.05, 0.1],
  threequarter: [CAMERA_DIST * 0.72, 0.12, CAMERA_DIST * 0.72],
};
const norm = (s: string) =>
  s
    .replace(/^mixamorig1?:?/i, "")
    .replace(/[^a-z0-9]/gi, "")
    .toLowerCase();
const findSkinned = (root: Object3D): SkinnedMesh | null => {
  let m: SkinnedMesh | null = null;
  root.traverse((o) => {
    if (!m && (o as SkinnedMesh).isSkinnedMesh) m = o as SkinnedMesh;
  });
  return m;
};
const findBone = (root: Object3D, names: string[]) => {
  let hit: Object3D | null = null;
  const want = names.map((n) => norm(n));
  root.traverse((o) => {
    if (!hit && (o as { isBone?: boolean }).isBone && want.includes(norm(o.name))) hit = o;
  });
  return hit as Object3D | null;
};
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

function Rig({ sign, freezeT, speed }: { sign: string; freezeT?: number; speed: number }) {
  const xbotGltf = useGLTF("/avatar/xbot.glb");
  const srcGltf = useGLTF(`/3dlex/${sign}.glb`);
  const xbot = useMemo(() => cloneSkinned(xbotGltf.scene) as Group, [xbotGltf.scene]);
  const src = useMemo(() => cloneSkinned(srcGltf.scene) as Group, [srcGltf.scene]);
  const xbotWrap = useRef<Group>(null);
  const mixer = useMemo(() => new AnimationMixer(xbot), [xbot]);
  const action = useRef<ReturnType<AnimationMixer["clipAction"]> | null>(null);
  const framed = useRef(false);

  useEffect(() => {
    framed.current = false;
  }, [sign]);

  useEffect(() => {
    const xMesh = findSkinned(xbot);
    const sMesh = findSkinned(src);
    const clip = srcGltf.animations[0];
    if (!xMesh || !sMesh || !clip) return;
    // pearl on X Bot
    const mat = pearl();
    xbot.traverse((o) => {
      const m = o as { isMesh?: boolean; material?: unknown; frustumCulled?: boolean };
      if (m.isMesh) {
        m.material = mat;
        m.frustumCulled = false;
      }
    });
    // names: TARGET (xbot) bone name -> SOURCE (3D-LEX) bone name, matched by normalized name
    const srcByNorm = new Map<string, string>();
    sMesh.skeleton.bones.forEach((b) => srcByNorm.set(norm(b.name), b.name));
    const names: Record<string, string> = {};
    let hip = "Hips";
    xMesh.skeleton.bones.forEach((b) => {
      const s = srcByNorm.get(norm(b.name));
      if (s) {
        names[b.name] = s;
        if (norm(b.name) === "hips") hip = s;
      }
    });
    xbot.updateMatrixWorld(true);
    src.updateMatrixWorld(true);
    const newClip: AnimationClip = retargetClip(xMesh, sMesh, clip, { names, hip });
    const a = mixer.clipAction(newClip);
    a.reset().play();
    a.timeScale = speed;
    action.current = a;
  }, [xbot, src, mixer, srcGltf.animations, speed]);

  useFrame((_, dt) => {
    if (freezeT != null) {
      if (action.current) action.current.time = freezeT;
      mixer.update(0);
    } else mixer.update(dt);
    const g = xbotWrap.current;
    if (!g || framed.current) return;
    g.quaternion.identity();
    g.position.set(0, 0, 0);
    g.scale.setScalar(1);
    g.updateMatrixWorld(true);
    const hips = findBone(xbot, ["Hips"]);
    const neck = findBone(xbot, ["Neck", "Head", "Spine2"]);
    const lsh = findBone(xbot, ["LeftArm"]);
    const rsh = findBone(xbot, ["RightArm"]);
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
    const bb = () => {
      const b = new Box3();
      const v = new Vector3();
      xbot.traverse((o) => {
        if ((o as { isBone?: boolean }).isBone) {
          o.getWorldPosition(v);
          b.expandByPoint(v);
        }
      });
      return b;
    };
    let box = bb();
    let size = box.getSize(new Vector3());
    const s = TARGET_HEIGHT / Math.max(size.y, 1e-3);
    g.scale.setScalar(s);
    g.updateMatrixWorld(true);
    box = bb();
    size = box.getSize(new Vector3());
    const c = box.getCenter(new Vector3());
    g.position.set(-c.x, -(box.min.y + size.y * FRAME_ANCHOR), -c.z);
    g.updateMatrixWorld(true);
    framed.current = true;
  });

  return (
    <group ref={xbotWrap}>
      <primitive object={xbot} />
    </group>
  );
}

function Inner() {
  const q = useSearchParams();
  const sign = q.get("sign") ?? "clean";
  const cam = CAM[q.get("cam") ?? "front"] ?? CAM.front;
  const tParam = q.get("t");
  const freezeT = tParam != null ? parseFloat(tParam) : undefined;
  const speed = parseFloat(q.get("speed") ?? "0.5");
  const [f, setF] = useState(sign);
  return (
    <div style={{ width: "100vw", height: "100vh", background: "#f2ede2", overflow: "hidden" }}>
      <div
        style={{
          position: "absolute",
          top: 8,
          left: 12,
          zIndex: 10,
          fontFamily: "monospace",
          color: "#2a2620",
        }}
      >
        X Bot ← 3D-LEX (retargetClip): <strong>{f}</strong>
        <input
          value={f}
          onChange={(e) => setF(e.target.value)}
          style={{ marginLeft: 8, fontFamily: "monospace", padding: "2px 6px" }}
        />
      </div>
      <Canvas
        camera={{ position: cam, fov: 30 }}
        dpr={[1, 2]}
        gl={{ antialias: true, alpha: true }}
        style={{ width: "100%", height: "100%", background: "transparent" }}
      >
        <ambientLight intensity={0.8} />
        <directionalLight position={[2.5, 3, 2.5]} intensity={1.6} color="#ffffff" />
        <directionalLight position={[-3, 1.5, -1]} intensity={0.7} color="#b48aff" />
        <pointLight position={[0, 1.2, 2]} intensity={0.4} color="#7b5cff" />
        <Suspense fallback={null}>
          <Rig key={f} sign={f} freezeT={freezeT} speed={speed} />
        </Suspense>
      </Canvas>
    </div>
  );
}
export default function XbotMocapPage() {
  return (
    <Suspense fallback={null}>
      <Inner />
    </Suspense>
  );
}
