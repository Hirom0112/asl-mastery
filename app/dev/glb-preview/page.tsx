"use client";

// The CORRECT-MOTION avatar: 3D-LEX real Vicon+glove mocap, played NATIVELY on
// its own rig (correct for every sign), stood upright + framed, pearl finish.
// Picker over all covered signs. ?sign=clean&cam=front|side|threequarter&speed=0.5

import { Canvas, useFrame } from "@react-three/fiber";
import { useAnimations, useGLTF } from "@react-three/drei";
import { Suspense, useEffect, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
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
const CAM: Record<string, [number, number, number]> = {
  front: [0.05, 0.0, CAMERA_DIST],
  side: [CAMERA_DIST, 0.05, 0.1],
  threequarter: [CAMERA_DIST * 0.72, 0.12, CAMERA_DIST * 0.72],
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
const findBone = (root: Object3D, names: string[]) => {
  let hit: Object3D | null = null;
  const want = names.map((n) => n.toLowerCase());
  root.traverse((o) => {
    if (!hit && want.includes(o.name.toLowerCase())) hit = o;
  });
  return hit as Object3D | null;
};

const REST_REF = "clean"; // any mocap GLB — shared body; frozen at t=0 = neutral stance

// Per-sign face-contact correction. On these signs the mocap fingertip lands ON the
// captured signer's face, but the render head sits a few cm proud → fingers bury.
// We lift the signing hand OUT along the outward face normal (≈ +Z, toward camera)
// by a hand-tuned amount, ONLY while the hand is near the head (gated + ramped), so
// approach/retract and the other 60+ signs are untouched. World-space units = framed
// model space (height 1.7). Whole Hand bone moves → handshape stays rigid.
const FACE_FIX: Record<string, { hand: "Left" | "Right"; offset: [number, number, number] }> = {
  // mouth/nose contacts → lift toward camera (+Z) enough to clearly occlude the face from front
  water: { hand: "Left", offset: [0, 0, 0.05] }, // W at the mouth/nose (0.02 buried the fingers in the face, 0.07 floated them off; 0.05 = at the lips)
  red: { hand: "Left", offset: [0, 0, 0.03] }, // index at the lips
  kid: { hand: "Left", offset: [0, 0, 0.06] }, // index at the nose (deep)
  sweet: { hand: "Left", offset: [0, -0.005, 0.035] }, // middle at the chin
  // cheek/side contacts → push sideways OFF the cheek (±X) + a little forward
  fruit: { hand: "Left", offset: [0.045, 0, 0.075] }, // F-hand at the cheek (screen-right)
  drink: { hand: "Left", offset: [0, 0, 0.02] }, // gentle forward lift only (big offset stretched the arm)
  flower: { hand: "Right", offset: [-0.045, 0, 0.03] }, // at the nose/cheek (screen-left)
  hair: { hand: "Left", offset: [0.03, 0, 0.02] }, // at the side of the head (screen-right)
};
const FIX_GATE = 0.2; // fingertip→Head distance (framed space) below which the lift ramps in
const FIX_SMOOTH = 0.3; // per-frame lerp toward the gated target (no pop)

function Mocap({
  sign,
  speed,
  rest = false,
  fixEnabled = true,
  freezeFrac = null,
}: {
  sign: string;
  speed: number;
  rest?: boolean;
  fixEnabled?: boolean;
  freezeFrac?: number | null;
}) {
  const { scene, animations } = useGLTF(`/3dlex/${sign}.glb`);
  const model = scene as Group;
  const wrap = useRef<Group>(null);
  const { actions, names } = useAnimations(animations, model);
  const framed = useRef(false);
  const settle = useRef(0);
  const push = useRef(new Vector3()); // smoothed face-contact lift offset (world space)

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
  useEffect(() => {
    const a = actions[names[0]];
    if (a) {
      // three.js AnimationAction is configured by mutation (drei useAnimations);
      // block-disable survives prettier reformatting (unlike a -next-line).
      /* eslint-disable react-hooks/immutability */
      a.reset().play();
      a.timeScale = speed;
      if (rest) {
        a.time = 0;
        a.paused = true;
      } else if (freezeFrac != null) {
        a.time = a.getClip().duration * freezeFrac;
        a.paused = true;
      }
      /* eslint-enable react-hooks/immutability */
    }
    return () => {
      a?.stop();
    };
  }, [actions, names, speed, rest, freezeFrac]);

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
  useFrame(() => {
    if (!framed.current) return;
    const fix = fixEnabled ? FACE_FIX[sign] : undefined;
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

const btn: React.CSSProperties = {
  cursor: "pointer",
  border: "1px solid #8b4f2e",
  background: "#ece5d5",
  color: "#2a2620",
  borderRadius: 8,
  padding: "6px 12px",
  fontFamily: "monospace",
};

interface Vocab {
  sign: string;
  label: string;
  category: string;
  mocap: boolean;
}

function Inner() {
  const q = useSearchParams();
  const cam = CAM[q.get("cam") ?? "front"] ?? CAM.front;
  const speed = parseFloat(q.get("speed") ?? "0.5");
  const fixEnabled = q.get("fix") !== "0"; // face-contact lift on by default; ?fix=0 to compare
  const tParam = q.get("t");
  const freezeFrac = tParam != null ? parseFloat(tParam) : null; // ?t=0..1 freezes at that clip fraction
  const [vocab, setVocab] = useState<Vocab[]>([]);
  const [idx, setIdx] = useState(0);

  useEffect(() => {
    fetch("/3dlex/_vocab_order.json")
      .then((r) => (r.ok ? r.json() : []))
      .then((v: Vocab[]) => {
        setVocab(v);
        const want = q.get("sign");
        const i = want ? v.findIndex((x) => x.sign === want) : v.findIndex((x) => x.mocap);
        setIdx(i >= 0 ? i : 0);
      })
      .catch(() => {});
  }, [q]);

  const cur = vocab[idx];
  const go = (d: number) => {
    if (vocab.length) setIdx((idx + d + vocab.length) % vocab.length);
  };
  // category optgroups, in order
  const grouped: { cat: string; items: { i: number; v: Vocab }[] }[] = [];
  vocab.forEach((v, i) => {
    const g = grouped[grouped.length - 1];
    if (g && g.cat === v.category) g.items.push({ i, v });
    else grouped.push({ cat: v.category, items: [{ i, v }] });
  });

  return (
    <div style={{ width: "100vw", height: "100vh", background: "#f2ede2", overflow: "hidden" }}>
      <div
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          right: 0,
          zIndex: 10,
          display: "flex",
          gap: 12,
          alignItems: "center",
          padding: "10px 16px",
          fontFamily: "monospace",
          color: "#2a2620",
          background: "rgba(242,237,226,.9)",
        }}
      >
        <button onClick={() => go(-1)} style={btn}>
          ← prev
        </button>
        <button onClick={() => go(1)} style={btn}>
          next →
        </button>
        <select
          value={idx}
          onChange={(e) => setIdx(Number(e.target.value))}
          style={{ ...btn, padding: "6px 8px", maxWidth: 260 }}
        >
          {grouped.map((g) => (
            <optgroup key={g.cat} label={g.cat}>
              {g.items.map(({ i, v }) => (
                <option key={v.sign} value={i}>
                  {v.label}
                  {v.mocap ? "" : "  (no mocap)"}
                </option>
              ))}
            </optgroup>
          ))}
        </select>
        {cur && <strong style={{ fontSize: 18 }}>{cur.label}</strong>}
        {cur && (
          <span style={{ opacity: 0.6 }}>
            {idx + 1}/{vocab.length} · {cur.category}
          </span>
        )}
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
        {cur?.mocap && (
          <Suspense fallback={null}>
            <Mocap
              key={cur.sign}
              sign={cur.sign}
              speed={speed}
              fixEnabled={fixEnabled}
              freezeFrac={freezeFrac}
            />
          </Suspense>
        )}
        {cur && !cur.mocap && (
          <Suspense fallback={null}>
            <Mocap key={`rest:${REST_REF}`} sign={REST_REF} speed={speed} rest />
          </Suspense>
        )}
      </Canvas>
      {cur && !cur.mocap && (
        <div
          style={{
            position: "absolute",
            bottom: 24,
            left: 0,
            right: 0,
            textAlign: "center",
            fontFamily: "monospace",
            color: "#8b4f2e",
            opacity: 0.8,
            pointerEvents: "none",
          }}
        >
          resting — no mocap yet for “{cur.label}”
        </div>
      )}
    </div>
  );
}

export default function GlbPreviewPage() {
  return (
    <Suspense fallback={null}>
      <Inner />
    </Suspense>
  );
}
