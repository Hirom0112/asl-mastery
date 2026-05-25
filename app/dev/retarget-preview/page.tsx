"use client";

// Dev route: drive the Mixamo "X Bot" mannequin with real 3D-LEX sign MOCAP.
//
// Loads the X Bot (public/avatar/xbot.glb) and a per-sign 3D-LEX GLB
// (public/3dlex/<sign>.glb), builds a remapped AnimationClip via lib/retarget3dlex
// (ABSOLUTE world-orientation copy — see that file's header for why a naive local
// copy or a rest-relative delta both fail here), plays it on the X Bot, and frames
// the upper body the way the practice sign-avatar does so it reads at that bar.
//
// Query params:
//   ?sign=clean|understand|red|help|where     which 3D-LEX clip to retarget
//   ?cam=front|side|threequarter|top|full      camera preset
//   ?mirror=1                                  learner's-reflection mirror (L↔R)
//   ?compensate=0                              A/B: rest-RELATIVE delta instead of
//                                              absolute copy (arms read as T-pose)
//   ?speed=0.5                                 playback timeScale (default 0.5)
//   ?t=N                                       freeze the clip at time N (capture)

import { Canvas, useFrame } from "@react-three/fiber";
import { useGLTF } from "@react-three/drei";
import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import {
  AnimationMixer,
  Color,
  MeshPhysicalMaterial,
  Vector3,
  type AnimationAction,
  type Group,
  type Object3D,
} from "three";
import { useThree } from "@react-three/fiber";
import { clone as cloneSkinned } from "three/examples/jsm/utils/SkeletonUtils.js";
import { buildRetargetedClip } from "@/lib/retarget3dlex";

const XBOT_URL = "/avatar/xbot.glb";
const TARGET_HEIGHT = 1.7;
const CAMERA_DIST = 2.0;

const CAM: Record<string, [number, number, number]> = {
  front: [0.1, 0.0, CAMERA_DIST],
  side: [CAMERA_DIST, 0.1, 0.15], // avatar's right; reveals palm facing + reach depth
  threequarter: [CAMERA_DIST * 0.72, 0.15, CAMERA_DIST * 0.72],
  top: [0.1, CAMERA_DIST * 0.92, 0.55],
  full: [0.2, 0.0, 3.6], // whole body, front (QA orientation check)
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

function Retargeted({
  sign,
  mirror,
  compensate,
  speed,
  freezeT,
  noclip,
}: {
  sign: string;
  mirror: boolean;
  compensate: boolean;
  speed: number;
  freezeT?: number;
  noclip?: boolean;
}) {
  // X Bot is the TARGET rig; the 3D-LEX GLB supplies the source clip + skeleton.
  const { scene: xbotScene } = useGLTF(XBOT_URL);
  const { scene: srcScene, animations: srcAnims } = useGLTF(`/3dlex/${sign}.glb`);

  const invalidate = useThree((s) => s.invalidate);

  // Fresh skinned clone of the X Bot so multiple instances / param changes don't
  // share mutated bone state. Wrapped in an OUTER group we own for world framing,
  // so the retargeted clip is free to drive the (inner) skeleton's local axes
  // however the source authored them.
  const model = useMemo(() => cloneSkinned(xbotScene) as Group, [xbotScene]);
  const outer = useRef<Group>(null);

  const mixerRef = useRef<AnimationMixer | null>(null);
  const actionRef = useRef<AnimationAction | null>(null);

  const framedRef = useRef(false);

  // pearl material once.
  useEffect(() => {
    const mat = pearl();
    model.traverse((o) => {
      const m = o as { isMesh?: boolean; material?: unknown; frustumCulled?: boolean };
      if (m.isMesh) {
        m.material = mat;
        m.frustumCulled = false;
      }
    });
    framedRef.current = false;
  }, [model]);

  // Build + play the retargeted clip.
  useEffect(() => {
    framedRef.current = false;
    if (noclip) return;
    const srcClip = srcAnims[0];
    if (!srcClip) return;

    // Default: WORLD-SPACE rest retarget (the two rigs share hierarchy + names but
    // NOT rest orientation, so a naive local copy twists the body — see CONCLUSION).
    // ?compensate=0 falls back to the naive direct copy for the A/B writeup.
    const { clip, stats } = buildRetargetedClip(srcClip, srcScene, model, {
      compensateRest: compensate,
      mirror,
      // World-orientation neutralization in-skeleton: drop the Armature (Z-up
      // container) and anchor Hips to clip frame 0, so the X Bot stands upright
      // in its own native Mixamo frame and only the sign motion plays on top.
      dropContainer: true,
    });
     
    console.log(
      `[retarget] sign=${sign} mode=${stats.mode} mirror=${mirror} ` +
        `matched=${stats.matched.length}/${stats.totalSourceTracks} ` +
        `droppedNonRot=${stats.droppedNonRotation} ` +
        `unmatched=[${[...new Set(stats.unmatchedSourceTracks)].join(",")}] ` +
        `dur=${clip.duration.toFixed(2)}`,
    );

    const mixer = new AnimationMixer(model);
    mixerRef.current = mixer;
    const action = mixer.clipAction(clip);
    action.timeScale = speed;
    action.play();
    actionRef.current = action;
    invalidate();

    return () => {
      action.stop();
      mixer.stopAllAction();
      mixerRef.current = null;
      actionRef.current = null;
    };
  }, [model, srcScene, srcAnims, sign, compensate, mirror, speed, noclip, invalidate]);

  // advance / freeze the clip, then frame ONCE per load. The retargeted body is
  // POSE-correct but carries the source's world up-axis + facing (it's authored
  // Z-up, facing away). We correct that rigidly on the OUTER group from landmark
  // joints measured at clip frame 0 (clip-invariant → size/pose never jump):
  //   stand  = rotate Hips→Head onto +Y; face = rotate the chest normal onto +Z;
  //   scale  = (TARGET_HEIGHT·0.43)/|Hips→Head| (hip→head ≈ 43% of stature);
  //   anchor = put the head near the top so the legs fall out of frame.
  useFrame((_, delta) => {
    const mixer = mixerRef.current;
    const action = actionRef.current;
    if (!mixer) return;
    if (freezeT != null && action) {
      const dur = action.getClip().duration;
      mixer.setTime(((freezeT % dur) + dur) % dur);
    } else {
      mixer.update(delta);
    }

    const grp = outer.current;
    if (!grp || framedRef.current) return;
    const savedTime = action ? action.time : 0;
    if (action) mixer.setTime(0);
    grp.rotation.set(0, 0, 0);
    grp.position.set(0, 0, 0);
    grp.scale.setScalar(1);
    grp.updateMatrixWorld(true);

    const getNode = (names: string[]) => {
      let n: Object3D | null = null;
      grp.traverse((o) => {
        const k = o.name.replace(/[^a-z0-9]/gi, "").toLowerCase();
        if (names.includes(k)) n = o;
      });
      return n as Object3D | null;
    };
    const get = (names: string[]) => {
      const n = getNode(names);
      return n ? n.getWorldPosition(new Vector3()) : null;
    };
    const headP = get(["mixamorighead", "head"]);
    const hipsP = get(["mixamorighips", "hips"]);
    const neckP = get(["mixamorigneck", "neck"]) ?? get(["mixamorigspine2", "spine2"]);
    const lShP = get(["mixamorigleftshoulder", "leftshoulder"]);
    const rShP = get(["mixamorigrightshoulder", "rightshoulder"]);
    if (!headP || !hipsP || !neckP || !lShP || !rShP) {
      if (action) mixer.setTime(savedTime);
      return;
    }
    const hipHead = headP.clone().sub(hipsP).length();

    // STAND: align the TORSO axis (hips→neck, which is steadier than hips→head —
    // the head/neck tilt during signing) to world +Y, so the figure stands upright
    // regardless of the source up-axis (Z-up here).
    const bodyUp = neckP.clone().sub(hipsP);
    if (bodyUp.lengthSq() < 1e-6) {
      if (action) mixer.setTime(savedTime);
      return;
    }
    bodyUp.normalize();
    grp.quaternion.setFromUnitVectors(bodyUp, new Vector3(0, 1, 0));
    grp.updateMatrixWorld(true);

    // FACE: yaw so the chest normal (L→R shoulder × up) points to the camera (+Z).
    const lS = lShP.clone().applyQuaternion(grp.quaternion);
    const rS = rShP.clone().applyQuaternion(grp.quaternion);
    const across = rS.clone().sub(lS).setY(0).normalize();
    const fwd = across
      .clone()
      .cross(new Vector3(0, 1, 0))
      .normalize();
    if (fwd.lengthSq() > 1e-6) {
      grp.rotateY(-Math.atan2(fwd.x, fwd.z));
      grp.updateMatrixWorld(true);
    }

    // scale so the full stature ≈ TARGET_HEIGHT (hip→head ≈ 43% of stature). Then
    // back off (FRAME_FILL) so the spread signing arms stay inside frame.
    const FRAME_FILL = 0.58;
    const s = ((TARGET_HEIGHT * 0.43) / hipHead) * FRAME_FILL;
    grp.scale.setScalar(s);
    grp.updateMatrixWorld(true);

    // anchor from the POSED torso bones (head, both shoulders, hips) measured at
    // frame 0: center the torso horizontally and put the head ~near the top of the
    // viewport so the chest + arms (signing space) sit centered and the legs drop
    // out of frame. Using the torso (not the spread arms) keeps the centering stable.
    const headW2 = get(["mixamorighead", "head"])!;
    const hipsW2 = get(["mixamorighips", "hips"])!;
    const lSh2 = get(["mixamorigleftshoulder", "leftshoulder"])!;
    const rSh2 = get(["mixamorigrightshoulder", "rightshoulder"])!;
    const cx = (headW2.x + hipsW2.x + lSh2.x + rSh2.x) / 4;
    const cz = (headW2.z + hipsW2.z + lSh2.z + rSh2.z) / 4;
    // place the top of the head near the top of the viewport. At CAMERA_DIST=2.0,
    // fov 30, the look-at plane spans y ∈ ±0.54; HEAD_TOP_Y just under that keeps
    // the crown in frame with the signing space below it.
    const HEAD_TOP_Y = 0.28;
    grp.position.set(-cx, HEAD_TOP_Y - headW2.y, -cz);
    grp.updateMatrixWorld(true);

    if (action) mixer.setTime(savedTime);
    framedRef.current = true;
    invalidate();
  });

  return (
    <group ref={outer}>
      <primitive object={model} />
    </group>
  );
}

function Inner() {
  const q = useSearchParams();
  const sign = q.get("sign") ?? "clean";
  const cam = CAM[q.get("cam") ?? "front"] ?? CAM.front;
  const mirror = q.get("mirror") === "1";
  const compensate = q.get("compensate") !== "0"; // default ON: world-space rest retarget
  const speed = q.get("speed") != null ? parseFloat(q.get("speed")!) : 0.5;
  const tParam = q.get("t");
  const freezeT = tParam != null ? parseFloat(tParam) : undefined;
  const noclip = q.get("noclip") === "1";

  const [s, setS] = useState(sign);

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
        3D-LEX → X Bot retarget: <strong>{s}</strong>{" "}
        <input
          value={s}
          onChange={(e) => setS(e.target.value)}
          style={{ marginLeft: 8, fontFamily: "monospace", padding: "2px 6px" }}
        />
        <span style={{ marginLeft: 12, opacity: 0.7 }}>
          cam={q.get("cam") ?? "front"} mirror={mirror ? 1 : 0} compensate={compensate ? 1 : 0}
        </span>
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
          <Retargeted
            key={`${s}:${mirror}:${compensate}:${speed}:${freezeT ?? ""}:${noclip ? 1 : 0}`}
            sign={s}
            mirror={mirror}
            compensate={compensate}
            speed={speed}
            freezeT={freezeT}
            noclip={noclip}
          />
        </Suspense>
      </Canvas>
    </div>
  );
}

export default function RetargetPreviewPage() {
  return (
    <Suspense fallback={null}>
      <Inner />
    </Suspense>
  );
}

useGLTF.preload(XBOT_URL);
