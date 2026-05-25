// Retargeting 3D-LEX per-sign MOCAP onto the Mixamo "X Bot" mannequin.
//
// THE PROBLEM. The 3D-LEX GLBs and the X Bot are both Mixamo-named with an
// identical bone hierarchy, but their REST poses differ. The X Bot's rest is a
// clean T-pose where every bone's LOCAL rotation is IDENTITY (the rig shape
// lives entirely in the bone offset translations). The 3D-LEX rig is a
// "characterized" rig whose bones carry NON-identity rest-local rotations (e.g.
// LeftShoulder rest-local ≈ [-0.505,-0.489,0.515,-0.49], Spine ≈ [-0.069,0,0,
// 0.998]) and bakes its WORLD up-axis (Z-up→Y-up) into a separate "Armature"
// container node. So a naive local-rotation copy lays the body on its back AND
// twists every limb, because the source's animated local rotation already folds
// in that bone's rest orientation, which is wrong for the X Bot's identity rest.
//
// THE FIX — WORLD-SPACE REST RETARGET. Copying local rotations only works when
// both rigs share a bone's REST ORIENTATION; here they don't, and a per-bone
// local-rest offset is NOT enough because a bone's true orientation depends on
// its whole parent chain. The correct, general method works in world space:
//
//   srcAnimWorld[b]  = Π(rootΒb) srcAnimLocal      (compose anim locals)
//   srcRestWorld[b]  = Π(rootΒb) srcRestLocal      (compose rest locals)
//   srcDeltaWorld[b] = srcAnimWorld[b] · srcRestWorld[b]⁻¹   (rotation from rest)
//   tgtAnimWorld[b]  = srcDeltaWorld[b] · tgtRestWorld[b]
//   tgtAnimLocal[b]  = tgtParentAnimWorld[b]⁻¹ · tgtAnimWorld[b]
//
// i.e. we measure how far each source bone has rotated AWAY FROM ITS REST in
// world space, then apply that same world rotation to the X Bot's rest and bring
// it back to a parent-relative local rotation the AnimationMixer can play.
//
// Because the X Bot's rest-local is identity for every bone, its rest WORLD
// orientation is identity too, so tgtRestWorld[b] = I and the last two lines
// reduce to tgtAnimWorld[b] = srcDeltaWorld[b] and
// tgtAnimLocal[b] = srcDeltaWorld[parent]⁻¹ · srcDeltaWorld[b]. We implement the
// general form anyway (reading the X Bot rest world) so it stays correct if the
// target rig ever changes. The differing world up-axis is irrelevant: it lives
// in the Armature/Hips, and the delta-from-rest cancels any constant container
// orientation — we additionally drop the Armature track and anchor the Hips to
// clip frame 0 so the X Bot stands upright in its own native Mixamo frame.

import { AnimationClip, Quaternion, QuaternionKeyframeTrack, type Object3D } from "three";

// Normalize a bone name for cross-rig matching: strip the mixamorig prefix,
// lowercase, drop every non-alphanumeric. "mixamorig:LeftHandIndex1" and the
// 3D-LEX "LeftHandIndex1" both collapse to "lefthandindex1". The glTF loader
// also sanitizes ":" out of names, so we must be tolerant of both forms.
export function normBone(name: string): string {
  return name
    .replace(/mixamorig1?:?/i, "")
    .replace(/[^a-z0-9]/gi, "")
    .toLowerCase();
}

function boneMap(root: Object3D): Map<string, Object3D> {
  const m = new Map<string, Object3D>();
  root.traverse((o) => {
    if (o.name) {
      const k = normBone(o.name);
      if (!m.has(k) || (o as { isBone?: boolean }).isBone) m.set(k, o);
    }
  });
  return m;
}

export interface RetargetStats {
  matched: string[];
  unmatchedSourceTracks: string[];
  droppedNonRotation: number;
  totalSourceTracks: number;
  mode: string;
}

// World rest orientation of a node = product of rest-local quats from the scene
// root down to (and including) the node. Computed from the node initial
// quaternions (bind pose), which SkeletonUtils.clone preserves.
function restWorldQuat(node: Object3D): Quaternion {
  const chain: Object3D[] = [];
  let n: Object3D | null = node;
  while (n) {
    chain.push(n);
    n = n.parent;
  }
  const q = new Quaternion();
  for (let i = chain.length - 1; i >= 0; i--) q.multiply(chain[i].quaternion);
  return q;
}

/**
 * Remap a source AnimationClip (3D-LEX "Unreal Take") onto the X Bot skeleton,
 * returning a new clip whose tracks address the X Bot nodes and play correctly.
 *
 * @param opts.compensateRest  use the WORLD-SPACE rest retarget (default true).
 *        Pass false for a NAIVE direct local-rotation copy (for the A/B writeup).
 * @param opts.mirror  mirror L↔R so the avatar is the learner's reflection.
 * @param opts.dropContainer  drop the Armature container + anchor Hips to clip
 *        frame 0 (world-orientation neutralization). Default true.
 */
export function buildRetargetedClip(
  sourceClip: AnimationClip,
  sourceRoot: Object3D,
  targetRoot: Object3D,
  opts: {
    compensateRest?: boolean;
    mirror?: boolean;
    dropContainer?: boolean;
  } = {},
): { clip: AnimationClip; stats: RetargetStats } {
  const compensateRest = opts.compensateRest ?? true;
  const mirror = opts.mirror ?? false;
  const dropContainer = opts.dropContainer ?? true;

  const targetBones = boneMap(targetRoot);
  const sourceBones = boneMap(sourceRoot);

  const stats: RetargetStats = {
    matched: [],
    unmatchedSourceTracks: [],
    droppedNonRotation: 0,
    totalSourceTracks: sourceClip.tracks.length,
    mode: compensateRest ? "absoluteWorldCopy" : "restRelativeDelta",
  };

  // ---- index the source animation: bone-key → { times, values } ----
  type SrcTrack = { times: ArrayLike<number>; values: ArrayLike<number> };
  const animByKey = new Map<string, SrcTrack>();
  let times: number[] | null = null;
  for (const track of sourceClip.tracks) {
    const dot = track.name.lastIndexOf(".");
    const nodeName = track.name.slice(0, dot);
    const prop = track.name.slice(dot + 1);
    if (prop !== "quaternion") {
      stats.droppedNonRotation++;
      continue;
    }
    animByKey.set(normBone(nodeName), { times: track.times, values: track.values });
    if (!times) times = Array.from(track.times);
  }
  if (!times) {
    return {
      clip: new AnimationClip(`retarget_${sourceClip.name}`, sourceClip.duration, []),
      stats,
    };
  }
  const nFrames = times.length;

  // ---- which source bones do we drive? every animated source bone that maps to
  // an X Bot bone (after the mirror name swap). Skip the Armature container. ----
  type Driven = {
    srcKey: string; // source bone key (un-mirrored)
    tgtKey: string; // X Bot bone key (mirrored if requested)
    srcNode: Object3D;
    tgtNode: Object3D;
    srcRestWorldInv: Quaternion;
    tgtRestWorld: Quaternion;
    parentSrcKey: string | null; // nearest ANIMATED+driven ancestor (for world compose)
    aboveRestWorld?: Quaternion; // root only: rest-world ABOVE the root (the Armature
    // container tip) — seeds the root's anim-world so it shares the SAME frame as
    // its rest-world and the delta cancels the constant world tip.
  };

  const drivenByKey = new Map<string, Driven>();
  const driven: Driven[] = [];

  for (const [srcKey, srcNode] of sourceBones) {
    if (srcKey === "armature") continue;
    if (!animByKey.has(srcKey)) continue; // only animated source bones
    let tgtKey = srcKey;
    if (mirror) {
      if (tgtKey.startsWith("left")) tgtKey = "right" + tgtKey.slice(4);
      else if (tgtKey.startsWith("right")) tgtKey = "left" + tgtKey.slice(5);
    }
    const tgtNode = targetBones.get(tgtKey);
    if (!tgtNode) {
      stats.unmatchedSourceTracks.push(srcNode.name);
      continue;
    }
    const d: Driven = {
      srcKey,
      tgtKey,
      srcNode,
      tgtNode,
      srcRestWorldInv: restWorldQuat(srcNode).invert(),
      tgtRestWorld: restWorldQuat(tgtNode),
      parentSrcKey: null,
    };
    drivenByKey.set(srcKey, d);
    driven.push(d);
  }

  // resolve each driven bone's nearest DRIVEN ancestor in the source hierarchy
  // (so we can compose animated world rotations parent→child). Also flag the root.
  for (const d of driven) {
    let p = d.srcNode.parent;
    while (p) {
      const k = normBone(p.name);
      if (drivenByKey.has(k)) {
        d.parentSrcKey = k;
        break;
      }
      p = p.parent;
    }
  }

  // Root (Hips): seed its anim-world with the constant rest-world ABOVE it (the
  // Armature container) so EVERY bone's anim-world and rest-world are composed over
  // the identical full chain — making the per-bone delta-from-rest frame-consistent
  // (this is what gets the spine + arms right). The leftover constant world tip
  // (the source is Z-up + facing away) is then removed RIGIDLY at the render group.
  for (const d of driven) {
    if (d.parentSrcKey === null && d.srcNode.parent) {
      d.aboveRestWorld = restWorldQuat(d.srcNode.parent);
    }
  }

  // sort driven bones parent-before-child so the world compose is single-pass.
  const order: Driven[] = [];
  const seen = new Set<string>();
  const visit = (d: Driven) => {
    if (seen.has(d.srcKey)) return;
    if (d.parentSrcKey && drivenByKey.has(d.parentSrcKey)) visit(drivenByKey.get(d.parentSrcKey)!);
    seen.add(d.srcKey);
    order.push(d);
  };
  for (const d of driven) visit(d);

  // scratch
  const _animLocal = new Quaternion();
  const _animWorld = new Quaternion();
  const _deltaWorld = new Quaternion();
  const _tgtAnimWorld = new Quaternion();
  const _tgtParentWorldInv = new Quaternion();
  const _out = new Quaternion();

  // per-frame source ANIM-WORLD and the resulting X Bot ANIM-WORLD, keyed by bone.
  const srcAnimWorld = new Map<string, Quaternion>();
  const tgtAnimWorld = new Map<string, Quaternion>();
  const outValues = new Map<string, Float32Array>();
  for (const d of driven) outValues.set(d.srcKey, new Float32Array(nFrames * 4));

  const readAnimLocal = (key: string, frame: number, out: Quaternion) => {
    const t = animByKey.get(key);
    if (t) {
      const o = frame * 4;
      out.set(t.values[o], t.values[o + 1], t.values[o + 2], t.values[o + 3]);
    } else {
      // un-animated driven bone → constant rest-local (shouldn't happen: driven
      // requires an animated track, but keep it safe).
      out.copy(drivenByKey.get(key)!.srcNode.quaternion);
    }
  };

  for (let f = 0; f < nFrames; f++) {
    srcAnimWorld.clear();
    tgtAnimWorld.clear();
    for (const d of order) {
      // source animated WORLD = parent's source animated world · this anim-local.
      // The root is seeded with the constant above-root rest (the Armature tip) so
      // it matches its rest-world frame; the delta then cancels that tip.
      readAnimLocal(d.srcKey, f, _animLocal);
      _animWorld.copy(_animLocal);
      if (d.parentSrcKey && srcAnimWorld.has(d.parentSrcKey)) {
        _animWorld.premultiply(srcAnimWorld.get(d.parentSrcKey)!);
      } else if (d.aboveRestWorld) {
        _animWorld.premultiply(d.aboveRestWorld);
      }
      srcAnimWorld.set(d.srcKey, _animWorld.clone());

      if (compensateRest) {
        // ABSOLUTE world-orientation copy: the X Bot bone adopts the SAME world
        // orientation as the source bone. Because both rigs share the hierarchy +
        // bone names, this reproduces the source pose EXACTLY regardless of the
        // rest-pose difference (the source arm rest is A-pose-ish, the X Bot's is a
        // T-pose; a rest-RELATIVE delta would only preserve motion, leaving the X
        // Bot arms stuck near T-pose — see CONCLUSION). The constant world up-axis
        // tip is removed rigidly at the render group.
        _tgtAnimWorld.copy(_animWorld);
      } else {
        // A/B: rest-RELATIVE delta (preserves motion-from-rest, not absolute pose).
        // Kept to demonstrate why absolute-copy is required; arms read as T-pose.
        _deltaWorld.copy(_animWorld).multiply(d.srcRestWorldInv);
        _tgtAnimWorld.copy(_deltaWorld).multiply(d.tgtRestWorld);
      }
      tgtAnimWorld.set(d.srcKey, _tgtAnimWorld.clone());

      // target LOCAL = target-parent-anim-world⁻¹ · target-anim-world.
      if (d.parentSrcKey && tgtAnimWorld.has(d.parentSrcKey)) {
        _tgtParentWorldInv.copy(tgtAnimWorld.get(d.parentSrcKey)!).invert();
        _out.copy(_tgtParentWorldInv).multiply(_tgtAnimWorld);
      } else {
        _out.copy(_tgtAnimWorld);
      }

      if (mirror) {
        // reflect across the sagittal (YZ) plane → mirror image; the L↔R bone
        // swap (tgtKey) carries the chirality so handshapes are preserved.
        _out.set(_out.x, -_out.y, -_out.z, _out.w);
      }

      const arr = outValues.get(d.srcKey)!;
      const o = f * 4;
      arr[o] = _out.x;
      arr[o + 1] = _out.y;
      arr[o + 2] = _out.z;
      arr[o + 3] = _out.w;
    }
  }

  // ---- emit one quaternion track per driven bone, addressing the X Bot node ----
  const outTracks: QuaternionKeyframeTrack[] = [];
  for (const d of driven) {
    outTracks.push(
      new QuaternionKeyframeTrack(
        `${d.tgtNode.name}.quaternion`,
        times,
        Array.from(outValues.get(d.srcKey)!),
      ),
    );
    stats.matched.push(d.tgtNode.name);
  }
  if (dropContainer) stats.unmatchedSourceTracks.push("Armature(dropped:container)");

  const clip = new AnimationClip(`retarget_${sourceClip.name}`, sourceClip.duration, outTracks);
  return { clip, stats };
}
