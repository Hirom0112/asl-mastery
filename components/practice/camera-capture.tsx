"use client";

// Camera capture surface for the practice screen.
//
// Owns: getUserMedia stream, mirrored preview, green-box overlay,
// countdown, 2-second 30-fps frame grab into a hidden offscreen
// canvas, and the call into the MediaPipe extractor.
//
// Does NOT own: classification (the parent practice page calls
// classifier.predict / stubPredict on the keypoint tensor we return).
//
// Privacy: frames live only in this component. The hidden canvas
// carries the `no-track-canvas` class so PostHog's autocapture
// filter (Phase 7) cannot pick them up.

import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState } from "react";

import { TEMPORAL_LENGTH, type ClipKeypoints } from "@/lib/keypoints";
import { useLandmarkExtractor } from "@/hooks/use-landmark-extractor";

export type CaptureState =
  | "initializing"
  | "permission-denied"
  | "no-camera"
  | "ready"
  | "countdown"
  | "recording"
  | "extracting"
  | "model-load-error";

export interface CaptureResult {
  keypoints: ClipKeypoints;
  detectionFailed: boolean;
  handDetectionFailureRate: number;
  promptedAtIso: string;
  submittedAtIso: string;
}

export interface CameraCaptureHandle {
  startCapture: () => Promise<CaptureResult | null>;
}

const CAPTURE_MS = 2000;
const FRAME_INTERVAL_MS = CAPTURE_MS / TEMPORAL_LENGTH;

interface Props {
  onStateChange?: (s: CaptureState) => void;
  isLeftHanded?: boolean;
}

export const CameraCapture = forwardRef<CameraCaptureHandle, Props>(function CameraCapture(
  { onStateChange, isLeftHanded = false },
  ref,
) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [state, setState] = useState<CaptureState>("initializing");
  const [countdown, setCountdown] = useState<number | null>(null);

  const extractor = useLandmarkExtractor();

  const setStateAndNotify = useCallback(
    (s: CaptureState) => {
      setState(s);
      onStateChange?.(s);
    },
    [onStateChange],
  );

  useEffect(() => {
    let cancelled = false;
    async function init() {
      if (typeof navigator === "undefined" || !navigator.mediaDevices) {
        setStateAndNotify("no-camera");
        return;
      }
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 720, height: 720, frameRate: 30 },
          audio: false,
        });
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play().catch(() => undefined);
        }
        setStateAndNotify("ready");
        // Preload MediaPipe in the background so the first attempt is snappy.
        extractor.preload().catch(() => undefined);
      } catch (err) {
        const e = err as DOMException;
        if (e?.name === "NotAllowedError" || e?.name === "SecurityError") {
          setStateAndNotify("permission-denied");
        } else if (e?.name === "NotFoundError") {
          setStateAndNotify("no-camera");
        } else {
          setStateAndNotify("no-camera");
        }
      }
    }
    init();
    return () => {
      cancelled = true;
      const stream = streamRef.current;
      if (stream) stream.getTracks().forEach((t) => t.stop());
    };
    // We intentionally do not depend on `extractor` because its identity changes
    // each render and would re-init the camera; preload is best-effort.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [setStateAndNotify]);

  const startCapture = useCallback(async (): Promise<CaptureResult | null> => {
    if (state !== "ready") return null;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return null;

    // 3-2-1 countdown.
    setStateAndNotify("countdown");
    for (let n = 3; n > 0; n--) {
      setCountdown(n);
      await new Promise((r) => setTimeout(r, 700));
    }
    setCountdown(null);

    setStateAndNotify("recording");
    const promptedAt = new Date().toISOString();
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    canvas.width = 256;
    canvas.height = 256;

    // Grab TEMPORAL_LENGTH frames evenly across CAPTURE_MS.
    const frames: { image: HTMLCanvasElement; timestampMs: number }[] = [];
    const start = performance.now();
    for (let i = 0; i < TEMPORAL_LENGTH; i++) {
      // Draw the current video frame, undoing the visual mirror so MediaPipe sees a
      // non-mirrored input. If the user is left-handed, mirror so the model sees the
      // right-handed convention it was trained on.
      ctx.save();
      const shouldMirror = isLeftHanded;
      if (shouldMirror) {
        ctx.translate(canvas.width, 0);
        ctx.scale(-1, 1);
      }
      // Source video is also mirrored visually via CSS; we draw from raw video which is
      // un-mirrored, so the canvas is naturally un-mirrored unless we flip for lefties.
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      ctx.restore();

      // Clone the canvas at this moment because MediaPipe will read it later.
      const snapshot = document.createElement("canvas");
      snapshot.width = canvas.width;
      snapshot.height = canvas.height;
      snapshot.getContext("2d")?.drawImage(canvas, 0, 0);
      frames.push({ image: snapshot, timestampMs: Math.round(performance.now() - start) });

      // Wait for next frame slot.
      const target = start + (i + 1) * FRAME_INTERVAL_MS;
      const wait = target - performance.now();
      if (wait > 0) await new Promise((r) => setTimeout(r, wait));
    }

    setStateAndNotify("extracting");
    try {
      const clip = await extractor.extract(frames);
      const submittedAt = new Date().toISOString();
      setStateAndNotify("ready");
      return {
        keypoints: clip.keypoints,
        detectionFailed: clip.detectionFailed,
        handDetectionFailureRate: clip.handDetectionFailureRate,
        promptedAtIso: promptedAt,
        submittedAtIso: submittedAt,
      };
    } catch (err) {
      console.error("extractor failed", err);
      setStateAndNotify("model-load-error");
      return null;
    }
  }, [extractor, isLeftHanded, setStateAndNotify, state]);

  useImperativeHandle(ref, () => ({ startCapture }), [startCapture]);

  return (
    <div className="relative h-full w-full overflow-hidden bg-zinc-900">
      <video
        ref={videoRef}
        playsInline
        muted
        className="h-full w-full scale-x-[-1] transform object-cover"
      />
      {/* Green-box framing overlay. */}
      <div className="pointer-events-none absolute inset-[12%] rounded-xl border-2 border-emerald-400/80" />
      {state === "countdown" && countdown !== null ? (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
          <span className="text-7xl font-semibold text-white drop-shadow-md">{countdown}</span>
        </div>
      ) : null}
      {state === "recording" ? (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
          <span className="rounded-full bg-red-500/90 px-3 py-1 text-xs font-medium text-white">
            Recording…
          </span>
        </div>
      ) : null}
      {state === "extracting" ? (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
          <span className="rounded-full bg-zinc-900/80 px-3 py-1 text-xs font-medium text-white">
            Analyzing…
          </span>
        </div>
      ) : null}
      {state === "permission-denied" || state === "no-camera" ? (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 p-6 text-center text-zinc-100">
          <p className="text-sm font-medium">
            {state === "permission-denied"
              ? "Camera permission is required."
              : "No camera detected."}
          </p>
          <p className="text-xs text-zinc-300">
            Allow camera access in your browser settings and reload.
          </p>
        </div>
      ) : null}
      {/* Hidden capture canvas. The no-track-canvas class is the
          contract with PostHog Replay (per ARCHITECTURE §8). */}
      <canvas ref={canvasRef} className="no-track-canvas hidden" aria-hidden="true" />
    </div>
  );
});
