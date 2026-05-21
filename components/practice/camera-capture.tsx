"use client";

// Camera capture surface for the practice screen.
//
// Owns: getUserMedia stream, mirrored preview, green-box overlay,
// countdown, 2-second 30-fps frame grab, and packing the captured
// frames into the raw RGB video tensor the classifier consumes.
//
// Does NOT own: classification (the parent practice page calls
// classifier.predict / stubPredict on the video tensor we return).
//
// Under ADR 0010 (reversal of ADR 0006, restoring ADR 0001 Path B)
// the MediaPipe Holistic step that ran between capture and
// classification is removed. The output of this component is a
// Float32Array of shape (T=16, H, W, 3), values in [0, 1], laid out
// row-major frame-by-frame so the inference path can wrap it in an
// ort.Tensor of shape (1, T, H, W, 3) with no copy.
//
// Privacy: frames live only in this component. The hidden canvas
// carries the `no-track-canvas` class so PostHog's autocapture
// filter cannot pick them up. Frames never leave the device.

import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState } from "react";

import {
  DEFAULT_VIDEO_HEIGHT,
  DEFAULT_VIDEO_WIDTH,
  VIDEO_CHANNELS,
  VIDEO_TEMPORAL_LENGTH,
  videoTensorLength,
  type VideoTensor,
} from "@/lib/inference/classifier";

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
  videoTensor: VideoTensor;
  inputHeight: number;
  inputWidth: number;
  promptedAtIso: string;
  submittedAtIso: string;
}

export interface CameraCaptureHandle {
  startCapture: () => Promise<CaptureResult | null>;
}

const CAPTURE_MS = 2000;
const FRAME_INTERVAL_MS = CAPTURE_MS / VIDEO_TEMPORAL_LENGTH;

interface Props {
  onStateChange?: (s: CaptureState) => void;
  isLeftHanded?: boolean;
  // Per-version spatial dimensions; defaults to the classifier's
  // DEFAULT_VIDEO_*. v3 → v4 with a different crop size is a prop
  // change, not a code change.
  inputHeight?: number;
  inputWidth?: number;
}

export const CameraCapture = forwardRef<CameraCaptureHandle, Props>(function CameraCapture(
  {
    onStateChange,
    isLeftHanded = false,
    inputHeight = DEFAULT_VIDEO_HEIGHT,
    inputWidth = DEFAULT_VIDEO_WIDTH,
  },
  ref,
) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const captureCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [state, setState] = useState<CaptureState>("initializing");
  const [countdown, setCountdown] = useState<number | null>(null);

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
  }, [setStateAndNotify]);

  const startCapture = useCallback(async (): Promise<CaptureResult | null> => {
    if (state !== "ready") return null;
    const video = videoRef.current;
    const canvas = captureCanvasRef.current;
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
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) return null;
    canvas.width = inputWidth;
    canvas.height = inputHeight;

    // Pre-allocate the (T, H, W, 3) float32 tensor we'll return.
    // Layout: frame 0 row 0 [r,g,b], frame 0 row 0 col 1, ..., frame 1, ...
    const tensorLength = videoTensorLength(inputHeight, inputWidth);
    const tensor = new Float32Array(tensorLength);
    const frameStride = inputHeight * inputWidth * VIDEO_CHANNELS;

    // Grab VIDEO_TEMPORAL_LENGTH frames evenly across CAPTURE_MS.
    const start = performance.now();
    for (let i = 0; i < VIDEO_TEMPORAL_LENGTH; i++) {
      // Draw the current video frame into the model-sized canvas,
      // undoing the visual mirror so the model sees a non-mirrored
      // input. If the user is left-handed, mirror so the model sees
      // the right-handed convention it was trained on.
      ctx.save();
      const shouldMirror = isLeftHanded;
      if (shouldMirror) {
        ctx.translate(canvas.width, 0);
        ctx.scale(-1, 1);
      }
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      ctx.restore();

      // Extract the RGBA pixels and pack RGB into the tensor at this
      // frame's offset, normalizing [0, 255] uint8 → [0, 1] float32.
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
      const frameOffset = i * frameStride;
      // ImageData is RGBA in source order (row-major, top-to-bottom).
      // We write RGB triples into the tensor.
      for (let p = 0, t = frameOffset; p < imageData.length; p += 4, t += 3) {
        tensor[t] = imageData[p] / 255;
        tensor[t + 1] = imageData[p + 1] / 255;
        tensor[t + 2] = imageData[p + 2] / 255;
      }

      // Wait for next frame slot.
      const target = start + (i + 1) * FRAME_INTERVAL_MS;
      const wait = target - performance.now();
      if (wait > 0) await new Promise((r) => setTimeout(r, wait));
    }

    const submittedAt = new Date().toISOString();
    setStateAndNotify("ready");
    return {
      videoTensor: tensor,
      inputHeight,
      inputWidth,
      promptedAtIso: promptedAt,
      submittedAtIso: submittedAt,
    };
  }, [inputHeight, inputWidth, isLeftHanded, setStateAndNotify, state]);

  useImperativeHandle(ref, () => ({ startCapture }), [startCapture]);

  return (
    <div className="relative h-full w-full overflow-hidden bg-[#14222a]">
      <video
        ref={videoRef}
        playsInline
        muted
        // contain (not cover) so the learner's wrists and full upper
        // body stay in frame at any container aspect ratio. Mirrored
        // for the selfie-view convention (CSS scale-x:-1).
        className="h-full w-full scale-x-[-1] transform object-contain"
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
      <canvas ref={captureCanvasRef} className="no-track-canvas hidden" aria-hidden="true" />
    </div>
  );
});
