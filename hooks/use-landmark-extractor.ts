"use client";

// Client-only hook around the MediaPipe HolisticLandmarker singleton.
// Lazy-loads the WASM runtime + .task model on first use; subsequent
// mounts in the same tab share the loaded instance.

import { useCallback, useEffect, useRef, useState } from "react";

import type { HolisticLandmarker } from "@mediapipe/tasks-vision";

import { extractClip, type ClipResult } from "@/lib/mediapipe/extractor";
import { loadHolistic, type LoadOptions } from "@/lib/mediapipe/loader";

export type ExtractorStatus = "idle" | "loading" | "ready" | "error";

export interface UseLandmarkExtractor {
  status: ExtractorStatus;
  error: Error | null;
  // Trigger load eagerly (e.g. on page mount). Otherwise the loader
  // fires automatically on the first extract() call.
  preload: () => Promise<void>;
  extract: (
    frames: {
      image: HTMLVideoElement | HTMLCanvasElement | ImageBitmap;
      timestampMs: number;
    }[],
  ) => Promise<ClipResult>;
}

export function useLandmarkExtractor(options: LoadOptions = {}): UseLandmarkExtractor {
  const [status, setStatus] = useState<ExtractorStatus>("idle");
  const [error, setError] = useState<Error | null>(null);
  const landmarkerRef = useRef<HolisticLandmarker | null>(null);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const ensureLoaded = useCallback(async (): Promise<HolisticLandmarker> => {
    if (landmarkerRef.current) return landmarkerRef.current;
    if (mountedRef.current) setStatus("loading");
    try {
      const lm = await loadHolistic(options);
      landmarkerRef.current = lm;
      if (mountedRef.current) setStatus("ready");
      return lm;
    } catch (err) {
      const e = err instanceof Error ? err : new Error(String(err));
      if (mountedRef.current) {
        setStatus("error");
        setError(e);
      }
      throw e;
    }
  }, [options]);

  const preload = useCallback(async () => {
    await ensureLoaded();
  }, [ensureLoaded]);

  const extract = useCallback(
    async (
      frames: {
        image: HTMLVideoElement | HTMLCanvasElement | ImageBitmap;
        timestampMs: number;
      }[],
    ) => {
      const lm = await ensureLoaded();
      return extractClip(lm, frames);
    },
    [ensureLoaded],
  );

  return { status, error, preload, extract };
}
