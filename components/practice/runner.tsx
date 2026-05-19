"use client";

// Practice screen orchestration. Owns the per-attempt state machine:
// idle → recording (delegated to CameraCapture) → result (pass/fail/
// detection_failed) → next-item.

import { useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState, useTransition } from "react";

import {
  CameraCapture,
  type CameraCaptureHandle,
  type CaptureResult,
  type CaptureState,
} from "./camera-capture";
import { Button } from "@/components/ui/button";
import {
  predict,
  stubPredict,
  type ClassifierConfig,
  type ClassifierPrediction,
} from "@/lib/inference/classifier";
import { flagAttempt, recordAttempt, type NextItem } from "@/lib/scheduler/actions";

type Outcome =
  | { kind: "idle" }
  | {
      kind: "result";
      attemptId: string;
      prediction: ClassifierPrediction;
      capture: CaptureResult;
      reachedMastery: boolean;
    }
  | { kind: "detection_failed" };

interface Props {
  item: NextItem;
  isLeftHanded: boolean;
  activeModelVersionId: string | null;
  activeModelArtifactUrl: string | null;
  activeModelConfigUrl: string | null;
}

export function PracticeRunner({
  item,
  isLeftHanded,
  activeModelVersionId,
  activeModelArtifactUrl,
  activeModelConfigUrl,
}: Props) {
  const router = useRouter();
  const captureRef = useRef<CameraCaptureHandle>(null);
  const [captureState, setCaptureState] = useState<CaptureState>("initializing");
  const [outcome, setOutcome] = useState<Outcome>({ kind: "idle" });
  const [submitPending, startSubmit] = useTransition();
  const [classifierConfig, setClassifierConfig] = useState<ClassifierConfig | null>(null);

  // Lazy-fetch the classifier config from R2 once on mount (it's a
  // tiny JSON file). The ONNX itself is loaded by predict() on first
  // call and cached in the InferenceSession singleton.
  useEffect(() => {
    if (!activeModelConfigUrl) return;
    let cancelled = false;
    (async () => {
      try {
        const r = await fetch(activeModelConfigUrl);
        if (!r.ok) throw new Error(`config fetch ${r.status}`);
        const cfg = await r.json();
        if (!cancelled)
          setClassifierConfig({
            classes: cfg.classes,
            temperature: cfg.temperature ?? 1.0,
            perSignThresholds: cfg.per_sign_thresholds ?? {},
          });
      } catch (err) {
        console.error("classifier config fetch failed", err);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [activeModelConfigUrl]);

  const onRecord = useCallback(async () => {
    const cap = await captureRef.current?.startCapture();
    if (!cap) return;

    if (cap.detectionFailed) {
      setOutcome({ kind: "detection_failed" });
      return;
    }

    const useRealModel = activeModelArtifactUrl && classifierConfig;
    let prediction: ClassifierPrediction;
    try {
      prediction = useRealModel
        ? await predict(cap.keypoints, item.vocabId, activeModelArtifactUrl, classifierConfig)
        : stubPredict(item.vocabId);
    } catch (err) {
      console.error("classifier predict failed; falling back to stub", err);
      prediction = stubPredict(item.vocabId);
    }

    startSubmit(async () => {
      const result = await recordAttempt({
        vocabId: item.vocabId,
        promptedAtIso: cap.promptedAtIso,
        submittedAtIso: cap.submittedAtIso,
        predictedClassId: prediction.predictedClassId,
        confidence: prediction.confidence,
        passed: prediction.passed,
        hintShown: prediction.passed
          ? null
          : "Try matching the reference video's handshape and movement.",
        hintSource: prediction.passed ? "none" : "generic_failure",
        mediapipeDetectionFailed: cap.detectionFailed,
        modelVersionId: activeModelVersionId,
      });
      setOutcome({
        kind: "result",
        attemptId: result.attemptId,
        prediction,
        capture: cap,
        reachedMastery: result.reachedMastery,
      });
    });
  }, [activeModelArtifactUrl, activeModelVersionId, classifierConfig, item.vocabId]);

  const onNext = useCallback(() => {
    setOutcome({ kind: "idle" });
    router.refresh();
  }, [router]);

  return (
    <div className="grid gap-6 lg:grid-cols-[1.1fr_1fr]">
      <section className="flex flex-col gap-4">
        <header className="flex flex-col gap-1">
          <p className="text-xs font-medium tracking-widest text-zinc-500 uppercase dark:text-zinc-400">
            Sign this
          </p>
          <h1 className="text-4xl font-semibold tracking-tight text-zinc-950 dark:text-zinc-50">
            {item.displayGloss}
          </h1>
          <p className="text-sm text-zinc-600 dark:text-zinc-300">{item.category}</p>
        </header>

        {item.referenceVideoUrl ? (
          <video
            src={item.referenceVideoUrl}
            autoPlay
            loop
            muted
            playsInline
            // 30fps clips trimmed to ~2s play uncomfortably fast at 1×.
            // 0.5× makes the sign legible; the browser persists this
            // across loop restarts.
            ref={(el) => {
              if (el) el.playbackRate = 0.5;
            }}
            onLoadedMetadata={(e) => {
              (e.currentTarget as HTMLVideoElement).playbackRate = 0.5;
            }}
            className="aspect-video w-full rounded-2xl border border-zinc-200 bg-black object-cover dark:border-zinc-800"
          />
        ) : (
          <div className="flex aspect-video w-full items-center justify-center rounded-2xl border border-dashed border-zinc-300 text-sm text-zinc-500 dark:border-zinc-700 dark:text-zinc-400">
            Reference video pending Phase 3d ingestion.
          </div>
        )}

        {item.preAttemptHint ? (
          <div className="rounded-lg border border-zinc-200 bg-white p-4 text-sm text-zinc-700 dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-200">
            <p className="font-medium text-zinc-900 dark:text-zinc-100">Before you sign</p>
            <p className="mt-1">{item.preAttemptHint}</p>
          </div>
        ) : null}
      </section>

      <section className="flex flex-col gap-4">
        <CameraCapture
          ref={captureRef}
          onStateChange={setCaptureState}
          isLeftHanded={isLeftHanded}
        />

        {outcome.kind === "idle" ? (
          <Button
            size="lg"
            disabled={captureState !== "ready" || submitPending}
            onClick={onRecord}
            className="w-full"
          >
            {captureState === "ready"
              ? "Record attempt"
              : captureState === "initializing"
                ? "Setting up camera…"
                : captureState === "permission-denied" || captureState === "no-camera"
                  ? "Camera unavailable"
                  : "Working…"}
          </Button>
        ) : null}

        {outcome.kind === "detection_failed" ? (
          <div className="rounded-lg border border-amber-300 bg-amber-50 p-4 text-sm text-amber-900 dark:border-amber-700 dark:bg-amber-950 dark:text-amber-100">
            <p className="font-medium">We couldn&apos;t see your hands clearly.</p>
            <p className="mt-1">
              Adjust framing so both hands are inside the green box and the lighting is even, then
              try again.
            </p>
            <div className="mt-3 flex gap-2">
              <Button onClick={() => setOutcome({ kind: "idle" })}>Try again</Button>
            </div>
          </div>
        ) : null}

        {outcome.kind === "result" ? (
          <ResultPanel
            item={item}
            outcome={outcome}
            onRetry={() => setOutcome({ kind: "idle" })}
            onNext={onNext}
          />
        ) : null}
      </section>
    </div>
  );
}

function ResultPanel({
  item,
  outcome,
  onRetry,
  onNext,
}: {
  item: NextItem;
  outcome: Extract<Outcome, { kind: "result" }>;
  onRetry: () => void;
  onNext: () => void;
}) {
  const { prediction, reachedMastery } = outcome;
  if (prediction.passed) {
    return (
      <div className="rounded-lg border border-emerald-300 bg-emerald-50 p-4 text-sm text-emerald-900 dark:border-emerald-700 dark:bg-emerald-950 dark:text-emerald-100">
        <p className="text-base font-medium">Nice — {item.displayGloss}.</p>
        <p className="mt-1 text-xs">
          Confidence {(prediction.confidence * 100).toFixed(0)}% (threshold{" "}
          {(prediction.threshold * 100).toFixed(0)}%).
        </p>
        {reachedMastery ? (
          <p className="mt-2 font-medium">
            You just mastered this sign. It rotates out of active practice — your next review is
            scheduled.
          </p>
        ) : null}
        <div className="mt-3 flex gap-2">
          <Button onClick={onNext}>Next sign</Button>
        </div>
      </div>
    );
  }

  return <FailPanel item={item} outcome={outcome} onRetry={onRetry} onNext={onNext} />;
}

function FailPanel({
  item,
  outcome,
  onRetry,
  onNext,
}: {
  item: NextItem;
  outcome: Extract<Outcome, { kind: "result" }>;
  onRetry: () => void;
  onNext: () => void;
}) {
  const { prediction, attemptId } = outcome;
  const [flagged, setFlagged] = useState(false);
  const [flagPending, startFlag] = useTransition();

  function onFlag() {
    if (flagged || flagPending) return;
    startFlag(async () => {
      const r = await flagAttempt(attemptId);
      if (!r.error) setFlagged(true);
    });
  }

  return (
    <div className="rounded-lg border border-zinc-300 bg-white p-4 text-sm text-zinc-700 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-200">
      <p className="text-base font-medium text-zinc-900 dark:text-zinc-100">
        Not quite — try again.
      </p>
      <p className="mt-1 text-xs">
        Confidence {(prediction.confidence * 100).toFixed(0)}% (threshold{" "}
        {(prediction.threshold * 100).toFixed(0)}%).
      </p>
      <p className="mt-2">
        Watch the reference once more, then focus on the handshape and the direction of movement.
      </p>
      <div className="mt-3 flex flex-wrap gap-2">
        <Button variant="outline" onClick={onRetry}>
          Try again
        </Button>
        <Button onClick={onNext}>Skip for now</Button>
        {flagged ? (
          <span className="text-xs text-zinc-500 dark:text-zinc-400" role="status">
            Flagged for review — thanks.
          </span>
        ) : (
          <Button variant="ghost" size="sm" disabled={flagPending} onClick={onFlag}>
            I think I did this right
          </Button>
        )}
      </div>
      <p className="mt-2 text-[10px] text-zinc-500 dark:text-zinc-400">
        Flagging tells us this {item.displayGloss} attempt may be a false negative. We use the
        signal to identify signs that need more training data.
      </p>
    </div>
  );
}
