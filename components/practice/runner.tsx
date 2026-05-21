"use client";

// Practice screen orchestration. Owns the per-attempt state machine:
// idle → recording (delegated to CameraCapture) → result (pass/fail/
// detection_failed) → next-item.

import { useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState, useTransition } from "react";

import styles from "@/app/practice/practice.module.css";
import {
  CameraCapture,
  type CameraCaptureHandle,
  type CaptureResult,
  type CaptureState,
} from "./camera-capture";
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
          : (item.preAttemptHint ?? "Try matching the reference video's handshape and movement."),
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
  }, [
    activeModelArtifactUrl,
    activeModelVersionId,
    classifierConfig,
    item.preAttemptHint,
    item.vocabId,
  ]);

  const onNext = useCallback(() => {
    setOutcome({ kind: "idle" });
    router.refresh();
  }, [router]);

  const modelOffline = !activeModelArtifactUrl;

  return (
    <>
      {modelOffline ? (
        <div className={styles.offlineNotice} role="status" aria-live="polite">
          <strong>Model offline.</strong> The classifier is being rebuilt under a stricter
          no-pretrained-components constraint (ADR 0010). Pass/fail uses a deterministic stub until
          v3 ships.
        </div>
      ) : null}
      <header>
        <p className={styles.eyebrow}>Sign this</p>
        <h1 className={styles.gloss}>{item.displayGloss}</h1>
      </header>

      <div className={styles.stage}>
        <section className={styles.panel}>
          <p className={styles.panelTitle}>Reference</p>
          {item.referenceVideoUrl ? (
            <video
              src={item.referenceVideoUrl}
              autoPlay
              loop
              muted
              playsInline
              ref={(el) => {
                if (el) el.playbackRate = 0.5;
              }}
              onLoadedMetadata={(e) => {
                (e.currentTarget as HTMLVideoElement).playbackRate = 0.5;
              }}
              className={styles.referenceVideo}
            />
          ) : (
            <div className={`${styles.referenceVideo} ${styles.referenceMissing}`}>
              Reference clip not available
            </div>
          )}

          {item.preAttemptHint ? (
            <div className={styles.hintCard}>
              <p className={styles.hintLabel}>Before you sign</p>
              <p className={styles.hintBody}>{item.preAttemptHint}</p>
            </div>
          ) : null}
        </section>

        <section className={styles.panel}>
          <p className={styles.panelTitle}>Your camera</p>
          <div className={styles.cameraFrame}>
            <CameraCapture
              ref={captureRef}
              onStateChange={setCaptureState}
              isLeftHanded={isLeftHanded}
            />
          </div>

          {outcome.kind === "idle" ? (
            <button
              className={`${styles.btn} ${styles.btnPrimary}`}
              disabled={captureState !== "ready" || submitPending}
              onClick={onRecord}
            >
              {captureState === "ready"
                ? "Record attempt"
                : captureState === "initializing"
                  ? "Setting up camera…"
                  : captureState === "permission-denied" || captureState === "no-camera"
                    ? "Camera unavailable"
                    : "Working…"}
            </button>
          ) : null}

          {outcome.kind === "detection_failed" ? (
            <div className={`${styles.resultPanel} ${styles.resultDetect}`}>
              <h2 className={styles.resultHeadline}>
                We couldn&apos;t see your <em>hands</em>.
              </h2>
              <p className={styles.resultBody}>
                Adjust framing so both hands are inside the green box and the lighting is even, then
                try again.
              </p>
              <div className={styles.resultActions}>
                <button
                  className={`${styles.btn} ${styles.btnOutline}`}
                  onClick={() => setOutcome({ kind: "idle" })}
                >
                  Try again
                </button>
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
    </>
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
      <div className={`${styles.resultPanel} ${styles.resultPass}`}>
        <h2 className={styles.resultHeadline}>
          Nice — <em>{item.displayGloss}</em>.
        </h2>
        <p className={styles.resultMeta}>
          Confidence {(prediction.confidence * 100).toFixed(0)}% (threshold{" "}
          {(prediction.threshold * 100).toFixed(0)}%).
        </p>
        {reachedMastery ? (
          <p className={styles.resultBody}>
            You just mastered this sign. It rotates out of active practice — your next review is
            scheduled.
          </p>
        ) : null}
        <div className={styles.resultActions}>
          <button className={`${styles.btn} ${styles.btnPrimary}`} onClick={onNext}>
            Next sign →
          </button>
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
    <div className={`${styles.resultPanel} ${styles.resultFail}`}>
      <h2 className={styles.resultHeadline}>
        Not quite — <em>try again</em>.
      </h2>
      <p className={styles.resultMeta}>
        Confidence {(prediction.confidence * 100).toFixed(0)}% (threshold{" "}
        {(prediction.threshold * 100).toFixed(0)}%).
      </p>
      <p className={styles.resultBody}>
        Watch the reference once more, then focus on the handshape and the direction of movement.
      </p>
      <div className={styles.resultActions}>
        <button className={`${styles.btn} ${styles.btnOutline}`} onClick={onRetry}>
          Try again
        </button>
        <button className={`${styles.btn} ${styles.btnOutline}`} onClick={onNext}>
          Skip for now
        </button>
        {flagged ? (
          <span className={styles.flagged} role="status">
            Flagged for review — thanks.
          </span>
        ) : (
          <button
            className={`${styles.btn} ${styles.btnGhost}`}
            disabled={flagPending}
            onClick={onFlag}
          >
            I think I did this right
          </button>
        )}
      </div>
      <p className={styles.attribution}>
        Flagging tells us this {item.displayGloss} attempt may be a false negative. We use the
        signal to identify signs that need more training data.
      </p>
    </div>
  );
}
