"use client";

// Practice screen orchestration. Owns the per-attempt state machine:
// idle → recording (delegated to CameraCapture) → result (pass/fail)
// → next-item.
//
// Under ADR 0010 the `detection_failed` state that the ADR 0006
// landmark pipeline raised when MediaPipe missed both hands is gone
// — the 3D CNN classifier has an opinion on every clip.

import { useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState, useTransition } from "react";

import styles from "@/app/practice/practice.module.css";
import {
  CameraCapture,
  type CameraCaptureHandle,
  type CaptureResult,
  type CaptureState,
} from "./camera-capture";
import { SignAvatar } from "./sign-avatar";
import { stubPredict, type ClassifierPrediction } from "@/lib/inference/classifier";
import { predictFromFrames, preloadKeypointModels } from "@/lib/inference/keypoint-predict";
import { flagAttempt, recordAttempt, type NextItem } from "@/lib/scheduler/actions";

type Outcome =
  | { kind: "idle" }
  | {
      kind: "result";
      attemptId: string;
      prediction: ClassifierPrediction;
      capture: CaptureResult;
      reachedMastery: boolean;
    };

interface Props {
  item: NextItem;
  isLeftHanded: boolean;
  activeModelVersionId: string | null;
  activeModelArtifactUrl: string | null;
  activeModelConfigUrl: string | null;
}

export function PracticeRunner({ item, isLeftHanded, activeModelVersionId }: Props) {
  // activeModelArtifactUrl/activeModelConfigUrl (the old 3D-CNN R2 path) are no
  // longer read — the keypoint classifier loads from /public/models.
  const router = useRouter();
  const captureRef = useRef<CameraCaptureHandle>(null);
  const [captureState, setCaptureState] = useState<CaptureState>("initializing");
  const [outcome, setOutcome] = useState<Outcome>({ kind: "idle" });
  const [submitPending, startSubmit] = useTransition();
  // Warm the keypoint ONNX models (detector/landmark/pose + classifier) on
  // mount so the first recorded attempt isn't slowed by a cold model load.
  useEffect(() => {
    void preloadKeypointModels().catch(() => undefined);
  }, []);

  const onRecord = useCallback(async () => {
    const cap = await captureRef.current?.startCapture();
    if (!cap) return;

    // From-scratch keypoint pipeline (v3, 75.8% top1): recorded frames →
    // ONNX detector/landmark/pose → 108D features → sign_classifier_v3.onnx.
    // Models are served from /public/models; falls back to the stub on error.
    let prediction: ClassifierPrediction;
    try {
      prediction = await predictFromFrames(
        cap.frames,
        cap.frameWidth,
        cap.frameHeight,
        item.vocabId,
      );
    } catch (err) {
      console.error("keypoint predict failed; falling back to stub", err);
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
        // ADR 0010: no MediaPipe in the pipeline. The column is kept
        // on the row for historical attempts (v1/v2/v2.1 written
        // under ADR 0006) but new attempts always write false.
        mediapipeDetectionFailed: false,
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
  }, [activeModelVersionId, item.preAttemptHint, item.vocabId]);

  const onNext = useCallback(() => {
    setOutcome({ kind: "idle" });
    router.refresh();
  }, [router]);

  // The from-scratch keypoint classifier ships from /public/models, so
  // recognition is always available regardless of the model_versions DB row.
  const modelOffline = false;

  const coachingMessage =
    captureState === "ready"
      ? "You're in frame. Watch the avatar, then press record."
      : captureState === "initializing"
        ? "Setting up your camera…"
        : captureState === "permission-denied" || captureState === "no-camera"
          ? "Camera unavailable. Check your browser permissions."
          : "Hold still while we read your hands…";

  return (
    <>
      {modelOffline ? (
        <div className={styles.offlineNotice} role="status" aria-live="polite">
          <strong>Model offline.</strong> The classifier is being rebuilt under a stricter
          no-pretrained-components constraint (ADR 0010). Pass/fail uses a deterministic stub until
          v3 ships.
        </div>
      ) : null}
      <div className={styles.stage}>
        <section className={styles.centerColumn}>
          <header className={styles.header}>
            <p className={styles.eyebrow}>Sign this</p>
            <h1 className={styles.gloss}>{item.displayGloss}</h1>
          </header>

          <SignAvatar className={styles.referenceVideo} signId={item.displayGloss.toLowerCase()} />

          <div className={styles.hintCard}>
            <p className={styles.hintLabel}>Before you sign</p>
            <p className={styles.hintBody}>
              Find a well-lit spot and keep your head, hands, and upper body fully in frame. Watch
              the avatar a couple of times, then press record and sign along.
            </p>
          </div>
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

          {outcome.kind === "result" ? (
            <ResultPanel
              item={item}
              outcome={outcome}
              onRetry={() => setOutcome({ kind: "idle" })}
              onNext={onNext}
            />
          ) : null}

          <div className={styles.coachingBubble}>
            <span className={styles.coachingDot} aria-hidden="true" />
            <span>{coachingMessage}</span>
          </div>
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
