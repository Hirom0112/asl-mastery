"use client";

// Practice screen orchestration. Owns the per-attempt state machine:
// idle → recording (delegated to CameraCapture) → result (pass/fail)
// → next-item.
//
// The recognition pipeline returns a pass/fail prediction on every
// clip, so there is no separate "detection failed" state.

import { useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState, useTransition } from "react";

import styles from "@/app/practice/practice.module.css";
import {
  CameraCapture,
  type CameraCaptureHandle,
  type CaptureResult,
  type CaptureState,
} from "./camera-capture";
import { ReferenceStage } from "./reference-stage";
import { RealSignerPeek } from "./real-signer-peek";
import { stubPredict, type ClassifierPrediction } from "@/lib/inference/classifier";
import { predictFromFrames, preloadKeypointModels } from "@/lib/inference/keypoint-predict";
import { flagAttempt, recordAttempt, type NextItem } from "@/lib/scheduler/actions";
import { prefetch, speak } from "@/lib/tts";

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
  nextSignId: string | null;
  activeModelVersionId: string | null;
  activeModelArtifactUrl: string | null;
  activeModelConfigUrl: string | null;
}

const SKIP_AFTER_FAILS = 3; // only offer "Skip for now" after this many misses

export function PracticeRunner({ item, isLeftHanded, nextSignId, activeModelVersionId }: Props) {
  // activeModelArtifactUrl/activeModelConfigUrl (the old 3D-CNN R2 path) are no
  // longer read — the keypoint classifier loads from /public/models.
  const router = useRouter();
  const captureRef = useRef<CameraCaptureHandle>(null);
  const [captureState, setCaptureState] = useState<CaptureState>("initializing");
  const [outcome, setOutcome] = useState<Outcome>({ kind: "idle" });
  const [failCount, setFailCount] = useState(0);
  const [submitPending, startSubmit] = useTransition();

  // Reset per-sign state when the sign changes — render-time pattern (avoids a
  // setState-in-effect cascade; React re-renders immediately without committing).
  // Clearing `outcome` here is what un-sticks "Skip for now" / sidebar jumps: the
  // component is not remounted on navigation, so a stale result panel (and its
  // attemptId, from the PREVIOUS sign) would otherwise keep showing and hide the
  // Record button on the new sign.
  const [trackedSign, setTrackedSign] = useState(item.vocabId);
  if (item.vocabId !== trackedSign) {
    setTrackedSign(item.vocabId);
    setFailCount(0);
    setOutcome({ kind: "idle" });
  }
  // Warm the keypoint ONNX models (detector/landmark/pose + classifier) on
  // mount so the first recorded attempt isn't slowed by a cold model load.
  useEffect(() => {
    void preloadKeypointModels().catch(() => undefined);
    // Warm the TTS phrases so they play instantly (no synth latency on trigger).
    void prefetch("welcome");
    void prefetch("pass");
  }, []);

  // Speak a one-time welcome (OpenAI fable voice) when the camera becomes
  // ready. Fails silently if TTS is disabled or autoplay is blocked.
  const welcomedRef = useRef(false);
  useEffect(() => {
    if (captureState === "ready" && !welcomedRef.current) {
      welcomedRef.current = true;
      void speak("welcome");
    }
  }, [captureState]);

  const onRecord = useCallback(async () => {
    const cap = await captureRef.current?.startCapture();
    if (!cap) return;

    // From-scratch keypoint pipeline (v4 face-anchored, 81.6% top1): recorded
    // frames → ONNX detect + landmark + pose → 108D features → sign_classifier_v4.onnx.
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

    setFailCount((c) => (prediction.passed ? 0 : c + 1));
    if (prediction.passed) void speak("pass"); // congrats voice (matches the avatar box)

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

  // "Skip for now" (offered only after SKIP_AFTER_FAILS misses): advance to the
  // next sign by rank, bypassing the progression lock for this explicit skip.
  const onSkip = useCallback(() => {
    setOutcome({ kind: "idle" }); // clear the result panel immediately (don't wait for navigation)
    if (nextSignId) router.push(`/practice?sign=${nextSignId}&skip=1`);
    else router.refresh();
  }, [nextSignId, router]);

  const coachingMessage =
    captureState === "ready"
      ? "You're in frame. Watch the avatar, then press record."
      : captureState === "initializing"
        ? "Setting up your camera…"
        : captureState === "permission-denied" || captureState === "no-camera"
          ? "Camera unavailable. Check your browser permissions."
          : "Hold still while we read your hands…";

  // Pass feedback: on a pass we glow the camera green, replay the learner's own
  // captured frames ("here's how you did it"), and swap the avatar hint card to
  // a congratulations message.
  const result = outcome.kind === "result" ? outcome : null;
  const passed = !!result?.prediction.passed;
  const replayFrames = passed ? (result?.capture.frames ?? null) : null;

  return (
    <div className={styles.stage}>
      <section className={styles.centerColumn}>
        <header className={styles.header}>
          <p className={styles.eyebrow}>Sign this</p>
          <h1 className={styles.gloss}>{item.displayGloss}</h1>
        </header>

        <ReferenceStage
          className={styles.referenceVideo}
          signId={item.displayGloss.toLowerCase()}
        />

        <div className={styles.hintCard}>
          {passed ? (
            <>
              <p className={styles.hintLabel}>✓ Nice work</p>
              <p className={styles.hintBody}>
                Congratulations, you got it right! Let’s rewatch how you did it.
              </p>
            </>
          ) : (
            <>
              <p className={styles.hintLabel}>Before you sign</p>
              <p className={styles.hintBody}>
                Find a well-lit spot and keep your head, hands, and upper body fully in frame. Watch
                the avatar a couple of times, then press record and sign along.
              </p>
            </>
          )}
        </div>
      </section>

      <section className={styles.panel}>
        <p className={styles.panelTitle}>Your camera</p>
        <div className={`${styles.cameraFrame} ${passed ? styles.cameraFramePass : ""}`}>
          <CameraCapture
            ref={captureRef}
            onStateChange={setCaptureState}
            isLeftHanded={isLeftHanded}
            replayFrames={replayFrames}
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
            onSkip={onSkip}
            canSkip={failCount >= SKIP_AFTER_FAILS}
          />
        ) : null}

        <div className={styles.coachingBubble}>
          <span className={styles.coachingDot} aria-hidden="true" />
          <span style={{ flex: 1 }}>{coachingMessage}</span>
          <RealSignerPeek signId={item.displayGloss.toLowerCase()} />
        </div>
      </section>
    </div>
  );
}

function ResultPanel({
  item,
  outcome,
  onRetry,
  onNext,
  onSkip,
  canSkip,
}: {
  item: NextItem;
  outcome: Extract<Outcome, { kind: "result" }>;
  onRetry: () => void;
  onNext: () => void;
  onSkip: () => void;
  canSkip: boolean;
}) {
  const { prediction, reachedMastery } = outcome;
  if (prediction.passed) {
    return (
      <div className={`${styles.resultPanel} ${styles.resultPass}`}>
        <h2 className={styles.resultHeadline}>
          Nice — <em>{item.displayGloss}</em>.
        </h2>
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

  return (
    <FailPanel item={item} outcome={outcome} onRetry={onRetry} onSkip={onSkip} canSkip={canSkip} />
  );
}

function FailPanel({
  item,
  outcome,
  onRetry,
  onSkip,
  canSkip,
}: {
  item: NextItem;
  outcome: Extract<Outcome, { kind: "result" }>;
  onRetry: () => void;
  onSkip: () => void;
  canSkip: boolean;
}) {
  const { attemptId } = outcome;
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
      <p className={styles.resultBody}>
        Watch the reference once more, then focus on the handshape and the direction of movement.
      </p>
      <div className={styles.resultActions}>
        <button className={`${styles.btn} ${styles.btnOutline}`} onClick={onRetry}>
          Try again
        </button>
        {canSkip ? (
          <button className={`${styles.btn} ${styles.btnOutline}`} onClick={onSkip}>
            Skip for now
          </button>
        ) : null}
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
