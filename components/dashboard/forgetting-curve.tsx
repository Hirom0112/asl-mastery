// Mastery decay visualization — the slice-1 "thoughtful extra #1"
// per claude/CLAUDE.md §6. Renders a tiny SVG sparkline of the
// learner's predicted retention curve for one sign, with the next
// review scheduled date marked.
//
// The curve uses the canonical Ebbinghaus exponential decay form
// R(t) = exp(-t / S) where S (memory stability in days) is the
// learner's current intervalDays for the sign. The point is to make
// the spaced-retrieval theory visible — not to predict retention
// numerically. The curve restarts after each successful review.

import type { SignProgress } from "@/lib/scheduler/queries";

const WIDTH = 80;
const HEIGHT = 30;
const HORIZON_DAYS = 30;

export function ForgettingCurve({ sign }: { sign: SignProgress }) {
  if (sign.status === "untouched" || !sign.lastAttemptAt) {
    return (
      <div className="h-[30px] w-[80px] rounded bg-zinc-100 dark:bg-zinc-800" aria-hidden="true" />
    );
  }
  const stability = Math.max(sign.intervalDays || 1, 1); // days
  // Sample 16 points along the horizon.
  const points: string[] = [];
  for (let i = 0; i < 16; i++) {
    const t = (i / 15) * HORIZON_DAYS;
    const r = Math.exp(-t / stability);
    const x = (i / 15) * WIDTH;
    const y = HEIGHT - r * HEIGHT;
    points.push(`${x.toFixed(1)},${y.toFixed(1)}`);
  }

  const now = new Date();
  const nextReview = sign.nextReviewAt ? new Date(sign.nextReviewAt) : null;
  const reviewDays = nextReview
    ? Math.max(0, (nextReview.getTime() - now.getTime()) / (24 * 3600_000))
    : null;
  const reviewX =
    reviewDays !== null ? (Math.min(reviewDays, HORIZON_DAYS) / HORIZON_DAYS) * WIDTH : null;

  const stroke =
    sign.status === "mastered"
      ? "stroke-emerald-500"
      : sign.status === "reviewing"
        ? "stroke-amber-500"
        : "stroke-blue-500";

  return (
    <svg
      width={WIDTH}
      height={HEIGHT}
      className="overflow-visible"
      aria-label={`${sign.displayGloss} retention curve`}
    >
      <polyline points={points.join(" ")} fill="none" strokeWidth={1.5} className={`${stroke}`} />
      {reviewX !== null ? (
        <line
          x1={reviewX}
          x2={reviewX}
          y1={0}
          y2={HEIGHT}
          strokeDasharray="2 2"
          className="stroke-zinc-400 dark:stroke-zinc-500"
        />
      ) : null}
    </svg>
  );
}
