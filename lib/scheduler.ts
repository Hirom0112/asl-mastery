// Modified SM-2 scheduler per docs/ARCHITECTURE.md §4.
//
// Pure functions only: deterministic, side-effect-free, easy to test.
// Server actions in lib/scheduler/actions.ts persist the state
// transitions to Postgres and wrap the pure logic below.

import type { Database } from "@/lib/db/database.types";

export type MasteryStatus = "untouched" | "learning" | "reviewing" | "mastered";

export interface MasteryState {
  status: MasteryStatus;
  ease: number;
  intervalDays: number;
  nextReviewAt: Date | null;
  consecutivePasses: number;
  totalAttempts: number;
  totalPasses: number;
  lastAttemptAt: Date | null;
}

export const DEFAULT_MASTERY: MasteryState = {
  status: "untouched",
  ease: 1.3,
  intervalDays: 0,
  nextReviewAt: null,
  consecutivePasses: 0,
  totalAttempts: 0,
  totalPasses: 0,
  lastAttemptAt: null,
};

// Modified-SM-2 tunables per ARCHITECTURE §4. Centralized so a
// Phase 4 tuning pass can adjust them with a single PR.
export const SCHEDULER_TUNABLES = {
  EASE_MIN: 0.13,
  EASE_MAX: 2.5,
  EASE_PASS_BONUS: 0.1,
  EASE_FAIL_PENALTY: 0.2,
  FAIL_INTERVAL_DECAY: 0.3,
  LEARNING_TO_REVIEWING_PASSES: 2, // consecutive in-session
  MASTERY_INTERVAL_DAYS: 7,
  MASTERY_CONSECUTIVE_PASSES: 3,
  NEW_ITEM_PASS_RATE_FLOOR: 0.7,
  ROLLING_WINDOW: 10,
} as const;

function clamp(x: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, x));
}

function addDays(d: Date, days: number): Date {
  const out = new Date(d.getTime());
  out.setDate(out.getDate() + days);
  return out;
}

export function applyPass(state: MasteryState, now: Date, inSession: boolean = true): MasteryState {
  const t = SCHEDULER_TUNABLES;
  const next: MasteryState = { ...state };
  next.totalAttempts += 1;
  next.totalPasses += 1;
  next.lastAttemptAt = now;

  if (state.status === "untouched") {
    next.status = "learning";
    next.consecutivePasses = 1;
    next.intervalDays = 0;
    next.nextReviewAt = now;
    return next;
  }

  next.consecutivePasses += 1;

  if (state.status === "learning") {
    if (inSession && next.consecutivePasses >= t.LEARNING_TO_REVIEWING_PASSES) {
      next.status = "reviewing";
      next.intervalDays = 1;
      next.nextReviewAt = addDays(now, 1);
    }
    return next;
  }

  // status: reviewing or mastered (a pass at mastered just keeps it mastered).
  next.ease = clamp(state.ease + t.EASE_PASS_BONUS, t.EASE_MIN, t.EASE_MAX);
  next.intervalDays = Math.max(1, state.intervalDays * next.ease);
  next.nextReviewAt = addDays(now, Math.round(next.intervalDays));

  if (
    state.status === "reviewing" &&
    next.intervalDays >= t.MASTERY_INTERVAL_DAYS &&
    next.consecutivePasses >= t.MASTERY_CONSECUTIVE_PASSES
  ) {
    next.status = "mastered";
  }
  return next;
}

export function applyFail(state: MasteryState, now: Date): MasteryState {
  const t = SCHEDULER_TUNABLES;
  const next: MasteryState = { ...state };
  next.totalAttempts += 1;
  next.lastAttemptAt = now;
  next.consecutivePasses = 0;
  next.ease = clamp(state.ease - t.EASE_FAIL_PENALTY, t.EASE_MIN, t.EASE_MAX);
  next.intervalDays = Math.max(0, state.intervalDays * t.FAIL_INTERVAL_DECAY);

  // Status drops one level: mastered → reviewing → learning. untouched
  // stays untouched on a fail (you cannot fail what you have not seen).
  if (state.status === "untouched") {
    return next;
  }
  if (state.status === "mastered") {
    next.status = "reviewing";
    next.nextReviewAt = addDays(now, Math.round(next.intervalDays || 1));
    return next;
  }
  if (state.status === "reviewing") {
    next.status = "learning";
    next.nextReviewAt = now;
    return next;
  }
  // status: learning — stay in learning, drop interval.
  next.nextReviewAt = now;
  return next;
}

export interface AttemptHistoryEntry {
  passed: boolean;
  submittedAt: Date;
}

export function rollingPassRate(history: AttemptHistoryEntry[]): number {
  const window = history.slice(-SCHEDULER_TUNABLES.ROLLING_WINDOW);
  if (window.length === 0) return 1; // empty history doesn't gate anything
  const passes = window.filter((a) => a.passed).length;
  return passes / window.length;
}

export type CandidateItem = {
  vocabId: string;
  state: MasteryState;
};

export function pickNextItem(
  candidates: CandidateItem[],
  history: AttemptHistoryEntry[],
  now: Date,
): CandidateItem | null {
  if (candidates.length === 0) return null;

  // 1. Review-due items (status != untouched, next_review_at <= now).
  const dueReviews = candidates
    .filter(
      (c) =>
        c.state.status !== "untouched" &&
        c.state.nextReviewAt &&
        c.state.nextReviewAt.getTime() <= now.getTime(),
    )
    .sort((a, b) => {
      // Most overdue first.
      const at = a.state.nextReviewAt?.getTime() ?? 0;
      const bt = b.state.nextReviewAt?.getTime() ?? 0;
      return at - bt;
    });
  if (dueReviews.length > 0) return dueReviews[0];

  // 2. Decide between introducing a new item vs prioritizing a struggling
  //    "learning" item, based on rolling pass rate.
  const passRate = rollingPassRate(history);
  const learning = candidates.filter((c) => c.state.status === "learning");
  const untouched = candidates.filter((c) => c.state.status === "untouched");

  if (passRate < SCHEDULER_TUNABLES.NEW_ITEM_PASS_RATE_FLOOR && learning.length > 0) {
    return learning[0];
  }
  if (untouched.length > 0) return untouched[0];
  if (learning.length > 0) return learning[0];

  // Everything mastered or reviewing-not-due.
  return null;
}

// -- Postgres row ↔ MasteryState conversion --

type Row = Database["public"]["Tables"]["mastery_state"]["Row"];

export function fromRow(row: Row): MasteryState {
  return {
    status: row.status as MasteryStatus,
    ease: row.ease,
    intervalDays: row.interval_days,
    nextReviewAt: row.next_review_at ? new Date(row.next_review_at) : null,
    consecutivePasses: row.consecutive_passes,
    totalAttempts: row.total_attempts,
    totalPasses: row.total_passes,
    lastAttemptAt: row.last_attempt_at ? new Date(row.last_attempt_at) : null,
  };
}

export function toRowUpdate(state: MasteryState): {
  status: MasteryStatus;
  ease: number;
  interval_days: number;
  next_review_at: string | null;
  consecutive_passes: number;
  total_attempts: number;
  total_passes: number;
  last_attempt_at: string | null;
} {
  return {
    status: state.status,
    ease: state.ease,
    interval_days: state.intervalDays,
    next_review_at: state.nextReviewAt?.toISOString() ?? null,
    consecutive_passes: state.consecutivePasses,
    total_attempts: state.totalAttempts,
    total_passes: state.totalPasses,
    last_attempt_at: state.lastAttemptAt?.toISOString() ?? null,
  };
}
