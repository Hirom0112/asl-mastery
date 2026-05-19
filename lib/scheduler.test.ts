import { describe, expect, it } from "vitest";

import {
  applyFail,
  applyPass,
  DEFAULT_MASTERY,
  type MasteryState,
  pickNextItem,
  rollingPassRate,
  SCHEDULER_TUNABLES,
  type CandidateItem,
} from "./scheduler";

const NOW = new Date("2026-05-19T12:00:00Z");

describe("applyPass", () => {
  it("untouched → learning on first pass", () => {
    const out = applyPass(DEFAULT_MASTERY, NOW);
    expect(out.status).toBe("learning");
    expect(out.consecutivePasses).toBe(1);
    expect(out.totalAttempts).toBe(1);
    expect(out.totalPasses).toBe(1);
  });

  it("learning → reviewing after 2 in-session passes", () => {
    let s = applyPass(DEFAULT_MASTERY, NOW);
    s = applyPass(s, NOW, true);
    expect(s.status).toBe("reviewing");
    expect(s.intervalDays).toBe(1);
  });

  it("reviewing pass grows the interval by ease", () => {
    let s: MasteryState = { ...DEFAULT_MASTERY, status: "reviewing", intervalDays: 2, ease: 1.5 };
    s = applyPass(s, NOW);
    expect(s.intervalDays).toBeCloseTo(2 * (1.5 + SCHEDULER_TUNABLES.EASE_PASS_BONUS));
    expect(s.ease).toBeCloseTo(1.5 + SCHEDULER_TUNABLES.EASE_PASS_BONUS);
  });

  it("promotes to mastered when interval ≥ 7d and consecutive passes ≥ 3", () => {
    let s: MasteryState = {
      ...DEFAULT_MASTERY,
      status: "reviewing",
      intervalDays: 6,
      ease: 1.5,
      consecutivePasses: 2,
    };
    s = applyPass(s, NOW);
    expect(s.status).toBe("mastered");
    expect(s.consecutivePasses).toBe(3);
  });

  it("respects ease bounds", () => {
    let s: MasteryState = { ...DEFAULT_MASTERY, status: "reviewing", ease: 2.5, intervalDays: 5 };
    s = applyPass(s, NOW);
    expect(s.ease).toBeCloseTo(SCHEDULER_TUNABLES.EASE_MAX);
  });
});

describe("applyFail", () => {
  it("learning fail keeps status learning, decays interval by FAIL_INTERVAL_DECAY", () => {
    const s = applyFail(
      { ...DEFAULT_MASTERY, status: "learning", intervalDays: 1 } satisfies MasteryState,
      NOW,
    );
    expect(s.status).toBe("learning");
    expect(s.intervalDays).toBeCloseTo(SCHEDULER_TUNABLES.FAIL_INTERVAL_DECAY);
    expect(s.consecutivePasses).toBe(0);
  });

  it("reviewing fail drops to learning", () => {
    const s = applyFail(
      {
        ...DEFAULT_MASTERY,
        status: "reviewing",
        intervalDays: 5,
        ease: 1.6,
      } satisfies MasteryState,
      NOW,
    );
    expect(s.status).toBe("learning");
    expect(s.ease).toBeCloseTo(1.6 - SCHEDULER_TUNABLES.EASE_FAIL_PENALTY);
  });

  it("mastered fail drops to reviewing", () => {
    const s = applyFail(
      {
        ...DEFAULT_MASTERY,
        status: "mastered",
        intervalDays: 30,
        ease: 2.5,
      } satisfies MasteryState,
      NOW,
    );
    expect(s.status).toBe("reviewing");
  });
});

describe("rollingPassRate", () => {
  it("returns 1 on empty history", () => {
    expect(rollingPassRate([])).toBe(1);
  });
  it("uses only the last ROLLING_WINDOW attempts", () => {
    const history = Array.from({ length: SCHEDULER_TUNABLES.ROLLING_WINDOW + 5 }, (_, i) => ({
      passed: i % 2 === 0,
      submittedAt: NOW,
    }));
    const r = rollingPassRate(history);
    expect(r).toBeGreaterThanOrEqual(0);
    expect(r).toBeLessThanOrEqual(1);
  });
});

describe("pickNextItem", () => {
  function cand(vocabId: string, state: Partial<typeof DEFAULT_MASTERY>): CandidateItem {
    return { vocabId, state: { ...DEFAULT_MASTERY, ...state } };
  }
  const HISTORY_HIGH = Array.from({ length: 10 }, () => ({ passed: true, submittedAt: NOW }));
  const HISTORY_LOW = Array.from({ length: 10 }, () => ({ passed: false, submittedAt: NOW }));

  it("returns the most overdue review-due item first", () => {
    const earlier = new Date(NOW.getTime() - 3 * 86400_000);
    const later = new Date(NOW.getTime() - 1 * 86400_000);
    const picked = pickNextItem(
      [
        cand("a", { status: "reviewing", nextReviewAt: later }),
        cand("b", { status: "reviewing", nextReviewAt: earlier }),
      ],
      HISTORY_HIGH,
      NOW,
    );
    expect(picked?.vocabId).toBe("b");
  });

  it("with high pass rate and no due reviews, introduces an untouched item", () => {
    const picked = pickNextItem(
      [cand("x", { status: "untouched" }), cand("y", { status: "learning" })],
      HISTORY_HIGH,
      NOW,
    );
    expect(picked?.vocabId).toBe("x");
  });

  it("with low pass rate, prioritizes a struggling learning item over a new one", () => {
    const picked = pickNextItem(
      [cand("u", { status: "untouched" }), cand("l", { status: "learning" })],
      HISTORY_LOW,
      NOW,
    );
    expect(picked?.vocabId).toBe("l");
  });

  it("returns null when everything is mastered or scheduled in the future", () => {
    const tomorrow = new Date(NOW.getTime() + 86400_000);
    const picked = pickNextItem(
      [
        cand("m", { status: "mastered", nextReviewAt: tomorrow }),
        cand("r", { status: "reviewing", nextReviewAt: tomorrow }),
      ],
      HISTORY_HIGH,
      NOW,
    );
    expect(picked).toBeNull();
  });
});
