"use server";

// Sidebar progression query: all vocabulary items ordered by
// difficulty_rank, with each one tagged as "mastered" / "current" /
// "unlocked" / "locked" based on the learner's mastery state.
//
// Locking rule: a sign becomes available once the prior sign has been
// *passed* — i.e. reached `reviewing` (2 in-session passes) or `mastered`.
// (Full `mastered` needs 3 consecutive passes + a 7-day interval, which is a
// long-term goal, not a gate to the next word.) The first sign not yet
// passed (untouched/learning) in rank order is "current"; signs already
// reviewing/mastered are unlocked; everything after current is "locked."

import { createClient } from "@/lib/db/server";

export type VocabProgressState = "mastered" | "current" | "unlocked" | "locked";

export interface VocabProgressItem {
  id: string;
  displayGloss: string;
  difficultyRank: number | null;
  category: string;
  state: VocabProgressState;
}

export async function getVocabProgression(): Promise<VocabProgressItem[]> {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  const [vocabRes, masteryRes] = await Promise.all([
    supabase
      .from("vocabulary_items")
      .select("id, display_gloss, difficulty_rank, category")
      .eq("is_active_for_practice", true)
      .order("difficulty_rank", { ascending: true, nullsFirst: false }),
    user
      ? supabase.from("mastery_state").select("vocab_id, status").eq("user_id", user.id)
      : Promise.resolve({ data: [], error: null }),
  ]);

  const vocab = vocabRes.data ?? [];
  const masteryByVocab = new Map<string, string>(
    (masteryRes.data ?? []).map((m) => [m.vocab_id, m.status]),
  );

  const out: VocabProgressItem[] = [];
  let currentAssigned = false;

  for (const v of vocab) {
    const status = masteryByVocab.get(v.id);
    let state: VocabProgressState;

    if (status === "mastered") {
      // Long-term mastered.
      state = "mastered";
    } else if (status === "reviewing") {
      // Passed (≥2 in-session passes) — done enough to open the next sign.
      state = "unlocked";
    } else if (!currentAssigned) {
      // First sign not yet passed (untouched/learning) → the one to practice.
      state = "current";
      currentAssigned = true;
    } else {
      // Ranked after the current sign and not yet passed → locked.
      state = "locked";
    }

    out.push({
      id: v.id,
      displayGloss: v.display_gloss,
      difficultyRank: v.difficulty_rank,
      category: v.category ?? "other",
      state,
    });
  }

  return out;
}

// Helper used by the practice page when the user clicks a sidebar item:
// returns true iff the requested sign is available (current, already passed,
// or mastered). Locked signs cannot be practiced out of order.
export async function isSignUnlocked(signId: string): Promise<boolean> {
  const progression = await getVocabProgression();
  const item = progression.find((p) => p.id === signId);
  if (!item) return false;
  return item.state === "current" || item.state === "mastered" || item.state === "unlocked";
}
