"use server";

// Sidebar progression query: all vocabulary items ordered by
// difficulty_rank, with each one tagged as "mastered" / "current" /
// "unlocked" / "locked" based on the learner's mastery state.
//
// Locking rule: a sign becomes available once *all* signs ranked
// before it have reached `mastered`. The first un-mastered sign in
// rank order is "current"; everything after is "locked." If no
// mastery state exists yet, only rank-1 is unlocked.

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
  let lockedFromHere = false;
  let currentAssigned = false;

  for (const v of vocab) {
    const status = masteryByVocab.get(v.id);
    let state: VocabProgressState;

    if (status === "mastered") {
      state = "mastered";
    } else if (lockedFromHere) {
      state = "locked";
    } else if (!currentAssigned) {
      state = "current";
      currentAssigned = true;
      // Everything ranked after the current sign is locked until it's mastered.
      lockedFromHere = true;
    } else {
      // Shouldn't reach here because currentAssigned + !lockedFromHere is impossible,
      // but keep TS happy with an unreachable branch.
      state = "unlocked";
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
// returns true iff the requested sign is unlocked (mastered OR current)
// for the current user. Locked signs cannot be practiced out of order.
export async function isSignUnlocked(signId: string): Promise<boolean> {
  const progression = await getVocabProgression();
  const item = progression.find((p) => p.id === signId);
  if (!item) return false;
  return item.state === "current" || item.state === "mastered";
}
